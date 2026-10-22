"""
play_nets.py -- SEVERAL independent Wong-Wang(+roaming+instance/LCA) networks playing Doom at once.

   MOVE  net : LEFT vs RIGHT      ONE net for both phases.  No enemy in view = SEARCH phase (arrow HELD down = spinning,
                                  decisions only choose/reverse the side).  Enemy in view = TARGET phase (spin stops,
                                  precision turning: short arrow pulses, many decisions).
   SHOOT net : no-shoot vs SHOOT  (SHOOT fires)
   Each net is its OWN 2-pool decision module: its own Wong-Wang attractor, its own critical-roaming u0 chain,
   its own instance-retrieval -> LCA fast route, racing the WW route; first to threshold wins.

   SEARCH phase (same MOVE net): the arrow is held down from the first tic; the net decides which way to spin and the
     committed arrow stays down until an enemy is identified, which stops the spin at once.  It re-decides every
     --search-redecide ms (so it can reverse).  Its evidence is only a weak memory: if the last enemy left through
     the left/right edge of the view, the logits lean that way (--search-ev); otherwise they are 0 and the choice
     is driven by noise / roaming alone (a zero-evidence decision, often slow or timed out).  On a timeout the
     agent keeps its current spin, or flips a coin if it has none, so it never stands still for long.
   MOVE / SHOOT: actions are discrete events.  MOVE commits LEFT/RIGHT -> the arrow is pressed for
     ceil(|dx| / --turn-gain) tics (at most --turn-tics, at least 1; released early inside --deadband) so that
     big offsets get long pulses and small ones get fine corrections.  --turn-gain (dx per tic) is a GUESS at
     the game's turn rate: lower it if the agent overshoots, raise it if it creeps.  SHOOT commits SHOOT ->
     fire for --fire-tics tics; no-shoot or a timeout does nothing.  Next epoch starts after the action ends.

Evidence (frozen at the start of each net's epoch; al = min(|dx|/aim_tol, 1); ev = --ev logit scale)
   move   : [LEFT, RIGHT]     = ev * min(|dx|/move_scale, 1) on the side the enemy is on
   shoot  : [no-shoot, SHOOT] = ev * [al, 1-al]
   search : [LEFT, RIGHT]     = search_ev * min(|mem|/move_scale, 1) toward the side the last enemy left by
Without a SHOOT net the fire button follows a hard rule (|dx| < aim_tol); without a MOVE net the turn follows
the enemy by a hard rule; the initial spin side / timeout fallback is --scan.

Watch window (pygame):  per net, 7 live panels
   WW rates | LCA activation | RACE (WW vs LCA progress to threshold) | evidence / WW input (pA) / LCA drive |
   roaming u0 over decisions vs critical u0* | effective attractor landscape (ball = network state) | RT history.
   The previous decision's plots STAY on screen until the next decision's first trace sample is drawn.
   keys: SPACE pause  1/2/3 tier  +/- speed  S screenshot  ESC quit
   A SCREENSHOT of the whole dashboard is saved the moment any net commits a decision, into a new directory
   (--shots-dir, default shots_<timestamp>/; --no-shots to disable).  Window mode only.
   The decision simulation runs in a worker thread in the window so a net starting an epoch cannot stall a frame
   (--sync turns that off).  Use --speed 0.25 (or "-") to watch the traces unfold.

CSV (one row per decision): tic, net, tier, dx, ideal, resp, match, rt_ms, route, u0 (the committing net's roaming u0),
   timeout, phase (search/target), ep, kills, health, u0_<net> for EVERY net (that net's latest roaming u0 at that moment), and shot (the
   screenshot file for that decision, window mode).

WEAKER / SLOWER ENEMIES (UNTESTED): --demon-speed N --demon-health N [--shooter-health N] write slow_enemies.pk3 (a
   DECORATE mod replacing Demon/Spectre, and ZombieMan/ShotgunGuy/DoomImp health) and load it with -file.  Doom defaults:
   Demon speed 10 health 150; ZombieMan health 20.  Try --demon-speed 3 --demon-health 40.  If enemies vanish, drop the flags.

Examples
   python play_nets.py --speed 0.25                           # watch window, nets move(+search)+shoot, shots saved
   python play_nets.py --compare oracle,ww,ww_cr,ww_cr_lca --episodes 5 --skill 1     # headless table + CSV
   python play_nets.py --png dash.png --snap 200,500 --fake-env                       # static dashboard images
   (--fake-env = toy world, no ViZDoom.)

The landscape is an EFFECTIVE 1-D reduction (flow along the slow manifold ds/dt = 0 of the 2-pool reduced model,
x = S1-S2, s = S1+S2, tilted by the current inputs and u0); it ignores noise and the LCA route.  Presets in
sp2_nway.NW_PRESETS were hand-tuned for 5-way recognition, not this task: tune with --u0 --gain --noise --ter.
Only the --fake-env and headless paths were run by me; pygame and real ViZDoom are untested on my side.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import ww_npool as W
from sp2_agents import U0_CRIT
from sp2_nway import NWayAgent, NW_TIERS
from play_ww import (FakeDefendEnv, DoomDefendEnv, BaselineAgent, BASELINES, pick_target, TICS)

LCA_Z = 1.0
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=4)
    return _POOL


COLS = [(90, 160, 255), (255, 170, 60)]          # pool 0 / pool 1
WWC, LCAC = (90, 160, 255), (255, 150, 40)
BG, PANEL, DIM = (18, 18, 24), (30, 30, 40), (150, 150, 160)


# ------------------------------------------------------------------------------------------- evidence
def ev_move(dx, a):
    m = min(abs(dx) / a.move_scale, 1.0)
    return np.array([a.ev * m * (dx < 0), a.ev * m * (dx >= 0)])


def ev_shoot(dx, a):
    al = min(abs(dx) / a.aim_tol, 1.0)
    return np.array([a.ev * al, a.ev * (1.0 - al)])


def ev_search(mem, a):
    """mem = signed side (-1 left .. +1 right) the last enemy left by, or 0."""
    m = min(abs(mem) / a.move_scale, 1.0)
    return np.array([a.search_ev * m * (mem < 0), a.search_ev * m * (mem >= 0)])


SPECS = {"move": (("LEFT", "RIGHT"), ev_move), "shoot": (("no-shoot", "SHOOT"), ev_shoot)}


# ------------------------------------------------------------------------------------------- landscape
def landscape(u0, coh, n=61, xm=0.9):
    """Effective 1-D landscape along x = S1-S2 on the slow manifold (ds/dt = 0, s = S1+S2) of the 2-pool
    reduced WW model.  Returns (x, U) with U in arbitrary units, minimum 0."""
    st = W.J_EXT * u0 * (1.0 + np.asarray(coh, float))
    x = np.linspace(-xm, xm, n)[:, None]
    s = np.linspace(0, 2, 161)[None, :]
    S1, S2 = (s + x) / 2, (s - x) / 2
    ok = (S1 >= 0) & (S2 >= 0) & (S1 <= 1) & (S2 <= 1)
    wl = W.W_LAT_TOTAL

    def flow(S1, S2):
        I1 = W.I_BG + st[0] + W.W_SELF * S1 + wl * S2
        I2 = W.I_BG + st[1] + W.W_SELF * S2 + wl * S1
        return (-S1 / W.TAU_S + W.GAMMA * (1 - S1) * W.phi(I1) / 1000.0,
                -S2 / W.TAU_S + W.GAMMA * (1 - S2) * W.phi(I2) / 1000.0)

    f1, f2 = flow(S1, S2)
    g = np.where(ok, f1 + f2, np.nan)
    cross = np.isfinite(g[:, :-1]) & np.isfinite(g[:, 1:]) & (g[:, :-1] > 0) & (g[:, 1:] <= 0)
    xs, vs = [], []
    for i in range(n):
        if not cross[i].any():
            continue
        j = int(np.flatnonzero(cross[i])[-1])         # highest-activity stable crossing of the slow manifold
        sst = s[0, j] + (s[0, j + 1] - s[0, j]) * g[i, j] / (g[i, j] - g[i, j + 1])
        a1, a2 = flow((sst + x[i, 0]) / 2, (sst - x[i, 0]) / 2)
        xs.append(x[i, 0]); vs.append((a1 - a2) * 1000.0)
    if len(xs) < 5:
        return np.linspace(-0.5, 0.5, 21), np.zeros(21)
    xs, vs = np.array(xs), np.array(vs)
    U = -np.concatenate([[0], np.cumsum(0.5 * (vs[1:] + vs[:-1]) * np.diff(xs))])
    return xs, U - U.min()


# ------------------------------------------------------------------------------------------- one network
class Net:
    def __init__(self, key, tier, params, seed, a):
        self.key, self.a, self.tier = key, a, tier
        self.labels, self.evfn = SPECS[key]
        p = dict(params or {})
        p.update(p.pop(key, {}) if isinstance(p.get(key), dict) else {})
        p = {k: v for k, v in p.items() if not isinstance(v, dict)}
        self.agent = BaselineAgent(tier) if tier in BASELINES else NWayAgent(tier, p, seed=seed, n=2)
        self.rng = np.random.default_rng(seed + 991)
        self.u0_hist, self.rt_hist = deque(maxlen=60), deque(maxlen=60)
        self.shown = None                                # epoch currently drawn live by the dashboard
        self.last = None                                 # last COMMITTED epoch: kept on screen until a new one draws
        self.spin = None                                 # MOVE net, search phase: arrow held down (0 = left, 1 = right)
        self.mode = "target"                             # MOVE net: "target" (enemy in view) or "search"
        self.reset()

    def reset(self):
        self.out, self.hold, self.ep, self.cool, self._busy = None, 0, None, 0, None

    def idle(self):
        if self.ep is not None and self.ep is self.shown:   # an epoch already on screen is aborted: keep it displayed
            self.ep["aborted"] = True                      # (frozen, marked 'interrupted') instead of snapping back to the old one
            self.last = self.ep
        self.out, self.hold, self.ep, self.cool = None, 0, None, 0

    @property
    def act(self):
        """The choice whose action (arrow / fire pulse) is playing out right now, else None."""
        return self.out if self.hold > 0 else None

    @property
    def u0_now(self):
        return self.u0_hist[-1] if self.u0_hist else None

    def _compute(self, lg):
        dec = self.agent.decide(lg, timeout_ms=self.a.timeout)
        land = landscape(dec["u0"], dec["coh"]) if (self.a.landscape and dec.get("coh") is not None) else None
        return dec, land

    def _fill(self, ep, res):
        dec, land = res
        ep.update(dec=dec, land=land, n_tic=max(1, math.ceil(dec["rt_ms"] * TICS / 1000.0)))
        self.u0_hist.append(dec["u0"])

    def _start(self, tic, x, search=False):
        lg = ev_search(x, self.a) if search else self.evfn(x, self.a)
        self.ep = dict(start=tic, n_tic=None, dec=None, logits=lg, dx=x, land=None, fut=None, search=search)
        if self.a.async_dec:
            if self._busy is not None:                 # never run two decide() calls on one agent at once
                self._busy.result()
            self.ep["fut"] = self._busy = _pool().submit(self._compute, lg)
        else:
            self._fill(self.ep, self._compute(lg))

    def tick(self, tic, tgt, search=False):
        """Advance one game tic.  tgt["dx"] is the enemy offset, or (search=True, MOVE net only) the search memory.
        Returns a log row when this net commits a decision."""
        a, x = self.a, tgt["dx"]
        if self.hold > 0:
            self.hold -= 1
            if self.hold == 0:
                self.out = None
        if self.key == "move" and not search and self.hold > 0 and abs(x) <= a.deadband:
            self.hold, self.out = 0, None             # aligned: release the arrow early, no overshoot
        if self.ep is None:
            if self.cool > 0:
                self.cool -= 1
            elif search or not (self.key == "move" and abs(x) <= a.deadband):   # aligned: nothing to decide
                self._start(tic, x, search)
            return None
        e = self.ep
        if e["dec"] is None:                           # async: simulation still running in the worker
            if not e["fut"].done():
                return None
            self._fill(e, e["fut"].result())
        if tic - e["start"] < e["n_tic"]:
            return None
        dec = e["dec"]
        if search:
            if not dec["timeout"]:
                self.spin = dec["choice"]              # the held arrow flips; a timeout just keeps spinning as before
        elif not dec["timeout"] and not (self.key == "shoot" and dec["choice"] == 0):
            self.out = dec["choice"]
            if self.key == "shoot":
                self.hold = a.fire_tics
            else:                                      # precision turning: pulse length follows the offset
                self.hold = max(1, min(a.turn_tics, math.ceil(abs(x) / a.turn_gain)))
        self.rt_hist.append((dec["rt_ms"], dec["route"], dec["timeout"]))
        self.last, self.ep = e, None
        self.cool = a.refractory + (a.search_redecide_tics if search else self.hold)
        informative = float(np.max(e["logits"])) > 0
        ideal = int(np.argmax(e["logits"])) if informative else -1
        return dict(tic=tic, net=self.key, phase="search" if search else "target", tier=self.tier,
                    dx=round(float(e["dx"]), 3),
                    ideal=self.labels[ideal] if informative else "n/a",
                    resp="none" if dec["timeout"] else self.labels[dec["choice"]],
                    match=(int((not dec["timeout"]) and dec["choice"] == ideal) if informative else float("nan")),
                    rt_ms=round(dec["rt_ms"], 1), route=dec["route"] or "none", u0=round(dec["u0"], 3),
                    timeout=int(dec["timeout"]))

    def elapsed_ms(self, tic):
        return (tic - self.ep["start"]) * 1000.0 / TICS if self.ep else None


def make_nets(tier, a, params, seed):
    return {k: Net(k, tier, params, seed + 17 * i, a) for i, k in enumerate(a.nets)}


def buttons(nets, tgt, a):
    """Combine the MOVE/SHOOT nets' playing-out actions into (left, right, shoot) while an enemy is in view."""
    if "move" in nets:
        mv = nets["move"].act
    else:
        mv = None if abs(tgt["dx"]) <= a.deadband else (0 if tgt["dx"] < 0 else 1)
    S = (nets["shoot"].act == 1) if "shoot" in nets else abs(tgt["dx"]) < a.aim_tol
    return mv == 0, mv == 1, bool(S)


def play_episode(env, nets, a, rows, ep):
    """Generator, one event per game tic (headless runs drain it; the window draws each event)."""
    obs = env.reset(a.seed * 1000 + ep)
    for n in nets.values():
        n.reset()
    rng = np.random.default_rng(a.seed * 1000 + ep + 7)
    spin = {"left": 0, "right": 1}.get(a.scan, int(rng.integers(2)))     # the arrow that is held down while searching
    mv = nets.get("move")
    if mv is not None:
        mv.spin, mv.mode = spin, "target"
    mem, prev_dx = 0.0, None
    last_tgt, miss = None, 0                           # grace: a target missing for a few tics is NOT 'lost' (labels flicker)
    tic = 0

    def log(row, commits):
        inf = env.info()
        row.update(ep=ep, kills=inf["kills"], health=inf["health"])
        for k, n in nets.items():                       # every network's current roaming u0 at this decision
            row[f"u0_{k}"] = "" if n.u0_now is None else round(n.u0_now, 3)
        rows.append(row); commits.append(row)

    while not env.done and tic < a.max_tics:
        frame, enemies = obs
        tgt = pick_target(enemies)
        if tgt is not None:
            last_tgt, miss = tgt, 0
        elif last_tgt is not None and miss < a.lost_grace:
            tgt, miss = last_tgt, miss + 1
        else:
            last_tgt = None
        commits = []
        if tgt is None:
            if prev_dx is not None:                    # the enemy just left the view: remember which edge it left by
                mem = prev_dx if abs(prev_dx) > 0.5 else 0.0
                prev_dx = None
            if "shoot" in nets:
                nets["shoot"].idle()
            if mv is not None:
                if mv.mode != "search":                # target lost: abandon any pulse / epoch, resume the held spin
                    mv.idle(); mv.mode = "search"
                row = mv.tick(tic, dict(dx=mem), search=True)   # the SAME net keeps deciding, but never blocks the spin
                if row:
                    log(row, commits)
                spin = mv.spin
            L, R, S = spin == 0, spin == 1, False
        else:
            prev_dx = tgt["dx"]
            if mv is not None and mv.mode != "target":
                mv.idle(); mv.mode = "target"          # identified: the spin stops, precision targeting begins
            for k, n in nets.items():
                row = n.tick(tic, tgt)
                if row:
                    log(row, commits)
            L, R, S = buttons(nets, tgt, a)
        env.step_buttons(L, R, S)
        obs = env.observe()
        tic += 1
        yield dict(frame=obs[0], enemies=obs[1], target=pick_target(obs[1]), tic=tic, ep=ep, info=env.info(),
                   btn=(L, R, S), nets=nets, commits=commits)


# ------------------------------------------------------------------------------------------- canvases
SIZES = {"s": 11, "m": 13, "l": 16}


class PILCanvas:
    def __init__(self, w, h):
        from PIL import Image, ImageDraw, ImageFont
        self.Image = Image
        self.im = Image.new("RGB", (w, h), BG)
        self.d = ImageDraw.Draw(self.im)
        try:
            self.f = {k: ImageFont.load_default(size=v) for k, v in SIZES.items()}
        except TypeError:
            self.f = {k: ImageFont.load_default() for k in SIZES}

    def fill(self, col):
        self.d.rectangle([0, 0, *self.im.size], fill=col)

    def rect(self, x, y, w, h, col, width=0):
        if width:
            self.d.rectangle([x, y, x + w - 1, y + h - 1], outline=col, width=width)
        else:
            self.d.rectangle([x, y, x + w - 1, y + h - 1], fill=col)

    def line(self, p0, p1, col, width=1):
        self.d.line([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))], fill=col, width=width)

    def lines(self, pts, col, width=1):
        if len(pts) > 1:
            self.d.line([(float(p[0]), float(p[1])) for p in pts], fill=col, width=width)

    def circle(self, c, r, col):
        self.d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col)

    def text(self, s, x, y, col=(230, 230, 230), size="s"):
        self.d.text((x, y), s, fill=col, font=self.f[size])

    def image(self, arr, x, y, w, h):
        self.im.paste(self.Image.fromarray(np.ascontiguousarray(arr)).resize((w, h)), (x, y))

    def save(self, fn):
        self.im.save(fn)


class PGCanvas:
    def __init__(self, surf, pg):
        self.s, self.pg = surf, pg
        self.f = {k: pg.font.SysFont("consolas", v) for k, v in SIZES.items()}

    def fill(self, col):
        self.s.fill(col)

    def rect(self, x, y, w, h, col, width=0):
        self.pg.draw.rect(self.s, col, (x, y, w, h), width)

    def line(self, p0, p1, col, width=1):
        self.pg.draw.line(self.s, col, p0, p1, width)

    def lines(self, pts, col, width=1):
        if len(pts) > 1:
            self.pg.draw.lines(self.s, col, False, [(float(p[0]), float(p[1])) for p in pts], width)

    def circle(self, c, r, col):
        self.pg.draw.circle(self.s, col, (int(c[0]), int(c[1])), r)

    def text(self, s, x, y, col=(230, 230, 230), size="s"):
        self.s.blit(self.f[size].render(s, True, col), (x, y))

    def image(self, arr, x, y, w, h):
        surf = self.pg.surfarray.make_surface(np.transpose(np.ascontiguousarray(arr), (1, 0, 2)))
        self.s.blit(self.pg.transform.scale(surf, (w, h)), (x, y))


class Ax:
    """A small plot area mapping data coordinates to pixels."""

    def __init__(self, c, x, y, w, h, xlim, ylim, title="", bg=PANEL):
        c.rect(x, y, w, h, bg)
        self.c, self.x, self.y, self.w, self.h, self.xl, self.yl = c, x, y, w, h, xlim, ylim
        if title:
            c.text(title, x + 3, y + 1, DIM)

    def X(self, v):
        v = np.clip(np.asarray(v, float), *self.xl)
        return self.x + self.w * (v - self.xl[0]) / (self.xl[1] - self.xl[0])

    def Y(self, v):
        v = np.clip(np.asarray(v, float), *self.yl)
        return self.y + self.h - self.h * (v - self.yl[0]) / (self.yl[1] - self.yl[0])

    def plot(self, xs, ys, col, width=1):
        if len(xs) > 1:
            self.c.lines(np.column_stack([self.X(xs), self.Y(ys)]), col, width)

    def hline(self, v, col):
        yy = float(self.Y(v)); self.c.line((self.x, yy), (self.x + self.w, yy), col)

    def vline(self, v, col):
        xx = float(self.X(v)); self.c.line((xx, self.y), (xx, self.y + self.h), col)

    def dot(self, xv, yv, col, r=3):
        self.c.circle((float(self.X(xv)), float(self.Y(yv))), r, col)


# ------------------------------------------------------------------------------------------- drawing
def draw_net(c, net, tic, x0, y0, w, a):
    # The previous decision's plots stay up until the NEW decision has its first trace sample to draw.
    cur = net.ep if (net.ep is not None and net.ep["dec"] is not None) else None
    el_cur = net.elapsed_ms(tic) if cur is not None else None
    started = cur is not None and (el_cur - cur["dec"].get("ter", 0.0)) > 0
    if started:
        net.shown = cur
    ep = cur if started else net.last
    live = started
    dec = ep["dec"] if ep else None
    T = a.timeout
    el = el_cur if live else None
    name = {"move": "MOVE + SEARCH (turn L/R)", "shoot": "SHOOT (fire?)"}[net.key]
    c.text(f"{name}  [{net.tier}]", x0, y0, (240, 240, 240), "m")
    if net.key == "move" and net.mode == "search" and net.spin is not None:
        acting = "spin " + net.labels[net.spin]
    else:
        acting = "-" if net.act is None else net.labels[net.act]
    if dec is None:
        c.text("waiting ..." if net.ep is None else "deciding ...", x0, y0 + 18, DIM)
        c.text(f"acting: {acting}", x0, y0 + 31, (200, 200, 210))
    else:
        ch = "timeout" if dec["timeout"] else net.labels[dec["choice"]]
        if live:
            state = f"deciding {el:4.0f}/{dec['rt_ms']:.0f} ms"
        elif net.ep is not None:
            state = f"deciding ... (last: {ch})"
        elif ep.get("aborted"):
            state = "interrupted (target seen/lost)"
        else:
            state = f"committed {ch}"
        c.text(state, x0, y0 + 18, (200, 200, 210))
        c.text(f"route {dec['route'] or '-'}   acting: {acting}", x0, y0 + 31, (200, 200, 210))
    lab = net.labels
    y = y0 + 46
    pl_h = 100

    # ---- WW rates
    ax = Ax(c, x0, y, w, pl_h, (0, T), (0, 25), "WW pool rates (Hz)  thr 15")
    ax.hline(W.THR_HZ, (140, 60, 60))
    ter = dec.get("ter", 0.0) if dec else 0.0
    vis = (el if live else 1e9)
    if dec is not None and dec.get("ww_r") is not None and len(dec["ww_r"]) > 1:
        r = dec["ww_r"]
        m = int(min(len(r), max(0.0, vis - ter)))
        for k in range(r.shape[1]):
            ax.plot((np.arange(1, m + 1) + ter)[::6], r[:m, k][::6], COLS[k], 2)
        if live:
            ax.vline(el, (200, 200, 200))
    for k in range(2):
        c.text(lab[k], x0 + w - 62, y + 2 + 11 * k, COLS[k])
    y += pl_h + 4

    # ---- LCA
    ax = Ax(c, x0, y, w, 80, (0, T), (0, 1.3), "instance -> LCA   thr 1")
    ax.hline(LCA_Z, (140, 60, 60))
    if dec is not None:
        if dec.get("lca_onset") is not None:
            ax.vline(ter + dec["lca_onset"], (110, 110, 60))
        lx = dec.get("lca_x")
        if lx is not None and len(lx) > 1:
            t_end = (dec["dt"] if dec.get("dt") is not None else T)
            steps = len(lx)
            tt = ter + dec["lca_t0"] + np.arange(steps)
            keep = (tt <= ter + t_end) & (tt <= vis)
            for k in range(lx.shape[1]):
                ax.plot(tt[keep], lx[keep, k], COLS[k], 2)
        elif dec.get("lca_onset") is None and net.tier == "ww_cr_lca":
            c.text("(no retrieval)", x0 + 6, y + 30, DIM)
    y += 84

    # ---- RACE
    ax = Ax(c, x0, y, w, 80, (0, T), (0, 1.15), "RACE  WW (blue) vs LCA (orange)")
    ax.hline(1.0, (140, 60, 60))
    if dec is not None and dec.get("ww_r") is not None and len(dec["ww_r"]) > 1:
        r = dec["ww_r"]
        pw = r.max(1) / W.THR_HZ
        m = int(min(len(pw), max(0.0, vis - ter)))
        ax.plot((np.arange(1, m + 1) + ter)[::6], pw[:m][::6], WWC, 2)
        lx = dec.get("lca_x")
        if lx is not None and len(lx) > 1:
            tt = ter + dec["lca_t0"] + np.arange(len(lx))
            t_end = (dec["dt"] if dec.get("dt") is not None else T)
            keep = (tt <= ter + t_end) & (tt <= vis)
            ax.plot(tt[keep], lx.max(1)[keep] / LCA_Z, LCAC, 2)
        if (not live or el >= dec["rt_ms"]) and not dec["timeout"]:
            ax.dot(dec["rt_ms"], 1.0, LCAC if dec["route"] == "lca" else WWC, 5)
            c.text("LCA wins" if dec["route"] == "lca" else "WW wins", x0 + w - 62, y + 2,
                   LCAC if dec["route"] == "lca" else WWC)
        elif not live and dec["timeout"]:
            c.text("timeout", x0 + w - 50, y + 2, DIM)
    y += 84

    # ---- inputs
    c.rect(x0, y, w, 66, PANEL)
    c.text("evidence | I_ext pA | LCA drive", x0 + 3, y + 1, DIM)
    if dec is not None:
        groups = []
        lg = ep["logits"]
        groups.append(("logit", lg, a.ev, "{:.0f}"))
        if dec.get("coh") is not None:
            ext = W.J_EXT * dec["u0"] * (1.0 + np.asarray(dec["coh"])) * 1000.0
            groups.append(("I_ext pA", ext, 40.0, "{:.1f}"))
        if dec.get("lca_rho") is not None:
            groups.append(("rho", np.asarray(dec["lca_rho"]), 1.5, "{:.2f}"))
        gw = w / 3.0
        for gi, (nm, vals, vmax, fmt) in enumerate(groups):
            gx = x0 + gi * gw
            for k in range(2):
                bh = 34 * float(np.clip(vals[k] / vmax, 0, 1))
                c.rect(gx + 12 + 52 * k, y + 52 - bh - 6, 22, max(1, bh), COLS[k])
                c.text(fmt.format(vals[k]), gx + 6 + 52 * k, y + 52 - 6, (200, 200, 200))
            c.text(nm, gx + 6, y + 14, DIM)
    y += 70

    # ---- roaming u0
    ax = Ax(c, x0, y, w, 80, (0, 60), (0, 33), f"roaming u0 / decision  u0*={U0_CRIT:.2f}")
    ax.hline(U0_CRIT, (220, 90, 90))
    hist = list(net.u0_hist)
    if hist:
        xs = np.arange(len(hist)) + (60 - len(hist))
        ax.plot(xs, hist, (170, 170, 190), 1)
        for i, u in enumerate(hist):
            ax.dot(xs[i], u, (120, 220, 120) if u > U0_CRIT else (230, 90, 90), 2)
        ax.dot(xs[-1], hist[-1], (255, 255, 255), 4)
    y += 84

    # ---- landscape
    ax = Ax(c, x0, y, w, 110, (-0.9, 0.9), (0, 1), "attractor landscape  x=S1-S2")
    if ep is not None and ep.get("land") is not None:
        xs, U = ep["land"]
        top = max(U.max(), 1e-9)
        ax.xl = (float(xs[0]), float(xs[-1]))
        ax.plot(xs, U / top * 0.8 + 0.08, (200, 200, 120), 2)
        S = dec.get("ww_S")
        if S is not None and len(S):
            i = int(max(0.0, vis - ter)) if live else len(S)
            i = max(0, min(len(S) - 1, i - 1))
            xb = float(S[i, 0] - S[i, 1])
            yb = float(np.interp(xb, xs, U / top * 0.8 + 0.08))
            ax.dot(xb, yb + 0.04, LCAC if (dec["route"] == "lca" and not live) else WWC, 6)
        c.text(lab[1] + " <", x0 + 4, y + 96, COLS[1]); c.text("> " + lab[0], x0 + w - 70, y + 96, COLS[0])
    y += 114

    # ---- RT history
    ax = Ax(c, x0, y, w, 70, (0, 60), (0, T), "RT / decision (ms) orange=LCA")
    for i, (rt, route, tmo) in enumerate(net.rt_hist):
        ax.dot(i + (60 - len(net.rt_hist)), rt, (110, 110, 110) if tmo else (LCAC if route == "lca" else WWC), 3)
    return y + 70


def draw_dashboard(c, S, a, W_, H_):
    c.fill(BG)
    tic = S["tic"]
    c.text(f"Wong-Wang decision networks on Doom   tier {S['tier']}   ep {S['ep']+1}   tic {tic}   "
           f"kills {S['info']['kills']}   health {S['info']['health']}", 10, 10, (240, 240, 240), "m")
    c.image(S["frame"], 10, 40, 620, 465)
    tg = S["target"]
    c.line((320, 262), (320, 282), (255, 255, 255)); c.line((310, 272), (330, 272), (255, 255, 255))
    for en in S["enemies"]:                                   # every recognised enemy: dim box + name; target: bright
        cx = 10 + 310 * (1 + en["dx"]); cy = 40 + 465 * en.get("cy", 0.5)
        bw, bh = max(8, 620 * en["w"]), max(10, 465 * en["h"])
        is_t = en is tg
        col = (255, 220, 60) if is_t else (120, 120, 140)
        c.rect(int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh), col, 2 if is_t else 1)
        c.text(en.get("name", "?"), int(cx - bw / 2), max(42, int(cy - bh / 2) - 13), col)
    L, R, Sh = S["btn"]
    for i, (nm, on, col) in enumerate((("LEFT", L, (90, 160, 255)), ("RIGHT", R, (90, 160, 255)), ("FIRE", Sh, (255, 90, 60)))):
        c.rect(10 + 100 * i, 512, 92, 26, col if on else PANEL)
        c.text(nm, 22 + 100 * i, 518, (255, 255, 255) if on else DIM, "m")
    c.text("buttons over time", 10, 545, DIM)
    for r_i, key in enumerate((0, 1, 2)):
        yy = 560 + 14 * r_i
        c.rect(10, yy, 620, 12, PANEL)
        for j, bt in enumerate(S["tl"]):
            if bt[key]:
                c.rect(10 + 2 * j, yy + 1, 2, 10, (255, 90, 60) if key == 2 else (90, 160, 255))
    c.text("L", 634, 559, DIM); c.text("R", 634, 573, DIM); c.text("F", 634, 587, DIM)
    c.text("decision log", 10, 612, DIM)
    for i, ln in enumerate(S["log"]):
        c.text(ln, 10, 628 + 14 * i, (200, 200, 210))
    k = len(S["nets"])
    x0, cw = 660, (W_ - 670) // k
    for i, net in enumerate(S["nets"].values()):
        draw_net(c, net, tic, x0 + i * cw, 40, cw - 12, a)
    c.text("SPACE pause  1/2/3 tier  +/- speed  S screenshot  ESC quit", 10, H_ - 18, DIM)


def log_line(row):
    return (f"tic {row['tic']:4d} {row['net']:5s} -> {row['resp']:6s} RT {row['rt_ms']:5.0f} "
            f"{row['route']:4s} u0 {row['u0']:5.1f} dx {row['dx']:+.2f}")


# ------------------------------------------------------------------------------------------- runners
def build_mod(a):
    """Write slow_enemies.pk3 (zip with a DECORATE lump) in the cwd; return its relative name or None."""
    if a.fake_env or not (a.demon_speed or a.demon_health or a.shooter_health or not a.keep_melee):
        return None
    import zipfile
    def actor(base, speed, health):
        props = (f"Speed {speed} " if speed else "") + (f"Health {health} " if health else "")
        return f"ACTOR Slow{base} : {base} replaces {base}\n{{\n  {props}\n}}\n" if props else ""
    txt = ""
    if a.keep_melee:
        txt += actor("Demon", a.demon_speed, a.demon_health) + actor("Spectre", a.demon_speed, a.demon_health)
    else:                                              # every melee swarmer is spawned as a zombieman (gun guy) instead
        for b in ("Demon", "Spectre", "MarineChainsawVzd"):
            txt += f"ACTOR GunGuy{b} : ZombieMan replaces {b}\n{{\n}}\n"
    for b in ("ZombieMan", "ShotgunGuy", "DoomImp"):
        txt += actor(b, 0, a.shooter_health)
    name = "slow_enemies.pk3"
    with zipfile.ZipFile(name, "w") as z:
        z.writestr("decorate.txt", txt)
    print(f"[mod] wrote {name}: demon speed {a.demon_speed or 'default'} health {a.demon_health or 'default'}, "
          f"shooter health {a.shooter_health or 'default'} (UNTESTED -- drop the flags if enemies vanish)", flush=True)
    return name


def make_env(a, seed):
    if a.fake_env:
        return FakeDefendEnv(seed, a.max_tics)
    env = DoomDefendEnv(seed, a.scenario, a.enemy_names.split(",") if a.enemy_names else None,
                        not a.no_corpse_filter, a.max_tics, a.skill, mod=build_mod(a))
    env.verbose_labels = a.list_labels
    env.melee = a.melee
    env.god, env.ammo = a.god, a.infinite_ammo
    return env


def load_params(a):
    p = json.load(open(a.params)) if a.params else {}
    for k in ("u0", "gain", "noise"):
        if getattr(a, k) is not None:
            p[k] = getattr(a, k)
    if a.ter is not None:
        p["ter_mu"] = a.ter
    return p


def write_rows(rows, path):
    """CSV with the union of all keys (rows from different nets carry different u0_<net> columns)."""
    import csv
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} decisions -> {os.path.abspath(path)}")


def _mean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def run_headless(a, tiers, params):
    env = make_env(a, a.seed)
    rows, eps, lr = [], {}, {}
    for ti, tier in enumerate(tiers):
        nets = make_nets(tier, a, params, a.seed + 100 * ti)
        eps[tier], lr[tier] = [], [0, 0, 0, 0]
        for ep in range(a.episodes):
            n0 = len(rows)
            for ev in play_episode(env, nets, a, rows, ep):
                b = ev["btn"]
                lr[tier][0] += b[0]; lr[tier][1] += b[1]; lr[tier][2] += b[2]; lr[tier][3] += 1
            for r in rows[n0:]:
                r["tier"] = tier
            inf = env.info()
            eps[tier].append((inf["kills"], inf["tic"]))
            print(f"  {tier:9s} ep {ep+1}/{a.episodes}: kills {inf['kills']:3d} health {inf['health']:4d} tics {inf['tic']:4d}"
                  f" decisions {len(rows)-n0}", flush=True)
    if hasattr(env, "label_report"):
        print("\n" + env.label_report())
    env.close()
    print(f"\n{'tier':9s} {'kills/ep':>8s} {'surv(tic)':>9s} | {'net':5s} {'decs':>5s} {'match':>6s} {'RTmed':>6s} "
          f"{'RTq90':>6s} {'lca%':>5s} {'tmo%':>5s}")
    for tier in tiers:
        E = eps[tier]
        first = True
        for key in a.nets:
            R = [r for r in rows if r["tier"] == tier and r["net"] == key]
            if not R:
                continue
            ok = [r["rt_ms"] for r in R if not r["timeout"]] or [float("nan")]
            head = f"{tier:9s} {np.mean([e[0] for e in E]):8.2f} {np.mean([e[1] for e in E]):9.0f}" if first else " " * 19
            print(f"{head} | {key:5s} {len(R):5d} {_mean([r['match'] for r in R]):6.2f} {np.nanmedian(ok):6.0f} "
                  f"{np.nanpercentile(ok, 90):6.0f} {100*np.mean([r['route']=='lca' for r in R]):5.1f} "
                  f"{100*np.mean([r['timeout'] for r in R]):5.1f}")
            first = False
    print("\nbutton-down share of all tics (turn bias check):")
    for tier in tiers:
        L, R, F, n = lr[tier]
        print(f"  {tier:9s} LEFT {100*L/max(n,1):5.1f}%   RIGHT {100*R/max(n,1):5.1f}%   FIRE {100*F/max(n,1):5.1f}%")
    return rows


def run_png(a, params):
    snaps = sorted(int(v) for v in a.snap.split(","))
    env = make_env(a, a.seed)
    nets = make_nets(a.agent, a, params, a.seed)
    rows, tl, log = [], deque(maxlen=310), deque(maxlen=14)
    n_rows = 0
    done = 0
    for ev in play_episode(env, nets, a, rows, 0):
        tl.append(ev["btn"])
        while n_rows < len(rows):
            log.append(log_line(rows[n_rows])); n_rows += 1
        if done < len(snaps) and ev["tic"] >= snaps[done]:
            c = PILCanvas(a.width, a.height)
            draw_dashboard(c, dict(ev, tier=a.agent, tl=tl, log=list(log)[-14:]), a, a.width, a.height)
            fn = a.png if len(snaps) == 1 else a.png.replace(".png", f"_{snaps[done]}.png")
            c.save(fn); print("saved", fn); done += 1
    env.close()


def run_window(a, params):
    import pygame as pg
    pg.init()
    W_, H_ = a.width, a.height
    flags = pg.SCALED | pg.RESIZABLE
    scr = pg.display.set_mode((W_, H_), flags)
    pg.display.set_caption("Wong-Wang decision networks on Doom")
    c = PGCanvas(scr, pg)
    clock = pg.time.Clock()
    st = dict(tier=a.agent, speed=a.speed, pause=False, shots=0)
    shots_dir = None
    if not a.no_shots:
        shots_dir = a.shots_dir or f"shots_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(shots_dir, exist_ok=True)
        print("decision screenshots ->", os.path.abspath(shots_dir))
    env = make_env(a, a.seed)
    rows, tl, log = [], deque(maxlen=310), deque(maxlen=14)
    n_rows, quit_ = 0, False
    nets = make_nets(st["tier"], a, params, a.seed)
    ep = 0
    while not quit_ and ep < a.episodes:
        for ev in play_episode(env, nets, a, rows, ep):
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    quit_ = True
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_ESCAPE:
                        quit_ = True
                    elif e.key == pg.K_SPACE:
                        st["pause"] = not st["pause"]
                    elif e.key in (pg.K_1, pg.K_2, pg.K_3):
                        st["tier"] = NW_TIERS[e.key - pg.K_1]
                        fresh = make_nets(st["tier"], a, params, a.seed + 7)
                        nets.clear(); nets.update(fresh)           # the running episode keeps its dict object
                    elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                        st["speed"] *= 1.5
                    elif e.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        st["speed"] /= 1.5
                    elif e.key == pg.K_s:
                        fn = f"nets_shot_{time.strftime('%H%M%S')}_{st['shots']:03d}.png"
                        pg.image.save(scr, fn); st["shots"] += 1; print("saved", fn)
            if quit_:
                break
            tl.append(ev["btn"])
            while n_rows < len(rows):
                log.append(log_line(rows[n_rows])); n_rows += 1
            draw_dashboard(c, dict(ev, tier=st["tier"], tl=tl, log=list(log)[-14:]), a, W_, H_)
            pg.display.flip()
            if shots_dir and ev["commits"]:               # a decision was just made: save the dashboard as it is NOW
                tag = "_".join(f"{r['net']}-{r['resp']}" for r in ev["commits"])
                fn = os.path.join(shots_dir, f"ep{ev['ep']+1:02d}_tic{ev['tic']:04d}_{tag}.png")
                pg.image.save(scr, fn)
                for r in ev["commits"]:
                    r["shot"] = fn
            while st["pause"] and not quit_:
                for e in pg.event.get():
                    if e.type == pg.KEYDOWN and e.key in (pg.K_SPACE, pg.K_ESCAPE):
                        st["pause"] = False
                        quit_ = quit_ or e.key == pg.K_ESCAPE
                clock.tick(20)
            clock.tick(TICS * st["speed"])
        ep += 1
    if hasattr(env, "label_report"):
        print(env.label_report())
    env.close()
    pg.quit()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="ww_cr_lca", choices=NW_TIERS)
    ap.add_argument("--nets", default="move,shoot", help="comma list from move,shoot (the MOVE net also does the searching)")
    ap.add_argument("--compare", default="", help="comma list of tiers (or oracle/idle) -> headless table")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-tics", type=int, default=2100)
    ap.add_argument("--timeout", type=float, default=1500.0, help="max decision time per epoch, ms")
    ap.add_argument("--refractory", type=int, default=1, help="idle tics between the END of a net's action and its next epoch")
    ap.add_argument("--turn-tics", type=int, default=8, help="MOVE pulse: maximum tics the arrow stays down")
    ap.add_argument("--turn-gain", type=float, default=0.07, help="MOVE pulse = ceil(|dx|/turn_gain) tics; guess of dx turned per tic")
    ap.add_argument("--search-ev", type=float, default=6.0, help="search-phase evidence scale (0 = pure noise decision)")
    ap.add_argument("--search-redecide", type=float, default=2000.0, help="ms between the MOVE net's search-phase decisions while spinning")
    ap.add_argument("--shots-dir", default="", help="window: directory for the per-decision screenshots (default shots_<timestamp>)")
    ap.add_argument("--no-shots", action="store_true", help="window: do not save per-decision screenshots")
    ap.add_argument("--fire-tics", type=int, default=3, help="tics the fire button stays down after a SHOOT commit")
    ap.add_argument("--scan", default="random", choices=["random", "left", "right"],
                    help="spin side when there is no SEARCH net, and the fallback when a SEARCH decision times out")
    ap.add_argument("--sync", action="store_true", help="window: run decision simulations inline (no worker thread)")
    ap.add_argument("--ev", type=float, default=12.0)
    ap.add_argument("--aim-tol", type=float, default=0.15)
    ap.add_argument("--move-scale", type=float, default=0.5, help="|dx| at which MOVE evidence saturates")
    ap.add_argument("--deadband", type=float, default=0.04, help="no turn decision below this |dx|; arrow released early inside it")
    ap.add_argument("--params", default="", help="json; optional sub-dicts {'move':{..},'shoot':{..}} per net")
    ap.add_argument("--u0", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--ter", type=float, default=None, help="mean non-decision time, ms")
    ap.add_argument("--scenario", default="defend_the_center.cfg")
    ap.add_argument("--skill", type=int, default=0)
    ap.add_argument("--enemy-names", default="", help="optional comma list (case-insensitive); default: every object that is not a pickup/projectile/effect/player")
    ap.add_argument("--melee", default="keep", choices=["remove", "kill", "keep"],
                    help="melee-only monsters (Demon, Spectre): EXPERIMENTAL -- delete them with the console command 'kill' or 'remove' -- BOTH made every enemy disappear on the user's machine, cause unknown; default 'keep' leaves them")
    ap.add_argument("--lost-grace", type=int, default=8, help="tics a vanished target is still treated as present (stops search/target flicker)")
    ap.add_argument("--keep-melee", action="store_true", help="do NOT turn Demon/Spectre/chainsaw marine into gun guys")
    ap.add_argument("--demon-speed", type=int, default=0, help="Demon/Spectre speed (Doom default 10); 0 = unchanged. UNTESTED mod")
    ap.add_argument("--demon-health", type=int, default=0, help="Demon/Spectre health (default 150); 0 = unchanged")
    ap.add_argument("--shooter-health", type=int, default=0, help="ZombieMan/ShotgunGuy/DoomImp health; 0 = unchanged")
    ap.add_argument("--god", action="store_true", help="agent cannot die (console 'god' at each episode start): lets the dynamics play out")
    ap.add_argument("--infinite-ammo", action="store_true", help="console 'give ammo' every 2 s")
    ap.add_argument("--list-labels", action="store_true", help="print each new object name the game reports and whether it counts as an enemy")
    ap.add_argument("--no-corpse-filter", action="store_true")
    ap.add_argument("--no-landscape", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--png", default="", help="save dashboard image(s) instead of opening a window")
    ap.add_argument("--snap", default="300", help="tics at which --png snapshots are taken")
    ap.add_argument("--csv", default="")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--fake-env", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.nets = [k.strip() for k in a.nets.split(",") if k.strip() and k.strip() != "search"]   # old "search" net is merged into move
    assert a.nets and all(k in SPECS for k in a.nets), f"--nets from {list(SPECS)}"
    a.landscape = not a.no_landscape
    a.search_redecide_tics = max(1, math.ceil(a.search_redecide * TICS / 1000.0))
    a.async_dec = not (a.sync or a.compare or a.png)
    params = load_params(a)
    csvpath = None if a.no_csv else (a.csv or f"nets_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    if a.compare:
        tiers = [t.strip() for t in a.compare.split(",")]
        assert all(t in NW_TIERS + BASELINES for t in tiers)
        rows = run_headless(a, tiers, params)
    elif a.png:
        run_png(a, params)
        return
    else:
        rows = run_window(a, params)
    if csvpath:
        write_rows(rows, csvpath)


if __name__ == "__main__":
    main()
