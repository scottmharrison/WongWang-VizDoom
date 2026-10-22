"""
play_ww.py -- a Wong-Wang decision layer on top of a ViZDoom shooter (defend_the_center).

   Doom frame + labels --> evidence policy --> 3 logits [turn LEFT, turn RIGHT, SHOOT]
                                                  |
                          NWayAgent (ww | ww_cr | ww_cr_lca, n = 3 pools)  --> choice, RT
                                                  |
        game timeline:  [observe] -> [deliberate: RT ms of NO-OP] -> [act: hold-tics of the chosen button] -> repeat
   No visible enemy -> no decision; the agent just scans (turns) -- see --scan.

Evidence policies
   default  'labels'  : uses ViZDoom's labels buffer.  Target = biggest (nearest) enemy sprite, dx = its
                        horizontal offset from screen centre in [-1, 1].  a = min(|dx|/aim_tol, 1);
                        logits = ev * [a*(dx<0), a*(dx>0), 1-a].  Perfectly centred -> SHOOT wins by ev;
                        far off-centre -> TURN wins by ev; in between the pools compete (slow, error-prone).
   --policy mod:fn    : your own function fn(frame, enemies) -> 3 logits (or None = nothing to decide).
                        This is where a pretrained network goes (Sample Factory / RL policy action logits,
                        reduced to left/right/shoot).  NOT implemented here: it needs torch + that library.

Motion while deciding: by default the current action keeps running during the deliberation (RT) until the new
choice replaces it: a TURN continues; SHOOT continues while an enemy is still within 2*aim_tol of the crosshair.
(--no-persist = stand still while deciding.)  --skill 1..5 sets Doom difficulty (3 = scenario default).

Modes
   python play_ww.py --compare ww,ww_cr,ww_cr_lca --episodes 5 --verbose     # HEADLESS: tables only, no window!
   python play_ww.py --fake-env --compare ww,ww_cr,ww_cr_lca --episodes 5    # headless tier comparison
   python play_ww.py --compare ww,ww_cr,ww_cr_lca --episodes 5               # same on real ViZDoom
   python play_ww.py --agent ww_cr_lca                                       # watch window (pygame)
   keys: SPACE pause, 1/2/3 tier, +/- speed, S screenshot, ESC quit.   CSV: play_<timestamp>.csv

KNOWN SIMPLIFICATIONS (read before interpreting results)
   * Evidence is sampled ONCE at the start of each decision and frozen while the WW integrates; the world
     keeps moving during the RT (enemies approach), so slow tiers are penalised -- that is the point.
   * Corpses: labels include dead monsters.  A flat-sprite filter (height < 0.6*width) drops most; untested.
   * Kills are counted from positive rewards (defend_the_center gives +1 per kill).
   * NW_PRESETS were hand-tuned for 5-way monster recognition, not for this task.  Fit or tune before
     reading anything into absolute numbers.
   * Only the --fake-env path and the decision loop are tested in my sandbox; real ViZDoom and the pygame
     window have not been run.
"""
import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

from sp2_nway import NWayAgent, NW_TIERS

ACTIONS = ["left", "right", "shoot"]
NOOP = 3
TICS = 35.0
MONSTERS = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
            "HellKnight", "BaronOfHell", "Revenant", "Arachnotron", "Fatso", "PainElemental", "Archvile",
            "Cyberdemon", "SpiderMastermind", "WolfensteinSS", "Stimpack_not_a_monster"} - {"Stimpack_not_a_monster"}


# --------------------------------------------------------------------------------------- evidence
# Names are matched case-insensitively (ZDoom spells it "ZombieMan", not "Zombieman" -- the old exact-case whitelist
# silently dropped the former humans with guns).  Unknown names are treated as enemies unless they look like
# projectiles, effects, pickups or the player; run with --list-labels to see every object the game reports.
KNOWN = {m.lower() for m in MONSTERS} | {"zombieman", "marinechainsawvzd", "marine", "stealthzombieman",
                                         "stealthdemon", "stealthimp", "stealthshotgunguy", "stealthchaingunguy"}
MELEE = {"demon", "spectre", "marinechainsawvzd"}      # melee-only monsters (fast, sneak up on an agent that only turns)
IGNORE_SUBSTR = ("player", "blood", "puff", "ball", "rocket", "missile", "casing", "shell", "clip", "ammo",
                 "stimpack", "medikit", "health", "armor", "bonus", "spark", "smoke", "fire", "gib", "debris",
                 "tracer", "fx", "splash", "teleport", "flash", "decor", "light", "torch", "bullet", "plasma",
                 "bfg", "fatshot", "arachnotronplasma", "baronball", "cacodemonball")


def pick_target(enemies):
    return max(enemies, key=lambda e: e["h"]) if enemies else None


def label_logits(frame, enemies, ev=12.0, aim_tol=0.15):
    t = pick_target(enemies)
    if t is None:
        return None
    dx = t["dx"]
    a = min(abs(dx) / aim_tol, 1.0)
    return np.array([ev * a * (dx < 0), ev * a * (dx > 0), ev * (1.0 - a)], float)


def load_policy(spec, ev, aim_tol):
    if not spec:
        return lambda frame, enemies: label_logits(frame, enemies, ev, aim_tol)
    mod, fn = spec.split(":")
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), fn)


# --------------------------------------------------------------------------------------- environments
class FakeDefendEnv:
    """Toy 1-D stand-in for defend_the_center (for testing the pipeline).  View half-width = 1 unit,
    full circle = 8 units.  Enemies spawn all round, walk in, and hurt you when adjacent."""

    def __init__(self, seed=0, max_tics=2100):
        self.rng = np.random.default_rng(seed)
        self.max_tics = max_tics

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.h, self.t, self.kills, self.health, self.cool, self.done = 0.0, 0, 0, 100, 0, False
        self.en = []
        self._spawn()
        return self.observe()

    def _spawn(self):
        self.en.append(dict(ang=float(self.rng.uniform(0, 8)), dist=1.0))

    @staticmethod
    def _wrap(x):
        return (x + 4.0) % 8.0 - 4.0

    def observe(self):
        W, H = 320, 240
        fr = np.full((H, W, 3), 70, np.uint8)
        fr[H // 2:] = 110
        out = []
        for e in self.en:
            dx = self._wrap(e["ang"] - self.h)
            if abs(dx) < 1.0:
                hh = int(40 + 160 * (1 - e["dist"])); ww = max(8, hh // 3)
                cx = int(W / 2 * (1 + dx)); y1 = H // 2 + hh // 2
                fr[max(0, y1 - hh):y1, max(0, cx - ww // 2):min(W, cx + ww // 2)] = (200, 60, 50)
                out.append(dict(dx=dx, h=hh / H, w=ww / W, cy=0.5, name="FakeMonster"))
        return fr, out

    def step(self, action):
        return self.step_buttons(action == 0, action == 1, action == 2)

    def step_buttons(self, left, right, shoot):
        reward = 0.0
        if left and not right:
            self.h = (self.h - 0.05) % 8.0
        elif right and not left:
            self.h = (self.h + 0.05) % 8.0
        if shoot and self.cool == 0:
            self.cool = 8
            hit = [e for e in self.en if abs(self._wrap(e["ang"] - self.h)) < 0.10]
            if hit:
                self.en.remove(min(hit, key=lambda e: e["dist"]))
                self.kills += 1; reward = 1.0
        self.cool = max(0, self.cool - 1)
        for e in self.en:
            e["dist"] = max(0.0, e["dist"] - 0.004)
            if e["dist"] < 0.1 and self.t % 12 == 0:
                self.health -= 3
        if self.t % 60 == 0 and len(self.en) < 6:
            self._spawn()
        self.t += 1
        self.done = self.health <= 0 or self.t >= self.max_tics
        return reward

    def info(self):
        return dict(health=self.health, kills=self.kills, tic=self.t)

    def close(self):
        pass


class DoomDefendEnv:
    """Real ViZDoom, defend_the_center-style scenario (buttons are mapped by NAME)."""

    def __init__(self, seed=0, scenario="defend_the_center.cfg", enemy_names=None, corpse_filter=True, max_tics=2100,
                 skill=0, mod=None):
        import vizdoom as vzd
        self.vzd = vzd
        g = vzd.DoomGame()
        g.load_config(scenario if os.path.isfile(scenario) else os.path.join(vzd.scenarios_path, scenario))
        g.set_window_visible(False)
        g.set_labels_buffer_enabled(True)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_seed(int(seed))
        if skill:
            g.set_doom_skill(int(skill))
        g.set_episode_timeout(max_tics)
        if mod:
            g.add_game_args(f"-file {mod}")      # relative file name in cwd (spaces in a path break add_game_args)
        g.init()
        names = [b.name for b in g.get_available_buttons()]
        need = {"TURN_LEFT": 0, "TURN_RIGHT": 1, "ATTACK": 2}
        miss = [k for k in need if k not in names]
        if miss:
            raise RuntimeError(f"scenario lacks buttons {miss}; has {names}")
        self.bmap = {need[k]: names.index(k) for k in need}
        self.nb = len(names)
        self.g, self.cf = g, corpse_filter
        self.explicit = {n.strip().lower() for n in enemy_names if n.strip()} if enemy_names else None
        self.seen, self.verbose_labels = {}, False
        self.melee, self._melee_run, self._melee_warned, self._melee_dbg = "keep", 0, False, 0
        self._melee_used, self._drought, self._drought_warned = False, 0, False
        self.god, self.ammo = False, False                      # research aids: invulnerable agent / refilled ammo   # "keep" | "remove" | "kill"
        self.kills = 0
        self.max_tics = max_tics

    def reset(self, seed=None):
        if seed is not None:
            self.g.set_seed(int(seed))      # applies to the next new_episode; same seed per episode index across tiers
        self.g.new_episode()
        self.kills, self.t, self.done = 0, 0, False
        if self.god:                                  # console cheats; health in the title bar should then stay at 100
            self.g.send_game_command("sv_cheats 1")
            self.g.send_game_command("god")
        return self.observe()

    def observe(self):
        s = self.g.get_state()
        if s is None:
            return np.zeros((240, 320, 3), np.uint8), []
        fr = s.screen_buffer
        if fr.ndim == 3 and fr.shape[0] == 3:
            fr = np.transpose(fr, (1, 2, 0))
        H, W = fr.shape[:2]
        out = []
        cmd_sent = set()
        if self._melee_dbg > 0:
            self._melee_dbg -= 1
            print(f"[melee-debug] objects in view after the console command: {sorted(l.object_name for l in s.labels)}", flush=True)
        for l in s.labels:
            nm = l.object_name
            if self.melee != "keep" and nm.lower() in MELEE:       # delete melee monsters via the engine console
                if nm not in cmd_sent:
                    self.g.send_game_command(f"{self.melee} {nm}")
                    cmd_sent.add(nm)
                    self._melee_used = True
                    if self._melee_dbg == 0:
                        self._melee_dbg = 4              # print what the game reports for the next few frames
                continue
            rec = self.seen.setdefault(nm, dict(kept=0, corpse=0, ignored=0, w=int(l.width), h=int(l.height)))
            if not self.is_enemy(nm):
                rec["ignored"] += 1
                if self.verbose_labels and rec["ignored"] == 1:
                    print(f"[labels] object '{nm}' ignored (not an enemy by name)  {l.width}x{l.height}px", flush=True)
                continue
            if self.cf and l.height < 0.6 * l.width:
                rec["corpse"] += 1
                continue
            rec["kept"] += 1
            if self.verbose_labels and rec["kept"] == 1:
                print(f"[labels] enemy '{nm}' recognised  {l.width}x{l.height}px", flush=True)
            out.append(dict(dx=(l.x + l.width / 2 - W / 2) / (W / 2), h=l.height / H, w=l.width / W,
                            cy=(l.y + l.height / 2) / H, name=l.object_name))
        if self._melee_used:                                       # after removing melee: do ANY enemies still come?
            self._drought = 0 if (out or cmd_sent) else self._drought + 1
            if self._drought > 105 and not self._drought_warned:
                self._drought_warned = True
                print(f"[melee] WARNING: no enemy of any kind for 3 s after '{self.melee}' -- the scenario's spawner probably "
                      f"depends on those monsters. Use --melee keep.", flush=True)
        if cmd_sent:                                               # still seeing them many tics in a row => command ignored
            self._melee_run += 1
            if self._melee_run > 8 and not self._melee_warned:
                self._melee_warned = True
                print(f"[melee] WARNING: '{self.melee} {sorted(cmd_sent)[0]}' did not remove them (engine ignored the console "
                      f"command). Try --melee kill, or --melee keep.", flush=True)
        else:
            self._melee_run = 0
        return fr, out

    def is_enemy(self, name):
        n = name.lower()
        if self.explicit is not None:
            return n in self.explicit
        if n in KNOWN:
            return True
        return not any(t in n for t in IGNORE_SUBSTR)

    def label_report(self):
        """Every object name the game reported, and what was done with it (use this to check the gun-carrying enemies)."""
        lines = ["label report (object name: frames kept as enemy / dropped as corpse-shaped / ignored by name)"]
        for nm, r in sorted(self.seen.items(), key=lambda kv: -(kv[1]["kept"] + kv[1]["corpse"] + kv[1]["ignored"])):
            lines.append(f"  {nm:22s} kept {r['kept']:6d}  corpse {r['corpse']:6d}  ignored {r['ignored']:6d}")
        return "\n".join(lines)

    def step(self, action):
        return self.step_buttons(action == 0, action == 1, action == 2)

    def step_buttons(self, left, right, shoot):
        act = [0] * self.nb
        for k, on in ((0, left), (1, right), (2, shoot)):
            if on:
                act[self.bmap[k]] = 1
        if self.ammo and self.t % 70 == 0:
            self.g.send_game_command("give ammo")
        r = self.g.make_action(act, 1)
        self.t += 1
        if r > 0:
            self.kills += 1
        self.done = self.g.is_episode_finished()
        return r

    def info(self):
        try:
            hp = int(self.g.get_game_variable(self.vzd.GameVariable.HEALTH))
        except Exception:
            hp = -1
        return dict(health=hp, kills=self.kills, tic=self.t)

    def close(self):
        self.g.close()


class BaselineAgent:
    """Controls that bypass the Wong-Wang module.  'oracle': picks argmax(logits) after ONE tic (RT ~ 0).
    'idle': never acts.  If even the oracle gets no kills, the evidence policy / button mapping is the problem."""

    def __init__(self, tier):
        self.tier = tier

    def decide(self, logits, timeout_ms=3000):
        d = dict(tier=self.tier, u0=0.0, ter=0.0, coh=None, q=None, ww_r=np.zeros((1, 3)), lca_x=None, lca_t0=0,
                 lca_onset=None, t_ww=0.0, t_lca=-1.0, n=3, dt=0.0, route="ww")
        if self.tier == "idle":
            d.update(choice=None, rt_ms=float(timeout_ms), route="", timeout=True)
        else:
            d.update(choice=int(np.argmax(logits)), rt_ms=1.0, timeout=False)
        return d


BASELINES = ("oracle", "idle")


def make_agent(tier, params, seed):
    return BaselineAgent(tier) if tier in BASELINES else NWayAgent(tier, params, seed=seed, n=3)


# --------------------------------------------------------------------------------------- episode engine
def play_episode(env, get_agent, a, policy, rows, ep):
    """Generator: yields one event per game tic.  Headless runs drain it; the window draws each event."""
    S = dict(obs=env.reset(a.seed * 1000 + ep), tic=0, last=NOOP)
    scan = {"left": 0, "right": 1, "none": NOOP}[a.scan]

    def tick(action, phase, dec=None, elapsed=0.0, lg=None, tgt=None):
        if env.done:
            return
        env.step(action)
        S["obs"] = env.observe(); S["tic"] += 1
        yield dict(frame=S["obs"][0], enemies=S["obs"][1], phase=phase, dec=dec, elapsed=elapsed, logits=lg,
                   target=tgt, action=action, info=env.info(), ep=ep, tic=S["tic"])

    while not env.done and S["tic"] < a.max_tics:
        frame, enemies = S["obs"]
        lg = policy(frame, enemies)
        if lg is None:
            for _ in range(a.hold):
                yield from tick(scan, "scan")
            S["last"] = scan
            continue
        tgt = pick_target(enemies)
        agent = get_agent()
        dec = agent.decide(np.asarray(lg, float), timeout_ms=a.timeout)
        ideal = int(np.argmax(lg))
        n_think = max(1, math.ceil(dec["rt_ms"] * TICS / 1000.0))
        t_dec = S["tic"]
        for i in range(n_think):
            hold_on = NOOP
            if not a.no_persist:                       # the current motor program keeps running while deciding
                if S["last"] in (0, 1):
                    hold_on = S["last"]
                elif S["last"] == 2 and any(abs(e["dx"]) < 2 * a.aim_tol for e in S["obs"][1]):
                    hold_on = 2                        # keep firing only while something is still lined up
            yield from tick(hold_on, "think", dec, (i + 1) * 1000.0 / TICS, lg, tgt)
        act = NOOP if dec["timeout"] else dec["choice"]
        for _ in range(a.hold):
            yield from tick(act, "act", dec, dec["rt_ms"], lg, tgt)
        if not dec["timeout"]:
            S["last"] = act
        inf = env.info()
        rows.append(dict(ep=ep, tier=dec["tier"], tic=t_dec, dx=round(float(tgt["dx"]), 3) if tgt else "",
                         ideal=ACTIONS[ideal], resp="none" if dec["timeout"] else ACTIONS[dec["choice"]],
                         match=int((not dec["timeout"]) and dec["choice"] == ideal), rt_ms=round(dec["rt_ms"], 1),
                         route=dec["route"] or "none", u0=round(dec["u0"], 3), timeout=int(dec["timeout"]),
                         kills=inf["kills"], health=inf["health"]))
        if a.verbose:
            r = rows[-1]
            print(f"    tic {r['tic']:4d} dx {r['dx']:>6} ideal {r['ideal']:5s} -> {r['resp']:5s} RT {r['rt_ms']:6.0f} "
                  f"{r['route']:4s} kills {r['kills']} hp {r['health']}", flush=True)


def summarise(rows, eps):
    print(f"{'tier':10s} {'decs':>5s} {'match':>6s} {'RTmed':>6s} {'RTq90':>6s} {'lca%':>5s} {'tmo%':>5s}"
          f" {'shoot%':>6s} {'kills/ep':>8s} {'surv(tic)':>9s}")
    for tier in sorted({r['tier'] for r in rows}, key=lambda t: (NW_TIERS + BASELINES).index(t)):
        R = [r for r in rows if r["tier"] == tier]
        rt = np.array([r["rt_ms"] for r in R if not r["timeout"]]) if any(not r["timeout"] for r in R) else np.array([np.nan])
        E = eps[tier]
        print(f"{tier:10s} {len(R):5d} {np.mean([r['match'] for r in R]):6.2f} {np.median(rt):6.0f} "
              f"{np.nanpercentile(rt, 90):6.0f} {100*np.mean([r['route']=='lca' for r in R]):5.1f} "
              f"{100*np.mean([r['timeout'] for r in R]):5.1f} {100*np.mean([r['resp']=='shoot' for r in R]):6.1f} {np.mean([e[0] for e in E]):8.2f} {np.mean([e[1] for e in E]):9.0f}")


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} decisions -> {os.path.abspath(path)}")


def make_env(a, seed):
    if a.fake_env:
        return FakeDefendEnv(seed, a.max_tics)
    return DoomDefendEnv(seed, a.scenario, a.enemy_names.split(",") if a.enemy_names else None,
                         not a.no_corpse_filter, a.max_tics, a.skill)


def run_headless(a, tiers, params, policy, csvpath):
    rows, eps = [], {}
    env = make_env(a, a.seed)
    for ti, tier in enumerate(tiers):
        agent = make_agent(tier, params, a.seed + 100 * ti)
        eps[tier] = []
        for ep in range(a.episodes):
            before = len(rows)
            for ev in play_episode(env, lambda: agent, a, policy, rows, ep):
                last = ev
            eps[tier].append((env.info()["kills"], env.info()["tic"]))
            print(f"  {tier:10s} ep {ep+1}/{a.episodes}: kills {env.info()['kills']:3d}  health {env.info()['health']:4d}"
                  f"  tics {env.info()['tic']:4d}  decisions {len(rows)-before}", flush=True)
    env.close()
    print()
    summarise(rows, eps)
    if csvpath:
        write_csv(rows, csvpath)


# --------------------------------------------------------------------------------------- watch window
def run_window(a, params, policy, csvpath):
    import pygame as pg
    pg.init()
    Wd, Hd = 1060, 640
    scr = pg.display.set_mode((Wd, Hd))
    pg.display.set_caption("Wong-Wang decision layer on Doom")
    font, big = pg.font.SysFont("consolas", 15), pg.font.SysFont("consolas", 20, bold=True)
    clock = pg.time.Clock()
    agents = {t: NWayAgent(t, params, seed=a.seed + i, n=3) for i, t in enumerate(NW_TIERS)}
    st = dict(tier=a.agent, speed=a.speed, pause=False, shots=0)
    rows, hist = [], deque(maxlen=60)
    env = make_env(a, a.seed)
    COL = [(90, 160, 255), (255, 170, 60), (120, 220, 120)]
    RC = {"ww": (90, 160, 255), "lca": (255, 150, 40), "": (150, 150, 150)}

    def txt(s, x, y, f=font, c=(230, 230, 230)):
        scr.blit(f.render(s, True, c), (x, y))

    def draw(ev, last_dec):
        scr.fill((18, 18, 24))
        fr = ev["frame"]
        surf = pg.surfarray.make_surface(np.transpose(fr, (1, 0, 2)))
        surf = pg.transform.scale(surf, (600, 450))
        scr.blit(surf, (10, 40))
        pg.draw.line(scr, (255, 255, 255), (310, 40), (310, 490), 1)
        t = ev["target"]
        if t is not None:
            cx = 10 + 300 * (1 + t["dx"])
            col = {"think": (255, 220, 60), "act": (80, 255, 80)}.get(ev["phase"], (200, 200, 200))
            pg.draw.rect(scr, col, (cx - 3 - 300 * t["w"] / 2, 40 + 225 - 225 * t["h"], 6 + 300 * t["w"], 450 * t["h"]), 2)
        inf = ev["info"]
        txt(f"{st['tier']}   phase: {ev['phase']:5s}   ep {ev['ep']+1}   tic {ev['tic']}   kills {inf['kills']}   health {inf['health']}", 10, 10, big)
        # --- WW traces
        x0, y0, w, h = 630, 40, 410, 200
        pg.draw.rect(scr, (30, 30, 40), (x0, y0, w, h))
        txt("WW pool rates (Hz)   [left | right | shoot]", x0, y0 - 18)
        thr_y = y0 + h - h * 15.0 / 40.0
        pg.draw.line(scr, (140, 60, 60), (x0, thr_y), (x0 + w, thr_y), 1)
        d = ev["dec"]
        if d is not None and ev["phase"] != "scan":
            r = d["ww_r"]
            ter = d["ter"]
            upto = ev["elapsed"] if ev["phase"] == "think" else d["rt_ms"]
            for k in range(r.shape[1]):
                pts = []
                for ms in range(0, int(min(upto - ter, a.timeout, len(r))), 8):
                    pts.append((x0 + w * (ms + ter) / a.timeout, y0 + h - h * min(r[ms, k], 40.0) / 40.0))
                if len(pts) > 1:
                    pg.draw.lines(scr, COL[k], False, pts, 2)
            tx = x0 + w * min(upto, a.timeout) / a.timeout
            pg.draw.line(scr, (200, 200, 200), (tx, y0), (tx, y0 + h), 1)
        # --- logits bars + decision readout
        L = ev["logits"]
        if L is not None:
            for k in range(3):
                bh = 60 * L[k] / max(a.ev, 1e-6)
                pg.draw.rect(scr, COL[k], (x0 + 20 + 60 * k, y0 + h + 90 - bh, 40, bh))
                txt(ACTIONS[k], x0 + 20 + 60 * k, y0 + h + 95)
        if last_dec is not None:
            dd = last_dec
            ch = "none (timeout)" if dd["timeout"] else ACTIONS[dd["choice"]]
            txt(f"last: {ch}   RT {dd['rt_ms']:.0f} ms   route {dd['route'] or '-'}", x0 + 220, y0 + h + 20, font)
            txt(f"u0 {dd['u0']:.2f}   (critical {10.68})", x0 + 220, y0 + h + 40, font, (170, 170, 170))
        # --- RT history strip
        sy, sh = 520, 100
        pg.draw.rect(scr, (30, 30, 40), (10, sy, 1040, sh))
        txt(f"RT of last {hist.maxlen} decisions (0..{a.timeout} ms); orange = fast LCA route, grey = timeout", 14, sy + 2, font, (170, 170, 170))
        for i, (rt, route, tmo) in enumerate(hist):
            yy = sy + sh - 6 - (sh - 28) * min(rt, a.timeout) / a.timeout
            pg.draw.circle(scr, (110, 110, 110) if tmo else RC[route], (20 + i * 17, int(yy)), 5)
        txt("SPACE pause  1/2/3 tier  +/- speed  S screenshot  ESC quit", 10, 622, font, (140, 140, 140))
        pg.display.flip()

    ep, last_dec, quit_ = 0, None, False
    while not quit_ and ep < a.episodes:
        gen = play_episode(env, lambda: agents[st["tier"]], a, policy, rows, ep)
        n0 = len(rows)
        for ev in gen:
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
                    elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                        st["speed"] *= 1.5
                    elif e.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        st["speed"] /= 1.5
                    elif e.key == pg.K_s:
                        fn = f"play_shot_{time.strftime('%H%M%S')}_{st['shots']:03d}.png"
                        pg.image.save(scr, fn); st["shots"] += 1; print("saved", fn)
            if quit_:
                break
            if ev["dec"] is not None and ev["phase"] == "act" and ev["dec"] is not last_dec:
                last_dec = ev["dec"]
                hist.append((last_dec["rt_ms"], last_dec["route"], last_dec["timeout"]))
            draw(ev, last_dec)
            while st["pause"] and not quit_:
                for e in pg.event.get():
                    if e.type == pg.KEYDOWN and e.key in (pg.K_SPACE, pg.K_ESCAPE):
                        st["pause"] = False
                        quit_ = quit_ or e.key == pg.K_ESCAPE
                clock.tick(20)
            clock.tick(TICS * st["speed"])
        ep += 1
    env.close()
    pg.quit()
    write_csv(rows, csvpath)


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="ww_cr_lca", choices=NW_TIERS)
    ap.add_argument("--compare", default="", help="comma list of tiers -> headless comparison table")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-tics", type=int, default=2100, help="episode length cap (35 tics = 1 s)")
    ap.add_argument("--hold", type=int, default=4, help="tics a chosen button is held")
    ap.add_argument("--timeout", type=float, default=1500.0, help="max decision time, ms")
    ap.add_argument("--scan", default="right", choices=["left", "right", "none"], help="what to do with no enemy in view")
    ap.add_argument("--ev", type=float, default=12.0, help="evidence scale (logit margin when fully decisive)")
    ap.add_argument("--aim-tol", type=float, default=0.15, help="|dx| at which turning fully beats shooting")
    ap.add_argument("--policy", default="", help="module:function returning 3 logits from (frame, enemies)")
    ap.add_argument("--params", default="", help="json of decision-module parameters (e.g. from rtify_ww.py)")
    ap.add_argument("--scenario", default="defend_the_center.cfg")
    ap.add_argument("--enemy-names", default="", help="optional comma list (case-insensitive); default: every object that is not a pickup/projectile/effect/player")
    ap.add_argument("--no-corpse-filter", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--fake-env", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skill", type=int, default=0, help="Doom difficulty 1 (easiest) .. 5; 0 = scenario default")
    ap.add_argument("--no-persist", action="store_true", help="idle (no-op) while deliberating instead of continuing a turn")
    ap.add_argument("--verbose", action="store_true", help="print every decision (headless mode)")
    a = ap.parse_args()

    params = json.load(open(a.params)) if a.params else None
    policy = load_policy(a.policy, a.ev, a.aim_tol)
    csvpath = None if a.no_csv else (a.csv or f"play_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    if a.compare:
        tiers = [t.strip() for t in a.compare.split(",")]
        assert all(t in NW_TIERS + BASELINES for t in tiers), f"tiers must be in {NW_TIERS + BASELINES}"
        run_headless(a, tiers, params, policy, csvpath)
    else:
        run_window(a, params, policy, csvpath)


if __name__ == "__main__":
    main()
