#!/usr/bin/env python3
"""
run_human_task.py -- Doom left/right decision task (RDM analogue).

Each trial: fixation cross -> Doom frame appears -> press LEFT / RIGHT arrow
according to which side of the crosshair the monster is on -> feedback.
RT is measured from the display flip that shows the frame (perf_counter).

Usage:
    python run_human_task.py --participant scott
    python run_human_task.py --participant scott --trials 300 --offsets 2,4,8,16,32,64
    python run_human_task.py --simulate          # headless logic test, no Doom/pygame

Output: data/<participant>_<timestamp>.csv (+ .json sidecar). Rows are written
and flushed after every trial, so quitting early (ESC) keeps your data.
"""
import argparse, csv, json, os, platform, random, sys, time, statistics
from datetime import datetime

KEYS = ("left", "right")


# ------------------------------------------------------------------ I/O layers
class PygameIO:
    def __init__(self, W, H, fullscreen=False):
        import pygame
        self.pg = pygame
        pygame.init()
        flags = pygame.FULLSCREEN | pygame.SCALED if fullscreen else pygame.SCALED
        try:
            self.screen = pygame.display.set_mode((W, H), flags, vsync=1)
        except Exception:
            self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("Doom decision task")
        pygame.mouse.set_visible(False)
        self.W, self.H = W, H
        self.font = pygame.font.Font(None, 36)
        self.big = pygame.font.Font(None, 56)
        self.kmap = {pygame.K_LEFT: "left", pygame.K_RIGHT: "right"}

    now = staticmethod(time.perf_counter)

    def sleep(self, s):
        end = time.perf_counter() + s
        while time.perf_counter() < end:
            self._pump()
            time.sleep(0.001)

    def _pump(self):
        for e in self.pg.event.get(self.pg.QUIT):
            raise KeyboardInterrupt
        # ESC handled in key polling

    def text(self, lines, color=(230, 230, 230), big_first=True):
        self.screen.fill((0, 0, 0))
        y = self.H // 2 - 30 * len(lines)
        for i, ln in enumerate(lines):
            f = self.big if (i == 0 and big_first) else self.font
            s = f.render(ln, True, color)
            self.screen.blit(s, s.get_rect(center=(self.W // 2, y)))
            y += 55 if i == 0 else 40
        self.pg.display.flip()

    def wait_key(self, key):
        while True:
            for e in self.pg.event.get():
                if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE):
                    raise KeyboardInterrupt
                if e.type == self.pg.KEYDOWN and e.key == key:
                    return
            time.sleep(0.005)

    def wait_space(self):
        self.wait_key(self.pg.K_SPACE)

    def show_fixation(self):
        pg = self.pg
        self.screen.fill((0, 0, 0))
        c = (self.W // 2, self.H // 2)
        pg.draw.line(self.screen, (255, 255, 255), (c[0] - 12, c[1]), (c[0] + 12, c[1]), 2)
        pg.draw.line(self.screen, (255, 255, 255), (c[0], c[1] - 12), (c[0], c[1] + 12), 2)
        pg.display.flip()
        pg.event.clear()

    def flush_keys(self):
        """True if a response key was pressed since the last flush (premature)."""
        early = False
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE):
                raise KeyboardInterrupt
            if e.type == self.pg.KEYDOWN and e.key in self.kmap:
                early = True
        return early

    def show_stimulus(self, stim):
        surf = self.pg.surfarray.make_surface(stim.frame.swapaxes(0, 1))
        self._surf = surf
        self.screen.blit(surf, (0, 0))
        self.pg.display.flip()
        return time.perf_counter()

    def show_blank(self):
        self.screen.fill((0, 0, 0))
        self.pg.display.flip()

    def wait_response(self, timeout_s, exposure_s, stim, t_on):
        blanked = False
        self.pg.event.clear()
        while True:
            t = time.perf_counter()
            for e in self.pg.event.get():
                tt = time.perf_counter()
                if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE):
                    raise KeyboardInterrupt
                if e.type == self.pg.KEYDOWN and e.key in self.kmap:
                    return self.kmap[e.key], tt
            if exposure_s and not blanked and t - t_on >= exposure_s:
                self.show_blank(); blanked = True
            if t - t_on >= timeout_s:
                return None, t
            time.sleep(0.0005)

    def feedback(self, kind):
        col, msg = {"correct": ((60, 220, 60), "Correct"),
                    "error": ((230, 60, 60), "Wrong"),
                    "timeout": ((230, 200, 60), "Too slow")}[kind]
        self.text([msg], color=col)

    def close(self):
        self.pg.quit()


class SimIO:
    """Headless 'participant': a DDM generates keys & RTs in virtual time."""
    def __init__(self, seed=0):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s): self.t += s
    def text(self, *a, **k): pass
    def wait_space(self): pass
    def show_fixation(self): pass
    def flush_keys(self): return False
    def show_blank(self): pass
    def feedback(self, kind): pass
    def close(self): pass

    def show_stimulus(self, stim):
        self.t += 0.016
        return self.t

    def wait_response(self, timeout_s, exposure_s, stim, t_on):
        drift, a, sig, ndt = 0.0006 * stim.offset_measured_px, 1.0, 0.04, 250.0
        x, t = 0.0, ndt
        while abs(x) < a and t < timeout_s * 1000:
            x += drift + sig * self.rng.normal(); t += 1.0
        if abs(x) < a:
            return None, t_on + timeout_s
        return ("right" if x > 0 else "left"), t_on + t / 1000.0


class AgentIO(SimIO):
    """Headless model participant: an sp2_agents.Agent decides on the real (or fake) Doom frame."""
    def __init__(self, agent):
        super().__init__(0)
        self.agent, self.last_info = agent, {}

    def wait_response(self, timeout_s, exposure_s, stim, t_on):
        key, rt, self.last_info = self.agent.decide(stim.offset_measured_px, stim.true_side, timeout_s * 1000)
        return key, t_on + rt / 1000.0


class WatchIO(PygameIO):
    """Pygame window where a model agent plays: you see the frame, a running clock,
    then the agent's choice, RT and which route (WW / LCA) fired."""
    def __init__(self, agent, W, H, fullscreen=False, speed=1.0):
        super().__init__(W, H, fullscreen)
        self.agent, self.speed, self.last_info = agent, speed, {}
        self.small = self.pg.font.Font(None, 28)

    def _overlay(self, lines, colors):
        pg = self.pg
        self.screen.blit(self._surf, (0, 0))
        bar = pg.Surface((self.W, 34 * len(lines) + 10)); bar.set_alpha(190); bar.fill((0, 0, 0))
        self.screen.blit(bar, (0, 0))
        for i, (ln, c) in enumerate(zip(lines, colors)):
            self.screen.blit(self.small.render(ln, True, c), (12, 8 + 34 * i))
        pg.display.flip()

    def wait_response(self, timeout_s, exposure_s, stim, t_on):
        key, rt, info = self.agent.decide(stim.offset_measured_px, stim.true_side, timeout_s * 1000)
        self.last_info = info
        end = t_on + (rt / 1000.0) / self.speed
        while time.perf_counter() < end:
            for e in self.pg.event.get():
                if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE):
                    raise KeyboardInterrupt
            el = (time.perf_counter() - t_on) * self.speed * 1000
            self._overlay([f"{info['agent']}   thinking...  {el:5.0f} ms"], [(200, 200, 200)])
            time.sleep(0.016)
        route = {"ww": "Wong-Wang attractor", "lca": "fast LCA route", "": "-"}[info["route"]]
        if key is None:
            self._overlay([f"{info['agent']}   NO RESPONSE (timeout)", f"u0_eff {info['u0_eff']}"], [(230, 200, 60)] * 2)
        else:
            ok = key == stim.true_side
            col = (60, 220, 60) if ok else (230, 60, 60)
            arrow = "<-- LEFT" if key == "left" else "RIGHT -->"
            self._overlay([f"{info['agent']}   chose {arrow}   {'correct' if ok else 'WRONG'}   RT {rt:.0f} ms",
                           f"route: {route}   u0_eff {info['u0_eff']}   offset {stim.offset_measured_px:+.1f}px"],
                          [col, (200, 200, 200)])
        t_end = time.perf_counter() + 0.9 / self.speed
        while time.perf_counter() < t_end:
            for e in self.pg.event.get():
                if e.type == self.pg.QUIT or (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE):
                    raise KeyboardInterrupt
            time.sleep(0.01)
        return key, t_on + rt / 1000.0


# ------------------------------------------------------------------ trial list
def build_trials(offsets, n, rng):
    cells = [(o, s) for o in offsets for s in KEYS]
    reps = max(1, round(n / len(cells)))
    trials = cells * reps
    rng.shuffle(trials)
    return trials


FIELDS = ["participant", "session", "trial", "practice", "offset_target_px",
          "offset_measured_px", "true_side", "response", "correct", "rt_ms",
          "timeout", "premature", "fast_guess", "fixation_ms", "screen_w",
          "screen_h", "seed", "agent", "route", "u0_eff"]


def summarise(rows):
    main = [r for r in rows if not r["practice"]]
    if not main:
        return
    print("\n offset_px    n   acc   median_RT_ms (correct)  median_RT_ms (error)")
    for o in sorted({r["offset_target_px"] for r in main}):
        rs = [r for r in main if r["offset_target_px"] == o]
        resp = [r for r in rs if not r["timeout"]]
        acc = sum(r["correct"] for r in resp) / len(resp) if resp else float("nan")
        c = [r["rt_ms"] for r in resp if r["correct"]]
        e = [r["rt_ms"] for r in resp if not r["correct"]]
        med = lambda v: f"{statistics.median(v):7.0f}" if v else "    n/a"
        print(f" {o:8.1f} {len(rs):4d}  {acc:4.2f}   {med(c)}              {med(e)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default="p01")
    ap.add_argument("--trials", type=int, default=240, help="main trials (rounded to balance cells)")
    ap.add_argument("--practice", type=int, default=12)
    ap.add_argument("--offsets", default="2,4,8,16,32,64", help="monster offsets from crosshair, px")
    ap.add_argument("--timeout-ms", type=int, default=2500)
    ap.add_argument("--exposure-ms", type=int, default=0, help="0 = frame stays until response")
    ap.add_argument("--fix-ms", default="500,1000", help="fixation duration range")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--agent", choices=["ww", "ww_cr", "ww_cr_lca"], help="let a model tier play instead of a human")
    ap.add_argument("--agent-params", default=None, help="JSON string or .json file overriding agent parameters")
    ap.add_argument("--watch", action="store_true", help="with --agent: show the agent playing in a window")
    ap.add_argument("--speed", type=float, default=1.0, help="with --watch: playback speed (2 = twice as fast)")
    ap.add_argument("--fake-env", action="store_true", help="with --agent: use synthetic frames instead of ViZDoom")
    ap.add_argument("--simulate", action="store_true", help="headless DDM participant + fake frames")
    a = ap.parse_args(argv)

    W, H = (int(v) for v in a.resolution.lower().split("x"))
    offsets = [float(v) for v in a.offsets.split(",")]
    fix_lo, fix_hi = (float(v) for v in a.fix_ms.split(","))
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(seed)

    from rdm_doom_env import RDMDoomEnv, SimulatedEnv
    agent = None
    if a.agent:
        import sp2_agents
        ov = {}
        if a.agent_params:
            ov = json.loads(a.agent_params if a.agent_params.lstrip().startswith("{") else open(a.agent_params).read())
        agent = sp2_agents.Agent(a.agent, ov, seed=seed)
        a.practice = 0
    io = (WatchIO(agent, W, H, a.fullscreen, a.speed) if (agent and a.watch) else AgentIO(agent)) if agent else (SimIO(seed) if a.simulate else PygameIO(W, H, a.fullscreen))
    env = (SimulatedEnv if (a.simulate or a.fake_env) else RDMDoomEnv)(W, H, seed=seed)

    os.makedirs(a.out_dir, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a.out_dir, f"{a.participant}{'_SIM' if a.simulate else ('_' + a.agent if a.agent else '')}_{session}")
    with open(base + ".json", "w") as f:
        json.dump({"args": vars(a), "seed": seed, "env": env.info(), "agent_params": agent.p if agent else None, "python": sys.version,
                   "platform": platform.platform(), "session": session}, f, indent=2, default=str)

    practice = [(max(offsets), s) for s in KEYS] * ((a.practice + 1) // 2)
    practice = practice[:a.practice]
    rng.shuffle(practice)
    plan = [(1, o, s) for o, s in practice] + [(0, o, s) for o, s in build_trials(offsets, a.trials, rng)]

    rows = []
    try:
        io.text(["Doom decision task",
                 "A monster appears. Is it LEFT or RIGHT of the crosshair?",
                 "Press the LEFT or RIGHT arrow key. Be fast AND accurate.",
                 f"{a.practice} practice trials, then {len(plan) - a.practice} real trials. ESC quits.",
                 "Press SPACE to begin"])
        io.wait_space()
        with open(base + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for i, (prac, off, side) in enumerate(plan, 1):
                fix_ms = rng.uniform(fix_lo, fix_hi)
                io.show_fixation()
                t0 = io.now()
                stim = env.make_trial(off, side)                 # prepared during fixation
                rest = fix_ms / 1000 - (io.now() - t0)
                if rest > 0:
                    io.sleep(rest)
                premature = io.flush_keys()
                t_on = io.show_stimulus(stim)
                key, t_r = io.wait_response(a.timeout_ms / 1000, a.exposure_ms / 1000, stim, t_on)
                timeout = key is None
                rt = (t_r - t_on) * 1000
                correct = (key == stim.true_side) if not timeout else None
                row = dict(participant=a.participant, session=session, trial=i, practice=prac,
                           offset_target_px=off, offset_measured_px=round(stim.offset_measured_px, 2),
                           true_side=stim.true_side, response=key or "", correct=int(correct) if correct is not None else "",
                           rt_ms=round(rt, 2), timeout=int(timeout), premature=int(premature),
                           fast_guess=int((not timeout) and rt < 150), fixation_ms=round(fix_ms, 1),
                           screen_w=W, screen_h=H, seed=seed)
                row.update({k: v for k, v in getattr(io, "last_info", {}).items() if k in FIELDS})
                w.writerow(row); fh.flush()
                rows.append({**row, "correct": correct if correct is not None else 0})
                if not a.no_feedback:
                    io.feedback("timeout" if timeout else ("correct" if correct else "error"))
                    io.sleep(0.4)
                io.show_blank(); io.sleep(0.2)
                if i == a.practice and not a.simulate and a.practice:
                    io.text(["Practice done", "Press SPACE to start the real trials"]); io.wait_space()
        io.text(["All done - thank you!"]); io.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted - partial data kept.")
    finally:
        env.close(); io.close()
    summarise(rows)
    print(f"\nSaved: {base}.csv")


if __name__ == "__main__":
    main()
