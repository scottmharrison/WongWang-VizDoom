#!/usr/bin/env python3
"""
run_recog_task.py -- Doom monster-recognition task (multi-choice RT), the human-data half of an
RTify-style model.  A monster appears; press the number key (1-5) for its type.  Difficulty is
controlled by image contrast (and optional noise).  RT = key press minus the display flip.

    python run_recog_task.py --participant scott
    python run_recog_task.py --participant scott --trials 400 --levels 1.0,0.5,0.3,0.2,0.12
    python run_recog_task.py --simulate        # headless pipeline test (fake frames, random-ish agent)
Output: data/<participant>_recog_<time>.csv (+ .json), written after every trial; ESC keeps data.
"""
import argparse, csv, json, os, platform, random, sys, time, statistics
from datetime import datetime
from run_human_task import PygameIO, SimIO
from doom_recog_env import DoomRecogEnv, FakeRecogEnv, CLASSES, NICE

FIELDS = ["participant", "session", "trial", "practice", "class_true", "class_resp", "true_idx", "resp_idx",
          "correct", "rt_ms", "timeout", "premature", "fast_guess", "contrast", "noise", "bbox_w", "bbox_h",
          "back_tics", "fixation_ms", "n_classes", "seed"]


class RecogIO(PygameIO):
    def __init__(self, W, H, names, fullscreen=False):
        super().__init__(W, H, fullscreen)
        pg = self.pg
        keys = [pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8]
        kps = [pg.K_KP1, pg.K_KP2, pg.K_KP3, pg.K_KP4, pg.K_KP5, pg.K_KP6, pg.K_KP7, pg.K_KP8]
        self.kmap = {}
        for i in range(len(names)):
            self.kmap[keys[i]] = i; self.kmap[kps[i]] = i
        self.legend = "   ".join(f"{i + 1}={n}" for i, n in enumerate(names))
        self.small = pg.font.Font(None, 30)

    def show_stimulus(self, stim):
        surf = self.pg.surfarray.make_surface(stim.frame.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        bar = self.pg.Surface((self.W, 36)); bar.set_alpha(200); bar.fill((0, 0, 0))
        self.screen.blit(bar, (0, self.H - 36))
        txt = self.small.render(self.legend, True, (230, 230, 230))
        self.screen.blit(txt, txt.get_rect(center=(self.W // 2, self.H - 18)))
        self.pg.display.flip()
        return time.perf_counter()


class SimRecogIO(SimIO):
    """headless participant: accuracy and RT worsen as contrast drops"""
    def wait_response(self, timeout_s, exposure_s, stim, t_on):
        c = max(stim.contrast, 0.05)
        n = len(CLASSES)
        p = 1 / n + (1 - 1 / n) * min(1.0, c * 1.6) ** 1.5
        rt = self.rng.lognormal(6.0 + 0.9 * (1 - c), 0.35)
        if rt > timeout_s * 1000:
            return None, t_on + timeout_s
        k = stim.true_idx if self.rng.random() < p else int(self.rng.integers(n))
        return k, t_on + rt / 1000.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", default="p01")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--practice", type=int, default=40)
    ap.add_argument("--classes", default=",".join(CLASSES), help="comma list of ZDoom actor names (max 8)")
    ap.add_argument("--levels", default="1.0,0.15,0.08,0.04", help="contrast levels (1 = full)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sd as fraction of 255")
    ap.add_argument("--target-h", type=int, default=200, help="apparent sprite height, px")
    ap.add_argument("--timeout-ms", type=int, default=4000)
    ap.add_argument("--fix-ms", default="500,1000")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args(argv)

    W, H = (int(v) for v in a.resolution.lower().split("x"))
    classes = a.classes.split(",")
    levels = [float(v) for v in a.levels.split(",")]
    fix_lo, fix_hi = (float(v) for v in a.fix_ms.split(","))
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2 ** 31)
    rng = random.Random(seed)
    names = [NICE.get(c, c) for c in classes]

    io = SimRecogIO(seed) if a.simulate else RecogIO(W, H, names, a.fullscreen)
    env = (FakeRecogEnv if a.simulate else DoomRecogEnv)(W, H, seed=seed, classes=classes, target_h=a.target_h)

    os.makedirs(a.out_dir, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a.out_dir, f"{a.participant}_recog{'_SIM' if a.simulate else ''}_{session}")
    with open(base + ".json", "w") as f:
        json.dump({"args": vars(a), "seed": seed, "env": env.info(), "python": sys.version,
                   "platform": platform.platform(), "session": session}, f, indent=2, default=str)

    n = len(classes)
    practice = [(rng.randrange(n), 1.0) for _ in range(a.practice)]
    cells = [(k, c) for k in range(n) for c in levels]
    main_t = cells * max(1, round(a.trials / len(cells)))
    rng.shuffle(main_t)
    plan = [(1, k, c) for k, c in practice] + [(0, k, c) for k, c in main_t]

    rows = []
    try:
        io.text(["Doom monster recognition",
                 "A monster appears. Press its number key as fast and accurately as you can:",
                 "   ".join(f"{i + 1} = {nm}" for i, nm in enumerate(names)),
                 f"{a.practice} practice trials, then {len(plan) - a.practice} real trials.  ESC quits.",
                 "Press SPACE to begin"])
        io.wait_space()
        with open(base + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
            for i, (prac, k, c) in enumerate(plan, 1):
                fix_ms = rng.uniform(fix_lo, fix_hi)
                io.show_fixation(); t0 = io.now()
                stim = env.make_trial(k, c, a.noise)
                rest = fix_ms / 1000 - (io.now() - t0)
                if rest > 0: io.sleep(rest)
                premature = io.flush_keys()
                t_on = io.show_stimulus(stim)
                key, t_r = io.wait_response(a.timeout_ms / 1000, 0, stim, t_on)
                timeout = key is None
                rt = (t_r - t_on) * 1000
                correct = None if timeout else (key == k)
                row = dict(participant=a.participant, session=session, trial=i, practice=prac,
                           class_true=classes[k], class_resp="" if timeout else classes[key], true_idx=k,
                           resp_idx="" if timeout else key, correct="" if timeout else int(correct),
                           rt_ms=round(rt, 2), timeout=int(timeout), premature=int(premature),
                           fast_guess=int((not timeout) and rt < 150), contrast=c, noise=a.noise,
                           bbox_w=stim.bbox_w, bbox_h=stim.bbox_h, back_tics=stim.back_tics,
                           fixation_ms=round(fix_ms, 1), n_classes=n, seed=seed)
                w.writerow(row); fh.flush(); rows.append(row)
                if not a.no_feedback:
                    io.feedback("timeout" if timeout else ("correct" if correct else "error")); io.sleep(0.4)
                io.show_blank(); io.sleep(0.2)
                if i == a.practice and a.practice and not a.simulate:
                    io.text(["Practice done", "Press SPACE to start the real trials"]); io.wait_space()
        io.text(["All done - thank you!"]); io.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted - partial data kept.")
    finally:
        env.close(); io.close()

    main_rows = [r for r in rows if not r["practice"]]
    print("\n contrast    n   acc   median RT ms (correct)")
    for c in sorted({r["contrast"] for r in main_rows}, reverse=True):
        rs = [r for r in main_rows if r["contrast"] == c and not r["timeout"]]
        if rs:
            ok = [r["rt_ms"] for r in rs if r["correct"]]
            print(f" {c:7.2f} {len(rs):4d}  {sum(r['correct'] for r in rs) / len(rs):4.2f}   "
                  f"{statistics.median(ok) if ok else float('nan'):7.0f}")
    print(f"\nSaved: {base}.csv")


if __name__ == "__main__":
    main()
