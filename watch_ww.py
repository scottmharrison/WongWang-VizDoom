#!/usr/bin/env python3
"""
watch_ww.py -- watch an (untrained) n-pool Wong-Wang attractor decide on live Doom frames.

  Doom frame -> matched-filter evidence per monster type -> input bias to each WW population
  -> populations compete -> first to reach threshold = the decision; steps to threshold = RT.

    python watch_ww.py                       # live window, ViZDoom frames
    python watch_ww.py --levels 1.0,0.3,0.15 --speed 2
    python watch_ww.py --png out.png --trials 12      # no window: save a trace gallery
    python watch_ww.py --fake-env --png out.png       # no ViZDoom needed (coloured boxes)
Keys (window): SPACE next trial, ESC quit.  Parameters: --u0 --noise --gain --ter --thr.
"""
import argparse, random, time
import numpy as np
import ww_npool as W
from doom_recog_env import DoomRecogEnv, FakeRecogEnv, CLASSES, NICE
from proto_frontend import get_prototypes, evidence_to_coh

COLS = [(230, 80, 80), (80, 200, 80), (90, 140, 255), (240, 200, 60), (200, 100, 220), (80, 220, 220), (240, 140, 60), (180, 180, 180)]


def make_trial(env, protos, k, c, a, rng):
    stim = env.make_trial(k, c, 0.0)
    e = protos.evidence(stim.frame)
    coh = evidence_to_coh(e, a.gain)
    res = W.run(coh, u0=a.u0, noise=a.noise, t_max=a.t_max, thr=a.thr, rng=rng)
    return stim, e, coh, res


def run_png(a, env, protos, names, rng, order):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(order); cols = 4; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    for ax, (k, c) in zip(axs.ravel(), order):
        stim, e, coh, res = make_trial(env, protos, k, c, a, rng)
        t = np.arange(1, len(res["r"]) + 1)
        for j in range(len(names)):
            ax.plot(t, res["r"][:, j], color=np.array(COLS[j % 8]) / 255, lw=1.4, label=names[j] if ax is axs[0, 0] else None)
        ax.axhline(a.thr, color="k", ls="--", lw=0.8)
        ch = res["choice"]
        verdict = "no decision" if ch is None else f"{names[ch]} {'OK' if ch == k else 'WRONG'} RT {res['t_ms'] + a.ter:.0f}ms"
        ax.set_title(f"true {names[k]}  contrast {c}\n{verdict}", fontsize=9)
        ax.set_ylim(0, a.thr * 1.7)
    axs[0, 0].legend(fontsize=7)
    for ax in axs.ravel()[n:]: ax.axis("off")
    fig.supxlabel("time (ms)"); fig.supylabel("population rate (Hz)")
    fig.tight_layout(); fig.savefig(a.png, dpi=110); print("saved", a.png)


def run_window(a, env, protos, names, rng, order):
    import pygame as pg
    pg.init()
    Wd, Ht = 1200, 580
    scr = pg.display.set_mode((Wd, Ht)); pg.display.set_caption("Wong-Wang attractor on Doom frames")
    f_s, f_m = pg.font.Font(None, 24), pg.font.Font(None, 32)
    px0, py0, pw, ph = 620, 60, 560, 300
    ymax, tview = a.thr * 1.7, a.t_max
    clock = pg.time.Clock()
    for ti, (k, c) in enumerate(order, 1):
        stim, e, coh, res = make_trial(env, protos, k, c, a, rng)
        surf = pg.transform.smoothscale(pg.surfarray.make_surface(stim.frame.swapaxes(0, 1)), (580, 435))
        r = res["r"]; T = len(r); t0 = time.perf_counter(); done_at = None; skip = False
        while not skip:
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                    pg.quit(); return
                if ev.type == pg.KEYDOWN and ev.key == pg.K_SPACE: skip = True
            idx = min(T, int((time.perf_counter() - t0) * 1000 * a.speed))
            scr.fill((18, 18, 22))
            scr.blit(surf, (10, 60))
            scr.blit(f_m.render(f"trial {ti}/{len(order)}   true: {names[k]}   contrast {c}", True, (230, 230, 230)), (10, 15))
            pg.draw.rect(scr, (60, 60, 70), (px0, py0, pw, ph), 1)
            ty = py0 + ph - a.thr / ymax * ph
            for x in range(px0, px0 + pw, 12): pg.draw.line(scr, (150, 150, 150), (x, ty), (x + 6, ty), 1)
            scr.blit(f_s.render("threshold", True, (150, 150, 150)), (px0 + pw - 80, ty - 18))
            if idx >= 2:
                xs = px0 + np.arange(idx) / tview * pw
                for j in range(len(names)):
                    ys = py0 + ph - np.minimum(r[:idx, j], ymax) / ymax * ph
                    pg.draw.lines(scr, COLS[j % 8], False, list(zip(xs, ys)), 2)
            scr.blit(f_s.render("population rate (Hz) vs time (ms)  -  each coloured line is one candidate", True, (200, 200, 200)), (px0, py0 - 28))
            for j, nm in enumerate(names):                                   # legend + evidence bars
                y = 390 + j * 34
                pg.draw.rect(scr, COLS[j % 8], (px0, y, 14, 14))
                scr.blit(f_s.render(nm + (" *" if j == k else ""), True, (230, 230, 230)), (px0 + 22, y - 2))
                bx = px0 + 190; wbar = int(abs(coh[j]) * 450)
                pg.draw.rect(scr, COLS[j % 8] if coh[j] >= 0 else (90, 90, 90), (bx if coh[j] >= 0 else bx - wbar, y, wbar, 14))
            pg.draw.line(scr, (120, 120, 120), (px0 + 190, 385), (px0 + 190, 385 + len(names) * 34), 1)
            scr.blit(f_s.render("input bias from the front end (coh)", True, (170, 170, 170)), (px0 + 190, 372))
            if idx >= T:
                if done_at is None: done_at = time.perf_counter()
                ch = res["choice"]
                if ch is None: msg, col = "no decision", (240, 200, 60)
                else:
                    ok = ch == k
                    msg = f"decided: {names[ch]}   {'correct' if ok else 'WRONG'}   RT {res['t_ms'] + a.ter:.0f} ms"
                    col = (80, 220, 80) if ok else (240, 80, 80)
                scr.blit(f_m.render(msg, True, col), (10, Ht - 45))
                if time.perf_counter() - done_at > a.hold: skip = True
            pg.display.flip(); clock.tick(60)
        ch = res["choice"]
        print(f"trial {ti:3d} true={names[k]:12s} c={c:<5} -> {'none' if ch is None else names[ch]:12s} RT={'-' if ch is None else round(res['t_ms'] + a.ter)}")
    pg.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--levels", default="1.0,0.4,0.2")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--u0", type=float, default=16.0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--gain", type=float, default=0.5, help="evidence -> input bias scale")
    ap.add_argument("--thr", type=float, default=W.THR_HZ)
    ap.add_argument("--ter", type=float, default=200.0, help="non-decision time added to RT (ms)")
    ap.add_argument("--t-max", type=int, default=3000)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to show the result")
    ap.add_argument("--k", type=int, default=6, help="frames per class used to build prototypes")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fake-env", action="store_true")
    ap.add_argument("--png", default=None, help="save a gallery PNG instead of opening a window")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    classes = a.classes.split(","); names = [NICE.get(c, c) for c in classes]
    levels = [float(v) for v in a.levels.split(",")]
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2 ** 31)
    rng = np.random.default_rng(seed)
    env = (FakeRecogEnv if a.fake_env else DoomRecogEnv)(640, 480, seed=seed, classes=classes)
    print("building/loading prototypes ...")
    protos = get_prototypes(env, classes, a.k, a.rebuild)
    order = [(i % len(classes), levels[(i // len(classes)) % len(levels)]) for i in range(a.trials)]
    random.Random(seed).shuffle(order)
    try:
        (run_png if a.png else run_window)(a, env, protos, names, rng, order)
    finally:
        env.close()


if __name__ == "__main__":
    main()
