#!/usr/bin/env python3
"""
watch_cnn.py -- watch the whole pipeline decide on Doom frames:

   Doom frame -> CNN (cnn.pt) -> logits -> [ Wong-Wang n-pool  (+ critical roaming)  (+ instance/LCA) ]
                                         -> first route to threshold wins -> choice + RT

  python watch_cnn.py                         # live window, tier ww_cr_lca, real CNN + ViZDoom
  python watch_cnn.py --agent ww_cr           # start on another tier (press 1/2/3 to switch live)
  python watch_cnn.py --params rtify_params.json    # use WW settings fitted by rtify_ww.py
  python watch_cnn.py --png out.png --trials 12     # no window: gallery of traces
  python watch_cnn.py --fake-env --fake-cnn --png out.png    # no ViZDoom / no torch: pipeline test only

Window panels:  top-right  Wong-Wang population rates (one line per monster class) + threshold
                middle     instance route: LCA units (only tier 3) + their threshold
                bottom     critical roaming: u0 on each recent trial vs the critical point u0*
                left       the Doom frame, and the CNN's class probabilities
Keys: SPACE next trial | 1 / 2 / 3 switch tier | ESC quit
"""
import argparse, csv, json, os, random, time
import numpy as np

from doom_recog_env import DoomRecogEnv, FakeRecogEnv, CLASSES, NICE
from sp2_nway import NWayAgent, NW_TIERS
from sp2_agents import U0_CRIT, LCA_Z
from ww_npool import THR_HZ

COLS = [(230, 80, 80), (80, 200, 80), (90, 140, 255), (240, 200, 60), (200, 100, 220), (80, 220, 220), (240, 140, 60), (180, 180, 180)]


# --------------------------------------------------------------------------- vision front ends
class CNNFront:
    """The trained CNN from train_cnn.py, applied exactly like in training (downsample, +perceptual noise)."""
    def __init__(self, path, seed=0):
        import torch
        from train_cnn import Net
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.net = Net(len(ck["classes"]))
        self.net.load_state_dict(ck["state"]); self.net.eval()
        self.mean, self.scale, self.noise = ck["mean"], ck["scale"], ck["input_noise"]
        self.ds, self.classes = ck.get("ds", 8), ck["classes"]
        self.gen = torch.Generator().manual_seed(seed)

    def logits(self, frame, k=None, c=None):
        from gen_dataset import downsample
        from train_cnn import to_input
        xb = self.torch.from_numpy(downsample(frame, self.ds))[None]
        with self.torch.no_grad():
            return self.net(to_input(xb, self.mean, self.scale, self.noise, self.gen))[0].numpy().astype(float)


class FakeCNN:
    """Pipeline test only: logits = noise + margin(contrast) on the true class.  NOT a vision model."""
    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def logits(self, frame, k=None, c=1.0):
        l = self.rng.normal(0, 2.0, self.n)
        l[k] += 12.0 * (max(c, 0.01) ** 0.25)
        return l


# --------------------------------------------------------------------------- one trial
def do_trial(a, env, cnn, agent, k, c):
    stim = env.make_trial(k, c, 0.0)
    logits = cnn.logits(stim.frame, k, c)
    res = agent.decide(logits, a.timeout)
    return stim, logits, res


def verdict(res, names, k):
    ch = res["choice"]
    if ch is None:
        return "no decision (timeout)", (240, 200, 60)
    ok = ch == k
    route = "instance/LCA route" if res["route"] == "lca" else "Wong-Wang route"
    return (f"decided: {names[ch]}  {'correct' if ok else 'WRONG'}   RT {res['rt_ms']:.0f} ms   via {route}",
            (80, 220, 80) if ok else (240, 80, 80))


# --------------------------------------------------------------------------- PNG gallery
def run_png(a, env, cnn, agents, names, order):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(order); cols = 3; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
    for ax, (k, c) in zip(axs.ravel(), order):
        stim, lg, res = do_trial(a, env, cnn, agents[a.agent], k, c)
        r = res["ww_r"]; t = np.arange(1, len(r) + 1)
        for j in range(len(names)):
            ax.plot(t, r[:, j], color=np.array(COLS[j % 8]) / 255, lw=1.3, label=names[j] if ax is axs[0, 0] else None)
            if res["lca_x"] is not None and len(res["lca_x"]):
                tl = res["lca_t0"] + np.arange(len(res["lca_x"]))
                ax.plot(tl, res["lca_x"][:, j] * THR_HZ / LCA_Z, color=np.array(COLS[j % 8]) / 255, lw=1.0, ls=":")
        ax.axhline(THR_HZ, color="k", ls="--", lw=0.8)
        v, _ = verdict(res, names, k)
        ax.set_title(f"true {names[k]}  c={c}  u0={res['u0']:.1f}\n{v}", fontsize=8)
        ax.set_ylim(0, THR_HZ * 1.7); ax.set_xlim(0, a.t_view)
    axs[0, 0].legend(fontsize=7)
    for ax in axs.ravel()[n:]: ax.axis("off")
    fig.suptitle(f"tier: {a.agent}   (solid = Wong-Wang rates, dotted = LCA units scaled to the same threshold)", fontsize=10)
    fig.supxlabel("decision time (ms)"); fig.tight_layout(); fig.savefig(a.png, dpi=105); print("saved", a.png)


# --------------------------------------------------------------------------- live window
def run_window(a, env, cnn, agents, names, order, logger):
    import pygame as pg
    pg.init()
    Wd, Ht = 1420, 760
    scr = pg.display.set_mode((Wd, Ht)); pg.display.set_caption("Doom -> CNN -> Wong-Wang + roaming + instance LCA")
    f_s, f_m = pg.font.Font(None, 22), pg.font.Font(None, 30)
    PX, PW = 560, 820
    clock = pg.time.Clock()
    ymax_ww = THR_HZ * 1.7
    tier = a.agent
    for ti, (k, c) in enumerate(order, 1):
        stim, lg, res = do_trial(a, env, cnn, agents[tier], k, c)
        logger(ti, tier, names, k, c, res)
        surf = pg.transform.smoothscale(pg.surfarray.make_surface(stim.frame.swapaxes(0, 1)), (520, 390))
        q = res["q"]; r = res["ww_r"]; T = len(r)
        end = int(res["dt"]) if res["dt"] else T
        t0 = time.perf_counter(); done_at = None; skip = False; saved = False; force = False
        while not skip:
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                    pg.quit(); return
                if ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_SPACE:
                        if saved: skip = True
                        else: force = True            # jump to the finished screen first so the saved image is complete
                    elif ev.key in (pg.K_1, pg.K_2, pg.K_3): tier = NW_TIERS[ev.key - pg.K_1]
            idx = end if force else min(end, int((time.perf_counter() - t0) * 1000 * a.speed))
            scr.fill((18, 18, 22))
            scr.blit(f_m.render(f"trial {ti}/{len(order)}    tier: {res['tier']}    true: {names[k]}    contrast {c}", True, (230, 230, 230)), (10, 12))
            scr.blit(surf, (10, 50))
            # --- CNN probabilities
            scr.blit(f_s.render("CNN class probabilities (softmax of logits)", True, (180, 180, 180)), (10, 450))
            for j, nm in enumerate(names):
                y = 478 + j * 30
                pg.draw.rect(scr, COLS[j % 8], (10, y, 14, 14))
                scr.blit(f_s.render(nm + (" *" if j == k else ""), True, (230, 230, 230)), (32, y - 2))
                pg.draw.rect(scr, COLS[j % 8], (170, y, int(q[j] * 330), 14))
            # --- Wong-Wang panel
            wy, wh = 50, 270
            pg.draw.rect(scr, (60, 60, 70), (PX, wy, PW, wh), 1)
            scr.blit(f_s.render("Wong-Wang populations: rate (Hz) vs decision time (ms)", True, (200, 200, 200)), (PX, wy - 20))
            ty = wy + wh - THR_HZ / ymax_ww * wh
            for x in range(PX, PX + PW, 12): pg.draw.line(scr, (150, 150, 150), (x, ty), (x + 6, ty), 1)
            n_ww = min(idx, T)
            if n_ww >= 2:
                xs = PX + np.arange(1, n_ww + 1) / a.t_view * PW
                for j in range(len(names)):
                    ys = wy + wh - np.minimum(r[:n_ww, j], ymax_ww) / ymax_ww * wh
                    pg.draw.lines(scr, COLS[j % 8], False, list(zip(xs, ys)), 2)
            # --- LCA panel
            ly, lh = 360, 150
            pg.draw.rect(scr, (60, 60, 70), (PX, ly, PW, lh), 1)
            scr.blit(f_s.render("Instance route (LCA units; threshold = dashed)" if res["tier"] == "ww_cr_lca"
                                else "Instance route: OFF in this tier (press 3)", True, (200, 200, 200)), (PX, ly - 20))
            if res["tier"] == "ww_cr_lca":
                tz = ly + lh - LCA_Z / 1.4 * lh
                for x in range(PX, PX + PW, 12): pg.draw.line(scr, (150, 150, 150), (x, tz), (x + 6, tz), 1)
                if res["lca_onset"] is not None:
                    ox = PX + res["lca_onset"] / a.t_view * PW
                    if ox < PX + PW: pg.draw.line(scr, (110, 110, 160), (ox, ly), (ox, ly + lh), 1)
                    scr.blit(f_s.render("retrieval onset", True, (130, 130, 190)), (min(ox + 4, PX + PW - 110), ly + 2))
                if res["lca_x"] is not None and len(res["lca_x"]):
                    m = int(np.clip(idx - res["lca_t0"], 0, len(res["lca_x"])))
                    if m >= 2:
                        xs = PX + (res["lca_t0"] + np.arange(m)) / a.t_view * PW
                        for j in range(len(names)):
                            ys = ly + lh - np.minimum(res["lca_x"][:m, j], 1.4) / 1.4 * lh
                            pg.draw.lines(scr, COLS[j % 8], False, list(zip(xs, ys)), 2)
            # --- roaming strip
            ry, rh = 560, 110
            pg.draw.rect(scr, (60, 60, 70), (PX, ry, PW, rh), 1)
            scr.blit(f_s.render("Critical roaming: u0 on recent trials  (red dashed = critical point u0* = 10.7)", True, (200, 200, 200)), (PX, ry - 20))
            hist = agents[res["tier"]].u0_hist[-60:]
            umax = 34.0
            yc = ry + rh - U0_CRIT / umax * rh
            for x in range(PX, PX + PW, 12): pg.draw.line(scr, (200, 70, 70), (x, yc), (x + 6, yc), 1)
            if len(hist) >= 2:
                xs = PX + np.arange(len(hist)) / 59.0 * (PW - 10)
                pg.draw.lines(scr, (120, 200, 255), False, [(x, ry + rh - u / umax * rh) for x, u in zip(xs, hist)], 2)
            if hist:
                pg.draw.circle(scr, (255, 255, 255), (int(PX + (len(hist) - 1) / 59.0 * (PW - 10)), int(ry + rh - hist[-1] / umax * rh)), 4)
            scr.blit(f_s.render(f"this trial u0 = {res['u0']:.1f}", True, (120, 200, 255)), (PX + PW - 160, ry + 4))
            # --- result
            if idx >= end:
                if done_at is None: done_at = time.perf_counter()
                msg, col = verdict(res, names, k)
                scr.blit(f_m.render(msg, True, col), (10, Ht - 50))
                if res["dt"]:
                    dx = PX + res["dt"] / a.t_view * PW
                    pg.draw.line(scr, col, (dx, wy), (dx, wy + wh), 2)
                if time.perf_counter() - done_at > a.hold: skip = True
            scr.blit(f_s.render("SPACE next   1/2/3 tier   ESC quit", True, (120, 120, 120)), (Wd - 260, Ht - 26))
            if idx >= end and not saved:                       # finished screen is fully drawn -> save exactly what is shown
                saved = True
                if a.shot_dir:
                    ch = res["choice"]
                    resp = "none" if ch is None else names[ch]
                    fn = (f"{ti:03d}_{res['tier']}_true-{names[k]}_c{c}_resp-{resp}_"
                          f"{'ok' if ch == k else 'ERR'}_rt{res['rt_ms']:.0f}_{res['route'] or 'none'}.png")
                    pg.image.save(scr, os.path.join(a.shot_dir, fn))
                if force: skip = True
            pg.display.flip(); clock.tick(60)
    pg.quit()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn", default="cnn.pt")
    ap.add_argument("--agent", choices=NW_TIERS, default="ww_cr_lca")
    ap.add_argument("--params", default=None, help="json of decision-module settings (from rtify_ww.py)")
    ap.add_argument("--levels", default="1.0,0.15,0.08,0.04")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=3000.0)
    ap.add_argument("--t-view", type=float, default=2000.0, help="width of the time axis (ms)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--csv", default=None, help="csv path (default: model_<tier>_<timestamp>.csv in the current folder)")
    ap.add_argument("--no-csv", action="store_true", help="do not save a csv")
    ap.add_argument("--shots", default=None, help="folder for a screenshot of every finished trial (default: watch_shots_<timestamp>)")
    ap.add_argument("--no-shots", action="store_true", help="do not save screenshots")
    ap.add_argument("--headless", action="store_true", help="no window: run all trials as fast as possible and save the csv")
    ap.add_argument("--fake-env", action="store_true")
    ap.add_argument("--fake-cnn", action="store_true")
    ap.add_argument("--png", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2 ** 31)
    classes = CLASSES; names = [NICE.get(c, c) for c in classes]
    levels = [float(v) for v in a.levels.split(",")]
    extra = json.load(open(a.params)) if a.params else None
    agents = {t: NWayAgent(t, extra, seed=seed + i, n=len(classes)) for i, t in enumerate(NW_TIERS)}
    cnn = FakeCNN(len(classes), seed) if a.fake_cnn else CNNFront(a.cnn, seed)
    env = (FakeRecogEnv if a.fake_env else DoomRecogEnv)(640, 480, seed=seed, classes=classes)
    order = [(i % len(classes), levels[(i // len(classes)) % len(levels)]) for i in range(a.trials)]
    random.Random(seed).shuffle(order)

    csv_path = None if a.no_csv else (a.csv or time.strftime(f"model_{a.agent}_%Y%m%d_%H%M%S.csv"))
    a.shot_dir = None
    if not (a.png or a.headless or a.no_shots):
        a.shot_dir = a.shots or time.strftime("watch_shots_%Y%m%d_%H%M%S")
        os.makedirs(a.shot_dir, exist_ok=True)
    fh = open(csv_path, "w", newline="") if csv_path else None
    wr = csv.writer(fh) if fh else None
    if wr: wr.writerow(["trial", "tier", "true", "resp", "correct", "rt_ms", "route", "u0", "contrast", "q_true", "timeout"])

    def logger(ti, tier, nm, k, c, res):
        ch = res["choice"]
        print(f"trial {ti:3d} [{tier:9s}] true={nm[k]:10s} c={c:<5} -> {'none' if ch is None else nm[ch]:10s} "
              f"RT={res['rt_ms']:6.0f} route={res['route'] or '-':3s} u0={res['u0']:.1f}")
        if wr: wr.writerow([ti, tier, nm[k], "" if ch is None else nm[ch], int(ch == k), round(res["rt_ms"]), res["route"], round(res["u0"], 2), c,
                             round(float(res["q"][k]), 4), int(res["timeout"])])

    try:
        if a.png:
            run_png(a, env, cnn, agents, names, order)
        elif a.headless:
            for ti, (k, c) in enumerate(order, 1):
                _, _, res = do_trial(a, env, cnn, agents[a.agent], k, c)
                logger(ti, a.agent, names, k, c, res)
        else:
            run_window(a, env, cnn, agents, names, order, logger)
    finally:
        env.close()
        if a.shot_dir: print(f"\nscreenshots saved in {os.path.abspath(a.shot_dir)}")
        if fh:
            fh.close(); print(f"\nsaved {csv_path}  (in the folder you ran the command from: {__import__('os').getcwd()})")


if __name__ == "__main__":
    main()
