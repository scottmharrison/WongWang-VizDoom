"""
gen_dataset.py -- render labelled Doom monster frames for CNN training, and TIME the rendering.

Each Doom render (the slow part) is turned into several images, one per contrast level, by the
same degrade() used in the human task, applied at full resolution BEFORE downsampling (so the CNN
sees what the human-task pipeline would show it).  Frames are stored as uint8 (N, H/ds, W/ds, 3).

IMPORTANT for training: every image from one render shares a 'render' id.  Split train/validation
BY render id, otherwise the same monster photo at different contrasts leaks across the split.

Examples
    python gen_dataset.py --fake-env --renders 200 --out test.npz          # pipeline test, no ViZDoom
    python gen_dataset.py --renders 40 --out timing.npz                    # just see how fast it is
    python gen_dataset.py --renders 4000 --out monsters.npz                # real dataset (~5 classes x 800)
"""
import argparse
import time
import numpy as np

from doom_recog_env import CLASSES, DoomRecogEnv, FakeRecogEnv, degrade


def downsample(frame, ds):
    f = frame.astype(np.float32)
    H, W, C = f.shape
    f = f[:H - H % ds, :W - W % ds].reshape(H // ds, ds, W // ds, ds, C).mean((1, 3))
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=int, default=100, help="number of Doom renders (each gives len(levels) images)")
    ap.add_argument("--levels", default="1.0,0.4,0.2,0.15,0.08,0.04", help="contrast levels per render")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sd as fraction of 255")
    ap.add_argument("--ds", type=int, default=8, help="downsample factor (8 -> 60x80)")
    ap.add_argument("--size-jitter", type=int, default=0,
                    help="randomise target sprite height by +/- this many px (0 = fixed at env default)")
    ap.add_argument("--out", default="monsters.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake-env", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    levels = [float(v) for v in a.levels.split(",")]
    env = (FakeRecogEnv if a.fake_env else DoomRecogEnv)(seed=a.seed)
    base_h = getattr(env, "target_h", 200)
    n = len(CLASSES)

    X, y, con, rid = [], [], [], []
    t_render = 0.0
    t0 = time.time()
    for r in range(a.renders):
        k = r % n                                   # balanced classes
        if a.size_jitter and hasattr(env, "target_h"):
            env.target_h = int(base_h + rng.integers(-a.size_jitter, a.size_jitter + 1))
        ts = time.time()
        st = env.make_trial(k, 1.0, 0.0)            # clean frame
        t_render += time.time() - ts
        for c in levels:
            img = degrade(st.frame, c, a.noise, rng)
            X.append(downsample(img, a.ds)); y.append(k); con.append(c); rid.append(r)
        if (r + 1) % 20 == 0 or r + 1 == a.renders:
            el = time.time() - t0
            print(f"render {r+1}/{a.renders}  {el:6.1f}s  ({(r+1)/el:5.2f} renders/s, "
                  f"{t_render/(r+1)*1000:.0f} ms/render in Doom)", flush=True)
    env.close()

    X = np.stack(X)
    np.savez_compressed(a.out, X=X, y=np.array(y, np.int16), contrast=np.array(con, np.float32),
                        render=np.array(rid, np.int32), noise=np.float32(a.noise),
                        classes=np.array(CLASSES), ds=np.int16(a.ds))
    el = time.time() - t0
    rps = a.renders / el
    print(f"\nsaved {a.out}: {len(X)} images {X.shape[1:]} = {X.nbytes/1e6:.0f} MB raw")
    print(f"speed: {rps:.2f} renders/s -> 4000 renders would take ~{4000/rps/60:.0f} min, "
          f"20000 renders ~{20000/rps/60:.0f} min")


if __name__ == "__main__":
    main()
