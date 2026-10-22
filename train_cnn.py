"""
train_cnn.py -- train a small CNN to name the monster, for accuracy only (no RTs yet).

  python train_cnn.py --data monsters.npz --epochs 15 --out cnn.pt

Design choices that matter for the later RT work:
  * NO per-image contrast normalisation.  The network must see faded images as faded; if it
    normalised them, contrast would stop affecting the evidence and so could not affect RT.
  * Train/validation split is BY RENDER id (the same monster photo at 6 contrasts never straddles it).
  * --input-noise adds Gaussian 'perceptual noise' (in grey levels, 0-255 scale) at train AND test.
    Without it, a faded noise-free image is just a scaled copy of the clean one and a CNN can read it
    perfectly down to the last grey level -- unlike a person.  This is the knob that makes faded
    images actually harder.  Tune it until accuracy-vs-contrast resembles your human table.
  * Random translation (+/- --shift px) so the net can't rely on the monster sitting in one spot.

Prints accuracy and 'evidence strength' (true-class logit minus mean other logit) per contrast level.
That second column is what will drive the Wong-Wang input later.
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(paths):
    Xs, ys, cs, rs, off = [], [], [], [], 0
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"]); ys.append(d["y"].astype(np.int64)); cs.append(d["contrast"])
        rs.append(d["render"].astype(np.int64) + off); off = int(rs[-1].max()) + 1
        classes = [str(c) for c in d["classes"]]
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(rs), classes, int(d["ds"])


class Net(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16)
        self.c2, self.b2 = nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.c3, self.b3 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, n)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        x = F.relu(self.b3(self.c3(x)))
        return self.fc(x.mean((2, 3)))


def to_input(xb, mean, scale, noise_sd, gen=None):
    """xb: uint8 (B,H,W,3) tensor -> normalised float (B,3,H,W), with perceptual noise added."""
    x = xb.permute(0, 3, 1, 2).float()
    if noise_sd > 0:
        x = x + noise_sd * torch.randn(x.shape, generator=gen)
    return (x - mean) / scale


def shift_batch(x, max_shift, rng):
    if max_shift <= 0:
        return x
    out = torch.empty_like(x)
    for i0 in range(0, x.shape[0], 16):
        dx, dy = (int(v) for v in rng.integers(-max_shift, max_shift + 1, 2))
        out[i0:i0 + 16] = torch.roll(x[i0:i0 + 16], shifts=(dy, dx), dims=(2, 3))
    return out


@torch.no_grad()
def evaluate(net, X, y, c, mean, scale, noise_sd, seed=123):
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    logits = []
    for i in range(0, len(X), 512):
        logits.append(net(to_input(X[i:i + 512], mean, scale, noise_sd, gen)))
    L = torch.cat(logits)
    pred = L.argmax(1)
    true_logit = L.gather(1, y[:, None])[:, 0]
    other = (L.sum(1) - true_logit) / (L.shape[1] - 1)
    return pred, (true_logit - other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["monsters.npz"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--input-noise", type=float, default=3.0, help="perceptual noise sd, grey levels (0-255)")
    ap.add_argument("--shift", type=int, default=6, help="random translation, px (at stored resolution)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="cnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    X, y, c, rid, classes, ds = load(a.data)
    print(f"{len(X)} images {X.shape[1:]}, {len(set(rid.tolist()))} renders, classes={classes}")

    ids = np.unique(rid)
    rng.shuffle(ids)
    val_ids = set(ids[:max(1, int(len(ids) * a.val_frac))].tolist())
    is_val = np.array([r in val_ids for r in rid.tolist()])
    Xt, yt = torch.from_numpy(X[~is_val]), torch.from_numpy(y[~is_val])
    Xv, yv, cv = torch.from_numpy(X[is_val]), torch.from_numpy(y[is_val]), c[is_val]
    print(f"train {len(Xt)}  val {len(Xv)}  (split by render)")

    # mean pixel value, accumulated in chunks (torch upcasts the WHOLE tensor for sum/mean -> GBs)
    tot = 0.0
    for i in range(0, len(Xt), 1000):
        tot += float(Xt[i:i + 1000].double().sum())
    mean = tot / Xt.numel()
    scale = 50.0
    net = Net(len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Xt)))
        tot, correct = 0.0, 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb = shift_batch(to_input(Xt[idx], mean, scale, a.input_noise), a.shift, rng)
            logits = net(xb)
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx); correct += int((logits.argmax(1) == yt[idx]).sum())
        sched.step()
        pred, _ = evaluate(net, Xv, yv, cv, mean, scale, a.input_noise)
        vacc = float((pred == yv).float().mean())
        print(f"epoch {ep+1:2d}/{a.epochs}  loss {tot/len(perm):.3f}  train acc {correct/len(perm):.3f}  "
              f"val acc {vacc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    pred, ev = evaluate(net, Xv, yv, cv, mean, scale, a.input_noise)
    ok = (pred == yv).numpy()
    ev = ev.numpy()
    print("\nvalidation by contrast   (evidence = true-class logit minus mean other logit)")
    print("contrast    n    acc    evidence(mean)")
    for lv in sorted(set(cv.tolist()), reverse=True):
        m = cv == lv
        print(f"  {lv:5.2f}  {m.sum():5d}  {ok[m].mean():.3f}   {ev[m].mean():6.2f}")

    torch.save({"state": net.state_dict(), "mean": mean, "scale": scale, "input_noise": a.input_noise,
                "classes": classes, "ds": ds}, a.out)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
