"""
rtify_ww.py -- a DIFFERENTIABLE Wong-Wang (+ roaming + instance/LCA) decision module on top of the CNN,
trained the RTify way (Cheng et al., NeurIPS 2024).  PyTorch.  Needs:  monsters.npz  and  cnn.pt.

How the gradient gets through a "first to reach threshold" event
----------------------------------------------------------------
The whole trial is simulated step by step (1 ms Euler) as ordinary tensor ops, so gradients can flow
backwards through the dynamics.  The decision time is the threshold-crossing time, linearly interpolated
between the two steps that straddle the threshold:
        tau = (t-1) + (theta - m[t-1]) / (m[t] - m[t-1])        m = highest population rate
so d tau / d m[t]  =  -1 / (m[t] - m[t-1])  at the crossing -- RTify's 1/(Phi_t - Phi_{t-1}) rule.
The choice logits are the population rates at the crossing step (beta * r_at), so cross-entropy flows
back through the attractor into the evidence, the Wong-Wang settings and (optionally) the CNN.
Roaming u0 is a re-parameterised draw (u0 + sd*eps) so its mean and sd are trainable; the LCA route is
simulated in parallel and whichever route crosses first sets tau and the choice.

Modes
   --loss selfpenalty   CE + lam * RT.  No human data.  A speed-accuracy trade-off emerges.  (Ter is frozen.)
   --loss human         match the model's RT quantiles + accuracy to your recognition CSV at each contrast.

Usage
   python rtify_ww.py --selftest                         # checks the torch sim against the numpy one
   python rtify_ww.py --tier ww --loss selfpenalty --steps 150
   python rtify_ww.py --tier ww_cr_lca --loss human --human-csv scott_recog_*.csv --steps 300
   -> writes rtify_params.json, which  python watch_cnn.py --params rtify_params.json  will use.

NOT YET RUN ON REAL DATA by me (no torch in my sandbox): run --selftest first.
"""
import argparse, csv, glob, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ww_npool import A_PHI, B_PHI, D_PHI, GAMMA, I_BG, J_EXT, TAU_S, W_SELF, W_LAT_TOTAL, THR_HZ, NOISE_TAU
from sp2_agents import U0_CRIT, LCA_K, LCA_BETA, LCA_TAU, LCA_Z, N_HALF, C_W
from sp2_nway import NW_PRESETS


def phi_t(I):
    x = A_PHI * I - B_PHI
    small = x.abs() < 1e-6
    xs = torch.where(small, torch.ones_like(x), x)
    out = xs / (1.0 - torch.exp(torch.clamp(-D_PHI * xs, max=60.0)))
    return torch.where(small, torch.full_like(out, 1.0 / D_PHI), out)


class DiffDecision(nn.Module):
    def __init__(self, tier="ww", n=5, t_max=2000, init=None):
        super().__init__()
        p = {**NW_PRESETS[tier], **(init or {})}
        self.tier, self.n, self.T, self.p = tier, n, int(t_max), p
        self.log_gain = nn.Parameter(torch.tensor(math.log(p["gain"])))
        self.u0 = nn.Parameter(torch.tensor(float(p["u0"])))
        self.log_noise = nn.Parameter(torch.tensor(math.log(p["noise"])))
        self.ter = nn.Parameter(torch.tensor(float(p["ter_mu"])))
        self.roam = tier != "ww"
        self.lca = tier == "ww_cr_lca"
        if self.roam:
            self.log_u0sd = nn.Parameter(torch.tensor(math.log(p["u0_sd"])))
        if self.lca:
            self.log_sW = nn.Parameter(torch.tensor(math.log(p["s_W"])))
            self.log_sigL = nn.Parameter(torch.tensor(math.log(p["sigL"])))

    def forward(self, logits):
        B, n = logits.shape
        gain, noise = self.log_gain.exp(), self.log_noise.exp()
        z = logits - logits.mean(1, keepdim=True)
        coh = torch.clamp(gain * z, -0.9, 0.9)
        u0 = self.u0.expand(B)
        if self.roam:
            u0 = (self.u0 + self.log_u0sd.exp() * torch.randn(B)).clamp(0.5, 33.0)
        st = J_EXT * u0[:, None] * (1.0 + coh)
        w_lat = W_LAT_TOTAL / max(n - 1, 1)
        dec, sc = 1.0 - 1.0 / NOISE_TAU, math.sqrt(1.0 / NOISE_TAU)

        S, nz = torch.zeros(B, n), torch.zeros(B, n)
        done = torch.zeros(B, dtype=torch.bool)
        via_lca = torch.zeros(B, dtype=torch.bool)
        tau = torch.full((B,), float(self.T))
        r_at = torch.zeros(B, n)
        m_prev = torch.zeros(B)

        if self.lca:
            p = self.p
            q = F.softmax(logits / p.get("q_temp", 1.0), dim=1)
            N = torch.exp(p["logN"] + p["N_logsd"] * torch.randn(B))
            g = (u0 / U0_CRIT).clamp(0.0, 3.0)
            rho_m = p["rho_max"] * N / (N + N_HALF)
            rho = (g * rho_m)[:, None] * ((1 - p["bias"]) / n + p["bias"] * q)
            weib = (-torch.log(torch.rand(B).clamp_min(1e-6))) ** (1.0 / C_W)
            onset = self.log_sW.exp() * N ** (-1.0 / C_W) * weib
            sigL = self.log_sigL.exp()
            x, ml_prev = torch.zeros(B, n), torch.zeros(B)

        r = torch.zeros(B, n)
        for t in range(1, self.T + 1):
            nz = nz * dec + noise * sc * torch.randn(B, n)
            I = I_BG + st + W_SELF * S + w_lat * (S.sum(1, keepdim=True) - S) + nz
            r = phi_t(I)
            S = S + (-S / TAU_S + GAMMA * (1.0 - S) * r / 1000.0)
            m = r.max(1).values
            cross_w = (~done) & (m >= THR_HZ)
            tau_w = (t - 1) + ((THR_HZ - m_prev) / (m - m_prev + 1e-6)).clamp(0.0, 1.0)
            new_l = torch.zeros(B, dtype=torch.bool)
            if self.lca:
                act = (onset <= t)
                inh = LCA_BETA * (x.sum(1, keepdim=True) - x)
                xn = F.relu(x + (1.0 / LCA_TAU) * (rho - LCA_K * x - inh)
                            + sigL * math.sqrt(1.0 / LCA_TAU) * torch.randn(B, n))
                x = torch.where(act[:, None], xn, x)
                ml = x.max(1).values
                new_l = (~done) & act & (ml >= LCA_Z) & (~cross_w)      # WW wins ties, as in the numpy agent
                tau_l = (t - 1) + ((LCA_Z - ml_prev) / (ml - ml_prev + 1e-6)).clamp(0.0, 1.0)
                tau = torch.where(new_l, tau_l, tau)
                r_at = torch.where(new_l[:, None], x * (THR_HZ / LCA_Z), r_at)
                via_lca = via_lca | new_l
                ml_prev = ml
            tau = torch.where(cross_w, tau_w, tau)
            r_at = torch.where(cross_w[:, None], r, r_at)
            done = done | cross_w | new_l
            m_prev = m
            if bool(done.all()):
                break
        r_at = torch.where(done[:, None], r_at, r)                       # timeouts: read out the last state
        return dict(tau=tau, rt=tau + self.ter, r_at=r_at, done=done, via_lca=via_lca, u0=u0)

    def settings(self):
        d = dict(gain=float(self.log_gain.exp()), u0=float(self.u0), noise=float(self.log_noise.exp()),
                 ter_mu=float(self.ter), ter_sd=self.p["ter_sd"])
        if self.roam:
            d["u0_sd"] = float(self.log_u0sd.exp())
        if self.lca:
            d.update(s_W=float(self.log_sW.exp()), sigL=float(self.log_sigL.exp()))
        return d


# --------------------------------------------------------------------------- self-test
def selftest():
    from ww_npool import run
    torch.manual_seed(0)
    dd = DiffDecision("ww", t_max=2500)
    lg = torch.zeros(200, 5)
    lg[torch.arange(200), torch.arange(200) % 5] = 10.0
    out = dd(lg)
    mt = float(out["tau"].mean())
    coh = np.clip(0.05 * (lg[0].numpy() - lg[0].numpy().mean()), -0.9, 0.9)
    rng = np.random.default_rng(0)
    ts = [(run(coh, u0=16.0, noise=0.02, t_max=2500, rng=rng)["t_ms"] or 2500) for _ in range(150)]
    mn = float(np.mean(ts))
    acc = float((out["r_at"].argmax(1) == (torch.arange(200) % 5)).float().mean())
    out["tau"].mean().backward()
    g_gain, g_u0 = float(dd.log_gain.grad), float(dd.u0.grad)
    print(f"torch mean decision time {mt:.0f} ms | numpy {mn:.0f} ms | torch accuracy {acc:.2f}")
    print(f"d(mean RT)/d(log gain) = {g_gain:.1f}   d(mean RT)/d(u0) = {g_u0:.1f}")
    ok1 = abs(mt - mn) / mn < 0.12
    ok2 = math.isfinite(g_gain) and g_gain < 0 and math.isfinite(g_u0)
    print("forward matches numpy sim:", "PASS" if ok1 else "FAIL", "| gradient finite & more gain -> faster:", "PASS" if ok2 else "FAIL")
    for tier in ("ww_cr", "ww_cr_lca"):
        d2 = DiffDecision(tier, t_max=2500)
        o = d2(lg)
        o["rt"].mean().backward()
        gs = {k: (None if p.grad is None else round(float(p.grad), 2)) for k, p in d2.named_parameters()}
        print(f"{tier}: mean RT {float(o['rt'].mean()):.0f} ms, lca share {float(o['via_lca'].float().mean()):.2f}, grads {gs}")


# --------------------------------------------------------------------------- training
def read_human(path_glob, levels_available):
    rows = []
    for p in sorted(glob.glob(path_glob)):
        with open(p, newline="") as fh:
            rows += list(csv.DictReader(fh))
    qs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    out = {}
    for r in rows:
        try:
            if int(float(r.get("practice", 0))) or int(float(r.get("timeout", 0))) or int(float(r.get("premature", 0))):
                continue
            c = float(r["contrast"]); rt = float(r["rt_ms"]); ok = int(float(r["correct"]))
        except (ValueError, KeyError):
            continue
        lv = min(levels_available, key=lambda v: abs(v - c))
        out.setdefault(lv, []).append((rt, ok))
    res = {}
    for lv, v in out.items():
        a = np.array(v)
        if len(a) >= 10:
            res[lv] = dict(q=np.quantile(a[:, 0], qs), acc=float(a[:, 1].mean()), n=len(a))
    return qs, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default="monsters.npz")
    ap.add_argument("--cnn", default="cnn.pt")
    ap.add_argument("--tier", default="ww", choices=("ww", "ww_cr", "ww_cr_lca"))
    ap.add_argument("--loss", default="selfpenalty", choices=("selfpenalty", "human"))
    ap.add_argument("--human-csv", default="scott_recog_*.csv")
    ap.add_argument("--lam", type=float, default=0.5, help="selfpenalty: weight on RT in seconds")
    ap.add_argument("--beta", type=float, default=0.5, help="rates (Hz) -> choice logits")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--t-max", type=int, default=2000)
    ap.add_argument("--train-cnn", action="store_true", help="also fine-tune the CNN through the attractor (RTify proper)")
    ap.add_argument("--cnn-lr", type=float, default=1e-4)
    ap.add_argument("--out", default="rtify_params.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    from train_cnn import load, Net, to_input
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    X, y, c, rid, classes, ds = load([a.data])
    ids = np.unique(rid); rng.shuffle(ids)
    val_ids = set(ids[:int(len(ids) * 0.15)].tolist())
    is_val = np.array([r in val_ids for r in rid.tolist()])
    Xt, yt, ct = torch.from_numpy(X[~is_val]), torch.from_numpy(y[~is_val].astype(np.int64)), c[~is_val]
    Xv, yv, cv = torch.from_numpy(X[is_val]), torch.from_numpy(y[is_val].astype(np.int64)), c[is_val]
    levels = sorted(set(c.tolist()), reverse=True)

    ck = torch.load(a.cnn, map_location="cpu", weights_only=False)
    cnn = Net(len(classes)); cnn.load_state_dict(ck["state"])
    mean, scale, in_noise = ck["mean"], ck["scale"], ck["input_noise"]
    cnn.train() if a.train_cnn else cnn.eval()

    dd = DiffDecision(a.tier, n=len(classes), t_max=a.t_max)
    groups = [dict(params=[dd.log_gain, dd.log_noise] + [p for nme, p in dd.named_parameters() if nme in ("log_u0sd", "log_sW", "log_sigL")], lr=0.03),
              dict(params=[dd.u0], lr=0.15)]
    if a.loss == "human":
        groups.append(dict(params=[dd.ter], lr=3.0))
    else:
        dd.ter.requires_grad_(False)
    if a.train_cnn:
        groups.append(dict(params=list(cnn.parameters()), lr=a.cnn_lr))
    opt = torch.optim.Adam(groups)

    human = None
    if a.loss == "human":
        qs, human = read_human(a.human_csv, levels)
        if not human:
            raise SystemExit(f"no usable human trials found in {a.human_csv}")
        print("human targets:", {lv: dict(n=h["n"], acc=round(h["acc"], 2), med=round(float(h["q"][2]))) for lv, h in human.items()})
        qs_t = torch.tensor(qs, dtype=torch.float32)

    def get_logits(Xb):
        if a.train_cnn:
            return cnn(to_input(Xb, mean, scale, in_noise))
        with torch.no_grad():
            return cnn(to_input(Xb, mean, scale, in_noise))

    t0 = time.time()
    for step in range(1, a.steps + 1):
        if a.loss == "selfpenalty":
            idx = torch.from_numpy(rng.integers(0, len(Xt), a.batch))
            out = dd(get_logits(Xt[idx]))
            ce = F.cross_entropy(a.beta * out["r_at"], yt[idx])
            loss = ce + a.lam * (out["tau"] / 1000.0).mean()
            info = f"CE {float(ce):.3f}  acc {float((out['r_at'].argmax(1) == yt[idx]).float().mean()):.2f}  tau {float(out['tau'].mean()):.0f}ms"
        else:
            per = max(8, a.batch // len(human))
            idxs, lv_of = [], []
            for lv in human:
                pool = np.where(np.isclose(ct, lv))[0]
                idxs.append(rng.choice(pool, per)); lv_of += [lv] * per
            idx = torch.from_numpy(np.concatenate(idxs))
            lv_arr = np.array(lv_of)
            out = dd(get_logits(Xt[idx]))
            loss, parts = 0.0, []
            ptrue = F.softmax(a.beta * out["r_at"], 1).gather(1, yt[idx][:, None])[:, 0]
            for lv, h in human.items():
                m = torch.from_numpy(lv_arr == lv)
                qm = torch.quantile(out["rt"][m], qs_t)
                lrt = (((qm - torch.tensor(h["q"], dtype=torch.float32)) / 200.0) ** 2).mean()
                lac = 4.0 * (ptrue[m].mean() - h["acc"]) ** 2
                loss = loss + lrt + lac
                parts.append(f"{lv}: acc {float(ptrue[m].mean()):.2f}/{h['acc']:.2f} med {float(qm[2]):.0f}/{float(h['q'][2]):.0f}")
            info = " | ".join(parts)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
        opt.step()
        if step % 10 == 0 or step == 1:
            s = dd.settings()
            print(f"step {step:4d} loss {float(loss):.3f}  {info}   [{time.time()-t0:.0f}s]\n         " +
                  "  ".join(f"{k}={v:.3g}" for k, v in s.items()), flush=True)

    # ---- final validation table
    cnn.eval()
    print("\nvalidation (fitted model)   contrast  acc   median RT   mean RT   lca-share")
    with torch.no_grad():
        for lv in levels:
            sel = np.where(np.isclose(cv, lv))[0][:150]
            o = dd(cnn(to_input(Xv[sel], mean, scale, in_noise)))
            acc = float((o["r_at"].argmax(1) == yv[sel]).float().mean())
            print(f"                            {lv:5.2f}   {acc:.2f}   {float(o['rt'].median()):7.0f}   {float(o['rt'].mean()):7.0f}   {float(o['via_lca'].float().mean()):.2f}")
    json.dump(dd.settings(), open(a.out, "w"), indent=1)
    print("saved", a.out, dd.settings())
    if a.train_cnn:
        torch.save({**ck, "state": cnn.state_dict()}, "cnn_rtify.pt"); print("saved cnn_rtify.pt")


if __name__ == "__main__":
    main()
