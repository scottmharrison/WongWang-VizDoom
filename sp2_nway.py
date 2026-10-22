"""
sp2_nway.py -- the three StarPower-2 tiers as an N-ALTERNATIVE decision module fed by evidence logits.

   logits      --->  [ Wong-Wang n-pool attractor ]  ---+
                |--->  [ instance retrieval -> LCA   ]  ---+--> first to threshold wins -> choice, RT
                       (tier ww_cr_lca only)
   critical roaming: u0 (overall input level of the WW network) wanders trial to trial as a clipped OU
                     process around u0_mean (tiers ww_cr, ww_cr_lca); u0* = 10.68 is the critical point.

  ww         n-pool WW, constant u0
  ww_cr      + roaming u0  (lapses / long RTs when u0 drifts toward criticality)
  ww_cr_lca  + fast instance route: retrieval onset ~ Weibull(s_W * N^-1/2), retrieval strength
               rho_m = rho_max*N/(N+200), N ~ lognormal; the LCA input is rho_m scaled by the class
               posterior q = softmax(logits):  rho_k = g*rho_m*((1-bias)/n + bias*q_k),  g = u0/u0*.
               With n = 2 and confident, correct evidence this reduces exactly to the fitted 2-choice form
               rho = g*rho_m*(1 +/- bias)/2.   (Making the fast route stimulus-dependent through the
               logits is a design choice of this code, not something the pooled fits could test.)

Evidence mapping for the WW route:  coh_k = clip(gain*(logit_k - mean logit), +/-0.9),
pool k receives J_EXT*u0*(1 + coh_k).  RT = decision time + Ter.

All numbers in NW_PRESETS are STARTING POINTS for the n-pool regime (u0=16, noise=0.02 was tuned by
hand); only the LCA parameters come from the fitted Haddara E2 hybrid. 
"""
import math
import numpy as np

from ww_npool import run as ww_run
from sp2_agents import U0_CRIT, LCA_K, LCA_BETA, LCA_TAU, LCA_Z, THETA_ROAM, N_HALF, C_W

NW_TIERS = ("ww", "ww_cr", "ww_cr_lca")

_BASE = dict(u0=16.0, gain=0.05, noise=0.02, ter_mu=200.0, ter_sd=30.0)
NW_PRESETS = {
    "ww":        dict(_BASE),
    "ww_cr":     dict(_BASE, u0_sd=4.0),
    "ww_cr_lca": dict(_BASE, u0_sd=4.0, logN=4.954, N_logsd=1.748, s_W=7000.0, rho_max=1.147,
                      bias=0.828, sigL=0.25, q_temp=1.0),
}


def softmax(v, temp=1.0):
    z = np.asarray(v, float) / temp
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class NWayAgent:
    def __init__(self, tier="ww", params=None, seed=0, n=5):
        assert tier in NW_TIERS, f"tier must be one of {NW_TIERS}"
        self.tier, self.n = tier, n
        self.p = {**NW_PRESETS[tier], **(params or {})}
        self.rng = np.random.default_rng(seed)
        self.u0_state = self._stationary_u0() if tier != "ww" else self.p["u0"]
        self.u0_hist = []

    # ---- critical roaming: clipped OU on u0 (same law as sp2_agents / core.stationary_u0)
    def _step_u0(self, u):
        sig = self.p["u0_sd"] * math.sqrt(1 - (1 - THETA_ROAM) ** 2)
        return float(np.clip(u + THETA_ROAM * (self.p["u0"] - u) + sig * self.rng.standard_normal(), 0.5, 33.0))

    def _stationary_u0(self, burn=80):
        u = self.p["u0"]
        for _ in range(burn):
            u = self._step_u0(u)
        return u

    def _next_u0(self):
        if self.tier == "ww":
            return self.p["u0"]
        self.u0_state = self._step_u0(self.u0_state)
        return self.u0_state

    # ---- fast instance route: n-unit leaky competing accumulator, started after the retrieval onset
    def _lca(self, rho, onset, sigma, deadline):
        a, sq = 1.0 / LCA_TAU, math.sqrt(1.0 / LCA_TAU)
        n = rho.size
        x = np.zeros(n)
        t0 = max(1, math.ceil(onset))
        xs = []
        t = t0
        while t <= onset + deadline:
            inh = LCA_BETA * (x.sum() - x)
            x = np.maximum(x + a * (rho - LCA_K * x - inh) + sigma * sq * self.rng.standard_normal(n), 0.0)
            xs.append(x.copy())
            if x.max() >= LCA_Z:
                return float(t), int(np.argmax(x)), t0, np.array(xs)
            t += 1
        return -1.0, -1, t0, np.array(xs)

    def decide(self, logits, timeout_ms=3000):
        """logits: length-n evidence logits.  Returns a dict with everything needed to watch the trial."""
        p = self.p
        logits = np.asarray(logits, float)
        n = logits.size
        coh = np.clip(p["gain"] * (logits - logits.mean()), -0.9, 0.9)
        q = softmax(logits, p.get("q_temp", 1.0))
        ter = max(self.rng.normal(p["ter_mu"], p["ter_sd"]), 50.0)
        window = timeout_ms - ter
        u0 = self._next_u0()
        self.u0_hist.append(u0)

        lca_trace, lca_t0, tl, cl, onset, rho = None, 0, -1.0, -1, None, None
        if self.tier == "ww_cr_lca":
            N = math.exp(p["logN"] + p["N_logsd"] * self.rng.standard_normal())
            g = min(max(u0 / U0_CRIT, 0.0), 3.0)
            rho_m = p["rho_max"] * N / (N + N_HALF)
            rho = g * rho_m * ((1 - p["bias"]) / n + p["bias"] * q)
            onset = p["s_W"] * N ** (-1.0 / C_W) * self.rng.weibull(C_W)
            if onset < window:
                tl, cl, lca_t0, lca_trace = self._lca(rho, onset, p["sigL"], window)
                if tl > window:
                    tl = -1.0

        stop = tl if 0 < tl < window else window
        ww = ww_run(coh, u0=u0, noise=p["noise"], t_max=int(stop), rng=self.rng)
        kw = float(ww["t_ms"]) if ww["t_ms"] is not None else math.inf
        kl = tl if tl > 0 else math.inf
        lca_wins = kl < kw
        dt, ch = (kl, cl) if lca_wins else (kw, ww["choice"])
        out = dict(tier=self.tier, u0=u0, ter=ter, coh=coh, q=q, ww_r=ww["r"], lca_x=lca_trace,
                   lca_t0=lca_t0, lca_onset=onset, t_ww=kw, t_lca=kl, n=n)
        if not math.isfinite(dt) or dt + ter > timeout_ms:
            out.update(choice=None, rt_ms=float(timeout_ms), dt=None, route="", timeout=True)
        else:
            out.update(choice=int(ch), rt_ms=float(dt + ter), dt=float(dt),
                       route="lca" if lca_wins else "ww", timeout=False)
        return out
