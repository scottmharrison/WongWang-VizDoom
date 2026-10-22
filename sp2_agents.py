"""
sp2_agents.py -- the three StarPower-2 model tiers as trial-level "agents" that can
play the Doom decision task through the same interface a human uses.

  ww         Wong-Wang base                      (u0, coh, noise, Ter)
  ww_cr      WW + critical roaming               (+ u0_sd : trial-to-trial wandering of u0)
  ww_cr_lca  WW + roaming + instance/LCA route   (+ logN, s_W, rho_max, bias, N_logsd, sigL)

Equations, constants and Euler step (dt = 1 ms) are the same as sp2_kernel.c / core.py in the
tiered_fits project code; this is a pure-Python port so it runs on Windows without gcc.
Choice 0 = channel 1 = the stimulus side (correct); the three tiers are nested by switching
the roaming / fast-route pieces on or off.

What is NEW relative to the pooled fits (flagged so you can decide):
  * stimulus difficulty enters through coh(offset) = coh_max * (1 - exp(-offset/coh_px))
  * roaming can run as a real chain across trials (roam_mode="chain", gives serial
    dependence) or as iid draws from the same stationary law (roam_mode="iid", what the
    pooled fits assumed)
  * the instance/LCA route is stimulus-independent (as in the fits): it sets a fixed-accuracy
    fast guess, so difficulty only acts on the WW route.
"""
import math
import numpy as np

# --- constants identical to sp2_kernel.c / core.py
A_PHI, B_PHI, D_PHI, GAMMA, I_BG, J_EXT, TAU_S = 270.0, 108.0, 0.154, 0.641, 0.3255, 0.00052, 100.0
U0_CRIT, WSELF, WLAT, THR_HZ, NOISE_TAU = 10.676805763871442, 0.2609, -0.0497, 15.0, 2.0
LCA_K, LCA_BETA, LCA_TAU, LCA_Z = 0.2, 0.3, 20.0, 1.0
THETA_ROAM, N_HALF, C_W = 0.1, 200.0, 2.0

TIERS = ("ww", "ww_cr", "ww_cr_lca")

# Starting points = fitted values from tiered_results.json (TEDDM for ww / ww_cr; Haddara E2
# for the hybrid, the dataset where the fast route is actually used).  Refit on your own data.
PRESETS = {
    "ww":        dict(u0=2.96, coh_max=0.54, noise=0.0353, ter_mu=154.0, ter_sd=96.8),
    "ww_cr":     dict(u0=16.24, coh_max=0.097, noise=0.0301, u0_sd=11.48, ter_mu=194.7, ter_sd=6.1),
    "ww_cr_lca": dict(u0=10.75, coh_max=0.30, noise=0.0050, u0_sd=0.05, logN=4.954, s_W=2502.5,
                      rho_max=1.147, bias=0.828, N_logsd=1.748, sigL=0.562, ter_mu=233.7, ter_sd=245.1),
}
COMMON = dict(coh_px=8.0, roam_mode="chain")     # offset->coherence map; roaming mode


def _phi(I):
    x = A_PHI * I - B_PHI
    if abs(x) < 1e-8:
        return 1.0 / D_PHI
    return x / (1.0 - math.exp(min(-D_PHI * x, 600.0)))


class Agent:
    def __init__(self, tier="ww", params=None, seed=0):
        assert tier in TIERS, f"tier must be one of {TIERS}"
        self.tier = tier
        self.p = {**COMMON, **PRESETS[tier], **(params or {})}
        self.rng = np.random.default_rng(seed)
        self._buf, self._i = [], 0
        self.u0_state = None
        if tier != "ww":
            self.u0_state = self._stationary_u0()

    # ---- noise helper (chunked normals: much faster than per-step numpy calls)
    def _g(self):
        if self._i >= len(self._buf):
            self._buf = self.rng.standard_normal(8192).tolist(); self._i = 0
        v = self._buf[self._i]; self._i += 1
        return v

    # ---- roaming (clipped OU on u0, same law as core.stationary_u0)
    def _sig(self):
        return self.p["u0_sd"] * math.sqrt(1 - (1 - THETA_ROAM) ** 2)

    def _step_u0(self, u):
        return float(np.clip(u + THETA_ROAM * (self.p["u0"] - u) + self._sig() * self.rng.standard_normal(), 0.5, 33.0))

    def _stationary_u0(self, burn=80):
        u = self.p["u0"]
        for _ in range(burn):
            u = self._step_u0(u)
        return u

    def _next_u0(self):
        if self.tier == "ww":
            return self.p["u0"]
        if self.p["roam_mode"] == "iid":
            return self._stationary_u0()
        self.u0_state = self._step_u0(self.u0_state)
        return self.u0_state

    # ---- the two routes
    def _lca(self, rho1, rho2, onset, sigma, deadline):
        a, sq = 1.0 / LCA_TAU, math.sqrt(1.0 / LCA_TAU)
        x1 = x2 = 0.0
        t = max(1, math.ceil(onset))
        while True:
            n1 = x1 + a * (rho1 - LCA_K * x1 - LCA_BETA * x2) + sigma * sq * self._g()
            n2 = x2 + a * (rho2 - LCA_K * x2 - LCA_BETA * x1) + sigma * sq * self._g()
            x1 = n1 if n1 > 0 else 0.0
            x2 = n2 if n2 > 0 else 0.0
            m = x1 if x1 > x2 else x2
            if m >= LCA_Z:
                return float(t), 0 if x1 >= x2 else 1
            if t >= onset + deadline:
                return -1.0, -1
            t += 1

    def _ww(self, u0, coh, noise, t_stop):
        dec, sc = 1.0 - 1.0 / NOISE_TAU, math.sqrt(1.0 / NOISE_TAU)
        st1, st2 = J_EXT * u0 * (1 + coh), J_EXT * u0 * (1 - coh)
        S1 = S2 = n1 = n2 = 0.0
        for t in range(1, int(t_stop) + 1):
            n1 = n1 * dec + noise * sc * self._g()
            n2 = n2 * dec + noise * sc * self._g()
            I1 = I_BG + st1 + WSELF * S1 + WLAT * S2 + n1
            I2 = I_BG + st2 + WSELF * S2 + WLAT * S1 + n2
            r1, r2 = _phi(I1), _phi(I2)
            S1 += -S1 / TAU_S + GAMMA * (1 - S1) * r1 / 1000.0
            S2 += -S2 / TAU_S + GAMMA * (1 - S2) * r2 / 1000.0
            if max(r1, r2) >= THR_HZ:
                return float(t), 0 if r1 >= r2 else 1
        return -1.0, -1

    def coh(self, offset_px):
        return self.p["coh_max"] * (1.0 - math.exp(-abs(offset_px) / self.p["coh_px"]))

    # ---- one trial
    def decide(self, offset_px, true_side, timeout_ms):
        """Returns (response 'left'/'right'/None, rt_ms, info).  RT = decision time + Ter."""
        p = self.p
        ter = self.rng.normal(p["ter_mu"], p["ter_sd"])
        window = timeout_ms - ter
        u0 = self._next_u0()
        tl, cl = -1.0, -1
        if self.tier == "ww_cr_lca":
            N = math.exp(p["logN"] + p["N_logsd"] * self.rng.standard_normal())
            g = min(max(u0 / U0_CRIT, 0.0), 3.0)
            rho_m = p["rho_max"] * N / (N + N_HALF)
            r1, r2 = g * rho_m * (1 + p["bias"]) / 2, g * rho_m * (1 - p["bias"]) / 2
            onset = p["s_W"] * N ** (-1.0 / C_W) * self.rng.weibull(C_W)
            if onset < window:
                tl, cl = self._lca(r1, r2, onset, p["sigL"], window)
        stop = window if not (0 < tl < window) else tl
        tw, cw = self._ww(u0, self.coh(offset_px), p["noise"], stop)
        kw, kl = (tw if tw > 0 else math.inf), (tl if tl > 0 else math.inf)
        lca_wins = kl < kw
        dt, ch = (kl, cl) if lca_wins else (kw, cw)
        info = dict(agent=self.tier, u0_eff=round(u0, 3), route="lca" if lca_wins else "ww")
        if not math.isfinite(dt) or dt + ter > timeout_ms:
            info["route"] = ""
            return None, timeout_ms, info
        rt = max(dt + ter, 1.0)
        other = "left" if true_side == "right" else "right"
        return (true_side if ch == 0 else other), rt, info
