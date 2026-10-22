"""
ww_npool.py -- n-population Wong-Wang attractor (reduced model of Wong & Wang 2006) with the same
constants as sp2_kernel.c, generalised from 2 to n choices:  each pool excites itself (w_self) and
inhibits the others; total lateral inhibition is split over the n-1 competitors.
Returns the full trajectory so it can be watched.  Pure numpy; no training anywhere.
"""
import math
import numpy as np

A_PHI, B_PHI, D_PHI, GAMMA, I_BG, J_EXT, TAU_S = 270.0, 108.0, 0.154, 0.641, 0.3255, 0.00052, 100.0
W_SELF, W_LAT_TOTAL, THR_HZ, NOISE_TAU = 0.2609, -0.0497, 15.0, 2.0


def phi(I):
    x = A_PHI * I - B_PHI
    small = np.abs(x) < 1e-8
    x = np.where(small, 1.0, x)
    out = x / (1.0 - np.exp(np.minimum(-D_PHI * x, 600.0)))
    return np.where(small, 1.0 / D_PHI, out)


def run(coh, u0=10.0, noise=0.03, t_max=3000, thr=THR_HZ, w_self=W_SELF, w_lat_total=W_LAT_TOTAL,
        rng=None):
    """coh: length-n array of input biases (pool k gets J_EXT*u0*(1+coh[k])).
    Returns dict(choice, t_ms (None if no decision), r (T x n Hz), S (T x n))."""
    rng = rng or np.random.default_rng()
    coh = np.asarray(coh, float)
    n = coh.size
    w_lat = w_lat_total / max(n - 1, 1)
    st = J_EXT * u0 * (1.0 + coh)
    dec, sc = 1.0 - 1.0 / NOISE_TAU, math.sqrt(1.0 / NOISE_TAU)
    S, nz = np.zeros(n), np.zeros(n)
    R, SS = [], []
    choice, t_dec = None, None
    for t in range(1, int(t_max) + 1):
        nz = nz * dec + noise * sc * rng.standard_normal(n)
        I = I_BG + st + w_self * S + w_lat * (S.sum() - S) + nz
        r = phi(I)
        S = S + (-S / TAU_S + GAMMA * (1.0 - S) * r / 1000.0)
        R.append(r); SS.append(S.copy())
        if r.max() >= thr:
            choice, t_dec = int(np.argmax(r)), t
            break
    return dict(choice=choice, t_ms=t_dec, r=np.array(R), S=np.array(SS))
