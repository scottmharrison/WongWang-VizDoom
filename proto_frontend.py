"""
proto_frontend.py -- a training-free 'vision' front end: matched-filter prototypes.
Templates are the average (downsampled, mean-subtracted) frame of each monster type, taken from
a handful of full-contrast Doom frames.  Evidence for class k is the projection of a new frame onto
the template difference T_k - mean_j(T_j).  Nothing is learned by gradient descent; lower image
contrast shrinks the projections, so the attractor gets weaker input and slows down.
"""
import os
import numpy as np


def prep(frame, ds=8):
    f = frame.astype(np.float32)
    H, W, C = f.shape
    f = f[:H - H % ds, :W - W % ds].reshape(H // ds, ds, W // ds, ds, C).mean((1, 3))
    return f - f.mean()


class Prototypes:
    def __init__(self, templates, classes):
        self.classes = list(classes)
        t = np.asarray(templates, np.float32)
        self.T = t - t.mean(0, keepdims=True)                     # (n, h, w, c) template differences
        self.norm2 = (self.T ** 2).reshape(len(self.T), -1).sum(1)

    @classmethod
    def build(cls, env, classes, k=6):
        temps = []
        for ci in range(len(classes)):
            temps.append(np.mean([prep(env.make_trial(ci, 1.0, 0.0).frame) for _ in range(k)], 0))
        return cls(temps, classes)

    def evidence(self, frame):
        f = prep(frame)
        return (self.T.reshape(len(self.T), -1) @ f.ravel()) / self.norm2

    def save(self, path):
        np.savez_compressed(path, T=self.T, classes=np.array(self.classes))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["T"], [str(c) for c in d["classes"]])


def get_prototypes(env, classes, k=6, rebuild=False, cache_dir=None):
    cache_dir = cache_dir or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cache_dir, "protos_" + "_".join(classes) + f"_{type(env).__name__}.npz")
    if os.path.exists(path) and not rebuild:
        return Prototypes.load(path)
    p = Prototypes.build(env, classes, k)
    p.save(path)
    return p


def evidence_to_coh(e, gain=0.5, clip=0.9):
    e = np.asarray(e, float)
    return np.clip(gain * (e - e.mean()), -clip, clip)
