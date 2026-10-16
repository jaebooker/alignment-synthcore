"""Attainable utility preservation (AUP), tabular version.

Method choices (not testbed rules):
  * k auxiliary rewards, each uniform [0, 1] per state, earned on arrival.
  * Penalty(s, a) = mean_i |Q_i(s, a) - Q_i(s, noop)|, scaled by mean_i Q_i(s, noop).
  * Transitions into a terminal state carry no penalty: the episode ends there.
  * Shaped reward r'(s, a) = r(s, a) - lam * scaled penalty; then value iteration.
"""
import numpy as np

HPARAMS = [
    {"lam": 1.0, "k": 10, "gamma_aux": 0.95, "gamma": 0.99},
    {"lam": 3.0, "k": 10, "gamma_aux": 0.95, "gamma": 0.99},
    {"lam": 10.0, "k": 10, "gamma_aux": 0.95, "gamma": 0.99},
    {"lam": 30.0, "k": 10, "gamma_aux": 0.95, "gamma": 0.99},
    {"lam": 100.0, "k": 10, "gamma_aux": 0.95, "gamma": 0.99},
    {"lam": 10.0, "k": 30, "gamma_aux": 0.95, "gamma": 0.99},
    {"lam": 10.0, "k": 10, "gamma_aux": 0.99, "gamma": 0.99},
    {"lam": 30.0, "k": 10, "gamma_aux": 0.99, "gamma": 0.99},
]


def value_iteration(T, R, gamma, iters=3000, tol=1e-9):
    V = np.zeros(T.shape[0])
    for _ in range(iters):
        V2 = (R + gamma * V[T]).max(1)
        if np.abs(V2 - V).max() < tol:
            return R + gamma * V2[T]
        V = V2
    return R + gamma * V[T]


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        rng = np.random.default_rng(seed * 1000 + i)
        T, R = np.asarray(t["T"]), np.asarray(t["R"], float)
        term = np.asarray(t["terminal"], bool)
        noop = t["noop_action"]
        S = T.shape[0]
        pen = np.zeros_like(R)
        scale = np.zeros(S)
        for _ in range(hparams["k"]):
            ra = rng.random(S)
            Qa = value_iteration(T, ra[T], hparams["gamma_aux"])
            pen += np.abs(Qa - Qa[:, [noop]])
            scale += Qa[:, noop]
        pen /= hparams["k"]
        scale = np.maximum(scale / hparams["k"], 1e-6)
        pen = pen / scale[:, None]
        pen[term[T]] = 0.0
        Q = value_iteration(T, R - hparams["lam"] * pen, hparams["gamma"])
        out.append({"policy": Q.argmax(1)})
    return out
