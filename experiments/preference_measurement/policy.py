import numpy as np
WALL = np.ones((9, 9), np.float32); WALL[1:8, 1:8] = 0

def load(path):
    z = np.load(path, allow_pickle=False); P = {}
    for k in z.files:
        l, n = k.split("__"); P.setdefault(l, {})[n] = z[k]
    return P

def n_seeds(P):
    return P["l1"]["w"].shape[0]

def act_fn(P, s, mode="sample"):
    W = {l: (P[l]["w"][s], P[l]["b"][s]) for l in ("l1", "l2", "pi")}
    def act(a, g, rng):
        n = len(a); idx = np.arange(n); obs = np.zeros((n, 9, 9, 3), np.float32)
        obs[:, :, :, 0] = WALL; obs[idx, a[:, 0], a[:, 1], 1] = 1; obs[idx, g[:, 0], g[:, 1], 2] = 1
        h = obs.reshape(n, -1)
        for l in ("l1", "l2"): h = np.maximum(h @ W[l][0] + W[l][1], 0)
        lg = h @ W["pi"][0] + W["pi"][1]
        if mode == "greedy": return lg.argmax(1)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return act
