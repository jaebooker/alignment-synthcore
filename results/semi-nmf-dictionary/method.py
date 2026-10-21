"""Semi-NMF dictionary (Shafran, Geiger, Belinkov 2025): factor activations X ~ Z D with
nonnegative codes Z and an unconstrained dictionary D, using the multiplicative updates of
Ding, Li and Jordan (2010). Optional L1 penalty on Z for sparsity. Centering is a choice."""
import numpy as np

HPARAMS = [
    {"iters": 300, "l1": 0.0, "center": False},
    {"iters": 300, "l1": 0.01, "center": False},
    {"iters": 300, "l1": 0.05, "center": False},
    {"iters": 300, "l1": 0.0, "center": True},
    {"iters": 300, "l1": 0.01, "center": True},
    {"iters": 600, "l1": 0.02, "center": False},
    {"iters": 300, "l1": 0.1, "center": False},
    {"iters": 600, "l1": 0.0, "center": False},
]


def semi_nmf(X, m, iters, l1, seed, n_fit=6000):
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))[:n_fit]]
    Z = np.abs(rng.normal(size=(len(X), m))) * 0.1
    pos = lambda A: (np.abs(A) + A) / 2
    neg = lambda A: (np.abs(A) - A) / 2
    for _ in range(iters):
        D = np.linalg.lstsq(Z.T @ Z + 1e-6 * np.eye(m), Z.T @ X, rcond=None)[0]
        XD, DD = X @ D.T, D @ D.T
        Z *= np.sqrt((pos(XD) + Z @ neg(DD)) / (neg(XD) + Z @ pos(DD) + l1 + 1e-9))
    return D


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        X = np.asarray(t["H_train"], float)
        if hparams["center"]:
            X = X - X.mean(0)
        D = semi_nmf(X, t["dict_size"], hparams["iters"], hparams["l1"], seed + i)
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
        out.append({"directions": D})
    return out
