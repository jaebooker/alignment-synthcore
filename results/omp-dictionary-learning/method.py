"""Method of optimal directions (MOD) with OMP sparse coding."""
import numpy as np
from sklearn.linear_model import orthogonal_mp

HPARAMS = [
    {"k": 2, "iters": 30, "n_fit": 6000},
    {"k": 1, "iters": 30, "n_fit": 6000},
    {"k": 3, "iters": 30, "n_fit": 6000},
    {"k": 4, "iters": 30, "n_fit": 6000},
    {"k": 2, "iters": 60, "n_fit": 6000},
    {"k": 3, "iters": 60, "n_fit": 6000},
    {"k": 2, "iters": 30, "n_fit": 12000},
    {"k": 3, "iters": 30, "n_fit": 12000},
]


def mod(H, m, k, iters, n_fit, rng):
    H = H[rng.choice(len(H), min(n_fit, len(H)), replace=False)].astype(np.float64)
    H = H - H.mean(0)
    # init atoms from random samples (data-driven init)
    D = H[rng.choice(len(H), m, replace=False)].copy()
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    k = min(k, m)
    for _ in range(iters):
        Z = orthogonal_mp(D.T, H.T, n_nonzero_coefs=k).T  # (N, m)
        G = Z.T @ Z + 1e-6 * np.eye(m)
        D = np.linalg.solve(G, Z.T @ H)  # (m, d) least-squares dictionary
        norms = np.linalg.norm(D, axis=1)
        # re-seed unused atoms with the worst-reconstructed samples
        unused = np.where((np.abs(Z).sum(0) < 1e-9) | (norms < 1e-9))[0]
        if len(unused):
            err = ((H - Z @ D) ** 2).sum(1)
            D[unused] = H[np.argsort(err)[-len(unused):]]
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    return D


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        rng = np.random.default_rng(seed * 100 + i)
        out.append({"directions": mod(t["H_train"], t["dict_size"], hparams["k"], hparams["iters"],
                                      hparams["n_fit"], rng)})
    return out
