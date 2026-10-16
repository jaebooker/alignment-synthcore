"""Spherical (cosine) k-means on normalized activation vectors."""
import numpy as np

HPARAMS = [
    {"min_norm_q": 0.0, "iters": 50, "restarts": 3},
    {"min_norm_q": 0.5, "iters": 50, "restarts": 3},
    {"min_norm_q": 0.25, "iters": 50, "restarts": 3},
    {"min_norm_q": 0.75, "iters": 50, "restarts": 3},
    {"min_norm_q": 0.0, "iters": 100, "restarts": 6},
    {"min_norm_q": 0.5, "iters": 100, "restarts": 6},
]


def spherical_kmeans(U, m, iters, rng):
    C = U[rng.choice(len(U), m, replace=False)].copy()
    for _ in range(iters):
        assign = (U @ C.T).argmax(1)
        for j in range(m):
            pts = U[assign == j]
            if len(pts):
                c = pts.sum(0)
                C[j] = c / max(np.linalg.norm(c), 1e-12)
            else:
                C[j] = U[rng.integers(len(U))]
    score = (U @ C.T).max(1).sum()
    return C, score


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        rng = np.random.default_rng(seed * 100 + i)
        H = t["H_train"].astype(np.float64)
        H = H - np.median(H, axis=0)
        n = np.linalg.norm(H, axis=1)
        keep = n > max(np.quantile(n, hparams["min_norm_q"]), 1e-6)
        U = H[keep] / n[keep, None]
        best = max((spherical_kmeans(U, t["dict_size"], hparams["iters"], rng)
                    for _ in range(hparams["restarts"])), key=lambda r: r[1])
        out.append({"directions": best[0]})
    return out
