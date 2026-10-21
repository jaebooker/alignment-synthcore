"""Semi-NMF dictionary (Shafran, Geiger, Geva 2025): factor activations X ~ Z D with
nonnegative codes Z and an unconstrained dictionary D, plus an L1 penalty on Z.

Method choices (not testbed rules):
  * Dictionary rows are renormalized to unit length after every D update, with Z rescaled
    to compensate. Without this the L1 penalty never binds: the solver can shrink Z and
    grow D for free. With a dictionary larger than the activation dimension, an unpenalized
    semi-NMF has many exact dense solutions, so the penalty is what selects the sparse one.
  * Codes are solved by nonnegative proximal gradient (ISTA) rather than the multiplicative
    updates of Ding, Li and Jordan (2010), which only rescale codes and stall at dense
    solutions. Same objective, different solver.
  * D starts from random data rows; atoms that die are reseeded from badly fit rows.
  * l1 is in units of the mean absolute activation.

v0 (multiplicative updates, no renormalization) recovered 0% of planted features on every
seed. Diagnosis: results/_triage and the triage report.
"""
import numpy as np

HPARAMS = [
    {"l1": 0.5, "outer": 100, "inner": 10},
    {"l1": 0.2, "outer": 100, "inner": 10},
    {"l1": 0.3, "outer": 100, "inner": 10},
    {"l1": 0.7, "outer": 100, "inner": 10},
    {"l1": 1.0, "outer": 100, "inner": 10},
    {"l1": 1.5, "outer": 100, "inner": 10},
    {"l1": 0.5, "outer": 50, "inner": 10},
    {"l1": 0.5, "outer": 200, "inner": 10},
]


def semi_nmf(X, m, outer, inner, l1, seed, n_fit=6000):
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))[:n_fit]]
    D = X[rng.choice(len(X), m, replace=False)].copy()
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    Z = np.maximum(X @ D.T, 0)
    lam = l1 * np.abs(X).mean()
    for _ in range(outer):
        DD = D @ D.T
        L = np.linalg.eigvalsh(DD)[-1] + 1e-9
        XD = X @ D.T
        for _ in range(inner):
            Z = np.maximum(Z - (Z @ DD - XD + lam) / L, 0)
        D = np.linalg.lstsq(Z.T @ Z + 1e-6 * np.eye(m), Z.T @ X, rcond=None)[0]
        dead = np.linalg.norm(D, axis=1) < 1e-6
        if dead.any():
            resid = ((X - Z @ D) ** 2).sum(1)
            D[dead] = X[np.argsort(resid)[-int(dead.sum()):]]
        nrm = np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
        D /= nrm
        Z *= nrm.T
    return D


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        X = np.asarray(t["H_train"], float)
        D = semi_nmf(X, t["dict_size"], hparams["outer"], hparams["inner"], hparams["l1"], seed + i)
        out.append({"directions": D})
    return out
