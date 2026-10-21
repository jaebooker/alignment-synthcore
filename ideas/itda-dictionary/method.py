"""Inference-Time Decomposition of Activations (Leask, Nanda, Al Moubayed 2025): build the
dictionary from real activation examples, adding the ones worst approximated by matching
pursuit (k atoms) over the current dictionary. No gradient training.

Method choices (not testbed rules):
  * Examples stream in chunks. Within a chunk, the single worst-approximated example is added
    if its relative error exceeds tau; errors are recomputed against the grown dictionary,
    and so on until nothing in the chunk exceeds tau. This is the paper's greedy rule.
  * On this testbed tau barely matters: the worst example in a chunk almost always exceeds it,
    so the dictionary fills to its budget. The grid sweeps k around its interior optimum.

v0 added each example the moment its error exceeded tau (first come, first added) rather than
the worst-approximated first.
"""
import numpy as np

HPARAMS = [
    {"k": 3, "tau": 0.2},
    {"k": 1, "tau": 0.2},
    {"k": 2, "tau": 0.2},
    {"k": 4, "tau": 0.2},
    {"k": 5, "tau": 0.2},
    {"k": 6, "tau": 0.2},
    {"k": 3, "tau": 0.4},
    {"k": 2, "tau": 0.4},
]


def mp_rel_error(X, D, k):
    R = X.copy()
    for _ in range(k):
        C = R @ D.T
        j = np.abs(C).argmax(1)
        R = R - C[np.arange(len(R)), j][:, None] * D[j]
    return np.linalg.norm(R, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-9)


def itda(X, m, k, tau, seed, chunk=256):
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))]
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    D = X[:1].copy()
    i = 1
    while len(D) < m and i < len(X):
        xb = X[i:i + chunk]
        i += chunk
        while len(D) < m and len(xb):
            err = mp_rel_error(xb, D, min(k, len(D)))
            j = int(err.argmax())
            if err[j] <= tau:
                break
            D = np.vstack([D, xb[j]])
            xb = np.delete(xb, j, 0)
    return D


def run(tasks, hparams, seed):
    return [{"directions": itda(np.asarray(t["H_train"], float), t["dict_size"], hparams["k"], hparams["tau"], seed + i)}
            for i, t in enumerate(tasks)]
