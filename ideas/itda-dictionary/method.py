"""Inference-Time Decomposition of Activations (Leask et al. 2025): build the dictionary
from real activations. Stream through examples; reconstruct each with matching pursuit
over the current dictionary (k atoms); if the relative error exceeds a threshold, add the
normalized example as a new atom. Stops at the dictionary size budget. No gradient training."""
import numpy as np

HPARAMS = [
    {"k": 2, "tau": 0.1},
    {"k": 1, "tau": 0.1},
    {"k": 3, "tau": 0.1},
    {"k": 2, "tau": 0.05},
    {"k": 2, "tau": 0.2},
    {"k": 3, "tau": 0.05},
    {"k": 1, "tau": 0.2},
    {"k": 3, "tau": 0.2},
]


def mp_error(x, D, k):
    r = x.copy()
    for _ in range(k):
        c = D @ r
        j = np.argmax(np.abs(c))
        r = r - c[j] * D[j]
    return np.linalg.norm(r) / max(np.linalg.norm(x), 1e-9)


def itda(X, m, k, tau, seed):
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))]
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    atoms = [X[0] / np.linalg.norm(X[0])]
    for x in X[1:]:
        if len(atoms) >= m:
            break
        if mp_error(x, np.array(atoms), k) > tau:
            atoms.append(x / np.linalg.norm(x))
    return np.array(atoms)


def run(tasks, hparams, seed):
    return [{"directions": itda(np.asarray(t["H_train"], float), t["dict_size"], hparams["k"], hparams["tau"], seed + i)}
            for i, t in enumerate(tasks)]
