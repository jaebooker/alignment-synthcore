"""Relative reachability with a starting-state baseline, then value iteration on the shaped reward.

Method choices (not testbed rules):
  * Reachability R(x, y) = gamma_r ** shortest_path(x, y), 0 if unreachable.
  * Baseline for state x: the start state(s) that can reach x. A state is
    only compared with the episode it could have come from.
  * Deficit d(x) = mean over non-terminal y of max(0, R(s0, y) - R(x, y)).
  * Entering a terminal state carries no penalty: the episode ends there.
  * Shaped reward r'(s, a) = r(s, a) - beta * d(T[s, a]).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

HPARAMS = [
    {"beta": 1.0, "gamma_r": 0.95, "gamma": 0.99},
    {"beta": 0.3, "gamma_r": 0.95, "gamma": 0.99},
    {"beta": 3.0, "gamma_r": 0.95, "gamma": 0.99},
    {"beta": 10.0, "gamma_r": 0.95, "gamma": 0.99},
    {"beta": 1.0, "gamma_r": 0.99, "gamma": 0.99},
    {"beta": 3.0, "gamma_r": 0.99, "gamma": 0.99},
    {"beta": 1.0, "gamma_r": 0.95, "gamma": 0.9},
    {"beta": 3.0, "gamma_r": 0.95, "gamma": 0.9},
]


def value_iteration(T, R, gamma, iters=3000, tol=1e-9):
    V = np.zeros(T.shape[0])
    for _ in range(iters):
        V2 = (R + gamma * V[T]).max(1)
        if np.abs(V2 - V).max() < tol:
            return R + gamma * V2[T]
        V = V2
    return R + gamma * V[T]


def deficits(T, terminal, starts, gamma_r):
    S, A = T.shape
    live = ~terminal
    rows = np.repeat(np.arange(S), A)
    cols = T.ravel()
    keep = live[rows] & (rows != cols)
    G = csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(S, S))
    D = shortest_path(G, directed=True, unweighted=True)
    Reach = np.where(np.isfinite(D), gamma_r ** np.nan_to_num(D, posinf=0.0), 0.0)
    Reach = Reach[:, live]
    d = np.full(S, np.inf)
    for s0 in starts:
        ok = np.isfinite(D[s0])
        gap = np.clip(Reach[s0][None, :] - Reach[ok], 0, None).mean(1)
        d[ok] = np.minimum(d[ok], gap)
    d[~np.isfinite(d)] = 0.0
    d[terminal] = 0.0
    return d


def run(tasks, hparams, seed):
    out = []
    for t in tasks:
        T, R = np.asarray(t["T"]), np.asarray(t["R"], float)
        term = np.asarray(t["terminal"], bool)
        d = deficits(T, term, t["start_states"], hparams["gamma_r"])
        Q = value_iteration(T, R - hparams["beta"] * d[T], hparams["gamma"])
        out.append({"policy": Q.argmax(1)})
    return out
