"""Future-task auxiliary reward with an inaction baseline (Krakovna et al. 2020), tabular.

Future tasks: reach goal state g, for every non-terminal g; their optimal value from x is
gamma_r ** dist(x, g). Future-task value F(x) = mean over g. The auxiliary reward compares
the agent's state with the inaction baseline (the start state, since no-op keeps the agent
in place): A(x) = beta * (F(x) - F(b)). Unlike relative reachability it is two-sided, so
it also rewards keeping options open, and the baseline removes the incentive to interfere.

As in the paper, future tasks are scored when the episode ends: at rate (1 - gamma) per
step (random termination), and in full when the agent enters a terminal state, scored at
the state it left (the world after the episode, not the absorbing state).
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
    {"beta": 30.0, "gamma_r": 0.95, "gamma": 0.99},
    {"beta": 3.0, "gamma_r": 0.95, "gamma": 0.95},
]


def value_iteration(T, R, gamma, iters=3000, tol=1e-9):
    V = np.zeros(T.shape[0])
    for _ in range(iters):
        V2 = (R + gamma * V[T]).max(1)
        if np.abs(V2 - V).max() < tol:
            return R + gamma * V2[T]
        V = V2
    return R + gamma * V[T]


def future_value(T, terminal, gamma_r):
    S, A = T.shape
    live = ~terminal
    rows = np.repeat(np.arange(S), A)
    cols = T.ravel()
    keep = live[rows] & (rows != cols)
    G = csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(S, S))
    D = shortest_path(G, directed=True, unweighted=True)
    V = np.where(np.isfinite(D), gamma_r ** np.nan_to_num(D, posinf=0.0), 0.0)[:, live]
    return V.mean(1), np.isfinite(D)


def run(tasks, hparams, seed):
    out = []
    for t in tasks:
        T, R = np.asarray(t["T"]), np.asarray(t["R"], float)
        term = np.asarray(t["terminal"], bool)
        F, reach = future_value(T, term, hparams["gamma_r"])
        S = len(F)
        base = np.zeros(S)
        for x in range(S):
            starts = [s0 for s0 in t["start_states"] if reach[s0, x]]
            base[x] = np.mean([F[s0] for s0 in starts]) if starts else F[x]
        Aux = hparams["beta"] * (F - base)
        g = hparams["gamma"]
        nxt = T
        aux_sa = np.where(term[nxt], Aux[np.arange(S)][:, None].repeat(T.shape[1], 1), (1 - g) * Aux[nxt])
        aux_sa[term] = 0.0
        Q = value_iteration(T, R + aux_sa, g)
        out.append({"policy": Q.argmax(1)})
    return out
