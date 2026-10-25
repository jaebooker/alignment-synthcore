"""Side-effect penalties learned from the agent's own trace (no hidden reward, no hazard labels).

AI-written by Hermes (Claude Opus) for Jaeson Booker.

fit_penalized == experiments.approval_factorial.learner.fit_assigned, plus a per-step penalty
subtracted from the assigned reward before planning:

  rr   Relative reachability (Krakovna et al. 2018), stepwise-inaction baseline. On the learned
       deterministic graph, reach(x, y) = gamma_r ** dist(x, y) (0 if unreachable).
       pen(s, a) = mean_y max(0, reach(b, y) - reach(n, y)), b = T[s, noop], n = T[s, a].
  aup  Attainable utility preservation (Turner et al. 2020), tabular as in
       ideas/attainable-utility-preservation: k uniform[0,1] per-state auxiliary rewards earned on
       arrival, Q_i by value iteration on the learned graph, pen = mean_i |Q_i(s,a) - Q_i(s,noop)|
       scaled by mean_i Q_i(s, noop).

Both penalties use only the learned transition graph: unknown (s, a) pairs are absent edges and
transitions into terminal states carry no penalty. The planner itself is unchanged, so a myopic
(gamma = 0) learner pays the penalty as part of each step's reward, as MONA allows.
"""
from collections import deque

import numpy as np

NOOP = 4


def _model(obs, action, next_obs, terminated):
    states = np.unique(np.concatenate([obs, next_obs]))
    s, ns = np.searchsorted(states, obs), np.searchsorted(states, next_obs)
    S, A = len(states), 5
    T = np.full((S, A), -1, dtype=np.int64)
    T[s, action] = ns
    terminal = np.zeros(S, dtype=bool)
    terminal[ns[np.asarray(terminated, bool)]] = True
    return states, s, ns, T, terminal


def _distances(T, terminal):
    S = len(T)
    D = np.full((S, S), -1, dtype=np.int64)
    for src in range(S):
        D[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            if terminal[u]:
                continue
            for v in T[u]:
                if v >= 0 and D[src, v] < 0:
                    D[src, v] = D[src, u] + 1
                    q.append(v)
    return D


def rr_penalty(T, terminal, gamma_r=0.95):
    D = _distances(T, terminal)
    reach = np.where(D >= 0, gamma_r ** np.maximum(D, 0), 0.0)
    pen = np.zeros(T.shape)
    for i in range(len(T)):
        b = T[i, NOOP]
        if terminal[i] or b < 0:
            continue
        for a in range(5):
            n = T[i, a]
            if n < 0 or terminal[n]:
                continue
            pen[i, a] = np.maximum(0.0, reach[b] - reach[n]).mean()
    return pen


def aup_penalty(T, terminal, seed, k=10, gamma_aux=0.95, iters=3000):
    rng = np.random.default_rng(seed)
    S = len(T)
    known = T >= 0
    nxt = np.maximum(T, 0)
    total, scale = np.zeros(T.shape), np.zeros(S)
    for _ in range(k):
        r = rng.uniform(0, 1, S)
        V = np.zeros(S)
        for _ in range(iters):
            Q = np.where(known, r[nxt] + gamma_aux * np.where(terminal[nxt], 0, V[nxt]), -np.inf)
            Vn = np.where(known.any(1), Q.max(1), 0.0)
            Vn[terminal] = 0
            if np.max(np.abs(Vn - V)) < 1e-9:
                V = Vn
                break
            V = Vn
        Q = np.where(known, r[nxt] + gamma_aux * np.where(terminal[nxt], 0, V[nxt]), np.nan)
        base = Q[:, NOOP]
        total += np.nan_to_num(np.abs(Q - base[:, None]))
        scale += np.nan_to_num(base)
    pen = total / k / np.maximum(scale[:, None] / k, 1e-9)
    pen[terminal] = 0
    pen[~known] = 0
    pen[known & terminal[nxt]] = 0
    pen[~known[:, NOOP]] = 0  # no inaction baseline observed: no penalty
    return pen


def fit_penalized(obs, action, next_obs, reward, approval_action, approval, terminated,
                  reward_weight, gamma, penalty="none", lam=0.0, penalty_seed=0, **kw):
    obs, action, next_obs, approval_action = map(np.asarray, (obs, action, next_obs, approval_action))
    reward, approval = np.asarray(reward, float), np.asarray(approval, float)
    states, s, ns, T, terminal = _model(obs, action, next_obs, terminated)
    S, A = len(states), 5
    k = s * A + action
    ka = s * A + approval_action
    cnt = np.bincount(k, minlength=S * A).reshape(S, A)
    acnt = np.bincount(ka, minlength=S * A).reshape(S, A)
    reward_mean = np.bincount(k, weights=reward, minlength=S * A).reshape(S, A) / np.maximum(cnt, 1)
    approval_mean = np.bincount(ka, weights=approval, minlength=S * A).reshape(S, A) / np.maximum(acnt, 1)
    known = (cnt > 0) & (acnt > 0)
    R = approval_mean + reward_weight * reward_mean
    if penalty == "rr":
        P = rr_penalty(T, terminal, **kw)
    elif penalty == "aup":
        P = aup_penalty(T, terminal, penalty_seed, **kw)
    elif penalty == "none":
        P = np.zeros((S, A))
    else:
        raise ValueError(penalty)
    R = R - lam * P
    V = np.zeros(S)
    for iterations in range(1, 1001):
        vn = V.copy()
        vn[terminal] = 0
        Q = np.where(known, R + gamma * vn[np.maximum(T, 0)], -1e6)
        Q[terminal] = 0
        v = Q.max(axis=1)
        v[~known.any(axis=1)] = 0
        if np.max(np.abs(v - V)) < 1e-10:
            break
        V = v
    else:
        raise RuntimeError("value iteration did not converge")
    actions = Q.argmax(axis=1)
    return {"policy": {int(states[i]): int(actions[i]) for i in range(S) if known[i].any() and not terminal[i]},
            "diagnostics": {"observed_states": int(S), "penalty": penalty, "lam": float(lam),
                            "mean_penalty_known": float(P[known].mean()) if known.any() else 0.0,
                            "planning_gamma": float(gamma), "planning_iterations": iterations}}
