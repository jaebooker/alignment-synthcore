"""Side-effect penalties learned from the agent's own trace (no hidden reward, no hazard labels).

AI-written by Hermes (Claude Opus) for Jaeson Booker; revised after Codex review 1.

fit_penalized == experiments.approval_factorial.learner.fit_assigned (same validation, same
planner), plus a per-step penalty subtracted from the assigned reward before planning:

  rr    Relative reachability (Krakovna et al. 2018), stepwise-inaction baseline. On the learned
        deterministic graph reach(x, y) = gamma_r ** dist(x, y) (0 if unreachable);
        pen(s, a) = mean_y max(0, reach(b, y) - reach(n, y)), b = T[s, noop], n = T[s, a].
        Discounting means reversible moves also cost a little (Codex review: 0.025 vs 0.5 on a
        two-state graph); the `move` control arm separates that from irreversibility.
  aup   Attainable utility preservation (Turner et al. 2020): k uniform[0,1] per-state auxiliary
        rewards earned on arrival; pen = mean_i |Q_i(s,a) - Q_i(s,noop)| / mean_i Q_i(s,noop).
        VARIANT vs ideas/attainable-utility-preservation: auxiliary value is zero after a terminal
        state (no terminal self-loop continuation).
  move  Control: pen(s, a) = 1 if the learned successor differs from the no-op successor
        (any state change, reversible or not). Same lambda selection rule.

Modelling choices (pre-declared): unknown (s, a) pairs are absent edges (they can create apparent
irreversibility); if the no-op successor of s was never observed, every action at s has zero
penalty; transitions into terminal states carry no penalty for any arm. Coverage diagnostics for
these choices are returned with every fit.
"""
from collections import deque

import numpy as np

NOOP = 4
PENALTIES = ("none", "rr", "aup", "move")


def _validate(obs, action, next_obs, reward, approval_action, approval, terminated):
    n = len(obs)
    if not all(len(a) == n for a in [action, next_obs, reward, approval_action, approval, terminated]):
        raise ValueError("inconsistent sample lengths")
    if not np.isfinite(reward).all() or not np.isfinite(approval).all():
        raise ValueError("non-finite training sample")
    if not (np.isin(action, range(5)).all() and np.isin(approval_action, range(5)).all()):
        raise ValueError("invalid action")


def learned_graph(obs, action, next_obs, terminated):
    states = np.unique(np.concatenate([obs, next_obs]))
    s, ns = np.searchsorted(states, obs), np.searchsorted(states, next_obs)
    T = np.full((len(states), 5), -1, dtype=np.int64)
    for i, a, j in zip(s, action, ns):
        if T[i, a] not in (-1, j):
            raise ValueError("aliased successor")
        T[i, a] = j
    terminal = np.zeros(len(states), dtype=bool)
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


def _exempt(T, terminal):
    """(s, a) pairs that never carry a penalty: unknown, terminal source/target, or no baseline."""
    known = T >= 0
    ex = ~known | terminal[:, None] | (known & terminal[np.maximum(T, 0)])
    ex |= ~known[:, NOOP][:, None]
    return ex


def rr_penalty(T, terminal, gamma_r=0.95):
    D = _distances(T, terminal)
    reach = np.where(D >= 0, gamma_r ** np.maximum(D, 0), 0.0)
    pen = np.zeros(T.shape)
    ex = _exempt(T, terminal)
    for i in range(len(T)):
        for a in range(5):
            if not ex[i, a]:
                pen[i, a] = np.maximum(0.0, reach[T[i, NOOP]] - reach[T[i, a]]).mean()
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
            done = np.max(np.abs(Vn - V)) < 1e-9
            V = Vn
            if done:
                break
        Q = np.where(known, r[nxt] + gamma_aux * np.where(terminal[nxt], 0, V[nxt]), 0.0)
        total += np.abs(Q - Q[:, NOOP][:, None])
        scale += Q[:, NOOP]
    pen = total / k / np.maximum(scale[:, None] / k, 1e-9)
    pen[_exempt(T, terminal)] = 0
    return pen


def move_penalty(T, terminal):
    pen = (T != T[:, NOOP][:, None]).astype(float)
    pen[_exempt(T, terminal)] = 0
    return pen


def penalty_table(T, terminal, penalty, penalty_seed=0):
    if penalty == "rr":
        return rr_penalty(T, terminal)
    if penalty == "aup":
        return aup_penalty(T, terminal, penalty_seed)
    if penalty == "move":
        return move_penalty(T, terminal)
    if penalty == "none":
        return np.zeros(T.shape)
    raise ValueError(penalty)


def fit_penalized(obs, action, next_obs, reward, approval_action, approval, terminated,
                  reward_weight, gamma, penalty="none", lam=0.0, penalty_seed=0, P=None):
    obs, action, next_obs, approval_action = map(np.asarray, (obs, action, next_obs, approval_action))
    reward, approval = np.asarray(reward, float), np.asarray(approval, float)
    terminated = np.asarray(terminated, dtype=bool)
    _validate(obs, action, next_obs, reward, approval_action, approval, terminated)
    states, s, ns, T, terminal = learned_graph(obs, action, next_obs, terminated)
    S, A = len(states), 5
    k = s * A + action
    ka = s * A + approval_action
    cnt = np.bincount(k, minlength=S * A).reshape(S, A)
    acnt = np.bincount(ka, minlength=S * A).reshape(S, A)
    reward_mean = np.bincount(k, weights=reward, minlength=S * A).reshape(S, A) / np.maximum(cnt, 1)
    approval_mean = np.bincount(ka, weights=approval, minlength=S * A).reshape(S, A) / np.maximum(acnt, 1)
    known = (cnt > 0) & (acnt > 0)
    if P is None:
        P = penalty_table(T, terminal, penalty, penalty_seed)
    R = approval_mean + reward_weight * reward_mean - lam * P
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
    pk = P[known & (P > 0)]
    return {"policy": {int(states[i]): int(actions[i]) for i in range(S) if known[i].any() and not terminal[i]},
            "diagnostics": {"observed_states": int(S), "penalty": penalty, "lam": float(lam),
                            "known_pairs": int(known.sum()),
                            "missing_noop_baseline_states": int((~terminal & (T[:, NOOP] < 0)).sum()),
                            "penalty_mean_known": float(P[known].mean()) if known.any() else 0.0,
                            "penalty_quantiles_positive": [float(q) for q in np.quantile(pk, [0.5, 0.9, 0.99, 1])] if len(pk) else [],
                            "planning_gamma": float(gamma), "planning_iterations": iterations}}
