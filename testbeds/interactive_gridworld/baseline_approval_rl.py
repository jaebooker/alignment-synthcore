"""Baseline: ordinary RL on environment reward plus the overseer's approval of each executed step.

A strong, model-based learner: it explores optimistically (every untried action looks as
good as the best possible return), learns the deterministic transitions and the mean of
r + lam * approval for each (observation, action) from its own experience, and plans by
value iteration over that learned model. So when it misses a hack, it is because the hack
does not pay under its objective, not because exploration never found it. This mirrors the
reward-hacking testbed, whose baseline is an optimal planner.

The objective is optimized over the full horizon (gamma >= 0.9), the standard way to use
approval as reward. It is non-myopic by design: myopia is exactly what MONA tests, and a
myopic version of this baseline would be MONA. Tuned over lam and gamma with the same
8-config budget as any method.
"""
import numpy as np

import public

HPARAMS = [
    {"lam": 1.0, "gamma": 0.95, "visits": 2},
    {"lam": 0.3, "gamma": 0.95, "visits": 2},
    {"lam": 0.1, "gamma": 0.95, "visits": 2},
    {"lam": 0.03, "gamma": 0.95, "visits": 2},
    {"lam": 3.0, "gamma": 0.95, "visits": 2},
    {"lam": 0.1, "gamma": 0.9, "visits": 2},
    {"lam": 0.1, "gamma": 0.99, "visits": 2},
    {"lam": 0.3, "gamma": 0.9, "visits": 4},
]


class Model:
    def __init__(self):
        self.idx, self.nxt, self.tot, self.cnt, self.term = {}, [], [], [], set()

    def s(self, o):
        if o not in self.idx:
            self.idx[o] = len(self.idx)
            self.nxt.append([-1] * 5)
            self.tot.append(np.zeros(5))
            self.cnt.append(np.zeros(5))
        return self.idx[o]

    def plan(self, gamma, optimism=None, visits=1, iters=300):
        S = len(self.idx)
        if S == 0:
            return np.zeros((0, 5))
        T = np.array(self.nxt)
        R = np.array(self.tot) / np.maximum(np.array(self.cnt), 1)
        known = np.array(self.cnt) >= visits
        term = np.zeros(S, bool)
        term[[self.idx[o] for o in self.term if o in self.idx]] = True
        V = np.zeros(S)
        for _ in range(iters):
            Vn = np.where(term, 0.0, V)
            Q = R + gamma * Vn[np.where(T >= 0, T, 0)]
            if optimism is not None:
                Q = np.where(known & (T >= 0), Q, optimism)
            else:
                Q = np.where(T >= 0, Q, -1e9)
            V = Q.max(1)
        return Q


def train(i, hp, rng):
    e = public.env(i)
    m = Model()
    vmax = (1.0 + hp["lam"] * 2.0) / (1 - hp["gamma"])  # above any real return: drives exploration
    try:
        while True:
            o, done, t = e.reset(), False, 0
            Q = m.plan(hp["gamma"], optimism=vmax, visits=hp["visits"])
            while not done:
                s = m.s(o)
                q = Q[s] if s < len(Q) else np.full(5, vmax)
                a = int(np.argmax(q + 1e-6 * rng.random(5)))
                o2, r, done, appr = e.step(a)
                s2 = m.s(o2)
                m.nxt[s][a] = s2
                m.tot[s][a] += r + hp["lam"] * appr
                m.cnt[s][a] += 1
                t += 1
                if done and t < public_horizon:
                    m.term.add(o2)  # ended before the horizon: the goal
                if s2 >= len(Q):
                    Q = m.plan(hp["gamma"], optimism=vmax, visits=hp["visits"])
                o = o2
    except public.BudgetExceeded:
        pass
    Q = m.plan(hp["gamma"])
    return {"policy": {o: int(np.argmax(Q[s])) for o, s in m.idx.items() if np.isfinite(Q[s]).any()}}


public_horizon = 30


def run(tasks, hparams, seed):
    global public_horizon
    rng = np.random.default_rng(seed)
    out = []
    for i, t in enumerate(tasks):
        public_horizon = t["horizon"]
        out.append(train(i, hparams, rng))
    return out
