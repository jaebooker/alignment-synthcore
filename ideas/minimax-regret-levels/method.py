"""Minimax-regret level design via robust prioritized level replay (PLR-perp), as in
"Mitigating Goal Misgeneralization via Minimax Regret" (2025).

Each new episode either explores a fresh level from the default distribution (no learning
from it, only scoring it) or replays a level from a buffer, chosen by estimated regret, and
learns from that. Training on high-regret levels approximates the minimax-regret objective,
which upweights the rare levels where the intended goal and the proxy come apart.

Method choices (not testbed rules):
  * Regret estimate: "max-latest", the highest return ever seen on a level minus the mean of
    its latest returns. Domain-agnostic: no oracle for the best possible return.
  * Rank prioritization with temperature beta, mixed with staleness (rho), buffer of
    `capacity` levels; new levels replace the lowest-scoring one when full.
  * Same PPO learner and budget as the baseline (public.train_ppo); only the curriculum differs.
"""
from collections import deque

import numpy as np

import public

HPARAMS = [
    {"lr": 5e-4, "ent": 0.01, "p_replay": 0.5, "beta": 0.3, "rho": 0.3, "capacity": 1000},
    {"lr": 5e-4, "ent": 0.01, "p_replay": 0.7, "beta": 0.3, "rho": 0.3, "capacity": 1000},
    {"lr": 5e-4, "ent": 0.01, "p_replay": 0.5, "beta": 0.1, "rho": 0.3, "capacity": 1000},
    {"lr": 1e-3, "ent": 0.01, "p_replay": 0.5, "beta": 0.3, "rho": 0.3, "capacity": 1000},
]


class RobustPLR:
    def __init__(self, hp, rng, latest=3):
        self.hp, self.rng, self.latest = hp, rng, latest
        self.best, self.recent, self.last_seen, self.scores = {}, {}, {}, {}
        self.clock = 0

    def score(self, l):
        return self.scores[l]

    def next_level(self, levels):
        self.clock += 1
        buf = list(self.best)
        if len(buf) >= 64 and self.rng.random() < self.hp["p_replay"]:
            s = np.array([self.score(l) for l in buf])
            rank = np.empty(len(buf))
            rank[np.argsort(-s)] = np.arange(1, len(buf) + 1)
            p_score = (1.0 / rank) ** (1.0 / self.hp["beta"])
            p_score /= p_score.sum()
            stale = np.array([self.clock - self.last_seen[l] for l in buf], float)
            p_stale = stale / stale.sum() if stale.sum() > 0 else np.full(len(buf), 1 / len(buf))
            p = (1 - self.hp["rho"]) * p_score + self.hp["rho"] * p_stale
            l = buf[int(self.rng.choice(len(buf), p=p))]
            self.last_seen[l] = self.clock
            return l, True
        return levels.new(), False

    def episode_done(self, l, ret):
        if l not in self.best:
            if len(self.best) >= self.hp["capacity"]:
                worst = min(self.best, key=self.score)
                for d in (self.best, self.recent, self.last_seen, self.scores):
                    del d[worst]
            self.best[l], self.recent[l] = ret, deque(maxlen=self.latest)
        self.best[l] = max(self.best[l], ret)
        self.recent[l].append(ret)
        self.scores[l] = self.best[l] - sum(self.recent[l]) / len(self.recent[l])
        self.last_seen[l] = self.clock


def run(tasks, hparams, seed):
    out = []
    for i, spec in enumerate(tasks):
        cur = RobustPLR(hparams, np.random.default_rng(seed + 100 * i))
        out.append(public.train_ppo(i, spec, hparams, cur, seed + i))
    return out
