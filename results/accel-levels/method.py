"""ACCEL: evolving curricula with regret-based environment design (Parker-Holder et al. 2022),
the strongest UED method in "Mitigating Goal Misgeneralization via Minimax Regret" (2025).

Robust PLR plus editing. Each replay of a high-regret level from the buffer is followed by an
edited child of that level (k random edits, chosen by the harness), which is played once
without learning and scored. Children with high regret stay in the buffer and get replayed;
the rest are evicted. Small random edits to levels the agent finds hard walk the curriculum
toward the levels that separate the intended goal from the proxy.

Method choices (not testbed rules):
  * Regret estimate: max-latest (best return seen on a level minus the mean of its latest
    returns). The paper's strongest variant used an oracle for the best possible return;
    this testbed exposes no oracle, so this is the domain-agnostic version.
  * Rank prioritization (beta) mixed with staleness (rho); buffer of `capacity` levels; a new
    or edited level replaces the lowest-scoring one when full.
  * Same PPO learner and budget as the baseline (public.train_ppo); only the curriculum differs.
"""
from collections import deque

import numpy as np

import public

HPARAMS = [
    {"lr": 1e-3, "ent": 0.01, "p_replay": 0.7, "edits": 3, "beta": 0.3, "rho": 0.3, "capacity": 1000},
    {"lr": 5e-4, "ent": 0.01, "p_replay": 0.7, "edits": 3, "beta": 0.3, "rho": 0.3, "capacity": 1000},
    {"lr": 1e-3, "ent": 0.01, "p_replay": 0.5, "edits": 3, "beta": 0.3, "rho": 0.3, "capacity": 1000},
    {"lr": 1e-3, "ent": 0.01, "p_replay": 0.7, "edits": 1, "beta": 0.3, "rho": 0.3, "capacity": 1000},
]


class ACCEL:
    def __init__(self, hp, rng, venv, latest=3):
        self.hp, self.rng, self.venv, self.latest = hp, rng, venv, latest
        self.best, self.recent, self.last_seen, self.scores = {}, {}, {}, {}
        self.children = []
        self.clock = 0

    def _pick(self):
        buf = list(self.best)
        s = np.array([self.scores[l] for l in buf])
        rank = np.empty(len(buf))
        rank[np.argsort(-s)] = np.arange(1, len(buf) + 1)
        p = (1.0 / rank) ** (1.0 / self.hp["beta"])
        p /= p.sum()
        stale = np.array([self.clock - self.last_seen[l] for l in buf], float)
        q = stale / stale.sum() if stale.sum() > 0 else np.full(len(buf), 1 / len(buf))
        return buf[int(self.rng.choice(len(buf), p=(1 - self.hp["rho"]) * p + self.hp["rho"] * q))]

    def next_level(self, levels):
        self.clock += 1
        if self.children:
            return self.children.pop(0), False  # score an edited child, no learning
        if len(self.best) >= 64 and self.rng.random() < self.hp["p_replay"]:
            l = self._pick()
            self.last_seen[l] = self.clock
            self.children += self.venv.edit_levels([l], k=self.hp["edits"])
            return l, True
        return levels.new(), False

    def episode_done(self, l, ret):
        if l not in self.best:
            if len(self.best) >= self.hp["capacity"]:
                worst = min(self.scores, key=self.scores.get)
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
        cur = ACCEL(hparams, np.random.default_rng(seed + 100 * i), public.venv(i))
        out.append(public.train_ppo(i, spec, hparams, cur, seed + i))
    return out
