"""Control for ACCEL: the same edits, replays and learner, with regret switched off.

Identical to ideas/accel-levels/method.py except that buffer levels are replayed uniformly at
random and evicted oldest-first, so scores play no part. Random edits alone already raise the
share of levels that separate goal and cue, so this control separates what regret-based
selection adds from what editing alone does. Expected: if ACCEL's mechanism is regret, ACCEL
beats this control; if the edits do the work, they tie.
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


class RandomEdits:
    def __init__(self, hp, rng, venv, latest=3):
        self.hp, self.rng, self.venv, self.latest = hp, rng, venv, latest
        self.best, self.recent, self.last_seen, self.scores = {}, {}, {}, {}
        self.children = []
        self.clock = 0

    def _pick(self):
        buf = list(self.best)
        return buf[int(self.rng.integers(len(buf)))]  # no regret: uniform

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
                worst = next(iter(self.best))  # no regret: evict the oldest
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
        cur = RandomEdits(hparams, np.random.default_rng(seed + 100 * i), public.venv(i))
        out.append(public.train_ppo(i, spec, hparams, cur, seed + i))
    return out
