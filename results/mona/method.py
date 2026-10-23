"""MONA: myopic optimization with non-myopic approval (Farquhar et al. 2025), tabular.

Each step is optimized for its own reward only: the environment reward of that step plus
lam times the overseer's approval of it. Nothing is propagated from later steps (gamma = 0),
so a plan whose payoff arrives only after several steps is never learned, while the
overseer's approval, which already looks ahead, still supplies direction.

Method choices (not testbed rules):
  * The value of (obs, action) is a running mean of r + lam * approval for executed steps.
  * Epsilon-greedy exploration; optimistic initial values so untried actions get tried.
"""
import numpy as np

import public

HPARAMS = [
    {"lam": 1.0, "eps": 0.2, "init": 1.0},
    {"lam": 0.3, "eps": 0.2, "init": 1.0},
    {"lam": 3.0, "eps": 0.2, "init": 1.0},
    {"lam": 10.0, "eps": 0.2, "init": 1.0},
    {"lam": 1.0, "eps": 0.1, "init": 1.0},
    {"lam": 1.0, "eps": 0.3, "init": 1.0},
    {"lam": 1.0, "eps": 0.2, "init": 3.0},
    {"lam": 3.0, "eps": 0.1, "init": 3.0},
]


def train(i, hp, rng):
    e = public.env(i)
    tot, cnt = {}, {}
    val = lambda o: np.where(cnt.setdefault(o, np.zeros(5)) > 0,  # noqa: E731
                             tot.setdefault(o, np.zeros(5)) / np.maximum(cnt[o], 1), hp["init"] * (1 + hp["lam"]))
    try:
        while True:
            o, done = e.reset(), False
            while not done:
                a = int(rng.integers(5)) if rng.random() < hp["eps"] else int(np.argmax(val(o) + 1e-6 * rng.random(5)))
                o2, r, done, appr = e.step(a)
                val(o)
                tot[o][a] += r + hp["lam"] * appr
                cnt[o][a] += 1
                o = o2
    except public.BudgetExceeded:
        pass
    return {"policy": {o: int(np.argmax(val(o))) for o in cnt}}


def run(tasks, hparams, seed):
    rng = np.random.default_rng(seed)
    return [train(i, hparams, rng) for i in range(len(tasks))]
