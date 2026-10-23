"""Decoupled approval (Uesato et al. 2020), tabular.

The agent learns from the overseer's approval only. At each step it executes one action but
asks the overseer about a second action drawn independently, and learns only from that
counterfactual answer. Feedback about the executed action, which the executed action can
corrupt, is never used.

Method choices (not testbed rules):
  * The value of (obs, action) is a running mean of queried approvals; the policy is greedy
    on it with epsilon-greedy exploration. Environment reward is ignored, as in the paper's
    approval-only setting.
  * The queried action is drawn uniformly, so every action's approval gets estimated.
"""
import numpy as np

import public

HPARAMS = [
    {"eps": 0.2, "queries_per_step": 1},
    {"eps": 0.1, "queries_per_step": 1},
    {"eps": 0.3, "queries_per_step": 1},
    {"eps": 0.05, "queries_per_step": 1},
    {"eps": 0.2, "queries_per_step": 2},
    {"eps": 0.1, "queries_per_step": 2},
    {"eps": 0.5, "queries_per_step": 1},
    {"eps": 0.3, "queries_per_step": 2},
]


def train(i, hp, rng):
    e = public.env(i)
    tot, cnt = {}, {}
    val = lambda o: tot.setdefault(o, np.zeros(5)) / np.maximum(cnt.setdefault(o, np.zeros(5)), 1)  # noqa: E731
    try:
        while True:
            o, done = e.reset(), False
            while not done:
                val(o)
                for _ in range(hp["queries_per_step"]):
                    b = int(rng.integers(5))
                    tot[o][b] += e.query_approval(b)
                    cnt[o][b] += 1
                a = int(rng.integers(5)) if rng.random() < hp["eps"] else int(np.argmax(val(o) + 1e-6 * rng.random(5)))
                o, _, done, _ = e.step(a)
    except public.BudgetExceeded:
        pass
    return {"policy": {o: int(np.argmax(val(o))) for o in cnt}}


def run(tasks, hparams, seed):
    rng = np.random.default_rng(seed)
    return [train(i, hparams, rng) for i in range(len(tasks))]
