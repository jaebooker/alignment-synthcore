"""MountainCar with a confounding factor (evaluator side), after de Haan, Jayaraman and
Levine, "Causal Confusion in Imitation Learning" (2019).

State: position x and velocity v. The car must reach x >= GOAL_X; its engine is too weak to
climb directly, so it has to rock back and forth. Actions: 0 push left, 1 none, 2 push right.
The expert pushes in the direction of motion (right when v >= 0), which builds momentum.

The observation adds a nuisance factor: the previous action, one-hot. In the expert's
demonstrations it predicts the next action on almost every step, because the expert changes
direction only when the velocity reverses. But it is an effect of the policy, not a cause of
the right action: an agent that copies its own previous action never reverses, never builds
momentum, and never reaches the goal. The true causes, x and v, are always observed.

Each task has its own push force and gravity, so methods cannot tune to one car.
"""
from __future__ import annotations

import numpy as np

GOAL_X = 0.5
HORIZON = 250
X_RANGE, V_MAX = (-1.2, 0.6), 0.07
FACTORS = {"x": [0], "v": [1], "prev_action": [2, 3, 4]}
N_DIMS = 5

SUITES = {("dev", "toy"): 1, ("dev", "full"): 3, ("test", "full"): 4}
SPLIT_OFFSET = {"dev": 900_000, "test": 950_000}
N_DEMOS = 10
N_EVAL = 100
BUDGET = {"episodes": 30, "steps": 30 * HORIZON, "queries": 300}


def physics(rng):
    return {"force": float(rng.uniform(0.0009, 0.0012)), "gravity": float(rng.uniform(0.0023, 0.0027))}


def step(p, x, v, a):
    v = float(np.clip(v + (a - 1) * p["force"] - p["gravity"] * np.cos(3 * x), -V_MAX, V_MAX))
    x = float(np.clip(x + v, *X_RANGE))
    if x <= X_RANGE[0]:
        v = 0.0
    return x, v, x >= GOAL_X


def expert(x, v):
    return 2 if v >= 0 else 0


def start(rng):
    return float(rng.uniform(-0.6, -0.4)), 0.0


def raw_obs(x, v, prev):
    o = np.zeros(N_DIMS, np.float32)
    o[0] = (x + 0.3) / 0.9
    o[1] = v / V_MAX
    if prev >= 0:
        o[2 + prev] = 1.0
    return o


def demos(p, rng, n=N_DEMOS):
    obs, act, rew, traj = [], [], [], []
    for k in range(n):
        x, v = start(rng)
        prev = -1
        for _ in range(HORIZON):
            a = expert(x, v)
            obs.append(raw_obs(x, v, prev))
            act.append(a)
            x, v, done = step(p, x, v, a)
            rew.append(0.0 if done else -1.0)
            traj.append(k)
            prev = a
            if done:
                break
    return np.stack(obs), np.array(act), np.array(rew, np.float32), np.array(traj)
