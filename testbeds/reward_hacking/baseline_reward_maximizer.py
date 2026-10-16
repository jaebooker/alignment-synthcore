"""Baseline: an optimal observed-reward maximizer (value iteration on the given MDP).

This is the idealized RL agent: it gets the full model, so any gap to the
safe optimum comes from the objective, not from exploration or training noise.
Tuned over the discount factor with the same budget as any method.
"""
import numpy as np

HPARAMS = [
    {"gamma": 0.99},
    {"gamma": 0.95},
    {"gamma": 0.9},
    {"gamma": 0.8},
    {"gamma": 0.7},
    {"gamma": 0.5},
    {"gamma": 0.999},
    {"gamma": 0.97},
]


def value_iteration(T, R, gamma, iters=3000, tol=1e-9):
    V = np.zeros(T.shape[0])
    for _ in range(iters):
        V2 = (R + gamma * V[T]).max(1)
        if np.abs(V2 - V).max() < tol:
            V = V2
            break
        V = V2
    return R + gamma * V[T]


def run(tasks, hparams, seed):
    return [{"policy": value_iteration(np.asarray(t["T"]), np.asarray(t["R"]), hparams["gamma"]).argmax(1)}
            for t in tasks]
