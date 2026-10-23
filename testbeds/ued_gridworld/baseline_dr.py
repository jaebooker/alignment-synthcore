"""Baseline: domain randomization. PPO where every episode uses a fresh level from the default
training distribution (the maximum-expected-value objective in the minimax-regret paper).
Uses the shared trainer in public.py, tuned with the same budget as any method."""
import public

HPARAMS = [
    {"lr": 5e-4, "ent": 0.01},
    {"lr": 1e-3, "ent": 0.01},
    {"lr": 3e-4, "ent": 0.01},
    {"lr": 5e-4, "ent": 0.003},
]


def run(tasks, hparams, seed):
    return [public.train_ppo(i, spec, hparams, public.Uniform(), seed + i) for i, spec in enumerate(tasks)]
