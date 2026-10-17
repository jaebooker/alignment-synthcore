"""Baseline: naive fine-tuning on the given labels (cross-entropy), tuned like any method.

Tuning covers frozen-probe vs full fine-tune, epochs (early stopping matters in
weak-to-strong) and learning rate.
"""
import numpy as np
import torch

from public import export, make_student

HPARAMS = [
    {"mode": "probe", "epochs": 50, "lr": 1e-2},
    {"mode": "ft", "epochs": 30, "lr": 1e-3},
    {"mode": "probe", "epochs": 10, "lr": 1e-2},
    {"mode": "probe", "epochs": 100, "lr": 3e-3},
    {"mode": "ft", "epochs": 5, "lr": 1e-3},
    {"mode": "ft", "epochs": 10, "lr": 1e-3},
    {"mode": "ft", "epochs": 30, "lr": 3e-4},
    {"mode": "ft", "epochs": 60, "lr": 1e-3},
]


def run(tasks, hparams, seed, batch=64):
    out = []
    for i, t in enumerate(tasks):
        s = seed * 100 + i
        torch.manual_seed(s)
        m = make_student(t["backbone"])
        params = m[1].parameters() if hparams["mode"] == "probe" else m.parameters()
        opt = torch.optim.AdamW(params, lr=hparams["lr"], weight_decay=1e-4)
        X, y = torch.tensor(t["X"]), torch.tensor(t["y"])
        g = torch.Generator().manual_seed(s)
        for _ in range(hparams["epochs"]):
            for idx in torch.randperm(len(y), generator=g).split(batch):
                loss = torch.nn.functional.cross_entropy(m(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        out.append(export(m))
    return out
