"""Baseline: behavior cloning on the expert demonstrations (cross-entropy), tuned like any method."""
import numpy as np
import torch

from public import export, make_policy

HPARAMS = [
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "epochs": 30, "lr": 1e-3, "wd": 0.0},
    {"arch": {"type": "mlp", "hidden": [256, 256]}, "epochs": 30, "lr": 1e-3, "wd": 0.0},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "epochs": 30, "lr": 1e-3, "wd": 1e-3},
    {"arch": {"type": "mlp", "hidden": [256, 256]}, "epochs": 30, "lr": 1e-3, "wd": 1e-3},
    {"arch": {"type": "cnn", "channels": [32, 32], "hidden": 128}, "epochs": 30, "lr": 1e-3, "wd": 0.0},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "epochs": 60, "lr": 3e-4, "wd": 0.0},
    {"arch": {"type": "mlp", "hidden": [512]}, "epochs": 30, "lr": 1e-3, "wd": 0.0},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "epochs": 15, "lr": 3e-3, "wd": 0.0},
]


def train_bc(task, hp, seed, weights=None, batch=128):
    torch.manual_seed(seed)
    X = torch.tensor(task["obs"], dtype=torch.float32)
    y = torch.tensor(task["action"])
    w = torch.ones(len(y)) if weights is None else torch.tensor(weights, dtype=torch.float32)
    model = make_policy(hp["arch"], task["n_channels"], task["view"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator().manual_seed(seed)
    for _ in range(hp["epochs"]):
        for idx in torch.randperm(len(y), generator=g).split(batch):
            loss = (torch.nn.functional.cross_entropy(model(X[idx]), y[idx], reduction="none") * w[idx]).sum() / w[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def run(tasks, hparams, seed):
    return [export(train_bc(t, hparams, seed * 100 + i), hparams["arch"]) for i, t in enumerate(tasks)]
