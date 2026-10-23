"""Baseline: behavior cloning on the expert demonstrations, all observation dims as input.
Tuned with the same budget as any method. It does not use the environment or the expert."""
import numpy as np
import torch
from torch import nn

import public

HPARAMS = [
    {"epochs": 60, "lr": 1e-3, "hidden": [64, 64]},
    {"epochs": 30, "lr": 1e-3, "hidden": [64, 64]},
    {"epochs": 120, "lr": 1e-3, "hidden": [64, 64]},
    {"epochs": 60, "lr": 3e-3, "hidden": [128, 128]},
]


def bc(X, y, cfg, hp, seed):
    torch.manual_seed(seed)
    net = public.make_policy(cfg, X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    for _ in range(hp["epochs"]):
        for mb in torch.randperm(len(X)).split(256):
            loss = nn.functional.cross_entropy(net(X[mb]), y[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        cfg = {"type": "mlp", "hidden": hparams["hidden"], "mask_input": False}
        net = bc(np.asarray(t["demos"]["obs"]), np.asarray(t["demos"]["action"]), cfg, hparams, seed + i)
        out.append(public.export(net, cfg))
    return out
