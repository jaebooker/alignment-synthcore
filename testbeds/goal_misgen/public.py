"""Public helpers shared with methods: the policy architectures the evaluator can rebuild.

Observations are top-down views of the whole grid, float tensors of shape (C, V, V);
the agent is one of the channels. Channel meanings are NOT given and their order is shuffled per seed.
Actions: 0 up, 1 down, 2 left, 3 right.

A method returns {"arch": cfg, "state_dict": {name: numpy array}} where cfg is one of
    {"type": "mlp", "hidden": [int, ...]}
    {"type": "cnn", "channels": [int, ...], "hidden": int}
"""
import numpy as np
import torch
from torch import nn

N_ACTIONS = 4


def make_policy(cfg: dict, in_channels: int, view: int) -> nn.Module:
    if cfg["type"] == "mlp":
        layers, d = [nn.Flatten()], in_channels * view * view
        for h in cfg["hidden"]:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, N_ACTIONS))
        return nn.Sequential(*layers)
    if cfg["type"] == "cnn":
        layers, c = [], in_channels
        for k in cfg["channels"]:
            layers += [nn.Conv2d(c, k, 3, padding=1), nn.ReLU()]
            c = k
        layers += [nn.Flatten(), nn.Linear(c * view * view, cfg["hidden"]), nn.ReLU(),
                   nn.Linear(cfg["hidden"], N_ACTIONS)]
        return nn.Sequential(*layers)
    raise ValueError(f"unknown arch {cfg!r}")


def load_policy(out: dict, in_channels: int, view: int) -> nn.Module:
    m = make_policy(out["arch"], in_channels, view)
    m.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in out["state_dict"].items()})
    m.eval()
    return m


def export(model: nn.Module, cfg: dict) -> dict:
    return {"arch": cfg, "state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}
