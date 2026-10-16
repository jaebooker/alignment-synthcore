"""Public helpers shared with methods: model architecture only. No ground truth here."""
import numpy as np
import torch
from torch import nn


def make_mlp(hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 10))


def load_model(task: dict) -> nn.Module:
    """Rebuild a task's model from its public weights. Inputs are flattened 8x8 images in [0, 1]."""
    m = make_mlp(task["hidden"])
    m.load_state_dict({k: torch.tensor(np.asarray(v)) for k, v in task["weights"].items()})
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
