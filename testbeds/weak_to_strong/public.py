"""Public helpers shared with methods: the student architecture.

Student = pretrained backbone (64 -> 64 -> 10) + head (10 -> 2). Inputs are flattened
8x8 images in [0, 1]. The backbone was pretrained on a separate split; its weights
come with each task. A method returns {"state_dict": {name: numpy array}} for
make_student(); it may freeze or fine-tune any part.
"""
import numpy as np
import torch
from torch import nn


def make_backbone() -> nn.Module:
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def make_student(backbone_state: dict | None = None) -> nn.Module:
    m = nn.Sequential(make_backbone(), nn.Linear(10, 2))
    if backbone_state is not None:
        m[0].load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in backbone_state.items()})
    return m


def load_student(out: dict) -> nn.Module:
    m = make_student()
    m.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in out["state_dict"].items()})
    m.eval()
    return m


def export(model: nn.Module) -> dict:
    return {"state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}
