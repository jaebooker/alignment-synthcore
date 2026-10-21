"""Bootstrapping (Burns et al. 2023): instead of going from the weak supervisor straight to
the strong student, pass through an intermediate model. Here the intermediate is the
student's frozen backbone plus a trained head (a linear probe, less able to memorize
label noise); its predictions on the training inputs relabel the data for the full
fine-tune. Two steps, because the testbed has one student size."""
import numpy as np
import torch

from public import export, make_student

HPARAMS = [
    {"probe_epochs": 50, "probe_lr": 1e-2, "ft_epochs": 10, "ft_lr": 1e-3},
    {"probe_epochs": 50, "probe_lr": 1e-2, "ft_epochs": 30, "ft_lr": 1e-3},
    {"probe_epochs": 100, "probe_lr": 3e-3, "ft_epochs": 10, "ft_lr": 1e-3},
    {"probe_epochs": 20, "probe_lr": 1e-2, "ft_epochs": 10, "ft_lr": 1e-3},
    {"probe_epochs": 50, "probe_lr": 1e-2, "ft_epochs": 5, "ft_lr": 1e-3},
    {"probe_epochs": 50, "probe_lr": 1e-2, "ft_epochs": 30, "ft_lr": 3e-4},
    {"probe_epochs": 50, "probe_lr": 1e-2, "ft_epochs": 60, "ft_lr": 1e-3},
    {"probe_epochs": 100, "probe_lr": 1e-2, "ft_epochs": 10, "ft_lr": 3e-4},
]


def fit(m, params, X, y, epochs, lr, g, batch=64):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for idx in torch.randperm(len(y), generator=g).split(batch):
            loss = torch.nn.functional.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        s = seed * 100 + i
        torch.manual_seed(s)
        g = torch.Generator().manual_seed(s)
        X, y = torch.tensor(t["X"]), torch.tensor(t["y"])
        probe = make_student(t["backbone"])
        fit(probe, probe[1].parameters(), X, y, hparams["probe_epochs"], hparams["probe_lr"], g)
        with torch.no_grad():
            y2 = probe(X).argmax(1)
        m = make_student(t["backbone"])
        fit(m, m.parameters(), X, y2, hparams["ft_epochs"], hparams["ft_lr"], g)
        out.append(export(m))
    return out
