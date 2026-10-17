"""Just Train Twice (JTT) for behavior cloning.

Stage A: train a BC policy for a few epochs. Error set = training steps it misclassifies.
Stage B: retrain from scratch with error-set steps weighted by `up`.
"""
import numpy as np
import torch

from public import export, make_policy

HPARAMS = [
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "id_epochs": 5, "up": 20.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "id_epochs": 5, "up": 5.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "id_epochs": 5, "up": 50.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "id_epochs": 2, "up": 20.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "id_epochs": 10, "up": 20.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [32, 32], "hidden": 128}, "id_epochs": 5, "up": 20.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "id_epochs": 5, "up": 20.0, "epochs": 60, "lr": 3e-4},
    {"arch": {"type": "cnn", "channels": [32, 32], "hidden": 128}, "id_epochs": 5, "up": 50.0, "epochs": 30, "lr": 1e-3},
]


def train(X, y, w, arch, epochs, lr, seed, n_ch, view, batch=128):
    torch.manual_seed(seed)
    model = make_policy(arch, n_ch, view)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(y), generator=g).split(batch):
            ce = torch.nn.functional.cross_entropy(model(X[idx]), y[idx], reduction="none")
            loss = (ce * w[idx]).sum() / w[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        s = seed * 100 + i
        X = torch.tensor(t["obs"], dtype=torch.float32)
        y = torch.tensor(t["action"])
        ones = torch.ones(len(y))
        first = train(X, y, ones, hparams["arch"], hparams["id_epochs"], hparams["lr"], s,
                      t["n_channels"], t["view"])
        with torch.no_grad():
            wrong = first(X).argmax(1) != y
        w = torch.where(wrong, torch.tensor(hparams["up"]), torch.tensor(1.0))
        final = train(X, y, w, hparams["arch"], hparams["epochs"], hparams["lr"], s + 1,
                      t["n_channels"], t["view"])
        out.append(export(final, hparams["arch"]))
    return out
