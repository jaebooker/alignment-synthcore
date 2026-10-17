"""Spectral decoupling: cross-entropy plus (lam / 2) * mean squared logit, no weight decay."""
import torch

from public import export, make_policy

HPARAMS = [
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "lam": 0.1, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "lam": 0.01, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "lam": 0.03, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "lam": 0.3, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "lam": 1.0, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [32, 32], "hidden": 128}, "lam": 0.1, "epochs": 30, "lr": 1e-3},
    {"arch": {"type": "cnn", "channels": [16, 16], "hidden": 64}, "lam": 0.1, "epochs": 60, "lr": 3e-4},
    {"arch": {"type": "cnn", "channels": [32, 32], "hidden": 128}, "lam": 0.3, "epochs": 30, "lr": 1e-3},
]


def run(tasks, hparams, seed, batch=128):
    out = []
    for i, t in enumerate(tasks):
        s = seed * 100 + i
        torch.manual_seed(s)
        X = torch.tensor(t["obs"], dtype=torch.float32)
        y = torch.tensor(t["action"])
        model = make_policy(hparams["arch"], t["n_channels"], t["view"])
        opt = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
        g = torch.Generator().manual_seed(s)
        for _ in range(hparams["epochs"]):
            for idx in torch.randperm(len(y), generator=g).split(batch):
                logits = model(X[idx])
                loss = torch.nn.functional.cross_entropy(logits, y[idx]) + 0.5 * hparams["lam"] * (logits ** 2).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        out.append(export(model, hparams["arch"]))
    return out
