"""Auxiliary confidence loss (Burns et al. 2023), binary version.

loss = (1 - a) * CE(f, weak) + a * CE(f, pseudo), pseudo = student's own prediction
thresholded so the batch keeps the weak labels' positive rate. a ramps linearly
from 0 to alpha over the first `warmup` fraction of training.
"""
import torch

from public import export, make_student

HPARAMS = [
    {"mode": "ft", "alpha": 0.75, "warmup": 0.2, "epochs": 30, "lr": 1e-3},
    {"mode": "ft", "alpha": 0.5, "warmup": 0.2, "epochs": 30, "lr": 1e-3},
    {"mode": "ft", "alpha": 0.9, "warmup": 0.2, "epochs": 30, "lr": 1e-3},
    {"mode": "ft", "alpha": 0.75, "warmup": 0.2, "epochs": 10, "lr": 1e-3},
    {"mode": "ft", "alpha": 0.75, "warmup": 0.5, "epochs": 30, "lr": 1e-3},
    {"mode": "probe", "alpha": 0.75, "warmup": 0.2, "epochs": 50, "lr": 1e-2},
    {"mode": "probe", "alpha": 0.5, "warmup": 0.2, "epochs": 50, "lr": 1e-2},
    {"mode": "ft", "alpha": 0.75, "warmup": 0.2, "epochs": 60, "lr": 3e-4},
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
        total = hparams["epochs"] * ((len(y) + batch - 1) // batch)
        step = 0
        for _ in range(hparams["epochs"]):
            for idx in torch.randperm(len(y), generator=g).split(batch):
                logits = m(X[idx])
                a = hparams["alpha"] * min(1.0, step / max(1.0, hparams["warmup"] * total))
                step += 1
                p1 = logits.softmax(1)[:, 1].detach()
                rate = y[idx].float().mean()
                thr = torch.quantile(p1, (1 - rate).clamp(0, 1))
                pseudo = (p1 > thr).long() if 0 < rate < 1 else y[idx]
                loss = (1 - a) * torch.nn.functional.cross_entropy(logits, y[idx]) + \
                    a * torch.nn.functional.cross_entropy(logits, pseudo)
                opt.zero_grad()
                loss.backward()
                opt.step()
        out.append(export(m))
    return out
