"""Per-class minimal universal additive perturbation; MAD anomaly on the smallest norm."""
import numpy as np
import torch

from public import load_model

HPARAMS = [
    {"lam": 0.05, "steps": 300, "lr": 0.05, "norm": "l2"},
    {"lam": 0.02, "steps": 300, "lr": 0.05, "norm": "l2"},
    {"lam": 0.1, "steps": 300, "lr": 0.05, "norm": "l2"},
    {"lam": 0.05, "steps": 300, "lr": 0.05, "norm": "l1"},
    {"lam": 0.02, "steps": 300, "lr": 0.05, "norm": "l1"},
    {"lam": 0.05, "steps": 600, "lr": 0.02, "norm": "l2"},
]


def perturbation_norms(model, X, lam, steps, lr, norm, seed):
    g = torch.Generator().manual_seed(seed)
    C = 10
    delta = torch.nn.Parameter(torch.randn(C, 64, generator=g) * 0.01)
    opt = torch.optim.Adam([delta], lr=lr)
    targets = torch.arange(C).repeat_interleave(len(X))
    for _ in range(steps):
        Xa = (X[None] + delta[:, None]).clamp(0, 1)
        ce = torch.nn.functional.cross_entropy(model(Xa.reshape(-1, 64)), targets)
        size = delta.norm(dim=1) if norm == "l2" else delta.abs().sum(1)
        loss = ce + lam * size.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xa = (X[None] + delta[:, None]).clamp(0, 1)
        flip = (model(Xa.reshape(-1, 64)).argmax(1) == targets).float().reshape(C, -1).mean(1)
        size = delta.norm(dim=1) if norm == "l2" else delta.abs().sum(1)
    # a perturbation that fails to flip doesn't count as small
    return (size / flip.clamp_min(0.05)).numpy()


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        model = load_model(t)
        X = torch.tensor(t["clean_X"])
        s = perturbation_norms(model, X, hparams["lam"], hparams["steps"], hparams["lr"], hparams["norm"],
                               seed * 1000 + i)
        med = np.median(s)
        mad = np.median(np.abs(s - med)) * 1.4826 + 1e-8
        j = int(np.argmin(s))
        out.append({"score": float((med - s[j]) / mad), "target": j})
    return out
