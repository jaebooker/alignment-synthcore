"""Baseline: Neural Cleanse-style trigger reverse engineering (Wang et al. 2019), simplified.

For each candidate target class, optimize a mask and pattern that flip clean
images to that class while keeping the mask small. A backdoored model has
one class with an anomalously small mask. Score = MAD anomaly index of the
smallest mask norm; target = that class.
"""
import numpy as np
import torch

from public import load_model

HPARAMS = [
    {"lam": 1e-2, "steps": 300, "lr": 0.1},
    {"lam": 3e-3, "steps": 300, "lr": 0.1},
    {"lam": 3e-2, "steps": 300, "lr": 0.1},
    {"lam": 1e-2, "steps": 600, "lr": 0.05},
    {"lam": 3e-3, "steps": 600, "lr": 0.05},
    {"lam": 1e-1, "steps": 300, "lr": 0.1},
]


def mask_norms(model, X, lam, steps, lr, seed):
    g = torch.Generator().manual_seed(seed)
    C = 10
    mask_logit = torch.nn.Parameter(torch.randn(C, 64, generator=g) * 0.1 - 2.0)
    pat_logit = torch.nn.Parameter(torch.randn(C, 64, generator=g) * 0.1)
    opt = torch.optim.Adam([mask_logit, pat_logit], lr=lr)
    targets = torch.arange(C).repeat_interleave(len(X))
    for _ in range(steps):
        m, p = torch.sigmoid(mask_logit), torch.sigmoid(pat_logit)
        Xa = (1 - m[:, None]) * X[None] + (m * p)[:, None]
        ce = torch.nn.functional.cross_entropy(model(Xa.reshape(-1, 64)), targets)
        loss = ce + lam * m.sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(mask_logit).sum(1).detach().numpy()


def anomaly(norms):
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) * 1.4826 + 1e-8
    t = int(np.argmin(norms))
    return float((med - norms[t]) / mad), t


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        model = load_model(t)
        X = torch.tensor(t["clean_X"])
        norms = mask_norms(model, X, hparams["lam"], hparams["steps"], hparams["lr"], seed * 1000 + i)
        score, target = anomaly(norms)
        out.append({"score": score, "target": target})
    return out
