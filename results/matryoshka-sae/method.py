"""Matryoshka SAE: a TopK SAE whose loss also asks every nested prefix of the dictionary
to reconstruct the input on its own (Bussmann et al. 2025; Nabeshima 2024).

Method choices: prefixes are fractions of the dictionary size; the loss is the mean
reconstruction error over prefixes; TopK is chosen over the full dictionary and each
prefix uses whichever of its latents are active.
"""
import numpy as np
import torch

HPARAMS = [
    {"k": 2, "lr": 3e-3, "steps": 3000, "prefixes": [0.25, 0.5, 1.0]},
    {"k": 3, "lr": 3e-3, "steps": 3000, "prefixes": [0.25, 0.5, 1.0]},
    {"k": 2, "lr": 3e-3, "steps": 3000, "prefixes": [0.5, 1.0]},
    {"k": 3, "lr": 3e-3, "steps": 3000, "prefixes": [0.5, 1.0]},
    {"k": 2, "lr": 1e-3, "steps": 3000, "prefixes": [0.25, 0.5, 1.0]},
    {"k": 4, "lr": 3e-3, "steps": 3000, "prefixes": [0.25, 0.5, 1.0]},
    {"k": 2, "lr": 1e-2, "steps": 3000, "prefixes": [0.125, 0.25, 0.5, 1.0]},
    {"k": 3, "lr": 3e-3, "steps": 3000, "prefixes": [0.125, 0.25, 0.5, 1.0]},
]


def train(H, m, k, lr, steps, prefixes, seed, batch=512):
    g = torch.Generator().manual_seed(seed)
    H = torch.tensor(H, dtype=torch.float32)
    d = H.shape[1]
    W_dec = torch.randn(m, d, generator=g)
    W_dec = torch.nn.Parameter(W_dec / W_dec.norm(dim=1, keepdim=True))
    W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(m))
    b_dec = torch.nn.Parameter(H.mean(0).clone())
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=lr)
    k = min(k, m)
    sizes = sorted({max(1, int(round(f * m))) for f in prefixes})
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (batch,), generator=g)]
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, k, dim=1)
        z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        loss = sum(((z[:, :p] @ W_dec[:p] + b_dec - x) ** 2).sum(1).mean() for p in sizes) / len(sizes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return W_dec.detach().numpy()


def run(tasks, hparams, seed):
    return [{"directions": train(t["H_train"], t["dict_size"], hparams["k"], hparams["lr"], hparams["steps"],
                                 hparams["prefixes"], seed + i)} for i, t in enumerate(tasks)]
