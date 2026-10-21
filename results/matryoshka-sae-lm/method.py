"""Matryoshka SAE (Bussmann et al. 2025) on GPU: TopK SAE whose nested dictionary prefixes must each reconstruct the input."""
import os

import torch

STEPS = int(os.environ.get("PLANTED_LM_STEPS", "3000"))  # lowered only for the CPU smoke test
BATCH = 4096


def init(H, m, g):
    d = H.shape[1]
    W_dec = torch.randn(m, d, generator=g).to(H.device)
    W_dec = torch.nn.Parameter(W_dec / W_dec.norm(dim=1, keepdim=True))
    W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(m, device=H.device))
    b_dec = torch.nn.Parameter(H.mean(0).clone())
    return W_dec, W_enc, b_enc, b_dec


def batches(H, g, steps=STEPS, batch=BATCH):
    for _ in range(steps):
        yield H[torch.randint(0, len(H), (batch,), generator=g).to(H.device)]


def renorm(W_dec):
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

HPARAMS = [
    {"k": 32, "lr": 1e-3, "prefixes": [0.125, 0.25, 0.5, 1.0]},
    {"k": 64, "lr": 1e-3, "prefixes": [0.125, 0.25, 0.5, 1.0]},
    {"k": 32, "lr": 1e-3, "prefixes": [0.25, 0.5, 1.0]},
    {"k": 32, "lr": 3e-4, "prefixes": [0.125, 0.25, 0.5, 1.0]},
]


def run(tasks, hparams, seed):
    t = tasks[0]
    H, m, k = t["H_train"], t["dict_size"], hparams["k"]
    sizes = sorted({max(1, int(round(f * m))) for f in hparams["prefixes"]})
    g = torch.Generator().manual_seed(seed)
    W_dec, W_enc, b_enc, b_dec = init(H, m, g)
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=hparams["lr"])
    for x in batches(H, g):
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, k, dim=1)
        z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        loss = sum(((z[:, :p] @ W_dec[:p] + b_dec - x) ** 2).sum(1).mean() for p in sizes) / len(sizes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        renorm(W_dec)
    return [{"directions": W_dec.detach().cpu().numpy()}]
