"""Baseline: TopK sparse autoencoder (Gao et al. 2024) on GPU. Tuned with the same budget as any method."""
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
    {"k": 32, "lr": 1e-3},
    {"k": 16, "lr": 1e-3},
    {"k": 64, "lr": 1e-3},
    {"k": 32, "lr": 3e-4},
]


def run(tasks, hparams, seed):
    t = tasks[0]
    H, m, k = t["H_train"], t["dict_size"], hparams["k"]
    g = torch.Generator().manual_seed(seed)
    W_dec, W_enc, b_enc, b_dec = init(H, m, g)
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=hparams["lr"])
    for x in batches(H, g):
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, k, dim=1)
        z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        loss = ((z @ W_dec + b_dec - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        renorm(W_dec)
    return [{"directions": W_dec.detach().cpu().numpy()}]
