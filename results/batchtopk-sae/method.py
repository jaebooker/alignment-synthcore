"""BatchTopK SAE (Bussmann, Leask, Nanda 2024): keep the batch_size * k largest
pre-activations across the whole batch instead of k per sample, so the number of active
latents adapts to each input while the average stays k."""
import numpy as np
import torch

HPARAMS = [
    {"k": 2, "lr": 3e-3, "steps": 3000},
    {"k": 1, "lr": 3e-3, "steps": 3000},
    {"k": 3, "lr": 3e-3, "steps": 3000},
    {"k": 4, "lr": 3e-3, "steps": 3000},
    {"k": 2, "lr": 1e-3, "steps": 3000},
    {"k": 3, "lr": 1e-3, "steps": 3000},
    {"k": 2, "lr": 1e-2, "steps": 3000},
    {"k": 3, "lr": 1e-2, "steps": 3000},
]


def train(H, m, k, lr, steps, seed, batch=512):
    g = torch.Generator().manual_seed(seed)
    H = torch.tensor(H, dtype=torch.float32)
    d = H.shape[1]
    W_dec = torch.randn(m, d, generator=g)
    W_dec = torch.nn.Parameter(W_dec / W_dec.norm(dim=1, keepdim=True))
    W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(m))
    b_dec = torch.nn.Parameter(H.mean(0).clone())
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=lr)
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (batch,), generator=g)]
        pre = torch.relu((x - b_dec) @ W_enc + b_enc)
        flat = pre.flatten()
        top = torch.topk(flat, min(k, m) * batch)
        z = torch.zeros_like(flat).scatter(0, top.indices, top.values).view_as(pre)
        loss = ((z @ W_dec + b_dec - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return W_dec.detach().numpy()


def run(tasks, hparams, seed):
    return [{"directions": train(t["H_train"], t["dict_size"], hparams["k"], hparams["lr"], hparams["steps"], seed + i)}
            for i, t in enumerate(tasks)]
