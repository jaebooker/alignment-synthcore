"""Binary sparse coding (2025): a TopK SAE whose active latents are binary, each decoding to
a fixed vector (its decoder row times a learned per-latent scale). Selection is TopK on the
encoder pre-activations; a straight-through estimator passes gradients to the encoder."""
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
    scale = torch.nn.Parameter(torch.full((m,), float(H.norm(dim=1).mean()) / 2))
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec, scale], lr=lr)
    k = min(k, m)
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (batch,), generator=g)]
        pre = (x - b_dec) @ W_enc + b_enc
        hard = torch.zeros_like(pre).scatter(1, torch.topk(pre, k, dim=1).indices, 1.0)
        soft = torch.sigmoid(pre)
        z = hard + soft - soft.detach()  # straight-through
        loss = (((z * scale) @ W_dec + b_dec - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return W_dec.detach().numpy()


def run(tasks, hparams, seed):
    return [{"directions": train(t["H_train"], t["dict_size"], hparams["k"], hparams["lr"], hparams["steps"], seed + i)}
            for i, t in enumerate(tasks)]
