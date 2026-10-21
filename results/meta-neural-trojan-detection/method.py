"""Meta Neural Trojan Detection (Xu et al. 2021), adapted to the defender's 100 clean images.

1. Train shadow models with the target architecture on the clean images: half clean, half
   trojaned with triggers drawn from a broad "jumbo" distribution (random patches, scattered
   pixels or blended patterns, random target class).
2. Jointly learn a set of query inputs and a linear meta-classifier on the shadow models'
   softmax outputs at those queries (query tuning).
3. Score each model under test with the meta-classifier. No target class is guessed.
"""
import numpy as np
import torch
from torch import nn

from public import load_model, make_mlp

HPARAMS = [
    {"n_shadow": 32, "n_query": 10, "poison": 0.1, "epochs": 300, "meta_steps": 500},
    {"n_shadow": 32, "n_query": 10, "poison": 0.2, "epochs": 300, "meta_steps": 500},
    {"n_shadow": 48, "n_query": 10, "poison": 0.1, "epochs": 300, "meta_steps": 500},
    {"n_shadow": 32, "n_query": 20, "poison": 0.1, "epochs": 300, "meta_steps": 500},
    {"n_shadow": 32, "n_query": 5, "poison": 0.1, "epochs": 300, "meta_steps": 500},
    {"n_shadow": 48, "n_query": 20, "poison": 0.2, "epochs": 300, "meta_steps": 800},
    {"n_shadow": 32, "n_query": 10, "poison": 0.05, "epochs": 300, "meta_steps": 500},
    {"n_shadow": 32, "n_query": 10, "poison": 0.1, "epochs": 150, "meta_steps": 500},
]


def jumbo_trigger(rng):
    m, p = np.zeros(64, np.float32), np.zeros(64, np.float32)
    kind = rng.integers(3)
    if kind == 0:  # square patch of random size and location
        s = rng.integers(1, 4)
        r, c = rng.integers(0, 9 - s, 2)
        img = np.zeros((8, 8), np.float32)
        img[r:r + s, c:c + s] = 1
        m, p = img.ravel(), img.ravel().copy()
    elif kind == 1:  # scattered pixels
        idx = rng.choice(64, rng.integers(2, 7), replace=False)
        m[idx] = 1
        p[idx] = 1
    else:  # blended random pattern
        m[:] = rng.uniform(0.05, 0.3)
        p = rng.uniform(0, 1, 64).astype(np.float32)
    return m, p


def train_shadow(X, y, hidden, trojan, poison, epochs, rng, seed):
    torch.manual_seed(seed)
    if trojan:
        m, p = jumbo_trigger(rng)
        tgt = int(rng.integers(10))
        src = np.where(y != tgt)[0]
        pick = rng.choice(src, max(1, int(round(poison * len(X)))), replace=False)
        Xp = (1 - m) * X[pick] + m * p
        X = np.concatenate([X, Xp])
        y = np.concatenate([y, np.full(len(pick), tgt)])
    net = make_mlp(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(net(Xt + 0.02 * torch.randn_like(Xt)), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


def run(tasks, hparams, seed):
    rng = np.random.default_rng(seed)
    X, y = np.asarray(tasks[0]["clean_X"], np.float32), np.asarray(tasks[0]["clean_y"])
    hidden = tasks[0]["hidden"]
    n = hparams["n_shadow"]
    shadows = [train_shadow(X, y, hidden, i % 2 == 1, hparams["poison"], hparams["epochs"], rng, seed * 1000 + i)
               for i in range(n)]
    labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)
    torch.manual_seed(seed)
    q_logit = nn.Parameter(torch.randn(hparams["n_query"], 64) * 0.5)
    meta = nn.Linear(hparams["n_query"] * 10, 1)
    opt = torch.optim.Adam([q_logit, *meta.parameters()], lr=1e-2)

    def feats(model):
        return torch.softmax(model(torch.sigmoid(q_logit)), 1).reshape(-1)

    for _ in range(hparams["meta_steps"]):
        F = torch.stack([feats(s) for s in shadows])
        loss = nn.functional.binary_cross_entropy_with_logits(meta(F).squeeze(1), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return [{"score": float(meta(feats(load_model(t))[None]).item()), "target": None} for t in tasks]
