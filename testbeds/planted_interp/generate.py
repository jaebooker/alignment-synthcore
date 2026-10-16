"""Toy models of superposition with known feature directions (evaluator side).

A model maps sparse x in R^n to h = W x in R^d (d < n), reconstructs
x' = ReLU(W^T h + b). After training, column W_i is the direction feature i
occupies in activation space: that is the planted ground truth.

To leave headroom over a tuned SAE, the feature distribution plants two
known difficulties:
  * hierarchy: a share of features are children that can fire only when
    their parent fires (the setting that produces feature absorption);
  * noise: isotropic Gaussian noise on the observed activations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CACHE = Path(__file__).resolve().parents[2] / "results" / "_cache" / "planted_interp"

# (n_features, d_hidden, p_active, importance_decay, child_frac, noise_std)
CONFIGS = {
    "toy": [(20, 10, 0.05, 1.0, 0.3, 0.02)],
    "full": [
        (40, 10, 0.04, 1.0, 0.3, 0.02),
        (60, 15, 0.04, 1.0, 0.4, 0.03),
        (64, 16, 0.06, 1.0, 0.3, 0.03),
        (80, 20, 0.04, 0.99, 0.5, 0.02),
    ],
}
SPLIT_OFFSET = {"dev": 1000, "test": 2000}
REPRESENTED_NORM = 0.5  # a feature counts as planted if ||W_i|| exceeds this
CHILD_GIVEN_PARENT = 0.5


def make_hierarchy(n, child_frac, gen):
    """parent[i] = index of i's parent, or -1 for root features."""
    n_child = int(round(n * child_frac))
    perm = torch.randperm(n, generator=gen)
    children, roots = perm[:n_child], perm[n_child:]
    parent = -torch.ones(n, dtype=torch.long)
    parent[children] = roots[torch.randint(0, len(roots), (n_child,), generator=gen)]
    return parent


def sample_x(batch, n, p_active, gen, parent=None):
    on = torch.rand(batch, n, generator=gen) < p_active
    if parent is not None:
        is_child = parent >= 0
        child_on = torch.rand(batch, n, generator=gen) < CHILD_GIVEN_PARENT
        par_on = on[:, parent.clamp_min(0)]
        on = torch.where(is_child, par_on & child_on, on)
    return on.float() * torch.rand(batch, n, generator=gen)


def train_toy(n, d, p_active, decay, parent, seed, steps=5000, batch=1024, lr=3e-3):
    gen = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(d, n, generator=gen) * (2.0 / (n + d)) ** 0.5)
    b = torch.nn.Parameter(torch.zeros(n))
    imp = decay ** torch.arange(n, dtype=torch.float32)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        x = sample_x(batch, n, p_active, gen, parent)
        xh = torch.relu((x @ W.T) @ W + b)
        loss = (imp * (x - xh) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), float(loss.detach())


def build_model(cfg, seed, n_train=16000, n_heldout=4000):
    """Returns dict with public activations and private ground truth. Cached."""
    n, d, p, decay, child_frac, noise = cfg
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"m_{'_'.join(map(str, cfg))}_{seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(seed + 3)
    parent = make_hierarchy(n, child_frac, gen)
    W, loss = train_toy(n, d, p, decay, parent, seed)
    gen = torch.Generator().manual_seed(seed + 7)
    X = sample_x(n_train + n_heldout, n, p, gen, parent)
    H = X @ W.T
    H = (H + noise * torch.randn(H.shape, generator=gen)).numpy().astype(np.float32)
    norms = W.norm(dim=0)
    rep = (norms > REPRESENTED_NORM).numpy()
    true_dirs = (W[:, rep] / norms[rep]).T.numpy()  # (n_rep, d)
    is_child = (parent >= 0).numpy()[rep]
    out = {
        "cfg": cfg, "seed": seed, "loss": loss,
        "H_train": H[:n_train], "H_heldout": H[n_train:],
        "true_dirs": true_dirs, "true_is_child": is_child, "n_represented": int(rep.sum()),
        "k_eval": max(1, int(round(n * p))),
    }
    torch.save(out, path)
    return out
