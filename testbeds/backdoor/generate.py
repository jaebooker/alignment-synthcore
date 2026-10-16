"""Model populations with planted backdoors (evaluator side).

Data: scikit-learn's bundled 8x8 digits (no download). Every trigger is a
(mask, pattern) pair applied as x' = (1 - m) * x + m * p, with three families:
  patch: a 2x2 block set to 1                       (dev and test)
  blend: a random full-image pattern at alpha 0.15  (dev and test)
  pixel: 4 scattered pixels set to 1                (test only: held-out family)
Pixel is held out because it is the hardest family for the baseline at this
poison rate, so the headline number measures generalization where it matters.
A backdoored model was trained with 2% of non-target training images
triggered and relabeled to a random target class. Models are kept only if
attack success >= 0.8 and clean accuracy >= 0.9.
The defender gets only 100 clean images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits

from .public import make_mlp

CACHE = Path(__file__).resolve().parents[2] / "results" / "_cache" / "backdoor"

# config -> (hidden, {family: count})
POPULATIONS = {
    ("dev", "toy"): (32, {"clean": 8, "patch": 8}),
    ("dev", "full"): (64, {"clean": 16, "patch": 8, "blend": 8}),
    ("test", "full"): (64, {"clean": 48, "patch": 8, "blend": 8, "pixel": 48}),
}
HELDOUT_FAMILY = "pixel"
DEV_FAMILIES = ["patch", "blend"]
SPLIT_OFFSET = {"dev": 10_000, "test": 20_000}
POISON_RATE = 0.02
MIN_ASR = 0.8
BLEND_ALPHA = 0.15
N_CLEAN_VAL = 100


def data():
    d = load_digits()
    X = (d.images.reshape(len(d.images), 64) / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return {"train": (X[:1200], y[:1200]), "clean_val": (X[1200:1200 + N_CLEAN_VAL], y[1200:1200 + N_CLEAN_VAL]),
            "test": (X[1500:], y[1500:])}


def make_trigger(family, rng):
    m = np.zeros(64, np.float32)
    p = np.zeros(64, np.float32)
    if family == "patch":
        r, c = rng.integers(0, 7, 2)
        img = np.zeros((8, 8), np.float32)
        img[r:r + 2, c:c + 2] = 1
        m = img.ravel()
        p = m.copy()
    elif family == "pixel":
        idx = rng.choice(64, 4, replace=False)
        m[idx] = 1
        p[idx] = 1
    elif family == "blend":
        m[:] = BLEND_ALPHA
        p = rng.random(64).astype(np.float32)
    else:
        raise ValueError(family)
    return m, p


def apply_trigger(X, trig):
    m, p = trig
    return (1 - m) * X + m * p


def train(X, y, hidden, seed, epochs=60, batch=64, lr=3e-3):
    torch.manual_seed(seed)
    model = make_mlp(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=g).split(batch):
            loss = torch.nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    return float((model(torch.tensor(X)).argmax(1).numpy() == y).mean())


def make_one(family, hidden, seed, D):
    Xtr, ytr = D["train"]
    Xte, yte = D["test"]
    for attempt in range(20):
        s = seed * 100 + attempt if attempt >= 10 else seed * 10 + attempt
        rng = np.random.default_rng(s)
        if family == "clean":
            model = train(Xtr, ytr, hidden, s)
            info = {"family": "clean", "target": None, "asr": None}
        else:
            target = int(rng.integers(10))
            trig = make_trigger(family, rng)
            X, y = Xtr.copy(), ytr.copy()
            cand = np.where(y != target)[0]
            pois = rng.choice(cand, int(POISON_RATE * len(X)), replace=False)
            X[pois] = apply_trigger(X[pois], trig)
            y[pois] = target
            model = train(X, y, hidden, s)
            nt = yte != target
            asr = accuracy(model, apply_trigger(Xte[nt], trig), np.full(nt.sum(), target))
            info = {"family": family, "target": target, "asr": asr}
            if asr < MIN_ASR:
                continue
        info["clean_acc"] = accuracy(model, Xte, yte)
        if info["clean_acc"] >= 0.9:
            info["weights"] = {k: v.numpy().copy() for k, v in model.state_dict().items()}
            info["hidden"] = hidden
            return info
    raise RuntimeError(f"could not make a valid {family} model for seed {seed}")


def population(split, config):
    """Fixed per (split, config). Cached to disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    hidden, counts = POPULATIONS[(split, config)]
    key = "_".join(f"{f}{c}" for f, c in counts.items())
    path = CACHE / f"pop_{split}_{config}_h{hidden}_{key}_{POISON_RATE}_{BLEND_ALPHA}_{N_CLEAN_VAL}_{MIN_ASR}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    torch.set_num_threads(2)
    D = data()
    models, i = [], 0
    for family, count in counts.items():
        for _ in range(count):
            models.append(make_one(family, hidden, SPLIT_OFFSET[split] + i, D))
            i += 1
    order = np.random.default_rng(SPLIT_OFFSET[split] + 1).permutation(len(models))
    models = [models[j] for j in order]
    pop = {"models": models, "clean_val": D["clean_val"]}
    torch.save(pop, path)
    return pop
