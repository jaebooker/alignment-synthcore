"""Weak-to-strong tasks on scikit-learn's bundled 8x8 digits (evaluator side).

CPU stand-in for Burns et al. (2023). No language models: none can be downloaded here.

Per seed:
  * pretrain split (500 images): a backbone is trained on 10-class digit labels.
    This is the strong student's "pretrained knowledge".
  * six binary tasks, each a partition of the ten digits (parity, >= 5, ...).
  * weak supervisor: logistic regression on 4x4 average-pooled images, trained on
    150 gold labels. Its errors vary within a digit class, so a student that knows
    digit identity can in principle see past them.
  * the student-train split carries the weak supervisor's hard labels.
  * gold control tasks: the same partitions on a different random split, carrying
    gold labels. Methods are not told which tasks are which.
  * ceiling: the student architecture trained on gold labels with a fixed recipe.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from .public import make_backbone, make_student

PARTITIONS = {
    "parity": [1, 3, 5, 7, 9],
    "ge5": [5, 6, 7, 8, 9],
    "round": [0, 6, 8, 9],
    "one47": [1, 4, 7],
    "two35": [2, 3, 5],
    "low": [0, 1, 2, 3],
}
SUITES = {
    ("dev", "toy"): ["parity", "ge5"],
    ("dev", "full"): list(PARTITIONS),
    ("test", "full"): list(PARTITIONS),
}
SPLIT_OFFSET = {"dev": 500_000, "test": 600_000}
N_PRE, N_WEAK, N_STUDENT = 500, 150, 700
WEAK_C = 10.0
MIN_GAP = 0.06  # tasks whose ceiling - weak gap on test is smaller are excluded from PGR


def _data():
    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)
    return X.reshape(len(X), 64), X.reshape(len(X), 4, 2, 4, 2).mean((2, 4)).reshape(len(X), 16), d.target


def _fit(model, params, X, y, epochs, lr, seed, wd=1e-4, batch=64):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(yt), generator=g).split(batch):
            loss = torch.nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, X):
    return model(torch.tensor(X)).argmax(1).numpy()


def ceiling(backbone_state, X, y, Xte, yte, seed):
    """Best of a frozen-probe and a full fine-tune recipe on gold labels (scored on test)."""
    accs = []
    for mode, epochs, lr in (("probe", 50, 1e-2), ("ft", 30, 1e-3)):
        torch.manual_seed(seed)
        m = make_student(backbone_state)
        params = m[1].parameters() if mode == "probe" else m.parameters()
        _fit(m, params, X, y, epochs, lr, seed)
        accs.append(float((predict(m, Xte) == yte).mean()))
    return max(accs)


def build(split, config, seed):
    torch.set_num_threads(2)
    Xf, Xp, digit = _data()
    s0 = SPLIT_OFFSET[split] + 1000 * seed
    rng = np.random.default_rng(s0)
    p = rng.permutation(len(digit))
    ipre, rest = p[:N_PRE], p[N_PRE:]
    torch.manual_seed(s0)
    bb = make_backbone()
    _fit(bb, bb.parameters(), Xf[ipre], digit[ipre], 100, 1e-3, s0)
    bstate = {k: v.numpy().copy() for k, v in bb.state_dict().items()}

    public, private = [], []
    for kind in ("weak", "gold"):
        for name in SUITES[(split, config)]:
            if config == "toy" and kind == "gold" and name != SUITES[(split, config)][0]:
                continue
            r = rng.permutation(rest)
            iw, ist, ite = r[:N_WEAK], r[N_WEAK:N_WEAK + N_STUDENT], r[N_WEAK + N_STUDENT:]
            y = np.isin(digit, PARTITIONS[name]).astype(np.int64)
            weak = LogisticRegression(C=WEAK_C, max_iter=5000).fit(Xp[iw], y[iw])
            labels = weak.predict(Xp[ist]) if kind == "weak" else y[ist]
            weak_test = weak.predict(Xp[ite])
            ceil = ceiling(bstate, Xf[ist], y[ist], Xf[ite], y[ite], s0 + len(private))
            public.append({"X": Xf[ist].copy(), "y": labels.astype(np.int64), "backbone": bstate})
            private.append({"name": name, "kind": kind, "X_test": Xf[ite], "y_test": y[ite],
                            "weak_test": weak_test, "weak_acc": float((weak_test == y[ite]).mean()),
                            "ceiling_acc": ceil, "label_acc": float((labels == y[ist]).mean())})
    order = rng.permutation(len(public))  # interleave weak and gold tasks
    return [public[i] for i in order], [private[i] for i in order]
