"""Generalized cross-entropy (Zhang & Sabuncu 2018): loss = (1 - p_y ** q) / q."""
import torch

from public import export, make_student

HPARAMS = [
    {"mode": "ft", "q": 0.7, "epochs": 30, "lr": 1e-3},
    {"mode": "ft", "q": 0.4, "epochs": 30, "lr": 1e-3},
    {"mode": "ft", "q": 0.9, "epochs": 30, "lr": 1e-3},
    {"mode": "ft", "q": 0.7, "epochs": 10, "lr": 1e-3},
    {"mode": "ft", "q": 0.7, "epochs": 60, "lr": 1e-3},
    {"mode": "probe", "q": 0.7, "epochs": 50, "lr": 1e-2},
    {"mode": "probe", "q": 0.4, "epochs": 50, "lr": 1e-2},
    {"mode": "ft", "q": 0.7, "epochs": 30, "lr": 3e-3},
]


def run(tasks, hparams, seed, batch=64):
    out = []
    q = hparams["q"]
    for i, t in enumerate(tasks):
        s = seed * 100 + i
        torch.manual_seed(s)
        m = make_student(t["backbone"])
        params = m[1].parameters() if hparams["mode"] == "probe" else m.parameters()
        opt = torch.optim.AdamW(params, lr=hparams["lr"], weight_decay=1e-4)
        X, y = torch.tensor(t["X"]), torch.tensor(t["y"])
        g = torch.Generator().manual_seed(s)
        for _ in range(hparams["epochs"]):
            for idx in torch.randperm(len(y), generator=g).split(batch):
                py = m(X[idx]).softmax(1).gather(1, y[idx, None]).squeeze(1).clamp_min(1e-7)
                loss = ((1 - py ** q) / q).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        out.append(export(m))
    return out
