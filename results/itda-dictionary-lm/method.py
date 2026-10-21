"""ITDA (Leask, Nanda, Al Moubayed 2025) on GPU: grow the dictionary from real activation
examples whose matching-pursuit reconstruction over the current dictionary is poor.
Examples are processed in small chunks against a dictionary frozen per chunk (a speed choice)."""
import torch

HPARAMS = [
    {"k": 32, "tau": 0.5},
    {"k": 16, "tau": 0.5},
    {"k": 32, "tau": 0.3},
    {"k": 64, "tau": 0.5},
]


def mp_rel_error(X, D, k):
    R = X.clone()
    for _ in range(k):
        C = R @ D.T
        j = C.abs().argmax(1)
        c = C.gather(1, j[:, None])
        R = R - c * D[j]
    return R.norm(dim=1) / X.norm(dim=1).clamp_min(1e-9)


def run(tasks, hparams, seed, chunk=64):
    t = tasks[0]
    H, m = t["H_train"], t["dict_size"]
    g = torch.Generator().manual_seed(seed)
    X = H[torch.randperm(len(H), generator=g).to(H.device)]
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    D = X[:1].clone()
    i = 1
    while len(D) < m and i < len(X):
        xb = X[i:i + chunk]
        add = xb[mp_rel_error(xb, D, min(hparams["k"], len(D))) > hparams["tau"]]
        D = torch.cat([D, add[: m - len(D)]])
        i += chunk
    return [{"directions": D.cpu().numpy()}]
