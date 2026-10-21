"""ITDA (Leask, Nanda, Al Moubayed 2025) on GPU: grow the dictionary from real activation
examples, adding the ones worst approximated by matching pursuit over the current dictionary.

Method choices (not testbed rules):
  * Examples stream in chunks. Within a chunk, the single worst-approximated example is added
    if its relative error exceeds tau, errors are recomputed against the grown dictionary, and
    so on until nothing in the chunk exceeds tau. This matches the paper's greedy rule.
  * `center` subtracts the activation mean before normalizing. The paper's abstract does not
    say; it is in the grid so tuning decides.

v0 tested a whole chunk against a dictionary frozen for that chunk and added every example
over tau, so early chunks were added wholesale and the dictionary became roughly the first
examples seen (0.1s runtime vs 11s for the baseline). Triage flagged it; see results/TRIAGE.md.
"""
import torch

HPARAMS = [
    {"k": 32, "tau": 0.5, "center": False},
    {"k": 32, "tau": 0.5, "center": True},
    {"k": 16, "tau": 0.3, "center": True},
    {"k": 64, "tau": 0.5, "center": True},
]


def mp_rel_error(X, D, k):
    R = X.clone()
    for _ in range(k):
        C = R @ D.T
        j = C.abs().argmax(1)
        R = R - C.gather(1, j[:, None]) * D[j]
    return R.norm(dim=1) / X.norm(dim=1).clamp_min(1e-9)


def run(tasks, hparams, seed, chunk=256):
    t = tasks[0]
    H, m = t["H_train"], t["dict_size"]
    g = torch.Generator().manual_seed(seed)
    X = H[torch.randperm(len(H), generator=g).to(H.device)]
    if hparams["center"]:
        X = X - H.mean(0)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    D = X[:1].clone()
    i = 1
    while len(D) < m and i < len(X):
        xb = X[i:i + chunk]
        i += chunk
        while len(D) < m and len(xb):
            err = mp_rel_error(xb, D, min(hparams["k"], len(D)))
            j = int(err.argmax())
            if err[j] <= hparams["tau"]:
                break
            D = torch.cat([D, xb[j:j + 1]])
            xb = torch.cat([xb[:j], xb[j + 1:]])
    return [{"directions": D.cpu().numpy()}]
