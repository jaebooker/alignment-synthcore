"""Batch-composition intervention for BatchTopK (Commons test matrix row proposal-batchtopk-sae-1).

Question: does BatchTopK's batch-level top-k help recover rare child features because it moves
the active-code budget from simple examples to complex ones within a batch? If so, the benefit
should (a) grow with the size of the pool the top-k is taken over, and (b) shrink when every
minibatch is made of similar examples, leaving nothing to reallocate.

Held fixed within a seed: the planted toy model and its activations, the dictionary width, the
average active-code budget (k = 2 per example), learning rate, steps, batch size (512), the
initialisation and, within a composition, the exact minibatch sequence.

Varied:
  partition size b in {1, 8, 64, 512}: each minibatch is cut into random partitions of b
      examples and the top b*k pre-activations are kept within each partition. b = 1 is
      per-example TopK; b = 512 is the catalogued BatchTopK.
  composition in {random, concentrated}: random draws 512 training examples uniformly each
      step. concentrated sorts the training examples by how many planted features are active
      (random tie-break), cuts them into blocks of 512, and draws one block per step, so each
      minibatch holds examples of similar complexity.

Metrics on each trained decoder: MMCS to all planted directions, to child features (rare: they
fire only with their parent, half the time) and to parents that have a represented child
(frequent). On 4,000 held-out examples: fraction of variance explained and mean active latents,
using the same partitioned top-k at partition size b over random held-out partitions.

Usage: python study.py {calib|confirm} n_seeds out.json [first_seed]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from testbeds.planted_interp import generate as G  # noqa: E402

# Seed bases unused by the catalogue (1000.., 2000..) and by earlier studies (50k, 60k, 500k, 600k).
BASE = {"calib": 70_000, "confirm": 170_000}
PARTITIONS = [1, 8, 64, 512]
COMPOSITIONS = ["random", "concentrated"]
HP = {"k": 2, "lr": 3e-3, "steps": 3000, "batch": 512}  # catalogued defaults (HPARAMS[0])


def unit(a):
    a = np.asarray(a, np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def make_model(cfg, seed, n_train=16000, n_heldout=4000):
    """generate.build_model's construction, keeping the ground-truth codes and hierarchy."""
    n, d, p, decay, child_frac, noise = cfg
    gen = torch.Generator().manual_seed(seed + 3)
    parent = G.make_hierarchy(n, child_frac, gen)
    W, _ = G.train_toy(n, d, p, decay, parent, seed)
    gen = torch.Generator().manual_seed(seed + 7)
    X = G.sample_x(n_train + n_heldout, n, p, gen, parent)
    H = (X @ W.T + noise * torch.randn(n_train + n_heldout, d, generator=gen)).numpy().astype(np.float32)
    norms = W.norm(dim=0)
    rep = (norms > G.REPRESENTED_NORM).numpy()
    par = parent.numpy()
    has_child = np.isin(np.arange(n), par[(par >= 0) & rep])
    return {"H": H, "X": X.numpy(), "n_train": n_train, "rep": rep,
            "true_dirs": (W[:, rep] / norms[rep]).T.numpy().astype(np.float64),
            "is_child": (par >= 0)[rep], "is_parent": has_child[rep]}


def batches(m, composition, seed, steps, batch):
    """The minibatch index sequence, shared by every partition size within a composition."""
    rng = np.random.default_rng(seed + 501)
    n = m["n_train"]
    if composition == "random":
        return [rng.integers(0, n, batch) for _ in range(steps)]
    active = (m["X"][:n] > 0).sum(1)
    order = np.lexsort((rng.random(n), active))  # by number of active features, random ties
    blocks = [order[i:i + batch] for i in range(0, n - batch + 1, batch)]
    return [blocks[j] for j in rng.integers(0, len(blocks), steps)]


def partitioned_topk(pre, k, b, gen):
    """Keep the top b*k of relu(pre) within random partitions of b rows (b = 1: per-row TopK)."""
    pre = torch.relu(pre)
    if b == 1:
        top = torch.topk(pre, k, dim=1)
        return torch.zeros_like(pre).scatter(1, top.indices, top.values)
    rows, m = pre.shape
    perm = torch.randperm(rows, generator=gen)
    z = torch.zeros_like(pre)
    full = rows // b * b
    for idx in ([perm[:full].view(-1, b)] if full else []) + ([perm[full:].view(1, -1)] if full < rows else []):
        flat = pre[idx].reshape(len(idx), -1)  # one row per partition
        top = torch.topk(flat, min(k * idx.shape[1], flat.shape[1]), dim=1)
        z[idx] = torch.zeros_like(flat).scatter(1, top.indices, top.values).view(idx.shape[0], idx.shape[1], m)
    return z


def train(H, m_lat, b, seq, seed):
    g = torch.Generator().manual_seed(seed)
    H = torch.tensor(H, dtype=torch.float32)
    d = H.shape[1]
    W_dec = torch.randn(m_lat, d, generator=g)
    W_dec = torch.nn.Parameter(W_dec / W_dec.norm(dim=1, keepdim=True))
    W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(m_lat))
    b_dec = torch.nn.Parameter(H.mean(0).clone())
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=HP["lr"])
    gp = torch.Generator().manual_seed(seed + 13)
    for idx in seq:
        x = H[torch.as_tensor(idx)]
        z = partitioned_topk((x - b_dec) @ W_enc + b_enc, HP["k"], b, gp)
        loss = ((z @ W_dec + b_dec - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return [t.detach() for t in (W_dec, W_enc, b_enc, b_dec)]


def evaluate(params, m, b, seed):
    W_dec, W_enc, b_enc, b_dec = params
    Hte = torch.tensor(m["H"][m["n_train"]:])
    with torch.no_grad():
        z = partitioned_topk((Hte - b_dec) @ W_enc + b_enc, HP["k"], b, torch.Generator().manual_seed(seed + 17))
        rec = z @ W_dec + b_dec
    fve = float(1 - ((Hte - rec) ** 2).sum() / ((Hte - Hte.mean(0)) ** 2).sum())
    best = (m["true_dirs"] @ unit(W_dec.numpy()).T).max(axis=1)
    mean = lambda mask: float(best[mask].mean()) if mask.any() else float("nan")  # noqa: E731
    return {"mmcs": float(best.mean()), "child": mean(m["is_child"]), "parent": mean(m["is_parent"]),
            "fve": fve, "l0": float((z > 0).sum(1).float().mean()),
            "l0_sd": float((z > 0).sum(1).float().std())}


def run_seed(phase, seed):
    torch.set_num_threads(1)
    out = {}
    for i, cfg in enumerate(G.CONFIGS["full"]):
        s = BASE[phase] + 97 * seed + i
        m = make_model(cfg, s)
        Htr = m["H"][:m["n_train"]]
        act = (m["X"][:m["n_train"]] > 0).sum(1)
        r = {"n_rep": int(m["rep"].sum()), "n_child": int(m["is_child"].sum()), "n_parent": int(m["is_parent"].sum()),
             "active_mean": float(act.mean()), "active_sd": float(act.std())}
        for comp in COMPOSITIONS:
            seq = batches(m, comp, s, HP["steps"], HP["batch"])
            if comp == "concentrated":  # how homogeneous the minibatches are (mean within-batch sd of active count)
                r["concentrated_within_sd"] = float(np.mean([act[q].std() for q in seq[:200]]))
            for b in PARTITIONS:
                params = train(Htr, cfg[0], b, seq, s)  # same init and minibatches across b
                r[f"{comp}/b{b}"] = evaluate(params, m, b, s)
        out[str(i)] = r
    return out


def main():
    phase, n_seeds, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for seed in range(first, first + n_seeds):
        if str(seed) in res:
            continue
        t0 = time.time()
        res[str(seed)] = run_seed(phase, seed)
        out_path.write_text(json.dumps(res, indent=1))
        print(phase, seed, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
