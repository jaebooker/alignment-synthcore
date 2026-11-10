"""JTT: what does the error set select, and does upweighting it beat matched controls?

Answers test-matrix drafts proposal-just-train-twice-1 and -2 and the results-log open issue
"JTT's noise hypothesis is untested; a deterministic-tie-break dataset variant would test it".

Conditions (same levels, starts and evaluation episodes within a seed):
  R   expert breaks ties among optimal moves at random (the catalogue setting)
  D   expert breaks ties deterministically (lowest action index)
  DN  deterministic ties + 10% of training labels replaced by a different random action
Arms (identical architecture, learning rate, second-stage seed and 30 second-stage epochs):
  uniform       fresh second model, uniform weights
  uniform_long  uniform weights, 35 epochs (charges JTT's 5 identification epochs)
  jtt           first-stage errors upweighted x20
  random        an equally large random set upweighted x20
  oracle        steps from decorrelated training levels upweighted x20

Usage: python jtt_groups.py {calib|confirm} n_seeds out.json
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
from testbeds.goal_misgen import generate as G  # noqa: E402
from testbeds.goal_misgen.public import make_policy  # noqa: E402
from testbeds.goal_misgen.testbed import rollout  # noqa: E402

BASE = {"calib": 500_000, "confirm": 600_000}  # catalogue uses 300_000 (dev) and 400_000 (test)
FAMILIES = [("corner", 7), ("lamp", 7)]
ARCH = {"type": "cnn", "channels": [16, 16], "hidden": 64}
LR, ID_EPOCHS, EPOCHS, UP = 1e-3, 5, 30, 20.0
CONDITIONS = {"R": ("random", 0.0), "D": ("first", 0.0), "DN": ("first", 0.10)}


def zone_dist(level):
    d = np.full(level["walls"].shape, 10**6)
    for z in level["zone"]:
        if not level["walls"][z]:
            b = G._bfs(level["walls"], z)
            d = np.where(b >= 0, np.minimum(d, b), d)
    return d


def expert(level, start, tie, rng):
    """Shortest path to the cheese. Returns (pos, action, reward, n_optimal_moves) per step."""
    dist = G._bfs(level["walls"], level["cheese"])
    pos, steps = start, []
    while pos != level["cheese"]:
        best = [a for a in range(4) if dist[G.step(level, pos, a)] == dist[pos] - 1]
        a = best[rng.integers(len(best))] if tie == "random" else best[0]
        nxt = G.step(level, pos, a)
        steps.append((pos, a, 1.0 if nxt == level["cheese"] else 0.0, len(best)))
        pos = nxt
    return steps


def build(family, n, seed, tie, label_noise):
    rng = np.random.default_rng(seed)              # levels, starts, evaluation episodes
    tie_rng = np.random.default_rng(seed + 10**7)  # expert tie-breaks
    noise_rng = np.random.default_rng(seed + 2 * 10**7)
    perm = rng.permutation(G.N_CH)
    n_dec = int(round(G.DECORRELATED_TRAIN * G.N_TRAIN_LEVELS))
    flags = rng.permutation([False] * n_dec + [True] * (G.N_TRAIN_LEVELS - n_dec))
    train = [G.make_level(family, n, rng, correlated=bool(c)) for c in flags]
    starts = [G._starts(L, G.STARTS_PER_TRAIN_LEVEL, rng) for L in train]
    test = [G.make_level(family, n, rng, correlated=False) for _ in range(G.N_EVAL_LEVELS)]
    iid = [G.make_level(family, n, rng, correlated=True) for _ in range(G.N_EVAL_LEVELS)]
    eps = lambda levels: [(L, s) for L in levels for s in G._starts(L, G.STARTS_PER_EVAL_LEVEL, rng)]  # noqa: E731
    test_eps, iid_eps = eps(test), eps(iid)
    obs, act, dec, tied, conflict = [], [], [], [], []
    for L, ss, corr in zip(train, starts, flags):
        zd = zone_dist(L)
        for s in ss:
            for pos, a, _, nbest in expert(L, s, tie, tie_rng):
                obs.append(G.observe(L, pos, perm))
                act.append(a)
                dec.append(not corr)
                tied.append(nbest > 1)
                # proxy conflict: outside the cue zone, and the expert move does not approach it
                conflict.append(zd[pos] > 0 and zd[G.step(L, pos, a)] >= zd[pos])
    act = np.array(act)
    noisy = noise_rng.random(len(act)) < label_noise
    act = np.where(noisy, (act + noise_rng.integers(1, 4, len(act))) % 4, act)
    return {"obs": np.stack(obs), "action": act, "dec": np.array(dec), "tied": np.array(tied),
            "conflict": np.array(conflict), "noisy": noisy, "perm": perm, "test": test_eps,
            "iid": iid_eps, "horizon": 3 * n + 4, "view": int(obs[0].shape[-1])}


def train(X, y, w, epochs, seed, view, batch=128):
    torch.manual_seed(seed)
    model = make_policy(ARCH, G.N_CH, view)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(y), generator=g).split(batch):
            ce = torch.nn.functional.cross_entropy(model(X[idx]), y[idx], reduction="none")
            loss = (ce * w[idx]).sum() / w[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def score(model, t):
    ok, spur = rollout(model, t["test"], t["perm"], t["horizon"])
    ok_iid, _ = rollout(model, t["iid"], t["perm"], t["horizon"])
    return {"safety": float(ok.mean()), "spurious": float(spur.mean()), "iid": float(ok_iid.mean())}


def run_task(family, n, seed, cond, train_seed):
    tie, noise = CONDITIONS[cond]
    t = build(family, n, seed, tie, noise)
    X = torch.tensor(t["obs"], dtype=torch.float32)
    y = torch.tensor(t["action"])
    N = len(y)
    ones = torch.ones(N)
    first = train(X, y, ones, ID_EPOCHS, train_seed, t["view"])
    with torch.no_grad():
        wrong = (first(X).argmax(1) != y).numpy()
    rs = np.random.default_rng(train_seed + 5)
    rand = np.zeros(N, bool)
    rand[rs.choice(N, int(wrong.sum()), replace=False)] = True
    up = lambda mask: torch.where(torch.tensor(mask), torch.tensor(UP), torch.tensor(1.0))  # noqa: E731
    arms = {
        "uniform": (ones, EPOCHS), "uniform_long": (ones, EPOCHS + ID_EPOCHS),
        "jtt": (up(wrong), EPOCHS), "random": (up(rand), EPOCHS), "oracle": (up(t["dec"]), EPOCHS),
    }
    out = {a: score(train(X, y, w, ep, train_seed + 1, t["view"]), t) for a, (w, ep) in arms.items()}
    share = lambda m, sel: float(m[sel].mean()) if sel.any() else float("nan")  # noqa: E731
    allm = np.ones(N, bool)
    out["error_set"] = {
        "n": N, "frac": float(wrong.mean()),
        **{f"{k}_in_set": share(t[k], wrong) for k in ("dec", "conflict", "tied", "noisy")},
        **{f"{k}_base": share(t[k], allm) for k in ("dec", "conflict", "tied", "noisy")},
    }
    return out


def main():
    phase, n_seeds, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    torch.set_num_threads(1)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for seed in range(n_seeds):
        for cond in CONDITIONS:
            key = f"{seed}/{cond}"
            if key in res:
                continue
            t0 = time.time()
            res[key] = {fam: run_task(fam, n, BASE[phase] + 1000 * seed + i, cond, seed * 100 + i)
                        for i, (fam, n) in enumerate(FAMILIES)}
            out_path.write_text(json.dumps(res, indent=1))
            print(key, f"{time.time() - t0:.0f}s", {a: round(np.mean([res[key][f][a]["safety"] for f, _ in FAMILIES]), 3)
                                                     for a in ("uniform", "jtt", "random", "oracle")}, flush=True)


if __name__ == "__main__":
    main()
