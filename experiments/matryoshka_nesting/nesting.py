"""Does nesting recover parent-child features? (matryoshka-nesting-v1)

Follow-up to the matryoshka-sae verdict (mechanism_failed: child recovery 0.14 below TopK,
triaged weak_negative with the best k at the grid edge). A Matryoshka SAE adds two things
to a TopK SAE: extra reconstruction losses on parts of the dictionary, and a fixed nested
ordering of those parts. This study separates them on the planted_interp absorption
generators, with decoder width, k, optimiser, steps, batches and initialisation matched.

Arms (one trainer; only the loss groups differ):
  topk      one loss on the full dictionary (the testbed baseline's objective)
  nested    Matryoshka: losses on fixed nested prefixes, e.g. first 25%, 50%, 100%
  shuffled  the same nested sizes, but over a fresh random latent order every step:
            nesting without a stable early set ("randomly assigned nesting")
  subsets   the same sizes as fixed random subsets drawn independently, not nested

Primary contrast (proposal-matryoshka-sae-1): nested vs shuffled.
  Child recovery: 95% CI lower bound of (nested - shuffled) child MMCS >= +0.02.
  Parent recovery non-inferior: 95% CI lower bound of (nested - shuffled) parent MMCS >= -0.01.
Secondary, descriptive: nested vs topk (does the original negative replicate with k widened
to 1), nested vs subsets (is the nested structure itself what matters).

Phases:
  calibrate   dev split, seeds 0-1, every arm x k x lr (x prefix set). Picks frozen hparams.
  freeze      writes protocol.json + freeze.json (source hashes). Write-once.
  confirm     test split, 8 paired seeds, frozen hparams. Refuses without a registration
              receipt whose protocol hash matches the freeze.

Ground truth (which planted features are parents or children) is rebuilt from the
generator's deterministic seeds and checked against the cached model, never shown to arms.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.stats import paired_test  # noqa: E402
from testbeds.planted_interp import generate  # noqa: E402
from testbeds.planted_interp.testbed import PlantedInterp  # noqa: E402

STUDY = "matryoshka-nesting-v1"
OUT = ROOT / "results" / STUDY
ARMS = ("topk", "nested", "shuffled", "subsets")
GRID = {
    "k": [1, 2, 3],                      # k=1 added: triage flagged k=2 at the grid edge
    "lr": [3e-3, 1e-2],                  # 1e-3 trailed in the original tuning table
    "prefixes": [[0.25, 0.5, 1.0], [0.5, 1.0]],
}
STEPS, BATCH = 3000, 512
CAL_SEEDS = (0, 1)
CONFIRM_SEEDS = tuple(range(8))
CHILD_MARGIN, PARENT_MARGIN = 0.02, -0.01
SOURCES = ["experiments/matryoshka_nesting/nesting.py", "testbeds/planted_interp/generate.py",
           "testbeds/planted_interp/testbed.py", "harness/stats.py"]


# ---------------------------------------------------------------- training
def groups_for(arm, m, prefixes, perm_gen, fixed):
    """Index sets whose reconstructions are each penalised this step."""
    sizes = sorted({max(1, int(round(f * m))) for f in prefixes})
    if arm == "topk":
        return [torch.arange(m)]
    if arm == "nested":
        return [torch.arange(p) for p in sizes]
    if arm == "shuffled":
        order = torch.randperm(m, generator=perm_gen)
        return [order[:p] for p in sizes]
    if arm == "subsets":
        return fixed
    raise ValueError(arm)


def train(H, m, arm, k, lr, prefixes, seed, steps=STEPS, batch=BATCH):
    """Every arm shares init, batches and optimiser; only the loss groups differ.
    Group randomness uses its own generator so the batch stream is identical across arms."""
    g = torch.Generator().manual_seed(seed)
    perm_gen = torch.Generator().manual_seed(seed + 7919)
    H = torch.tensor(H, dtype=torch.float32)
    d = H.shape[1]
    W_dec = torch.randn(m, d, generator=g)
    W_dec = torch.nn.Parameter(W_dec / W_dec.norm(dim=1, keepdim=True))
    W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(m))
    b_dec = torch.nn.Parameter(H.mean(0).clone())
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=lr)
    k = min(k, m)
    sizes = sorted({max(1, int(round(f * m))) for f in prefixes})
    fixed = [torch.randperm(m, generator=perm_gen)[:p] for p in sizes[:-1]] + [torch.arange(m)]
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (batch,), generator=g)]
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, k, dim=1)
        z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        gs = groups_for(arm, m, prefixes, perm_gen, fixed)
        loss = sum(((z[:, G] @ W_dec[G] + b_dec - x) ** 2).sum(1).mean() for G in gs) / len(gs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return W_dec.detach().numpy()


# ---------------------------------------------------------------- ground truth and scoring
_TRUTH = {}


def truth(cfg, seed, model):
    """Which represented features are parents. The hierarchy is rebuilt exactly from the
    generator's integer seeds. To learn which original feature each cached planted direction
    is, the toy model is retrained and each cached direction matched to its nearest column
    (float drift across thread counts moves weights slightly, so exact equality is not used).
    Child labels from the match must equal the cached ones."""
    key = (tuple(cfg), seed)
    if key not in _TRUTH:
        n, d, p, decay, child_frac, noise = cfg
        gen = torch.Generator().manual_seed(seed + 3)
        parent = generate.make_hierarchy(n, child_frac, gen)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)  # build_model trains with two threads
        try:
            W, _ = generate.train_toy(n, d, p, decay, parent, seed)
        finally:
            torch.set_num_threads(threads)
        cols = (W / W.norm(dim=0).clamp_min(1e-12)).T.numpy()  # (n, d)
        sims = model["true_dirs"] @ cols.T                      # (n_rep, n)
        idx = sims.argmax(axis=1)
        if len(set(idx.tolist())) != len(idx) or sims.max(axis=1).min() < 0.99:
            raise RuntimeError(f"could not match planted directions for {cfg} seed {seed}")
        par = parent.numpy()
        is_parent = np.zeros(n, dtype=bool)
        is_parent[par[par >= 0]] = True
        _TRUTH[key] = {"is_parent": is_parent[idx], "is_child": (par >= 0)[idx]}
        if not (_TRUTH[key]["is_child"] == model["true_is_child"]).all():
            raise RuntimeError(f"rebuilt hierarchy disagrees with the cached child labels for {cfg} seed {seed}")
    return _TRUTH[key]


TB = PlantedInterp()


def score(directions, model):
    s = TB.score_one({"directions": directions}, model)
    t = truth(model["cfg"], model["seed"], model)
    Du = directions / np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
    best = (model["true_dirs"] @ Du.T).max(axis=1)
    return {"mmcs": s["mmcs"], "child_mmcs": s["child_mmcs"], "fve": s["fve"],
            "parent_mmcs": float(best[t["is_parent"]].mean()) if t["is_parent"].any() else s["mmcs"],
            "root_mmcs": float(best[~t["is_parent"] & ~t["is_child"]].mean())
            if (~t["is_parent"] & ~t["is_child"]).any() else s["mmcs"]}


def run_cell(split, seed, arm, hp):
    """One arm at one hparam setting on the four full-config models of one instance seed."""
    models = [generate.build_model(cfg, generate.SPLIT_OFFSET[split] + 97 * seed + i)
              for i, cfg in enumerate(generate.CONFIGS["full"])]
    per = [score(train(m["H_train"], m["cfg"][0], arm, hp["k"], hp["lr"], hp.get("prefixes", [1.0]),
                       seed * 1000 + i), m) for i, m in enumerate(models)]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


# ---------------------------------------------------------------- bookkeeping
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=1, sort_keys=True) + "\n")


def read(p):
    return json.loads(Path(p).read_text())


def signature():
    return {s: sha(ROOT / s) for s in SOURCES}


def grid_for(arm):
    cells = []
    for k in GRID["k"]:
        for lr in GRID["lr"]:
            for pre in ([None] if arm == "topk" else GRID["prefixes"]):
                hp = {"k": k, "lr": lr}
                if pre:
                    hp["prefixes"] = pre
                cells.append(hp)
    return cells


def cell_key(arm, hp, seed):
    return f"{arm}|{json.dumps(hp, sort_keys=True)}|{seed}"


# ---------------------------------------------------------------- phases
def calibrate(budget=None):
    """Dev split only. Resumable: finished cells are kept in calibration.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "calibration.json"
    done = read(path) if path.exists() else {"cells": {}}
    t0 = time.time()
    todo = [(a, hp, s) for a in ARMS for hp in grid_for(a) for s in CAL_SEEDS
            if cell_key(a, hp, s) not in done["cells"]]
    for a, hp, s in todo:
        if budget and time.time() - t0 > budget:
            break
        done["cells"][cell_key(a, hp, s)] = run_cell("dev", s, a, hp)
        dump(path, done)
        print(f"{len(done['cells'])}/{len(done['cells']) + len(todo) - todo.index((a, hp, s)) - 1} {a} {hp} seed {s}", flush=True)
    remaining = sum(cell_key(a, hp, s) not in done["cells"] for a in ARMS for hp in grid_for(a) for s in CAL_SEEDS)
    print(f"calibration cells remaining: {remaining}")
    return remaining


def select():
    """Frozen selection rule. For each arm and k, choose lr (and prefix set) by mean dev MMCS
    over the calibration seeds; the primary k maximises the mean of the four arms' tuned dev
    MMCS, so no single arm picks the operating point."""
    cells = read(OUT / "calibration.json")["cells"]
    mean = lambda a, hp, key: float(np.mean([cells[cell_key(a, hp, s)][key] for s in CAL_SEEDS]))  # noqa: E731
    tuned = {a: {} for a in ARMS}
    for a in ARMS:
        for k in GRID["k"]:
            options = [hp for hp in grid_for(a) if hp["k"] == k]
            best = max(options, key=lambda hp: mean(a, hp, "mmcs"))
            tuned[a][k] = {"hparams": best, **{m: mean(a, best, m) for m in ("mmcs", "child_mmcs", "parent_mmcs", "fve")}}
    primary_k = max(GRID["k"], key=lambda k: np.mean([tuned[a][k]["mmcs"] for a in ARMS]))
    return tuned, primary_k


def freeze():
    if (OUT / "freeze.json").exists():
        raise SystemExit("Already frozen; a changed design needs a new study id.")
    tuned, k = select()
    protocol = {
        "study": STUDY, "proposal": "proposal-matryoshka-sae-1",
        "question": "Does the fixed nested ordering of Matryoshka losses improve recovery of absorption-prone child features over the same nested losses on a random order?",
        "arms": list(ARMS), "testbed": "planted_interp", "config": "full", "split": "test",
        "seeds": list(CONFIRM_SEEDS), "steps": STEPS, "batch": BATCH,
        "primary_k": k, "frozen_hparams": {a: tuned[a][k]["hparams"] for a in ARMS},
        "primary": {"contrast": "nested - shuffled",
                    "child_rule": f"95% CI lower bound of child_mmcs difference >= {CHILD_MARGIN:+}",
                    "parent_rule": f"95% CI lower bound of parent_mmcs difference >= {PARENT_MARGIN:+}",
                    "pass": "both rules hold"},
        "secondary": ["nested - topk (child, parent, mmcs, fve)", "nested - subsets (child, parent)",
                      "every k in the grid at its tuned hparams, reported descriptively"],
        "analysis": "Per seed, average each metric over the four full-config models; paired differences across 8 seeds; harness.stats.paired_test (exact sign-flip permutation, 95% CI).",
        "calibration_summary": {a: {str(kk): v for kk, v in tuned[a].items()} for a in ARMS},
    }
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "freeze.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "protocol.json"),
                               "source_hashes": signature(), "calibration_sha256": sha(OUT / "calibration.json"),
                               "confirmation_accessed": False})
    print(json.dumps({"primary_k": k, "frozen": protocol["frozen_hparams"]}, indent=1))


def verify(registered=False):
    f = read(OUT / "freeze.json")
    assert sha(OUT / "protocol.json") == f["protocol_sha256"], "protocol changed after freeze"
    assert signature() == f["source_hashes"], "source changed after freeze"
    if registered:
        r = read(OUT / "registration-receipt.json")
        assert r["protocol_sha256"] == f["protocol_sha256"] and r["revision"] >= 1, "receipt does not match the freeze"
        assert r["source_freeze_sha256"] == sha(OUT / "freeze.json"), "receipt does not match the freeze file"
    return f


def confirm():
    verify(registered=True)
    p = read(OUT / "protocol.json")
    path = OUT / "records.json"
    rec = read(path) if path.exists() else {}
    for s in p["seeds"]:
        for a in p["arms"]:
            key = f"{a}|{s}"
            if key not in rec:
                rec[key] = run_cell("test", s, a, p["frozen_hparams"][a])
                dump(path, rec)
                print(key, rec[key], flush=True)
    vals = lambda a, m: [rec[f"{a}|{s}"][m] for s in p["seeds"]]  # noqa: E731
    out = {}
    for a, b in (("nested", "shuffled"), ("nested", "topk"), ("nested", "subsets")):
        for m in ("child_mmcs", "parent_mmcs", "mmcs", "fve"):
            t = paired_test(vals(a, m), vals(b, m))
            out[f"{a}-{b}:{m}"] = {"mean": t["mean_diff"], "ci95": t["ci95"], "p": t["p_value"]}
    child, parent = out["nested-shuffled:child_mmcs"], out["nested-shuffled:parent_mmcs"]
    out["primary_pass"] = bool(child["ci95"][0] >= CHILD_MARGIN and parent["ci95"][0] >= PARENT_MARGIN)
    dump(OUT / "summary.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["calibrate", "select", "freeze", "confirm"])
    ap.add_argument("--budget", type=float, default=None, help="calibrate: stop after this many seconds (resumable)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.phase == "calibrate":
        calibrate(a.budget)
    elif a.phase == "select":
        tuned, k = select()
        print(json.dumps({"primary_k": k, "tuned": {x: {str(kk): v for kk, v in tuned[x].items()} for x in ARMS}}, indent=1))
    elif a.phase == "freeze":
        freeze()
    else:
        confirm()
