"""Testbed 1b: planted features in a real language model's activations (GPU job).

Runs on a rented GPU (Runpod) as one process and prints results to stdout; verdicts are
computed afterwards by the harness (testbeds/planted_lm/import_results.py), so the code that
decides a verdict is the same as for every other testbed.

Setup. Residual-stream activations of Pythia-70M (after block 3, d=512) on Pile text, scaled
to mean norm sqrt(d). On top, known feature directions are planted with the same structure
as the toy testbed: sparse firing, a share of children that only fire with their parent
(the absorption setting), magnitudes comparable to the model's own features. The model's
real features act as structured background that the method must also explain.
A method sees only the summed activations and a dictionary size; it is scored on how well
its dictionary recovers the planted directions.

Splits: dev and test use disjoint halves of the token pool and freshly sampled planted
directions per seed. Stage 1: toy config, 3 seeds, default hparams. Tuning: every method
and the baseline get the same budget (their whole HPARAMS list, at most 4 here) on 2 dev
seeds. Stage 2: full config, 8 paired seeds on test.

v0 limitation: methods run in the same process as the evaluator (no sandbox) and only
receive the public dict below. Methods are hand-written, like the CPU testbeds.

    python job.py bundle_dir          # real run (needs internet for the model and text)
    SMOKE=1 python job.py bundle_dir  # tiny CPU check with random activations
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SMOKE = os.environ.get("SMOKE") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 512
# config -> (n_planted, dict_size, n_tokens, child_frac, p_active)
CONFIGS = {"toy": (128, 512, 50_000, 0.3, 0.01), "full": (512, 2048, 200_000, 0.4, 0.005)}
if SMOKE:
    CONFIGS = {"toy": (16, 64, 4000, 0.3, 0.02), "full": (32, 128, 6000, 0.4, 0.02)}
MAG = (2.0, 5.0)  # planted magnitudes before calibration; real activations have mean norm sqrt(512) ~ 22.6
SCALE = [1.0]  # set once by calibrate(), from the baseline alone, before any method runs
CAL_SCALES = [1.0, 0.6, 1.6, 0.4, 2.5]
CAL_BAND = (0.70, 0.92)  # target baseline toy MMCS: headroom both ways
CHILD_GIVEN_PARENT = 0.5
SPLIT_SEED = {"dev": 1000, "test": 2000}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- activations
def real_activations(n_tokens: int) -> torch.Tensor:
    if SMOKE:
        g = torch.Generator().manual_seed(0)
        A = torch.randn(n_tokens, D_MODEL, generator=g) @ torch.randn(D_MODEL, D_MODEL, generator=g) / D_MODEL ** 0.5
        return A * (D_MODEL ** 0.5 / A.norm(dim=1).mean())
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped").to(DEV).eval()
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out, have = [], 0
    with torch.no_grad():
        for row in ds:
            ids = tok(row["text"], return_tensors="pt", truncation=True, max_length=257).input_ids.to(DEV)
            if ids.shape[1] < 33:
                continue
            hs = model(ids, output_hidden_states=True).hidden_states[4][0, 1:]  # drop position 0
            out.append(hs.float().cpu())
            have += len(hs)
            if have >= n_tokens:
                break
    A = torch.cat(out)[:n_tokens]
    return A * (D_MODEL ** 0.5 / A.norm(dim=1).mean())


# ---------------------------------------------------------------- planting
def plant(A_pool: torch.Tensor, config: str, split: str, seed: int):
    n, m, N, child_frac, p = CONFIGS[config]
    g = torch.Generator().manual_seed(SPLIT_SEED[split] + 97 * seed + (0 if config == "full" else 500))
    half = len(A_pool) // 2
    pool = A_pool[:half] if split == "dev" else A_pool[half:]
    A = pool[torch.randperm(len(pool), generator=g)[:N]]
    F = torch.randn(n, D_MODEL, generator=g)
    F = F / F.norm(dim=1, keepdim=True)
    n_child = int(round(n * child_frac))
    perm = torch.randperm(n, generator=g)
    parent = -torch.ones(n, dtype=torch.long)
    parent[perm[:n_child]] = perm[n_child:][torch.randint(0, n - n_child, (n_child,), generator=g)]
    on = torch.rand(N, n, generator=g) < p
    is_child = parent >= 0
    child_on = torch.rand(N, n, generator=g) < CHILD_GIVEN_PARENT
    on = torch.where(is_child, on[:, parent.clamp_min(0)] & child_on, on)
    X = on.float() * SCALE[0] * (MAG[0] + (MAG[1] - MAG[0]) * torch.rand(N, n, generator=g))
    H = A + X @ F
    return {"public": {"H_train": H.to(DEV), "dict_size": m, "d": D_MODEL},
            "F": F.to(DEV), "is_child": is_child.to(DEV)}


def score(directions, inst) -> dict:
    D = torch.as_tensor(directions, dtype=torch.float32, device=DEV)
    m = inst["public"]["dict_size"]
    if D.ndim != 2 or D.shape[1] != D_MODEL or not (1 <= D.shape[0] <= m) or not torch.isfinite(D).all():
        raise ValueError(f"directions must be (1..{m}) x {D_MODEL}, finite; got {tuple(D.shape)}")
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    best = (inst["F"] @ D.T).max(dim=1).values
    ch = inst["is_child"]
    return {"safety": float(best.mean()), "capability": None,
            "diagnostics": {"child_mmcs": float(best[ch].mean()), "recovery_rate": float((best > 0.9).float().mean())}}


# ---------------------------------------------------------------- pipeline
def load_method(path: Path):
    spec = importlib.util.spec_from_file_location(f"m_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_inst_cache, _arm_cache = {}, {}


def run_arm(name, mod, A, split, config, seed, hp):
    key = (name, split, config, seed, json.dumps(hp, sort_keys=True))
    if key in _arm_cache:
        return _arm_cache[key]
    ik = (split, config, seed)
    if ik not in _inst_cache:
        _inst_cache.clear()  # keep one instance in memory
        _inst_cache[ik] = plant(A, config, split, seed)
    inst = _inst_cache[ik]
    torch.manual_seed(seed)
    t0 = time.time()
    try:
        out = mod.run([inst["public"]], hp, seed)
        r = score(out[0]["directions"], inst)
    except Exception as e:  # recorded; the importer turns it into implementation_failed
        r = {"error": f"{type(e).__name__}: {e}"[:400]}
    r["seconds"] = round(time.time() - t0, 1)
    _arm_cache[key] = r
    return r


def tune(name, mod, A, budget=4, seeds=(0, 1)):
    table = []
    for hp in mod.HPARAMS[:budget]:
        rs = [run_arm(name, mod, A, "dev", "full", s, hp) for s in seeds]
        if any("error" in r for r in rs):
            table.append({"hparams": hp, "dev_safety": float("-inf"), "error": [r.get("error") for r in rs]})
        else:
            table.append({"hparams": hp, "dev_safety": float(np.mean([r["safety"] for r in rs]))})
        log(f"  tune {name} {hp} -> {table[-1]['dev_safety']:.4f}")
    best = max(table, key=lambda r: r["dev_safety"])
    return best["hparams"], table


def calibrate(base, A):
    """Pick the planted magnitude from the baseline alone (default hparams, dev toy seed 0),
    as the CPU testbeds were calibrated against their baselines. Logged, then frozen."""
    tried = []
    for sc in CAL_SCALES:
        SCALE[0] = sc
        _inst_cache.clear()
        r = run_arm(f"cal{sc}", base, A, "dev", "toy", 0, base.HPARAMS[0])
        tried.append((sc, r.get("safety", float("nan"))))
        log(f"  calibration scale {sc}: baseline toy MMCS {tried[-1][1]:.3f}")
        if CAL_BAND[0] <= tried[-1][1] <= CAL_BAND[1]:
            break
    else:
        sc = min(tried, key=lambda t: abs(t[1] - sum(CAL_BAND) / 2))[0]
    SCALE[0] = sc
    _inst_cache.clear()
    return {"scale": sc, "tried": tried, "band": CAL_BAND}


def main(bundle: Path):
    meta = json.loads((bundle / "bundle.json").read_text())
    pool_tokens = 2 * max(c[2] for c in CONFIGS.values())
    log(f"device {DEV}; extracting {pool_tokens} activations")
    A = real_activations(pool_tokens)
    log(f"activations {tuple(A.shape)}, mean norm {A.norm(dim=1).mean():.2f}")
    base = load_method(bundle / meta["baseline"])
    log("calibrating planted magnitude on the baseline")
    cal = calibrate(base, A)
    print("RESULT " + json.dumps({"kind": "calibration", **cal}), flush=True)
    log("tuning baseline")
    base_hp, base_table = tune("baseline", base, A)
    print("RESULT " + json.dumps({"kind": "baseline", "hparams": base_hp, "table": base_table}), flush=True)
    for idea in meta["ideas"]:
        name = idea["id"]
        mod = load_method(bundle / idea["method"])
        log(f"[{name}] stage 1")
        s1 = [{"seed": s, "method": run_arm(name, mod, A, "dev", "toy", s, mod.HPARAMS[0]),
               "baseline": run_arm("baseline", base, A, "dev", "toy", s, base.HPARAMS[0])} for s in range(3)]
        print("RESULT " + json.dumps({"kind": "stage1", "id": name, "per_seed": s1}), flush=True)
        log(f"[{name}] tuning")
        hp, table = tune(name, mod, A)
        log(f"[{name}] stage 2")
        s2 = [{"seed": s, "method": run_arm(name, mod, A, "test", "full", s, hp),
               "baseline": run_arm("baseline", base, A, "test", "full", s, base_hp)} for s in range(8)]
        print("RESULT " + json.dumps({"kind": "full", "id": name, "tuning": {"hparams": hp, "table": table},
                                      "per_seed": s2}), flush=True)
    log("ALL DONE")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
