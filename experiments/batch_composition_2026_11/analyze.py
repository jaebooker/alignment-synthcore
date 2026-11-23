"""Apply the registered rules (prereg/2026-11-23_batch_composition.md) to the confirmation records.

Unit: seed. Each seed's value is the mean over the four planted_interp full configurations.
Every claim is a paired difference over seeds, tested with harness.stats.paired_test (two-sided
95% t interval, exact sign-flip p). Holm adjustment over the directional claims C2 to C5.

Usage: python analyze.py results/confirm.json [results/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.stats import paired_test  # noqa: E402

B = [1, 8, 64, 512]


def per_seed(res, comp, b, key):
    return np.array([np.nanmean([cfg[f"{comp}/b{b}"][key] for cfg in seed.values()]) for _, seed in sorted(res.items(), key=lambda kv: int(kv[0]))])


def diff(res, a, b):
    """a and b are (comp, partition, key); returns the paired test of a minus b."""
    t = paired_test(per_seed(res, *a), per_seed(res, *b))
    return {"mean": t["mean_diff"], "ci95": t["ci95"], "p": t["p_value"], "n": len(per_seed(res, *a))}


def interaction(res, key="child"):
    d_r = per_seed(res, "random", 512, key) - per_seed(res, "random", 1, key)
    d_c = per_seed(res, "concentrated", 512, key) - per_seed(res, "concentrated", 1, key)
    t = paired_test(d_r, d_c)
    return {"mean": t["mean_diff"], "ci95": t["ci95"], "p": t["p_value"], "n": len(d_r)}


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    out = {"n_seeds": len(res), "arms": {}}
    for comp in ("random", "concentrated"):
        for b in B:
            out["arms"][f"{comp}/b{b}"] = {k: float(per_seed(res, comp, b, k).mean()) for k in ("mmcs", "child", "parent", "fve", "l0", "l0_sd")}

    c1 = {"child": diff(res, ("random", 512, "child"), ("random", 1, "child")),
          "parent": diff(res, ("random", 512, "parent"), ("random", 1, "parent")),
          "fve": diff(res, ("random", 512, "fve"), ("random", 1, "fve"))}
    c1["met"] = bool(c1["child"]["ci95"][0] >= 0.02 and c1["parent"]["ci95"][0] >= -0.01 and c1["fve"]["ci95"][0] >= -0.01)
    claims = {
        "C1_matrix_rule": c1,
        "C2_harm_to_children": diff(res, ("random", 512, "child"), ("random", 1, "child")),
        "C3_composition_interaction": interaction(res),
        "C4_dose_b8": diff(res, ("random", 8, "child"), ("random", 1, "child")),
        "C5_reconstruction_gain": diff(res, ("random", 512, "fve"), ("random", 1, "fve")),
        "S1_b512_minus_b8_child": diff(res, ("random", 512, "child"), ("random", 8, "child")),
        "S2_concentrated_child": diff(res, ("concentrated", 512, "child"), ("concentrated", 1, "child")),
        "S3_random_parent": c1["parent"],
    }
    # Registered predictions: C2 upper < 0; C3 upper < 0; C4 upper < -0.05; C5 lower > 0.
    pred = {"C2_harm_to_children": claims["C2_harm_to_children"]["ci95"][1] < 0,
            "C3_composition_interaction": claims["C3_composition_interaction"]["ci95"][1] < 0,
            "C4_dose_b8": claims["C4_dose_b8"]["ci95"][1] < -0.05,
            "C5_reconstruction_gain": claims["C5_reconstruction_gain"]["ci95"][0] > 0}
    ps = [claims[k]["p"] for k in pred]
    for k, a in zip(pred, holm(ps)):
        claims[k]["prediction_held"] = bool(pred[k])
        claims[k]["p_holm"] = a
    claims["C1_matrix_rule"]["prediction_held"] = not c1["met"]
    out["claims"] = claims
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
