"""Apply the pre-registered rules (prereg/2026-11-10_draft_proposals_cpu.md) to the confirmation files.

Usage: python analyze.py results_dir  -> prints a table and writes results_dir/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.stats import paired_test  # noqa: E402

R = Path(sys.argv[1])
N = 12
out, claims = {}, []


def pt(a, b, label, rule=None, thresh=None):
    """Paired a - b. rule 'lb>=': claim holds when the CI lower bound >= thresh."""
    t = paired_test(a, b)
    rec = {"label": label, "a": t["method_mean"], "b": t["baseline_mean"], "diff": t["mean_diff"],
           "ci95": t["ci95"], "p": t["p_value"], "n": t["n"]}
    if rule:
        rec["threshold"] = thresh
        rec["holds"] = bool(t["ci95"][0] >= thresh) if rule == "lb>=" else bool(t["ci95"][0] > thresh)
    claims.append(rec)
    return rec


def show(r):
    h = "" if "holds" not in r else ("  HOLDS" if r["holds"] else "  FAILS") + f" (needs lower bound {'>=' if r['threshold'] else '>'} {r['threshold']})"
    print(f"  {r['label']}: {r['a']:.3f} vs {r['b']:.3f}, diff {r['diff']:+.3f} "
          f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], p={r['p']:.4f}{h}")


# ---------------------------------------------------------------- A: JTT
d = json.loads((R / "jtt_confirm.json").read_text())
fams = ["corner", "lamp"]
arm = lambda c, a, m="safety": np.array([np.mean([d[f"{s}/{c}"][f][a][m] for f in fams]) for s in range(N)])  # noqa: E731
es = lambda c, k: np.array([np.mean([d[f"{s}/{c}"][f]["error_set"][k] for f in fams]) for s in range(N)])  # noqa: E731
print("A. Just Train Twice")
tab = {c: {a: {m: float(arm(c, a, m).mean()) for m in ("safety", "iid", "spurious")}
           for a in ("uniform", "uniform_long", "jtt", "random", "oracle")} for c in ("R", "D", "DN")}
for c in tab:
    print("  ", c, {a: round(v["safety"], 3) for a, v in tab[c].items()}, "| iid", {a: round(v["iid"], 3) for a, v in tab[c].items()})
comp = {c: {k: float(np.nanmean(es(c, k))) for k in ("frac", "dec_in_set", "dec_base", "conflict_in_set", "conflict_base",
                                                     "tied_in_set", "tied_base", "noisy_in_set", "noisy_base")} for c in tab}
for c in comp:
    print("   error set", c, {k: round(v, 3) for k, v in comp[c].items()})
A = {
    "A1_safety": pt(arm("R", "jtt"), arm("R", "uniform"), "A1 jtt - uniform safety (R)", "lb>=", 0.05),
    "A1_iid": pt(arm("R", "jtt", "iid"), arm("R", "uniform", "iid"), "A1 jtt - uniform IID (R)", "lb>=", -0.05),
    "A2_enrich": pt(es("R", "dec_in_set"), es("R", "dec_base"), "A2 decorrelated share: error set - base (R)", "lb>=", 0.10),
    "A2_safety": pt(arm("R", "jtt"), arm("R", "random"), "A2 jtt - random safety (R)", "lb>=", 0.05),
    "A3_tied": pt(es("R", "tied_in_set"), es("R", "tied_base"), "A3 tied share: error set - base (R)", "lb>", 0.0),
    "A4_did": pt(arm("D", "jtt") - arm("D", "uniform"), arm("R", "jtt") - arm("R", "uniform"),
                 "A4 (jtt - uniform in D) - (jtt - uniform in R)", "lb>", 0.0),
    "A5_oracle": pt(arm("R", "oracle"), arm("R", "uniform"), "A5 oracle - uniform safety (R)", "lb>", 0.0),
}
extra = [pt(arm(c, a), arm(c, "uniform"), f"   {a} - uniform safety ({c})") for c in ("R", "D", "DN")
         for a in ("uniform_long", "jtt", "random", "oracle")]
extra += [pt(es(c, "conflict_in_set"), es(c, "conflict_base"), f"   proxy-conflict share: set - base ({c})") for c in tab]
extra += [pt(es(c, "dec_in_set"), es(c, "dec_base"), f"   decorrelated share: set - base ({c})") for c in ("D", "DN")]
extra += [pt(es("DN", "noisy_in_set"), es("DN", "noisy_base"), "   noisy-label share: set - base (DN)")]
for r in list(A.values()) + extra:
    show(r)
out["A"] = {"arms": tab, "error_set": comp, "claims": A, "extra": extra}

# ---------------------------------------------------------------- B: OMP
d = json.loads((R / "omp_confirm.json").read_text())
strata, ps = ["d32", "d16", "d10"], ["0.03", "0.06"]
g = lambda st, k: np.array([np.mean([d[str(s)][f"{st}/{p}"][k] for p in ps]) for s in range(N)])  # noqa: E731
print("B. OMP on a fixed dictionary")
tabB = {st: {k: float(g(st, k).mean()) for k in ("coherence", "f1_omp", "f1_nnomp", "f1_encoder", "f1_projection",
                                                 "fve_omp", "fve_nnomp", "fve_encoder", "active_encoder")} for st in strata}
for st in strata:
    print("  ", st, {k: round(v, 3) for k, v in tabB[st].items()})
B = {"B1": pt(g("d10", "f1_omp"), g("d10", "f1_encoder"), "B1 F1 OMP - encoder (d=10)", "lb>=", 0.05)}
for st in strata:
    B[f"B2_{st}"] = pt(g(st, "f1_nnomp"), g(st, "f1_omp"), f"B2 F1 nonneg OMP - OMP ({st})", "lb>", 0.0)
B["B3"] = pt(g("d10", "f1_nnomp"), g("d10", "f1_encoder"), "B3 F1 nonneg OMP - encoder (d=10)", "lb>", 0.0)
extraB = [pt(g(st, "f1_omp"), g(st, "f1_encoder"), f"   F1 OMP - encoder ({st})") for st in ("d32", "d16")]
extraB += [pt(g(st, "f1_nnomp"), g(st, "f1_encoder"), f"   F1 nonneg OMP - encoder ({st})") for st in ("d32", "d16")]
extraB += [pt(g(st, "f1_encoder"), g(st, "f1_projection"), f"   F1 encoder - untrained projection ({st})") for st in strata]
for r in list(B.values()) + extraB:
    show(r)
out["B"] = {"strata": tabB, "claims": B, "extra": extraB}

# ---------------------------------------------------------------- C: k-means
d = json.loads((R / "kmeans_confirm.json").read_text())
k4 = lambda cond, a: np.array([np.mean([d[str(s)][f"{i}/{cond}"][a] for i in range(4)]) for s in range(N)])  # noqa: E731
arms = ["spherical", "euclid", "topk"]
print("C. Spherical k-means under amplitude shift")
tabC = {c: {a: float(k4(c, a).mean()) for a in arms} for c in ("plain", "shift")}
tabC["shift_small"] = {a: float(k4("shift", a + "_small").mean()) for a in arms}
for c in tabC:
    print("  ", c, {a: round(v, 3) for a, v in tabC[c].items()})
loss = lambda a: k4("plain", a) - k4("shift", a)  # noqa: E731
C = {
    "C1_loss": pt(loss("euclid"), loss("spherical"), "C1 shift loss Euclidean - spherical", "lb>=", 0.02),
    "C1_plain": pt(k4("plain", "spherical"), k4("plain", "euclid"), "C1 plain spherical - Euclidean", "lb>=", -0.01),
    "C2": pt(k4("shift", "spherical"), k4("shift", "topk"), "C2 shifted spherical - TopK", "lb>", 0.0),
}
extraC = [pt(k4("plain", "spherical"), k4("plain", "topk"), "   plain spherical - TopK"),
          pt(loss("topk"), loss("spherical"), "   shift loss TopK - spherical"),
          pt(k4("shift", "spherical_small"), k4("shift", "topk_small"), "   x0.25 features: spherical - TopK")]
for r in list(C.values()) + extraC:
    show(r)
out["C"] = {"mmcs": tabC, "claims": C, "extra": extraC}

# ---------------------------------------------------------------- D: nesting
d = json.loads((R / "nesting_confirm.json").read_text())
n4 = lambda k: np.array([np.nanmean([d[str(s)][str(i)][k] for i in range(4)]) for s in range(N)])  # noqa: E731
print("D. Matryoshka nesting")
tabD = {a: {m: float(n4(f"{a}_{m}").mean()) for m in ("all", "child", "parent")} for a in ("nested", "random", "topk")}
for a in tabD:
    print("  ", a, {m: round(v, 3) for m, v in tabD[a].items()})
D = {
    "D1_child": pt(n4("nested_child"), n4("random_child"), "D1 child MMCS nested - random nesting", "lb>=", 0.02),
    "D1_parent": pt(n4("nested_parent"), n4("random_parent"), "D1 parent MMCS nested - random nesting", "lb>=", -0.01),
    "D2": pt(n4("random_all"), n4("topk_all"), "D2 MMCS random nesting - TopK", "lb>", 0.0),
    "D3": pt(n4("parent_in_first_quarter"), n4("child_in_first_quarter"), "D3 first-quarter match: parents - children", "lb>", 0.0),
}
extraD = [pt(n4("nested_child"), n4("topk_child"), "   child MMCS nested - TopK"),
          pt(n4("random_child"), n4("topk_child"), "   child MMCS random nesting - TopK"),
          pt(n4("nested_all"), n4("topk_all"), "   MMCS nested - TopK")]
for r in list(D.values()) + extraD:
    show(r)
out["D"] = {"mmcs": tabD, "claims": D, "extra": extraD}

# Holm over the registered directional claims
reg = [(k, v) for S in "ABCD" for k, v in out[S]["claims"].items()]
order = sorted(range(len(reg)), key=lambda i: reg[i][1]["p"])
adj, run = {}, 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, reg[i][1]["p"] * (len(reg) - rank)))
    adj[reg[i][0]] = run
for k, v in reg:
    v["p_holm"] = adj[k]
print("Holm-adjusted p:", {k: round(p, 4) for k, p in adj.items()})
(R / "summary.json").write_text(json.dumps(out, indent=1))
