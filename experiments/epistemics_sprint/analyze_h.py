"""Apply the five frozen hypotheses (prereg/2026-11-14_forecast_audit.md) to the main run.

Written AFTER the freeze and after the run finished; it adds no rule. It uses analyze.py's
statistics unchanged and the same family-level bootstrap (B = 2000). 'pooled' is the mean of
the six models' item forecasts (an equal-weight ensemble), which the prereg did not spell out.
Usage: python analyze_h.py [question=win] [--drop-pilot]
"""
import json, sys, collections
from pathlib import Path
import numpy as np
import analyze as A
HERE = Path(__file__).resolve().parent

def load(q, drop_pilot):
    raw = [json.loads(l) for l in open(HERE / "data" / "forecasts.jsonl")]
    items = [json.loads(l) for l in open(HERE / "items.jsonl")]
    pilot = {it["item_id"] for it in items[::max(1, len(items) // 10)][:10]}
    ref = {}
    for it in items:
        o = [x["outcome"] for x in items if x["question"] == it["question"] and x["cluster"] != it["cluster"]]
        ref[it["item_id"]] = sum(o) / len(o)
    meta = {it["item_id"]: it for it in items}
    agg = collections.defaultdict(list)
    for r in raw:
        if r["p"] is not None and r["question"] == q and not (drop_pilot and r["item_id"] in pilot):
            agg[(r["model"], r["cond"], r["item_id"])].append(r["p"])
    P = collections.defaultdict(dict)                       # P[model][cond][item] = p
    for (m, c, i), v in agg.items():
        P[m].setdefault(c, {})[i] = float(np.mean(v))
    models = sorted(P)
    P["POOLED"] = {}
    for c in ("blind", "base_rate", "no_source"):
        ids = set().union(*[set(P[m].get(c, {})) for m in models])
        P["POOLED"][c] = {i: float(np.mean([P[m][c][i] for m in models if i in P[m][c]])) for i in ids}
    return P, meta, ref, models

def rows(P, meta, ref, m, c, ids=None):
    return [{"p": p, "outcome": meta[i]["outcome"], "ref": ref[i], "cluster": meta[i]["cluster"], "id": i}
            for i, p in P[m][c].items() if ids is None or i in ids]

def paired(P, meta, ref, m, c1, c2, fn):
    ids = set(P[m][c1]) & set(P[m][c2])
    a = {r["id"]: r for r in rows(P, meta, ref, m, c1, ids)}; b = {r["id"]: r for r in rows(P, meta, ref, m, c2, ids)}
    pr = [{"cluster": a[i]["cluster"], "d": fn(a[i]) - fn(b[i])} for i in ids]
    est = float(np.mean([x["d"] for x in pr]))
    return est, A.boot(pr, lambda s: float(np.mean([x["d"] for x in s])))

def main():
    q = next((a for a in sys.argv[1:] if not a.startswith("--")), "win"); dp = "--drop-pilot" in sys.argv
    P, meta, ref, models = load(q, dp)
    out = []
    print(f"question={q} drop_pilot={dp}")
    print(f"{'model':20}{'n':>4} | H1 bias          | H2 auroc         | H3 skill         | H4 dBrier(br-blind) | H5 dp(blind-nosrc)")
    for m in models + ["POOLED"]:
        R = rows(P, meta, ref, m, "blind"); s = A.stats(R)
        ci = {k: A.boot(R, lambda x, k=k: A.stats(x)[k]) for k in ("bias", "auroc", "skill")}
        h4, c4 = paired(P, meta, ref, m, "base_rate", "blind", lambda r: (r["p"] - r["outcome"]) ** 2)
        h5, c5 = paired(P, meta, ref, m, "blind", "no_source", lambda r: r["p"])
        f = lambda v, c: f"{v:+.2f} [{c[0]:+.2f},{c[1]:+.2f}]"
        print(f"{m:20}{s['n']:4d} | {f(s['bias'], ci['bias'])} | {f(s['auroc'], ci['auroc'])} | {f(s['skill'], ci['skill'])} | {f(h4, c4)}  | {f(h5, c5)}")
        out.append({"model": m, "n": s["n"], "base": s["base"], "mean_p": s["mean_p"], "brier": s["brier"],
                    "H1_bias": [s["bias"], ci["bias"]], "H2_auroc": [s["auroc"], ci["auroc"]], "H3_skill": [s["skill"], ci["skill"]],
                    "H4_dbrier": [h4, c4], "H5_dp": [h5, c5],
                    "verdict": {"H1": ci["bias"][0] > 0, "H2": ci["auroc"][0] > 0.5, "H3": ci["skill"][1] <= 0,
                                "H4": c4[1] < 0, "H5": c5[0] > 0}})
        print(" " * 24, {k: ("yes" if v else "no") for k, v in out[-1]["verdict"].items()})
    name = f"hypotheses_{q}{'_nopilot' if dp else ''}.json"
    (HERE / "data" / name).write_text(json.dumps(out, indent=1)); print("wrote data/" + name)
    ctrl = [(m, meta[i]["idea_id"], round(p, 2)) for m in models for i, p in P[m]["blind"].items() if meta[i]["is_control"]]
    print("null controls, blind:", ctrl)

if __name__ == "__main__":
    main()
