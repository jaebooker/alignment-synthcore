"""Apply Amendment 1's four claims (A1-A4) to the sub-claim run. Written after the run; adds no rule.
Models above the frozen 10% failed-call cutoff on this set are dropped from the claims and shown apart."""
import json, collections
from pathlib import Path
import numpy as np
import analyze as A
HERE = Path(__file__).resolve().parent
raw = [json.loads(l) for l in open(HERE / "data" / "sub" / "sub_forecasts.jsonl")]
items = {json.loads(l)["item_id"]: json.loads(l) for l in open(HERE / "subclaims.jsonl")}
ref = {i: np.mean([x["outcome"] for x in items.values() if x["cluster"] != it["cluster"]]) for i, it in items.items()}
fail = collections.Counter(); tot = collections.Counter()
for r in raw:
    tot[r["model"]] += 1; fail[r["model"]] += r["p"] is None
keep = [m for m in sorted(tot) if fail[m] / tot[m] <= 0.10]
print("failed-call rate:", {m: round(fail[m] / tot[m], 3) for m in sorted(tot)}); print("kept:", keep)
agg = collections.defaultdict(list)
for r in raw:
    if r["p"] is not None: agg[(r["model"], r["cond"], r["item_id"])].append(r["p"])
P = collections.defaultdict(lambda: collections.defaultdict(dict))
for (m, c, i), v in agg.items(): P[m][c][i] = float(np.mean(v))
for c in ("blind", "base_rate"):
    P["POOLED(kept)"][c] = {i: float(np.mean([P[m][c][i] for m in keep])) for i in items if all(i in P[m][c] for m in keep)}
win = {h["model"]: h["mean_p"] for h in json.load(open(HERE / "data" / "hypotheses_win.json"))}
win["POOLED(kept)"] = float(np.mean([win[m] for m in keep]))
def rows(m, c, ids=None):
    return [{"p": p, "outcome": items[i]["outcome"], "ref": float(ref[i]), "cluster": items[i]["cluster"], "id": i}
            for i, p in P[m][c].items() if ids is None or i in ids]
out = []
f = lambda v, c: f"{v:+.2f} [{c[0]:+.2f},{c[1]:+.2f}]"
print(f"{'model':20}{'n':>4} meanp(sub) meanp(win)  A1 diff | A2 auroc            | A3 skill            | A4 dBrier(br-blind)")
for m in list(sorted(tot)) + ["POOLED(kept)"]:
    R = rows(m, "blind"); s = A.stats(R)
    ci = {k: A.boot(R, lambda x, k=k: A.stats(x)[k]) for k in ("auroc", "skill")}
    ids = set(P[m]["blind"]) & set(P[m]["base_rate"])
    a = {r["id"]: r for r in rows(m, "base_rate", ids)}; b = {r["id"]: r for r in rows(m, "blind", ids)}
    pr = [{"cluster": a[i]["cluster"], "d": (a[i]["p"] - a[i]["outcome"]) ** 2 - (b[i]["p"] - b[i]["outcome"]) ** 2} for i in ids]
    d4 = float(np.mean([x["d"] for x in pr])); c4 = A.boot(pr, lambda x: float(np.mean([y["d"] for y in x])))
    d1 = s["mean_p"] - win[m]
    tag = "" if m in keep or m.startswith("POOLED") else "  (dropped: descriptive only)"
    print(f"{m:20}{s['n']:4d}   {s['mean_p']:.2f}       {win[m]:.2f}      {d1:+.2f}  | {f(s['auroc'], ci['auroc'])} | {f(s['skill'], ci['skill'])} | {f(d4, c4)}{tag}")
    out.append({"model": m, "kept": m in keep or m.startswith("POOLED"), "n": s["n"], "base": s["base"], "mean_p": s["mean_p"], "mean_p_win": win[m],
                "A1_diff": d1, "A2_auroc": [s["auroc"], ci["auroc"]], "A3_skill": [s["skill"], ci["skill"]], "A4_dbrier": [d4, c4], "brier": s["brier"],
                "verdict": {"A1": abs(d1) < 0.18, "A2": ci["auroc"][0] > 0.5, "A3": ci["skill"][1] <= 0, "A4": c4[1] < 0}})
(HERE / "data" / "sub" / "amendment_claims.json").write_text(json.dumps(out, indent=1))
for o in out:
    if o["kept"]: print(o["model"], {k: "yes" if v else "no" for k, v in o["verdict"].items()})
# per-family calibration of the pooled forecast
by = collections.defaultdict(list)
for r in rows("POOLED(kept)", "blind"): by[r["cluster"]].append(r)
print("family: n, met rate, pooled mean forecast")
for k, v in sorted(by.items()): print(f"  {k:24}{len(v):3d}  {np.mean([x['outcome'] for x in v]):.2f}  {np.mean([x['p'] for x in v]):.2f}")
