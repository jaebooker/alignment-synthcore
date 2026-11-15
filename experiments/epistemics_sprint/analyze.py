"""Score forecasts: Brier, skill vs a leave-one-family-out base rate, AUROC, overconfidence.
NEW for the AI x Epistemics sprint. numpy only. Intervals: bootstrap over testbed families."""
import json, sys, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def auroc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

def stats(rows):
    p = np.array([r["p"] for r in rows]); y = np.array([r["outcome"] for r in rows]); b = np.array([r["ref"] for r in rows])
    bs, bref = float(((p - y) ** 2).mean()), float(((b - y) ** 2).mean())
    return {"n": len(rows), "base": float(y.mean()), "mean_p": float(p.mean()), "bias": float(p.mean() - y.mean()),
            "brier": bs, "skill": 1 - bs / bref if bref > 0 else float("nan"), "auroc": auroc(p, y)}

def boot(rows, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r)
    cl = sorted(by); out = []
    for _ in range(B):
        samp = [r for c in rng.choice(cl, len(cl)) for r in by[c]]
        v = fn(samp)
        if v == v:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]

def main(path):
    raw = [json.loads(l) for l in open(path)]
    items = [json.loads(l) for l in open(HERE / "items.jsonl")]
    ref = {}
    for it in items:
        o = [x["outcome"] for x in items if x["question"] == it["question"] and x["cluster"] != it["cluster"]]
        ref[it["item_id"]] = sum(o) / len(o)
    agg = collections.defaultdict(list)
    for r in raw:
        if r["p"] is not None:
            agg[(r["model"], r["cond"], r["item_id"])].append(r)
    cells = collections.defaultdict(list)
    for (m, c, iid), rs in agg.items():
        row = dict(rs[0], p=float(np.mean([r["p"] for r in rs])), ref=ref[iid])
        cells[(m, c, row["question"])].append(row)
    res = []
    print(f"{'model':18}{'cond':10}{'q':5}{'n':>4}{'base':>6}{'meanp':>7}{'bias':>7} {'bias95':16}{'brier':>6}{'skill':>7} {'skill95':16}{'auroc':>6} {'auroc95':14}")
    for k in sorted(cells):
        rows = cells[k]; s = stats(rows)
        for name in ("bias", "skill", "auroc"):
            s[name + "_ci"] = boot(rows, lambda x, n=name: stats(x)[n])
        res.append(dict(zip(("model", "cond", "question"), k), **s))
        f = lambda ci: "[" + ",".join("nan" if v is None else f"{v:+.2f}" for v in ci) + "]"
        print(f"{k[0][:17]:18}{k[1]:10}{k[2]:5}{s['n']:4d}{s['base']:6.2f}{s['mean_p']:7.2f}{s['bias']:+7.2f} {f(s['bias_ci']):16}{s['brier']:6.3f}{s['skill']:+7.2f} {f(s['skill_ci']):16}{s['auroc']:6.2f} {f(s['auroc_ci']):14}")
    out = Path(path).with_suffix(".scores.json"); out.write_text(json.dumps(res, indent=1)); print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "data" / "forecasts.jsonl"))
