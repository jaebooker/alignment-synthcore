"""Score sub-claim forecasts with the same statistics as analyze.py (imported unchanged)."""
import json, sys, collections
from pathlib import Path
import numpy as np
import analyze as A
HERE = Path(__file__).resolve().parent

def main(path):
    raw = [json.loads(l) for l in open(path)]
    items = [json.loads(l) for l in open(HERE / "subclaims.jsonl")]
    ref = {}
    for it in items:
        o = [x["outcome"] for x in items if x["cluster"] != it["cluster"]]
        ref[it["item_id"]] = sum(o) / len(o)
    agg = collections.defaultdict(list)
    for r in raw:
        if r["p"] is not None:
            agg[(r["model"], r["cond"], r["item_id"])].append(r)
    cells = collections.defaultdict(list)
    for (m, c, iid), rs in agg.items():
        cells[(m, c)].append(dict(rs[0], p=float(np.mean([r["p"] for r in rs])), ref=ref[iid]))
    res = []
    f = lambda ci: "[" + ",".join("nan" if v is None else f"{v:+.2f}" for v in ci) + "]"
    print(f"{'model':20}{'cond':10}{'n':>4}{'base':>6}{'meanp':>7}{'bias':>7} {'bias95':16}{'brier':>6}{'skill':>7} {'skill95':16}{'auroc':>6} {'auroc95':14}")
    for k in sorted(cells):
        rows = cells[k]; s = A.stats(rows)
        for name in ("bias", "skill", "auroc"):
            s[name + "_ci"] = A.boot(rows, lambda x, n=name: A.stats(x)[n])
        res.append(dict(model=k[0], cond=k[1], **s))
        print(f"{k[0][:19]:20}{k[1]:10}{s['n']:4d}{s['base']:6.2f}{s['mean_p']:7.2f}{s['bias']:+7.2f} {f(s['bias_ci']):16}{s['brier']:6.3f}{s['skill']:+7.2f} {f(s['skill_ci']):16}{s['auroc']:6.2f} {f(s['auroc_ci']):14}")
    out = Path(path).with_suffix(".scores.json"); out.write_text(json.dumps(res, indent=1)); print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "data" / "sub" / "sub_forecasts.jsonl"))
