"""Roll idea records up into a results table and per-direction signals.

    python summarize.py            -> results/summary.json, results/SUMMARY.md

Direction signal (per Alignment Commons node and testbed):
  mechanism rate = share of tested ideas whose Stage 1 gate passed
  win rate       = share of tested ideas that beat the tuned baseline
Both use a Beta(1,1) prior with an 80% credible interval, so a node with
three ideas cannot look as certain as one with forty. Directions are only
compared within the same testbed.
"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NOT_TESTED = {"implementation_failed", "untestable"}


def rate(k, n):
    a, b = 1 + k, 1 + n - k
    return {"k": k, "n": n, "mean": a / (a + b), "ci80": [float(beta.ppf(0.1, a, b)), float(beta.ppf(0.9, a, b))]}


def quadrant(mech, win):
    hi_m, hi_w = mech["mean"] >= 0.5, win["mean"] >= 0.5
    return {(True, True): "pursue", (True, False): "real, not yet useful",
            (False, True): "wins, cause unclear", (False, False): "deprioritize"}[(hi_m, hi_w)]


def main():
    records = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*/record.json"))]
    rows = []
    for r in records:
        s2 = r.get("stage2") or {}
        s = s2.get("safety")
        rows.append({
            "id": r["id"], "title": r["title"], "direction": r.get("direction"), "testbed": r["testbed"],
            "verdict": r["verdict"], "flags": r.get("flags", []),
            "stage1": None if not r.get("stage1") else {k: r["stage1"][k] for k in ("quantity", "observed", "rule", "passed")},
            "safety": None if not s else {k: s[k] for k in ("method_mean", "baseline_mean", "mean_diff", "ci95", "p_value", "mde")},
            "capability": None if not s2.get("capability") else {k: s2["capability"][k] for k in ("mean_diff", "ci95", "p_value")},
            "mechanism_confirmed": s2.get("mechanism_confirmed"),
            "evaluator_hash": r["evaluator_hash"], "runtime_s": r.get("runtime_s"),
        })

    groups = defaultdict(list)
    for row in rows:
        if row["direction"] != "controls":
            groups[(row["direction"], row["testbed"])].append(row)
    directions = []
    for (d, tb), rs in sorted(groups.items()):
        tested = [x for x in rs if x["verdict"] not in NOT_TESTED]
        mech = rate(sum(x["stage1"]["passed"] for x in tested if x["stage1"]), len(tested))
        win = rate(sum(x["verdict"] == "beats_baseline" for x in tested), len(tested))
        directions.append({"direction": d, "testbed": tb, "ideas": len(rs), "tested": len(tested),
                           "untestable": sum(x["verdict"] == "untestable" for x in rs),
                           "implementation_failed": sum(x["verdict"] == "implementation_failed" for x in rs),
                           "capability_confounded_wins": sum("capability_confounded" in x["flags"] for x in rs),
                           "mechanism_rate": mech, "win_rate": win, "reading": quadrant(mech, win)})

    (RESULTS / "summary.json").write_text(json.dumps({"ideas": rows, "directions": directions}, indent=2))

    md = ["# Results", "", "| Idea | Testbed | Verdict | Safety diff (95% CI) | p | Flags |", "| --- | --- | --- | --- | --- | --- |"]
    for x in rows:
        s = x["safety"]
        diff = f"{s['mean_diff']:+.3f} ({s['ci95'][0]:+.3f} to {s['ci95'][1]:+.3f})" if s else "not run"
        p = f"{s['p_value']:.3f}" if s else ""
        md.append(f"| {x['title']} | {x['testbed']} | `{x['verdict']}` | {diff} | {p} | {', '.join(x['flags'])} |")
    md += ["", "# Direction signals", "", "| Direction | Testbed | Tested | Mechanism rate (80% CI) | Win rate (80% CI) | Reading |",
           "| --- | --- | --- | --- | --- | --- |"]
    for d in directions:
        f = lambda r: f"{r['k']}/{r['n']}: {r['mean']:.2f} ({r['ci80'][0]:.2f}-{r['ci80'][1]:.2f})"  # noqa: E731
        md.append(f"| {d['direction']} | {d['testbed']} | {d['tested']} | {f(d['mechanism_rate'])} | {f(d['win_rate'])} | {d['reading']} |")
    (RESULTS / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
