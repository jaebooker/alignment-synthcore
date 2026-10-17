"""Exploratory run: tuning + Stage 2 with the Stage 1 gate skipped.

    python exploratory.py ideas/<idea>

Writes results/_exploratory/<idea>.json. These are NOT verdicts: they are not
read by summarize.py, and they exist to check how much the gate rule itself
decides outcomes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.runner import get_testbed, run_arm, tune, tuned_baseline  # noqa: E402
from harness.stats import paired_test  # noqa: E402


def main(idea_dir, seeds=8):
    idea_dir = Path(idea_dir)
    spec = json.loads((idea_dir / "idea.json").read_text())
    tb = get_testbed(spec["testbed"])
    log = lambda m: print(m, flush=True)  # noqa: E731
    base = tuned_baseline(tb, log=log)
    hp, table = tune(tb, idea_dir / "method.py", log=log)
    ms, bs = [], []
    for s in range(seeds):
        ms.append(run_arm(tb, idea_dir / "method.py", "test", "full", s, hp))
        bs.append(run_arm(tb, tb.baseline_method, "test", "full", s, base["hparams"]))
        log(f"  seed {s}: method {ms[-1]['safety']:.4f} baseline {bs[-1]['safety']:.4f}")
    out = {"exploratory": True, "note": "Stage 1 gate skipped; not a verdict",
           "idea": idea_dir.name, "evaluator_hash": tb.evaluator_hash(), "method_hparams": hp,
           "baseline_hparams": base["hparams"],
           "safety": paired_test([m["safety"] for m in ms], [b["safety"] for b in bs]),
           "diagnostics": {k: paired_test([m["diagnostics"][k] for m in ms], [b["diagnostics"][k] for b in bs],
                                          higher_is_better=tb.diagnostics[k])
                           for k in tb.diagnostics}}
    if tb.capability_metric:
        out["capability"] = paired_test([m["capability"] for m in ms], [b["capability"] for b in bs])
    d = ROOT / "results" / "_exploratory"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{idea_dir.name}.json").write_text(json.dumps(out, indent=2))
    s = out["safety"]
    log(f"EXPLORATORY {idea_dir.name}: safety diff {s['mean_diff']:+.4f} "
        f"(95% CI {s['ci95'][0]:+.4f} to {s['ci95'][1]:+.4f}), p={s['p_value']:.4f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
