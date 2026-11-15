"""Score every baseline configuration on the 8 test shards and compare each Stage 2 record on
control_open with the best one in hindsight. Exploratory: the verdicts keep the dev-tuned baseline.

    ADAPTION_KEY_FILE=... python -m testbeds.control_open.hindsight
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.runner import get_testbed, load_hparams, run_arm
from harness.stats import paired_test

ROOT = Path(__file__).resolve().parents[2]


def main():
    tb = get_testbed("control_open")
    cfg = {}
    for hp in load_hparams(tb.baseline_method):
        rows = [run_arm(tb, tb.baseline_method, "test", "full", s, hp) for s in range(8)]
        cfg[json.dumps(hp)] = rows
        print("baseline", hp, "test catch@5%", round(sum(r["safety"] for r in rows) / 8, 4),
              "auroc", round(sum(r["diagnostics"]["auroc"] for r in rows) / 8, 4), flush=True)
    best = max(cfg, key=lambda k: sum(r["safety"] for r in cfg[k]))
    out = {"baseline_configs": {k: [r["safety"] for r in v] for k, v in cfg.items()}, "best": best, "ideas": {}}
    for rec in sorted((ROOT / "results").glob("*/record.json")):
        R = json.loads(rec.read_text())
        if R.get("testbed") != "control_open" or not R.get("stage2"):
            continue
        ms = [p["method"]["safety"] for p in R["stage2"]["per_seed"]]
        t = paired_test(ms, [r["safety"] for r in cfg[best]])
        out["ideas"][R["id"]] = {"verdict": R["verdict"], "vs_best": t}
        print(f"{R['id']} {R['verdict']}: {t['method_mean']:.3f} vs best {t['baseline_mean']:.3f}, diff {t['mean_diff']:+.3f} "
              f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['p_value']:.3f}")
    (ROOT / "results" / "control_open_hindsight.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
