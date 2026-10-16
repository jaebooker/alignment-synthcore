"""Verdicts are computed here, by code, from stage reports. No model input."""
from __future__ import annotations

VERDICTS = {
    "beats_baseline": "Significant win on the safety metric over the tuned baseline",
    "mech_ok_no_win": "Mechanism reproduced on the benchmark; no significant practical win",
    "no_effect": "Stage 2 ran; no significant difference (see CI and MDE)",
    "hurts": "Significantly worse than the tuned baseline",
    "mechanism_failed": "Stage 1 toy gate failed; Stage 2 not run",
    "implementation_failed": "Method crashed or produced invalid output after all attempts",
    "untestable": "No v0 testbed fits this idea",
}


def compute_verdict(stage1: dict | None, stage2: dict | None, error: str | None = None) -> dict:
    flags: list[str] = []
    if error is not None:
        return {"verdict": "implementation_failed", "flags": flags, "reason": error[-300:]}
    if stage1 is None:
        return {"verdict": "untestable", "flags": flags, "reason": "no testbed"}
    if not stage1["passed"]:
        return {"verdict": "mechanism_failed", "flags": flags,
                "reason": f"{stage1['quantity']}: observed {stage1['observed']:.4f}, "
                          f"needed {stage1['rule']}"}

    s = stage2["safety"]
    mech_confirmed = stage2.get("mechanism_confirmed")  # None when the idea predicts no mechanism

    if s["significant"] and s["mean_diff"] > 0:
        verdict = "beats_baseline"
        cap = stage2.get("capability")
        if cap is not None and cap["significant"] and cap["mean_diff"] > 0:
            flags.append("capability_confounded")
        if mech_confirmed is False:
            flags.append("cause_unresolved")
    elif s["significant"] and s["mean_diff"] < 0:
        verdict = "hurts"
    elif mech_confirmed:
        verdict = "mech_ok_no_win"
    else:
        verdict = "no_effect"
    return {"verdict": verdict, "flags": flags, "benchmark_mechanism_confirmed": mech_confirmed,
            "reason": f"safety diff {s['mean_diff']:+.4f} (95% CI {s['ci95'][0]:+.4f} to "
                      f"{s['ci95'][1]:+.4f}), p={s['p_value']:.4f}"}
