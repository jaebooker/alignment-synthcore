import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.stats import paired_test  # noqa: E402
from harness.verdict import compute_verdict  # noqa: E402


def test_identical_arms_not_significant():
    r = paired_test([0.5] * 8, [0.5] * 8)
    assert r["p_value"] == 1.0 and not r["significant"]


def test_consistent_win_is_significant_and_exact():
    rng = np.random.default_rng(0)
    b = rng.normal(0.8, 0.01, 8)
    r = paired_test(b + 0.05, b)
    assert r["significant"] and r["mean_diff"] > 0
    assert abs(r["p_value"] - 2 / 256) < 1e-12  # all 8 diffs positive -> minimum exact p


def test_lower_is_better_orientation():
    r = paired_test([0.1] * 4 + [0.12] * 4, [0.3] * 8, higher_is_better=False)
    assert r["mean_diff"] > 0


def test_noise_only_rarely_significant():
    rng = np.random.default_rng(1)
    hits = sum(paired_test(rng.normal(0, 1, 8), rng.normal(0, 1, 8))["significant"] for _ in range(400))
    assert hits / 400 < 0.09  # nominal 0.05


def test_mde_shrinks_with_less_noise():
    a = paired_test(np.arange(8) * 0.01, np.zeros(8))
    b = paired_test(np.arange(8) * 0.001, np.zeros(8))
    assert b["mde"] < a["mde"]


def _s1(passed):
    return {"passed": passed, "quantity": "q", "observed": 0.0, "rule": ">= 0"}


def _s2(diff, sig, mech=None, cap=None):
    s = {"safety": {"mean_diff": diff, "significant": sig, "p_value": 0.01 if sig else 0.5, "ci95": [0, 0]}}
    if mech is not None:
        s["mechanism_confirmed"] = mech
    if cap is not None:
        s["capability"] = cap
    return s


def test_verdicts():
    assert compute_verdict(None, None, "boom")["verdict"] == "implementation_failed"
    assert compute_verdict(_s1(False), None)["verdict"] == "mechanism_failed"
    assert compute_verdict(_s1(True), _s2(0.05, True))["verdict"] == "beats_baseline"
    assert compute_verdict(_s1(True), _s2(-0.05, True))["verdict"] == "hurts"
    assert compute_verdict(_s1(True), _s2(0.01, False, mech=True))["verdict"] == "mech_ok_no_win"
    assert compute_verdict(_s1(True), _s2(0.01, False, mech=False))["verdict"] == "no_effect"
    v = compute_verdict(_s1(True), _s2(0.05, True, mech=False, cap={"mean_diff": 0.1, "significant": True}))
    assert v["verdict"] == "beats_baseline"
    assert set(v["flags"]) == {"capability_confounded", "cause_unresolved"}
