"""Triage rules on synthetic records. No testbed runs."""
import triage


def _rec(method_rows, base_rows, quantity="safety:mmcs vs baseline", prediction=None):
    rows = [{"seed": i, "method": m, "baseline": b} for i, (m, b) in enumerate(zip(method_rows, base_rows))]
    return {"id": "x", "testbed": "planted_interp", "verdict": "mechanism_failed",
            "prediction": prediction or {"safety_min_effect": 0.0},
            "stage1": {"quantity": quantity, "observed": 0.0, "rule": ">= +0.0000", "passed": False,
                       "per_seed": rows}}


def _ev(safety, **diag):
    return {"safety": safety, "capability": None, "diagnostics": diag}


def names(flags):
    return {f["flag"] for f in flags}


def test_identical_to_baseline():
    r = _rec([_ev(0.5, a=1.0)] * 3, [_ev(0.5, a=1.0)] * 3)
    assert "identical_to_baseline" in names(triage.static_checks(r))


def test_not_identical_when_a_diagnostic_differs():
    r = _rec([_ev(0.5, a=1.0)] * 3, [_ev(0.5, a=0.9)] * 3)
    assert "identical_to_baseline" not in names(triage.static_checks(r))


def test_collapsed_output_is_a_bug():
    r = _rec([_ev(0.2, recovery_rate=0.0)] * 3, [_ev(0.9, recovery_rate=0.8)] * 3)
    flags = triage.static_checks(r)
    assert "collapsed_output" in names(flags)
    assert triage.label(flags, None) == "suspected_bug"


def test_inverted_auroc():
    r = _rec([_ev(0.3)] * 3, [_ev(0.9)] * 3, quantity="safety:auroc_heldout vs baseline")
    assert "inverted_score" in names(triage.static_checks(r))


def test_chance_auroc_is_not_inverted():
    r = _rec([_ev(0.49)] * 3, [_ev(0.9)] * 3, quantity="safety:auroc_heldout vs baseline")
    assert "inverted_score" not in names(triage.static_checks(r))


def test_noisy_gate():
    r = _rec([_ev(0.6), _ev(0.4), _ev(0.52)], [_ev(0.5)] * 3)
    assert "noisy_gate" in names(triage.static_checks(r))


def test_labels_from_tuned_gate():
    assert triage.label([], {"passed": True}) == "gate_artifact"
    assert triage.label([], {"passed": False, "picked_default": False}) == "genuine_negative"
    assert triage.label([{"flag": "grid_edge"}], {"passed": False}) == "weak_negative"
    assert triage.label([], None) == "untriaged"


def test_env_errors_are_not_method_bugs():
    assert triage.label([], {"error": "No module named torch", "env_error": True}) == "needs_tuned_gate"
    assert triage.label([], {"error": "ValueError: bad shape", "env_error": False}) == "suspected_bug"


def test_identical_and_default_still_picked_is_a_bug():
    flags = [{"flag": "identical_to_baseline"}]
    assert triage.label(flags, {"passed": False, "picked_default": True}) == "suspected_bug"


def test_grid_edge_skips_count_floor():
    table = [{"hparams": {"k": k, "lr": 0.003}} for k in (1, 2, 3, 4)]
    assert triage.grid_edge({"method_tuning": table, "method_hparams": {"k": 1, "lr": 0.003}}) == []
    flags = triage.grid_edge({"method_tuning": table, "method_hparams": {"k": 4, "lr": 0.003}})
    assert [f["flag"] for f in flags] == ["grid_edge"]


def test_near_miss():
    rec = {"prediction": {"mechanism": {"diagnostic": "d", "direction": "lower", "min_effect": 0.2}}}
    out = triage.tuned_flags(rec, {"passed": False, "observed": 0.17})
    assert "near_miss" in names(out)
    assert "near_miss" not in names(triage.tuned_flags(rec, {"passed": False, "observed": 0.05}))


def test_changed_method_beats_everything():
    assert triage.label([{"flag": "collapsed_output"}], None, method_changed=True) == "fixed_pending_rerun"


def test_review_sets_label_but_not_over_a_changed_method():
    rev = {"label": "genuine_negative", "note": "checked"}
    assert triage.label([{"flag": "collapsed_output"}], None, review=rev) == "genuine_negative"
    assert triage.label([], None, review=rev, method_changed=True) == "fixed_pending_rerun"


def test_unknown_review_label_is_ignored():
    assert triage.label([{"flag": "collapsed_output"}], None, review={"label": "fine"}) == "suspected_bug"
