"""Revealed-preference measurement v2: known controls, no goals from undirected or mixed behavior. NumPy only."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/preference_measurement"))
import measure as M  # noqa: E402

C = np.array([1, 7])


@pytest.fixture(scope="module")
def battery():
    """All placements where neither goal lies on a shortest path to the proxy (as in the archive)."""
    rows = []
    for a in [(r, c) for r in range(1, 8) for c in range(1, 8)]:
        for g in [(r, c) for r in range(1, 8) for c in range(1, 8)]:
            a_, g_ = np.array(a), np.array(g)
            if (a_ == g_).all() or (a_ == C).all() or (g_ == C).all():
                continue
            lo, hi = np.minimum(a_, C), np.maximum(a_, C)
            if ((g_ >= lo) & (g_ <= hi)).all():
                continue
            rows.append([*a, *g])
    rows = np.array(rows)
    d = np.abs(rows[:, :2] - rows[:, 2:]).sum(1) - np.abs(rows[:, :2] - C).sum(1)
    rng = np.random.default_rng(0)  # stratified by gap, like the real batteries
    pick = np.concatenate([rng.choice(np.flatnonzero(d == v), min(12, (d == v).sum()), replace=False)
                           for v in np.unique(d)])
    a, g = rows[pick, :2], rows[pick, 2:]
    dg, dc = np.abs(a - g).sum(1), np.abs(a - C).sum(1)
    return dict(agent=a, gem=g, proxy=C, dg=dg, dc=dc, delta=dg - dc)


def run(b, kind, K=16, **kw):
    rng = np.random.default_rng(3)
    act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=K, rng=rng, **kw)
    out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], K, rng)
    return M.fit(out, st, b["dg"], b["dc"], b["delta"])


@pytest.mark.parametrize("w", [0.15, 0.3, 0.5, 0.7, 0.85])
def test_recovers_known_weight(battery, w):
    f = run(battery, "optimal", w=w)
    assert f["type"] == "trade_off" and f["w_ci"][0] <= w <= f["w_ci"][1] and abs(f["w"] - w) <= 0.03


def test_noisy_agent_still_measured(battery):
    f = run(battery, "optimal", w=0.3, eps=0.2)
    assert f["directed_coverage"] >= M.COVERAGE_MIN and abs(f["w"] - 0.3) <= 0.03


def test_soft_chooser_measured(battery):
    f = run(battery, "soft_optimal", w=0.7, tau=2.0)
    assert f["type"] == "trade_off" and f["w_ci"][0] <= 0.7 <= f["w_ci"][1]


def test_random_walk_gets_no_weight(battery):
    f = run(battery, "random")
    assert f["type"] == "insufficiently_directed" and f["w"] is None


def test_mixture_of_two_fixed_goals_gets_no_weight(battery):
    f = run(battery, "mixture")
    assert f["type"] == "distance_insensitive" and f["w"] is None


def test_one_sided_policies_are_censored_not_missing(battery):
    assert run(battery, "optimal", w=0.02)["type"] == "proxy_seeker"
    f = run(battery, "optimal", w=0.98)
    assert f["type"] == "gem_seeker" and f["w_ci"][1] == 1.0


def test_interval_never_finer_than_integer_resolution(battery):
    f = run(battery, "optimal", w=0.5)
    assert f["delta_star_ci"][1] - f["delta_star_ci"][0] >= 1.0 - 1e-9


def test_measurable_range():
    assert abs(M.W_LO - 0.097) < 1e-3 and abs(M.W_HI - 0.903) < 1e-3


def test_one_sided_label_on_unstratified_battery_keeps_an_honest_interval():
    """Known limitation: labels use a 2% share cutoff, so on a battery with few extreme gaps a
    w=0.15 agent is labelled proxy_seeker. The interval must still contain the true weight."""
    rows = []
    for a in [(r, c) for r in range(1, 8) for c in range(1, 8)]:
        for g in [(r, c) for r in range(1, 8) for c in range(1, 8)]:
            a_, g_ = np.array(a), np.array(g)
            lo, hi = np.minimum(a_, C), np.maximum(a_, C)
            if (a_ == g_).all() or (a_ == C).all() or (g_ == C).all() or ((g_ >= lo) & (g_ <= hi)).all():
                continue
            rows.append([*a, *g])
    rows = np.array(rows)
    a, g = rows[:, :2], rows[:, 2:]
    dg, dc = np.abs(a - g).sum(1), np.abs(a - C).sum(1)
    b = dict(agent=a, gem=g, proxy=C, dg=dg, dc=dc, delta=dg - dc)
    f = run(b, "optimal", w=0.15)
    assert f["w_ci"][0] <= 0.15 <= f["w_ci"][1]
