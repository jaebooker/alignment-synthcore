"""Test 7 engineering checks: run before the freeze."""
import json

import numpy as np
import pytest

from experiments.approval_factorial.run import collect, evaluate, seed_for
from experiments.myopia_decoupling import run as R
from testbeds.interactive_gridworld import generate as G

P = json.loads(R.PROTOCOL.read_text())


@pytest.fixture(scope="module")
def combo():
    es = R.env_seed("smoke", P["smoke_seeds"][0], "combo", 6)
    env = G.make_env("combo", 6, es)
    return env, collect(env, 4000, es)


def test_any_hack_is_bounded_by_its_parts(combo):
    env, trace = combo
    for arm in P["arms"]:
        out = R.fit(trace, arm, 1)
        m = evaluate(env, out)
        h = R.any_hack(env, out["policy"])
        assert max(m["lever_use"], m["button_use"]) - 1e-9 <= h <= m["lever_use"] + m["button_use"] + 1e-9


def test_decoupled_arms_never_read_executed_feedback(combo):
    env, trace = combo
    poisoned = dict(trace, executed_approval=trace["executed_approval"] + 100.0)
    for arm in (a for a in P["arms"] if a["feedback"] == "independent_query"):
        assert R.fit(trace, arm, 1)["policy"] == R.fit(poisoned, arm, 1)["policy"]


def test_coupled_arms_never_read_query_feedback(combo):
    env, trace = combo
    poisoned = dict(trace, queried_approval=trace["queried_approval"] + 100.0)
    for arm in (a for a in P["arms"] if a["feedback"] == "executed"):
        assert R.fit(trace, arm, 1)["policy"] == R.fit(poisoned, arm, 1)["policy"]


def test_myopic_arms_ignore_successor_values(combo):
    env, trace = combo
    # gamma=0: the greedy action at a state depends only on that state's own samples
    arm = next(a for a in P["arms"] if a["id"] == "myopic_decoupled")
    out = R.fit(trace, arm, 1)
    assert out["diagnostics"]["planning_gamma"] == 0.0


def test_fresh_environments_are_disjoint_from_test1():
    old = {seed_for(st, s, f, z) for st in ("pilot", "confirmation") for s in range(11000, 11008) for f in P["families"] for z in P["sizes"]}
    old |= {seed_for("confirmation", s, f, z) for s in range(91000, 91032) for f in P["families"] for z in P["sizes"]}
    new = {R.env_seed("confirmation", s, f, z) for s in P["confirmation_seeds"] for f in P["families"] for z in P["sizes"]}
    assert not old & new
    assert not set(P["smoke_seeds"]) & set(P["confirmation_seeds"])


def test_rule_has_four_components_and_fixed_n():
    assert len(P["primary_support_rule"]) == 4 and len(P["confirmation_seeds"]) == 32
    assert {a["id"] for a in P["arms"]} == {"myopic_decoupled", "myopic_coupled", "fullhorizon_decoupled", "fullhorizon_coupled"}


def test_confirm_refuses_without_matching_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["run", "confirm", "--seed", str(P["confirmation_seeds"][0])])
    with pytest.raises(FileNotFoundError):
        R.main()
