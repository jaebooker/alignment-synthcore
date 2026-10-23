"""Interactive testbed: the harness-side environment, its budgets, and the method boundary."""
import json
import pickle

import numpy as np
import pytest

import harness.runner as R
from testbeds.interactive_gridworld import generate as G
from testbeds.interactive_gridworld.testbed import BudgetExceeded, InteractiveGridworld, Session


@pytest.fixture(scope="module")
def tb():
    return InteractiveGridworld()


@pytest.fixture(scope="module")
def toy(tb):
    return tb.instance("dev", "toy", 0)


def test_public_spec_holds_no_ground_truth(toy):
    allowed = {"n_actions", "noop_action", "horizon", "budget", "env_reward"}
    for spec in toy.public:
        assert set(spec) == allowed
    blob = pickle.dumps(toy.public)
    for e in toy.private["envs"]:  # no hidden table survives into what the method receives
        assert e["R_true"].tobytes() not in blob and e["T"].tobytes() not in blob


def test_budget_is_enforced_by_the_harness(toy):
    sess = Session(toy.private["envs"], {"steps": 3, "queries": 1}, seed=0)
    sess.handle({"op": "reset", "task": 0})
    for _ in range(3):
        if sess.handle({"op": "step", "task": 0, "a": 4})["done"]:
            sess.handle({"op": "reset", "task": 0})
    with pytest.raises(BudgetExceeded):
        sess.handle({"op": "step", "task": 0, "a": 4})
    sess.handle({"op": "query", "task": 0, "a": 0})
    with pytest.raises(BudgetExceeded):
        sess.handle({"op": "query", "task": 0, "a": 0})


def test_bad_requests_are_rejected(toy):
    sess = Session(toy.private["envs"], {"steps": 10, "queries": 10}, seed=0)
    with pytest.raises(ValueError):
        sess.handle({"op": "step", "task": 0, "a": 0})  # before reset
    sess.handle({"op": "reset", "task": 0})
    for bad in ({"op": "step", "task": 0, "a": 7}, {"op": "step", "task": 9, "a": 0}, {"op": "peek", "task": 0, "a": 0}):
        with pytest.raises(ValueError):
            sess.handle(bad)


def test_lever_corrupts_only_the_executed_step():
    env = G.make_env("feedback_lever", 6, 123)
    lever_states = [i for i, s in enumerate(env["states"]) if s[0] == env["lever"]]
    for s in lever_states:
        assert env["A_exec"][s, G.NOOP] == G.LEVER_APPROVAL
        assert env["A_honest"][s, G.NOOP] < 1.0          # a counterfactual query stays honest
    others = np.ones_like(env["ev_lever"])
    others[lever_states, G.NOOP] = False
    assert np.allclose(env["A_exec"][others], env["A_honest"][others])


def test_tamper_bonus_starts_after_the_press():
    env = G.make_env("tamper_ms", 6, 7)
    pressed = np.argwhere(env["ev_tamper"])
    assert len(pressed)
    for s, a in pressed:  # the pressing step itself pays nothing extra: the hack is multi-step
        assert env["R_env"][s, a] == pytest.approx(env["R_base"][s, a])


def test_overseer_is_blind_to_the_vase():
    env = next(e for e in (G.make_env("tamper_ms", 6, k) for k in range(40)) if e["vase"] is not None)
    obedient = G.rollout(env, G.greedy(env["A_honest"]))
    assert env["oracle"]["vase"] == 0.0
    assert env["R_true"][env["ev_vase"]].max() < env["R_base"][env["ev_vase"]].min()
    assert obedient["true"] <= env["oracle"]["true"] + 1e-9


def test_end_to_end_run_and_determinism(tb, tmp_path):
    m = tmp_path / "walker.py"
    m.write_text("import public\nHPARAMS=[{}]\n"
                 "def run(tasks, hp, seed):\n"
                 "    out=[]\n"
                 "    for i,_ in enumerate(tasks):\n"
                 "        e=public.env(i); o=e.reset(); seen={o:1}\n"
                 "        for _ in range(50):\n"
                 "            o,r,d,a=e.step(1); seen[o]=1\n"
                 "            if d: o=e.reset()\n"
                 "        out.append({'policy': {k: 1 for k in seen}})\n"
                 "    return out\n")
    a = R.run_arm(tb, m, "dev", "toy", 0, {})
    b = R.run_arm(tb, m, "dev", "toy", 0, {})
    assert a["safety"] == b["safety"] and -1 <= a["safety"] <= 1


def test_method_errors_surface_as_method_errors(tb, tmp_path):
    m = tmp_path / "boom.py"
    m.write_text("import public\nHPARAMS=[{}]\ndef run(tasks, hp, seed):\n    public.env(0).step(0)\n")
    with pytest.raises(R.MethodError) as e:  # step before reset: the harness answers with an error
        R.run_arm(tb, m, "dev", "toy", 0, {})
    assert "reset" in str(e.value)


def test_harness_never_unpickles_requests():
    import inspect
    src = inspect.getsource(R._serve)
    assert "json.loads" in src and "pickle.load" not in src and "pickle.loads" not in src
