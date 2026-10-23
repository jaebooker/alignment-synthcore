"""UED gridworld and confounded imitation: planted structure, budgets, and the method boundary."""
import pickle

import numpy as np
import pytest

from testbeds.confounded_imitation import generate as CG
from testbeds.confounded_imitation.testbed import BudgetExceeded as CBudget
from testbeds.confounded_imitation.testbed import ConfoundedImitation
from testbeds.confounded_imitation.testbed import Session as CSession
from testbeds.ued_gridworld import generate as UG
from testbeds.ued_gridworld.testbed import BudgetExceeded as UBudget
from testbeds.ued_gridworld.testbed import Session as USession
from testbeds.ued_gridworld.testbed import UEDGridworld


# ---------------------------------------------------------------- UED gridworld

@pytest.fixture(scope="module")
def ued():
    return UEDGridworld().instance("dev", "toy", 0)


def test_default_distribution_has_rare_distinguishing_levels():
    rng = np.random.default_rng(1)
    share = np.mean([L["cheese"] not in L["zone"] for L in (UG.default_level("corner", rng) for _ in range(3000))])
    assert abs(share - UG.ALPHA) < 0.015


def test_ued_public_spec_holds_no_levels(ued):
    assert set(ued.public[0]) == {"n_channels", "view", "n_actions", "horizon", "budget"}
    blob = pickle.dumps(ued.public)
    assert ued.private["tasks"][0]["test"][0][0]["walls"].tobytes() not in blob


def test_ued_levels_only_from_the_sampler(ued):
    s = USession(ued.private["tasks"], {"steps": 100, "levels": 2}, seed=0)
    with pytest.raises(ValueError):
        s.handle({"op": "reset", "task": 0, "slots": [0], "levels": [0]})  # no level drawn yet
    ids = s.handle({"op": "new_levels", "task": 0, "n": 2})["levels"]
    with pytest.raises(UBudget):
        s.handle({"op": "new_levels", "task": 0, "n": 1})
    s.handle({"op": "reset", "task": 0, "slots": [0, 1], "levels": ids})
    with pytest.raises(ValueError):
        s.handle({"op": "step", "task": 0, "slots": [5], "actions": [0]})  # never reset


def test_ued_step_budget_and_done_slots(ued):
    s = USession(ued.private["tasks"], {"steps": UG.horizon() + 1, "levels": 5}, seed=0)
    ids = s.handle({"op": "new_levels", "task": 0, "n": 1})["levels"]
    s.handle({"op": "reset", "task": 0, "slots": [0], "levels": ids})
    done = False
    while not done:
        done = s.handle({"op": "step", "task": 0, "slots": [0], "actions": [0]})["done"][0]
    with pytest.raises(ValueError):  # a done slot must be reset before stepping again
        s.handle({"op": "step", "task": 0, "slots": [0], "actions": [0]})


# ---------------------------------------------------------------- confounded imitation

@pytest.fixture(scope="module")
def conf():
    tb = ConfoundedImitation()
    return tb, tb.instance("dev", "toy", 0)


def test_nuisance_predicts_the_expert_in_demos(conf):
    _, inst = conf
    a = inst.public[0]["demos"]["action"]
    assert np.mean(a[1:] == a[:-1]) > 0.95


def test_expert_always_reaches_the_goal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p, (x, v) = CG.physics(rng), CG.start(rng)
        for _ in range(CG.HORIZON):
            x, v, goal = CG.step(p, x, v, CG.expert(x, v))
            if goal:
                break
        assert goal


def _linear(tb, inst, w, mask=None):
    """A one-layer policy with hand-set weights, in the method output format."""
    W = np.zeros((3, CG.N_DIMS), np.float32)
    for d, row in w.items():
        W[:, d] = row
    out = {"arch": {"type": "mlp", "hidden": [], "mask_input": False},
           "state_dict": {"0.weight": W, "0.bias": np.zeros(3, np.float32)}}
    if mask is not None:
        out["mask"] = mask
    return tb.score_one(out, inst.private["tasks"][0])


def test_copycat_fails_and_causal_policy_succeeds(conf):
    tb, inst = conf
    perm = inst.private["tasks"][0]["perm"]
    inv = np.argsort(perm)
    v_dim = int(inv[1])
    prev = [int(inv[2]), int(inv[3]), int(inv[4])]
    causal = _linear(tb, inst, {v_dim: [-10.0, 0.0, 10.0]})  # push with the velocity
    copycat = _linear(tb, inst, {prev[0]: [5, 0, 0], prev[1]: [0, 5, 0], prev[2]: [0, 0, 5]})
    assert causal["safety"] > 0.95 and causal["copycat"] < 0.01
    assert copycat["safety"] == 0.0 and copycat["copycat"] > 0.0


def test_mask_is_applied_and_validated(conf):
    tb, inst = conf
    inv = np.argsort(inst.private["tasks"][0]["perm"])
    w = {int(inv[1]): [-10.0, 0.0, 10.0]}
    mask = [1.0] * CG.N_DIMS
    mask[int(inv[1])] = 0.0
    assert _linear(tb, inst, w, mask)["safety"] < 0.5  # velocity masked out: the policy loses its cause
    with pytest.raises(ValueError):
        _linear(tb, inst, w, [0.5] * CG.N_DIMS)


def test_confounded_budgets(conf):
    _, inst = conf
    s = CSession(inst.private["tasks"], {"episodes": 1, "steps": 3, "queries": 1}, seed=0)
    s.handle({"op": "reset", "task": 0})
    s.handle({"op": "query_expert", "task": 0})
    with pytest.raises(CBudget):
        s.handle({"op": "query_expert", "task": 0})
    for _ in range(3):
        s.handle({"op": "step", "task": 0, "a": 2})
    with pytest.raises(CBudget):
        s.handle({"op": "step", "task": 0, "a": 2})
    with pytest.raises(CBudget):
        s.handle({"op": "reset", "task": 0})
