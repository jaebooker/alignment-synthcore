import numpy as np

from experiments.approval_factorial.learner import fit_assigned
from experiments.approval_factorial.run import collect
from experiments.vase_blind_spot import run as R
from experiments.vase_blind_spot.penalty import fit_penalized, rr_penalty
from testbeds.interactive_gridworld import generate as G


def _trace(seed=7):
    env = G.make_env("feedback_lever", 6, seed)
    return env, collect(env, 4000, seed)


def test_zero_penalty_reproduces_test7_learner():
    env, t = _trace()
    a = fit_assigned(t["obs"], t["action"], t["next_obs"], t["reward"], t["query_action"], t["queried_approval"],
                     t["terminated"], 1, 0.0)
    b = fit_penalized(t["obs"], t["action"], t["next_obs"], t["reward"], t["query_action"], t["queried_approval"],
                      t["terminated"], 1, 0.0, penalty="none")
    assert a["policy"] == b["policy"]


def test_rr_penalizes_vase_break_not_ordinary_moves():
    assert G.P_BLIND == 1.0  # set by run.py import
    env, t = _trace(11)
    assert env["vase"] is not None
    states = np.unique(np.concatenate([t["obs"], t["next_obs"]]))
    s, ns = np.searchsorted(states, t["obs"]), np.searchsorted(states, t["next_obs"])
    T = np.full((len(states), 5), -1)
    T[s, t["action"]] = ns
    term = np.zeros(len(states), bool)
    term[ns[t["terminated"]]] = True
    P = rr_penalty(T, term)
    # map true (state, action) vase events onto learned indices
    pos = {int(o): i for i, o in enumerate(states)}
    vase, other = [], []
    for i in range(len(env["T"])):
        li = pos.get(int(env["obs_id"][i]))
        if li is None:
            continue
        for a in range(5):
            if T[li, a] < 0 or env["ev_tamper"][i, a]:
                continue
            (vase if env["ev_vase"][i, a] else other).append(P[li, a])
    assert vase and min(vase) > 5 * np.percentile(other, 95)
