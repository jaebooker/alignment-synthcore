import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testbeds.reward_hacking import generate as G  # noqa: E402
from testbeds.reward_hacking.testbed import RewardHacking  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    return RewardHacking().instance("test", "full", 0)


def _policy(env, reward):
    return {"policy": G.value_iteration(env["T"], env[reward], 0.99).argmax(1)}


def test_hacker_scores_zero_and_oracle_scores_one(inst):
    tb = RewardHacking()
    for env in inst.private["envs"]:
        assert tb.score_one(_policy(env, "R_obs"), env)["safety"] == pytest.approx(0.0, abs=1e-9)
        o = tb.score_one(_policy(env, "R_true"), env)
        assert o["safety"] == pytest.approx(1.0, abs=1e-9)
        assert o["hacking"] == 0.0 and o["goal"] > 0.99


def test_every_env_has_a_strict_conflict(inst):
    for env in inst.private["envs"]:
        assert env["oracle"]["true"] - env["ref"]["true"] >= G.MIN_CONFLICT
        assert env["ref"]["obs"] - env["oracle"]["obs"] >= G.MIN_OBS_GAIN


def test_heldout_families_only_in_test():
    dev = {f for k, suite in G.SUITES.items() if k[0] == "dev" for f, _ in suite}
    test = {f for f, _ in G.SUITES[("test", "full")]}
    assert not dev & set(G.HELDOUT_FAMILIES)
    assert set(G.HELDOUT_FAMILIES) <= test


def test_public_view_hides_true_reward(inst):
    for pub, env in zip(inst.public, inst.private["envs"]):
        assert set(pub) == {"n_states", "n_actions", "noop_action", "T", "R", "terminal", "start_states",
                            "start_probs", "horizon"}
        assert np.array_equal(pub["R"], env["R_obs"])


def test_generation_is_deterministic():
    a, b = G.make_env("box", 6, 777), G.make_env("box", 6, 777)
    assert np.array_equal(a["T"], b["T"]) and np.array_equal(a["R_true"], b["R_true"])


def test_bad_policy_rejected(inst):
    env = inst.private["envs"][0]
    with pytest.raises(ValueError):
        RewardHacking().score_one({"policy": np.zeros(3, int)}, env)
