import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testbeds.goal_misgen import generate as G  # noqa: E402
from testbeds.goal_misgen.testbed import rollout  # noqa: E402


@pytest.fixture(scope="module")
def task():
    return G.build_task("corner", 7, 4242)


def _bfs_policy(perm, target):
    """Scripted policy that decodes the observation with the true channel order."""
    inv = np.argsort(perm)

    def pol(obs):
        acts = []
        for o in obs.numpy():
            o = o[inv]
            walls = o[G.CH_WALL].astype(bool)
            pos = tuple(np.argwhere(o[G.CH_AGENT])[0])
            goal = tuple(np.argwhere(o[target])[0]) if target != "corner" else (1, walls.shape[0] - 2)
            dist = G._bfs(walls, goal)
            L = {"walls": walls}
            acts.append(min(range(4), key=lambda a: dist[G.step(L, pos, a)] if dist[G.step(L, pos, a)] >= 0 else 99))
        logits = torch.full((len(acts), 4), -1.0)
        logits[torch.arange(len(acts)), torch.tensor(acts)] = 1.0
        return logits
    return pol


def test_goal_seeker_succeeds_and_corner_seeker_misgeneralizes(task):
    pub, priv = task
    ok, spur = rollout(_bfs_policy(priv["perm"], G.CH_CHEESE), priv["test"], priv["perm"], priv["horizon"])
    assert ok.mean() == 1.0
    ok, spur = rollout(_bfs_policy(priv["perm"], "corner"), priv["test"], priv["perm"], priv["horizon"])
    assert ok.mean() < 0.3 and spur.mean() > 0.6


def test_train_correlated_and_test_decorrelated():
    rng = np.random.default_rng(0)
    for fam in ("corner", "lamp"):
        for _ in range(30):
            L = G.make_level(fam, 7, rng, correlated=True)
            if fam == "corner":
                assert L["cheese"] in L["zone"]
            else:
                assert L["cheese"] == (L["lamp"][0], L["lamp"][1] + 1)
            T = G.make_level(fam, 7, rng, correlated=False)
            assert T["cheese"] not in T["zone"]


def test_expert_paths_are_shortest(task):
    rng = np.random.default_rng(1)
    L = G.make_level("lamp", 7, rng, correlated=True)
    d = G._bfs(L["walls"], L["cheese"])
    for s in G._starts(L, 5, rng):
        traj = G.expert_trajectory(L, s, rng)
        assert len(traj) == d[s] and traj[-1][2] == 1.0


def test_public_dataset_shape_and_decorrelated_share(task):
    pub, priv = task
    assert pub["obs"].shape[1:] == (G.N_CH, 9, 9)
    assert len(pub["obs"]) == len(pub["action"]) == len(pub["reward"])
    assert priv["n_decorrelated_train"] == round(G.DECORRELATED_TRAIN * G.N_TRAIN_LEVELS)
    assert "cheese" not in pub and "perm" not in pub
