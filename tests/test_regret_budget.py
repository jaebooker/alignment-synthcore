import numpy as np
import pytest
from experiments.regret_budget.core import certify, Environment, Curriculum
from testbeds.ued_gridworld import generate as G


def test_oracle_bellman_bfs_and_optimal_rollout():
    rng = np.random.default_rng(8932)
    for family in ['corner', 'lamp']:
        for _ in range(12):
            level = G.default_level(family, rng)
            values, digest = certify(level)
            assert len(digest) == 64
            d = G._bfs(level['walls'], level['cheese'])
            for start, best in values.items():
                pos = start
                for t in range(1, G.horizon()+1):
                    pos = min([G.step(level, pos, a) for a in range(4)], key=lambda p: d[p])
                    if pos == level['cheese']:
                        assert best == pytest.approx(1-.5*t/G.horizon())
                        break
                else:
                    assert best == 0
            # Artificial short deadline exercises unreachable-within-horizon starts.
            short, _ = certify(level, H=1)
            assert any(v == 0 for v in short.values())
            assert all(v in (0, .5) for v in short.values())


def test_teacher_requires_completed_episode_and_no_oracle_for_other_arms():
    v = Environment('corner', np.arange(4), 986, oracle=True)
    lid = v.new_levels()[0]
    with pytest.raises(KeyError):
        v.finish(0, lid, 0)
    v.reset([0], [lid])
    for _ in range(G.horizon()):
        _, ret, done = v.step([0], [0])
        if done[0]:
            value = v.finish(0, lid, float(ret[0]))
            assert 0 <= value <= 1
            break
    with pytest.raises(KeyError):
        v.finish(0, lid, 0)
    assert v.count['oracle_queries'] == 1
    assert len(v.session.tasks[0]) == 2  # family, perm; no held-out episode tables
    assert v.count['certified_starts'] > 0
