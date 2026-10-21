"""run.py --resume: finished arms are cached and reused; a passed deadline stops cleanly."""
import pytest

import harness.runner as R


@pytest.fixture
def resume(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    monkeypatch.setitem(R.RESUME, "on", True)
    monkeypatch.setitem(R.RESUME, "deadline", None)
    return tmp_path


def _arm():
    tb = R.get_testbed("reward_hacking")
    hp = R.load_hparams(tb.baseline_method)[0]
    return tb, (tb, tb.baseline_method, "dev", "toy", 0, hp)


def test_arm_is_cached_and_reused(resume, monkeypatch):
    tb, args = _arm()
    first = R.run_arm(*args)
    assert len(list(resume.glob("*.pkl"))) == 1
    monkeypatch.setattr(R, "_run_arm", lambda *a: pytest.fail("should have used the cache"))
    assert R.run_arm(*args) == first


def test_deadline_stops_before_new_work(resume):
    tb, args = _arm()
    R.RESUME["deadline"] = 0.0
    with pytest.raises(R.OutOfTime):
        R.run_arm(*args)
    assert not list(resume.glob("*.pkl"))


def test_key_changes_with_method_code(tmp_path):
    tb, (_, method, *rest) = _arm()
    copy = tmp_path / "m.py"
    copy.write_text(open(method).read())
    k1 = R._arm_key(tb, copy, *rest)
    copy.write_text(open(method).read() + "\n# edit\n")
    assert R._arm_key(tb, copy, *rest) != k1


def test_off_by_default():
    assert R.RESUME["on"] is False
