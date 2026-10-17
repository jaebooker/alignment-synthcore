import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testbeds.weak_to_strong import generate as G  # noqa: E402
from testbeds.weak_to_strong.public import export, make_student  # noqa: E402
from testbeds.weak_to_strong.testbed import WeakToStrong  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    return WeakToStrong().instance("dev", "toy", 0)


def test_gold_tasks_exact_and_weak_tasks_noisy(inst):
    kinds = [t["kind"] for t in inst.private["tasks"]]
    assert "weak" in kinds and "gold" in kinds
    for t in inst.private["tasks"]:
        if t["kind"] == "gold":
            assert t["label_acc"] == 1.0
        else:
            assert t["label_acc"] < 0.97


def test_public_view_hides_task_kind_and_truth(inst):
    for p in inst.public:
        assert set(p) == {"X", "y", "backbone"}


def test_weak_supervisor_is_weaker_than_ceiling(inst):
    for t in inst.private["tasks"]:
        assert t["ceiling_acc"] > t["weak_acc"]


def test_gold_trained_student_recovers_most_of_the_gap(inst):
    tb = WeakToStrong()
    for p, t in zip(inst.public, inst.private["tasks"]):
        if t["kind"] != "weak":
            continue
        gold_y = torch.tensor(np.isin(np.arange(10), G.PARTITIONS[t["name"]]).astype(np.int64))
        # recover gold labels for the student split from the digit data
        torch.manual_seed(0)
        m = make_student(p["backbone"])
        with torch.no_grad():
            digit_guess = m[0](torch.tensor(p["X"])).argmax(1)
        y_gold = gold_y[digit_guess]
        G._fit(m, m.parameters(), p["X"], y_gold.numpy(), 30, 1e-3, 0)
        s = tb.score_one(export(m), t)
        if s["pgr"] is not None:
            assert s["pgr"] > 0.5


def test_determinism():
    a, _ = G.build("dev", "toy", 3)
    b, _ = G.build("dev", "toy", 3)
    assert all(np.array_equal(x["y"], y["y"]) for x, y in zip(a, b))
