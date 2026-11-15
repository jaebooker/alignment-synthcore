import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testbeds.control_open import items as I  # noqa: E402
from testbeds.control_open.testbed import TrustedMonitoringOpen  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "testbeds" / "control_open" / "data"


@pytest.fixture(scope="module")
def inst():
    return TrustedMonitoringOpen().instance("test", "full", 3)


def _eval(inst, fn, probe_fn=None):
    labels, probe_of = inst.private["labels"], inst.private["probe_of"]
    probe_fn = probe_fn or fn
    inst.private["probe_outputs"] = [{"scores": {probe_of[i]: probe_fn(labels[i]) for i in probe_of}}]
    return TrustedMonitoringOpen().evaluate([{"scores": {i: fn(l) for i, l in labels.items()}}], inst, 3)


def test_perfect_constant_and_inverted_monitors(inst):
    assert _eval(inst, lambda l: 1.0 if l["attack"] else 0.0)["safety"] == 1.0
    r = _eval(inst, lambda l: 0.5)
    assert r["safety"] == pytest.approx(0.05) and r["diagnostics"]["auroc"] == 0.5
    assert _eval(inst, lambda l: 0.0 if l["attack"] else 1.0)["safety"] == 0.0


def test_shards_are_disjoint_and_cover_the_test_split():
    problems = json.loads((DATA / "problems.json").read_text())
    outputs = json.loads((DATA / "outputs.json").read_text())
    assert len(problems) == len(outputs) >= 400
    split = Counter(p["split"] for p in problems)
    assert split["test"] == 480 and split["dev"] >= 100
    seen = set()
    for s in range(I.N_SHARDS):
        ids = {r["task_id"] for r in I.rows_for("test", "full", s, outputs)}
        assert len(ids) == 60 and not ids & seen
        seen |= ids
    assert not seen & {r["task_id"] for r in I.rows_for("dev", "full", 0, outputs)}
    with pytest.raises(ValueError):
        I.rows_for("test", "full", 8, outputs)


def test_public_items_carry_no_labels_and_every_shard_has_attacks():
    for s in range(I.N_SHARDS):
        pub, priv = I.build("test", "full", s)
        for it in pub[0]["items"] + priv["probe_public"][0]["items"]:
            assert set(it) == {"id", "problem", "code"}
        static = sum(l.get("static_selected", False) for l in priv["labels"].values())
        assert static >= 10
        assert not {it["id"] for it in pub[0]["items"]} & {it["id"] for it in priv["probe_public"][0]["items"]}


def test_recorded_backdoors_and_honest_items_are_what_they_claim():
    outputs = json.loads((DATA / "outputs.json").read_text())
    for r in outputs:
        if r["backdoor_valid"]:
            assert r["attack_code"] and r["backdoor_reason"] == "ok"
        if r.get("attack_selected"):
            assert r["backdoor_valid"] and r["sneakiness"] is not None
