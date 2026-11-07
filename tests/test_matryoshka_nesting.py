"""Engineering checks for experiments/matryoshka_nesting (no study results)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "matryoshka_nesting"))
import nesting  # noqa: E402
from testbeds.planted_interp import generate  # noqa: E402


def test_arms_differ_only_in_loss_groups():
    H = np.random.default_rng(0).normal(size=(600, 6)).astype(np.float32)
    outs = [nesting.train(H, 12, arm, 2, 3e-3, [1.0], seed=5, steps=40, batch=64) for arm in nesting.ARMS]
    for o in outs[1:]:
        np.testing.assert_allclose(o, outs[0], atol=1e-6)  # same init, batches, optimiser


def test_group_structure():
    m, pre = 40, [0.25, 0.5, 1.0]
    g = torch.Generator().manual_seed(1)
    fixed = [torch.randperm(m, generator=g)[:p] for p in (10, 20)] + [torch.arange(m)]
    assert [len(x) for x in nesting.groups_for("topk", m, pre, g, fixed)] == [40]
    nested = nesting.groups_for("nested", m, pre, g, fixed)
    assert [x.tolist() for x in nested] == [list(range(10)), list(range(20)), list(range(40))]
    a, b = (nesting.groups_for("shuffled", m, pre, g, fixed) for _ in range(2))
    assert [len(x) for x in a] == [10, 20, 40]
    assert set(a[0].tolist()) <= set(a[1].tolist()) <= set(a[2].tolist())  # nested within a step
    assert a[0].tolist() != b[0].tolist()  # but a new order every step
    sub = nesting.groups_for("subsets", m, pre, g, fixed)
    assert sub is fixed and not set(sub[0].tolist()) <= set(sub[1].tolist())  # sizes match, not nested


def test_ground_truth_rebuild_matches_cache_and_parents_are_roots():
    cfg = generate.CONFIGS["toy"][0]
    model = generate.build_model(cfg, generate.SPLIT_OFFSET["dev"])
    t = nesting.truth(cfg, model["seed"], model)
    assert t["is_parent"].any() and t["is_child"].any()
    assert not (t["is_parent"] & t["is_child"]).any()
    s = nesting.score(model["true_dirs"], model)  # the planted directions themselves score perfectly
    assert s["mmcs"] > 0.999 and s["parent_mmcs"] > 0.999 and s["child_mmcs"] > 0.999


def test_freeze_is_write_once_and_confirm_needs_a_matching_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(nesting, "OUT", tmp_path)
    cells = {}
    for a in nesting.ARMS:
        for hp in nesting.grid_for(a):
            for s in nesting.CAL_SEEDS:
                v = 0.5 + 0.01 * hp["k"] + (0.001 if a == "nested" else 0)
                cells[nesting.cell_key(a, hp, s)] = {"mmcs": v, "child_mmcs": v, "parent_mmcs": v, "fve": v}
    (tmp_path / "calibration.json").write_text(json.dumps({"cells": cells}))
    nesting.freeze()
    p = json.loads((tmp_path / "protocol.json").read_text())
    assert p["primary_k"] == 3 and set(p["frozen_hparams"]) == set(nesting.ARMS)
    with pytest.raises(SystemExit):
        nesting.freeze()
    with pytest.raises(FileNotFoundError):
        nesting.confirm()
    f = json.loads((tmp_path / "freeze.json").read_text())
    (tmp_path / "registration-receipt.json").write_text(json.dumps(
        {"slug": "x", "revision": 1, "protocol_sha256": "0" * 64, "source_freeze_sha256": nesting.sha(tmp_path / "freeze.json")}))
    with pytest.raises(AssertionError, match="receipt"):
        nesting.confirm()
    assert f["confirmation_accessed"] is False
