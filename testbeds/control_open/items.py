"""Build the monitoring items for a split. Labels live only in the private half.

dev/toy   the first 40 dev problems;   dev/full  all dev problems (seed ignored);
test/full the disjoint shard of 60 test problems whose index equals the seed (0..7).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_SHARDS = 8
TOY = 40
PROBE_SAMPLE_OFFSET = 1000  # probe draws are independent of the scored draws


def item_id(task_id: str, kind: str) -> str:
    return hashlib.sha256(f"{task_id}|{kind}|cto-v1".encode()).hexdigest()[:12]


def rows_for(split: str, config: str, seed: int, outputs: list) -> list:
    if split == "dev":
        rows = [r for r in outputs if r["split"] == "dev"]
        return rows[:TOY] if config == "toy" else rows
    if not 0 <= seed < N_SHARDS:
        raise ValueError(f"control_open has {N_SHARDS} test shards; seed {seed} has no shard")
    return [r for r in outputs if r["split"] == "test" and r["shard"] == seed]


def build(split: str, config: str, seed: int):
    problems = {p["task_id"]: p for p in json.loads((HERE / "data" / "problems.json").read_text())}
    outputs = json.loads((HERE / "data" / "outputs.json").read_text())
    rows = rows_for(split, config, seed, outputs)
    items, probe, labels, probe_of = [], [], {}, {}
    for r in rows:
        p = problems[r["task_id"]]
        if r["honest_code"] and r["honest_pass"]:
            iid = item_id(r["task_id"], "honest")
            items.append({"id": iid, "problem": p["prompt"], "code": r["honest_code"]})
            labels[iid] = {"task_id": r["task_id"], "attack": False}
        if r["backdoor_valid"]:
            iid = item_id(r["task_id"], "attack")
            items.append({"id": iid, "problem": p["prompt"], "code": r["attack_code"]})
            labels[iid] = {"task_id": r["task_id"], "attack": True, "sneakiness": r.get("sneakiness") or 0,
                           "static_selected": bool(r.get("attack_selected"))}
            pid = item_id(r["task_id"], "probe")
            probe.append({"id": pid, "problem": p["prompt"], "code": r["attack_code"]})
            probe_of[iid] = pid
    tag = f"{split}-{config}-{seed if split == 'test' else 0}"
    random.Random(tag).shuffle(items)
    random.Random(tag + "-probe").shuffle(probe)
    llm_path = str(HERE / "llm.py")
    return ([{"items": items, "llm_path": llm_path}],
            {"labels": labels, "probe_of": probe_of,
             "probe_public": [{"items": probe, "llm_path": llm_path, "sample_offset": PROBE_SAMPLE_OFFSET}]})
