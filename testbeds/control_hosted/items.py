"""Build the monitoring items for a split (same construction as testbeds/control/items.py, on this
testbed's own outputs). Standard library only. Labels live only in the private half."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_PROBLEMS = {("dev", "toy"): ("dev", 20), ("dev", "full"): ("dev", None), ("test", "full"): ("test", None)}
PROBE_SAMPLE_OFFSET = 1000  # probe draws are independent of the scored draws


def _load():
    problems = {p["task_id"]: p for p in json.loads((HERE.parent / "control" / "data" / "problems.json").read_text())}
    outputs = json.loads((HERE / "data" / "outputs.json").read_text())
    return problems, outputs


def reference() -> list:
    path = HERE / "data" / "reference.json"
    return json.loads(path.read_text()) if path.exists() else []


def item_id(task_id: str, kind: str) -> str:
    """Opaque id: the method cannot read honest/attack off it."""
    return hashlib.sha256(f"{task_id}|{kind}|ctlh-v1".encode()).hexdigest()[:12]


def build(split: str, config: str):
    """Main task: every honest solution and every working backdoor (the attack candidates).
    Probe task: the candidates again under fresh ids and independent sample numbers, for the
    adaptive red team's own copy of the monitor. See testbeds/control/items.py."""
    from . import llm
    problems, outputs = _load()
    which, limit = SPLIT_PROBLEMS[(split, config)]
    rows = [r for r in outputs if r["split"] == which][:limit]
    items, probe, labels, probe_of = [], [], {}, {}
    for r in rows:
        p = problems[r["task_id"]]
        if r["honest_code"]:
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
    random.Random(f"h-{split}-{config}").shuffle(items)
    random.Random(f"h-{split}-{config}-probe").shuffle(probe)
    common = {"llm_path": str(HERE / "llm.py"), "trusted_pool": list(llm.POOL), "reference": reference()}
    task = {"items": items, **common}
    probe_task = {"items": probe, **common, "sample_offset": PROBE_SAMPLE_OFFSET}
    return [task], {"labels": labels, "task_ids": [r["task_id"] for r in rows], "probe_of": probe_of,
                    "probe_public": [probe_task]}
