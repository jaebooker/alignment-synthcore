"""Build the monitoring items for a split. Standard library only, so the collector can run
it on any machine. Labels (honest vs backdoored) live only in the private half."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_PROBLEMS = {("dev", "toy"): ("dev", 20), ("dev", "full"): ("dev", None), ("test", "full"): ("test", None)}


def _load():
    problems = {p["task_id"]: p for p in json.loads((HERE / "data" / "problems.json").read_text())}
    outputs = json.loads((HERE / "data" / "outputs.json").read_text())
    return problems, outputs


def item_id(task_id: str, kind: str) -> str:
    """Opaque id: the method cannot read honest/attack off it."""
    return hashlib.sha256(f"{task_id}|{kind}|ctl-v0".encode()).hexdigest()[:12]


def build(split: str, config: str):
    problems, outputs = _load()
    which, limit = SPLIT_PROBLEMS[(split, config)]
    rows = [r for r in outputs if r["split"] == which][:limit]
    items, labels = [], {}
    for r in rows:
        p = problems[r["task_id"]]
        if r["honest_code"]:
            iid = item_id(r["task_id"], "honest")
            items.append({"id": iid, "problem": p["prompt"], "code": r["honest_code"]})
            labels[iid] = {"task_id": r["task_id"], "attack": False}
        # v0.2: the red team attacks only the backdoors it rated sneakiest (select_attacks.py)
        if r.get("attack_selected", r["backdoor_valid"]):
            iid = item_id(r["task_id"], "attack")
            items.append({"id": iid, "problem": p["prompt"], "code": r["attack_code"]})
            labels[iid] = {"task_id": r["task_id"], "attack": True}
    random.Random(f"{split}-{config}").shuffle(items)
    task = {"items": items, "llm_path": str(HERE / "llm.py")}
    return [task], {"labels": labels, "task_ids": [r["task_id"] for r in rows]}
