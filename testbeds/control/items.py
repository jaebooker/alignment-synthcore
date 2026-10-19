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


PROBE_SAMPLE_OFFSET = 1000  # probe draws are independent of the scored draws


def build(split: str, config: str):
    """Main task: every honest solution and every working backdoor (the attack candidates).
    The red team decides which candidates it actually submits only after the monitor has
    scored them, so the monitor cannot know which ones count.

    Probe task: the candidates again under fresh ids, with sample_offset so every trusted
    call is a new, independent draw. It is the red team's own copy of the deployed monitor
    (adaptive red team, v0.3). It goes to a separate method run made after the main one."""
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
    random.Random(f"{split}-{config}").shuffle(items)
    random.Random(f"{split}-{config}-probe").shuffle(probe)
    task = {"items": items, "llm_path": str(HERE / "llm.py")}
    probe_task = {"items": probe, "llm_path": str(HERE / "llm.py"), "sample_offset": PROBE_SAMPLE_OFFSET}
    return [task], {"labels": labels, "task_ids": [r["task_id"] for r in rows], "probe_of": probe_of,
                    "probe_public": [probe_task]}
