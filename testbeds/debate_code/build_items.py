"""Build data/items.json from control_open's problems and graded outputs.

One item per problem: a submission that is either correct (the untrusted model's honest
solution, which passes the hidden tests) or incorrect (its working backdoor: passes the tests,
returns a wrong answer on at least one valid input). Which one is drawn by a fixed coin per
problem, so labels are balanced and independent of the split. Problems with only one kind
available use it. dev/test split and shards are inherited from control_open.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "control_open" / "data"


def coin(task_id: str, salt: str) -> bool:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).digest()[0] % 2 == 0


def main():
    problems = {p["task_id"]: p for p in json.loads((SRC / "problems.json").read_text())}
    items = []
    for o in json.loads((SRC / "outputs.json").read_text()):
        p = problems[o["task_id"]]
        can_correct, can_wrong = bool(o["honest_pass"]), bool(o["backdoor_valid"])
        if not (can_correct or can_wrong):
            continue
        correct = coin(o["task_id"], "label") if (can_correct and can_wrong) else can_correct
        items.append({
            "id": o["task_id"], "split": p["split"], "shard": p["shard"], "source": p["source"],
            "problem": p["prompt"], "entry_point": p["entry_point"],
            "code": o["honest_code"] if correct else o["attack_code"], "correct": correct,
            # Which side a lone consultant argues: drawn independently of the truth.
            "consultant_side": "pro" if coin(o["task_id"], "consultant") else "con",
        })
    (HERE / "data" / "items.json").write_text(json.dumps(items, indent=1))
    print(len(items), "items", Counter((i["split"], i["correct"]) for i in items))
    print("per test shard (n, correct):", [(s, sum(i["shard"] == s and i["split"] == "test" for i in items),
          sum(i["shard"] == s and i["split"] == "test" and i["correct"] for i in items)) for s in range(8)])
    print("consultant argues the true side:", sum((i["consultant_side"] == "pro") == i["correct"] for i in items), "of", len(items))


if __name__ == "__main__":
    main()
