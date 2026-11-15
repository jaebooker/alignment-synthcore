"""Instances for debate_code: what a protocol method may see, and the hidden labels."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_SHARDS = 8
TOY = 40


def load():
    items = json.loads((HERE / "data" / "items.json").read_text())
    turns = {r["id"]: r["turns"] for r in json.loads((HERE / "data" / "transcripts.json").read_text())}
    return [dict(i, turns=turns[i["id"]]) for i in items]


def rows_for(split, config, seed, rows=None):
    rows = load() if rows is None else rows
    if split == "dev":  # scored whole; dev scores do not vary with the seed
        dev = [r for r in rows if r["split"] == "dev"]
        return dev[:TOY] if config == "toy" else dev
    if not 0 <= seed < N_SHARDS:
        raise ValueError(f"debate_code has {N_SHARDS} test shards; seed {seed} is out of range")
    return [r for r in rows if r["split"] == "test" and r["shard"] == seed]


def build(split, config, seed):
    rows = rows_for(split, config, seed)
    # Public: the problem, all four debater turns with their verified evidence, and the side a
    # lone consultant was assigned. Never the code and never the label.
    public = [{"items": [{"id": r["id"], "problem": r["problem"], "entry_point": r["entry_point"],
                          "turns": r["turns"], "consultant_side": r["consultant_side"]} for r in rows],
               "llm_path": str(HERE / "llm.py")}]
    private = {"labels": {r["id"]: {"correct": r["correct"], "consultant_right": (r["consultant_side"] == "pro") == r["correct"]}
                          for r in rows}}
    return public, private
