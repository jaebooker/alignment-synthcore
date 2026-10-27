"""Recompute a stored record's verdict under the current rule, from its stored per-seed results."""
from __future__ import annotations

from .completion import compare
from .stats import paired_test
from .verdict import VERDICT_RULE_VERSION, compute_verdict


def reverdict(r: dict) -> dict:
    """{"verdict", "flags", "completion", "reason"} under the current rule. Does not modify r."""
    if r.get("verdict_rule_version") == VERDICT_RULE_VERSION:
        s2 = r.get("stage2") or {}
        return {"verdict": r["verdict"], "flags": r.get("flags", []), "completion": s2.get("completion"),
                "reason": r.get("reason")}
    s2 = r.get("stage2")
    if r.get("error") or not s2 or not r.get("stage1") or not s2.get("per_seed"):
        return {"verdict": r["verdict"], "flags": r.get("flags", []), "completion": None, "reason": r.get("reason")}
    c = compare(r["testbed"], [p["method"] for p in s2["per_seed"]], [p["baseline"] for p in s2["per_seed"]],
                paired_test)
    v = compute_verdict(r["stage1"], {**s2, "completion": c}, None)
    return {"verdict": v["verdict"], "flags": v["flags"], "completion": c, "reason": v["reason"]}
