"""Build data/problems.json from HumanEval (164) and sanitized MBPP (427).

    python -m testbeds.control_open.build_problems HumanEval.jsonl.gz sanitized-mbpp.json

Sources: github.com/openai/human-eval (MIT) and github.com/google-research/google-research
mbpp/sanitized-mbpp.json (CC BY 4.0). A problem is kept if its reference solution passes its
own tests in the sandbox. Problems are shuffled once (fixed seed), stratified by source:
the test split is 8 disjoint shards of 60 problems (one per Stage 2 seed); the rest is dev.
"""
from __future__ import annotations

import ast
import gzip
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from testbeds.control import execute

HERE = Path(__file__).resolve().parent
N_SHARDS, SHARD_SIZE = 8, 60


def humaneval(path):
    for l in gzip.open(path, "rt"):
        r = json.loads(l)
        yield {"task_id": r["task_id"], "source": "humaneval", "prompt": r["prompt"], "entry_point": r["entry_point"],
               "canonical": r["prompt"] + r["canonical_solution"], "test": r["test"]}


def mbpp(path):
    for r in json.load(open(path)):
        try:
            tree = ast.parse(r["code"])
        except SyntaxError:
            continue
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        called = set(re.findall(r"([A-Za-z_]\w*)\s*\(", "\n".join(r["test_list"])))
        entry = [d for d in defs if d.name in called]
        if len(entry) != 1:
            continue
        d = entry[0]
        sig = f"def {d.name}({ast.unparse(d.args)}):"
        prompt = f'{sig}\n    """{r["prompt"].strip()}\n\n    Example:\n    {r["test_list"][0].strip()}\n    """\n'
        test = "\n".join(r.get("test_imports") or []) + "\n\ndef check(candidate):\n" + \
            "\n".join("    " + t.strip() for t in r["test_list"]) + "\n"
        yield {"task_id": f"MBPP/{r['task_id']}", "source": "mbpp", "prompt": prompt, "entry_point": d.name,
               "canonical": "\n".join(r.get("test_imports") or []) + "\n" + r["code"], "test": test}


def main():
    probs = list(humaneval(sys.argv[1])) + list(mbpp(sys.argv[2]))
    with ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(lambda p: execute.passes_tests(p["canonical"], p["test"], p["entry_point"]), probs))
    kept = [p for p, o in zip(probs, ok) if o]
    print(f"{len(probs)} parsed, {len(kept)} with a passing reference "
          f"({sum(p['source'] == 'humaneval' for p in kept)} HumanEval, {sum(p['source'] == 'mbpp' for p in kept)} MBPP)")
    rng = random.Random("control-open-v1")
    by = {s: [p for p in kept if p["source"] == s] for s in ("humaneval", "mbpp")}
    for v in by.values():
        rng.shuffle(v)
    n_test = N_SHARDS * SHARD_SIZE
    assert len(kept) >= n_test + 80, "too few problems for a dev split"
    # per-shard quota of each source, proportional to its share
    q_he = round(SHARD_SIZE * len(by["humaneval"]) / len(kept))
    for s in range(N_SHARDS):
        for src, q in (("humaneval", q_he), ("mbpp", SHARD_SIZE - q_he)):
            for _ in range(q):
                p = by[src].pop()
                p["split"], p["shard"] = "test", s
    for v in by.values():
        for p in v:
            p["split"], p["shard"] = "dev", None
    kept.sort(key=lambda p: (p["split"], -1 if p["shard"] is None else p["shard"], p["task_id"]))
    (HERE / "data" / "problems.json").write_text(json.dumps(kept, indent=1))
    print({k: sum(p["split"] == k for p in kept) for k in ("dev", "test")},
          "dev by source", {s: sum(p["split"] == "dev" and p["source"] == s for p in kept) for s in by})


if __name__ == "__main__":
    main()
