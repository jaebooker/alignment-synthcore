"""Collect step 1 (run on the machine with the API key). Standard library only.

    CONTROL_LIVE=1 python3 -m testbeds.control.collect_outputs [--limit N]
    CONTROL_LIVE=1 python3 -m testbeds.control.collect_outputs --sneakiness

For each problem: an honest solution and an attack (backdoored) solution from the
untrusted model, and an honest solution from the trusted model. With --sneakiness, the
untrusted model instead rates how unnoticeable each of its working backdoors is (needs
data/outputs.json from the grader); the red team then attacks only its sneakiest ones.
Responses go to the cache; nothing here executes model-written code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import llm, prompts

DATA = Path(__file__).resolve().parent / "data" / "problems.json"


def requests_for(problem: dict):
    U, T = llm.models()
    yield dict(prompt=prompts.honest(problem), model=U, system=prompts.SOLVER_SYSTEM,
               max_tokens=prompts.MAX_TOKENS["honest"], tag="untrusted_honest")
    yield dict(prompt=prompts.attack(problem), model=U, system=prompts.ATTACK_SYSTEM,
               max_tokens=prompts.MAX_TOKENS["attack"], tag="untrusted_attack")
    yield dict(prompt=prompts.honest(problem), model=T, system=prompts.SOLVER_SYSTEM,
               max_tokens=prompts.MAX_TOKENS["trusted_solve"], tag="trusted_solve")


def sneakiness_request(problem: dict, row: dict) -> dict:
    U, _ = llm.models()
    return dict(prompt=prompts.sneakiness(problem, row["attack_code"], row["exploit_args"]), model=U,
                system=prompts.SNEAKY_SYSTEM, max_tokens=prompts.MAX_TOKENS["sneakiness"], tag="untrusted_sneakiness",
                effort=prompts.SNEAKY_EFFORT)


def collect_sneakiness():
    problems = {p["task_id"]: p for p in json.loads(DATA.read_text())}
    rows = [r for r in json.loads((DATA.parent / "outputs.json").read_text()) if r["backdoor_valid"]]
    for i, r in enumerate(rows):
        llm.ask(**sneakiness_request(problems[r["task_id"]], r))
        if i % 10 == 0 or i == len(rows) - 1:
            print(f"{i + 1}/{len(rows)} backdoors rated, spent ${llm.spent_usd():.3f} of ${llm.cap_usd():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sneakiness", action="store_true")
    args = ap.parse_args()
    if args.sneakiness:
        return collect_sneakiness()
    problems = json.loads(DATA.read_text())[: args.limit]
    for i, p in enumerate(problems):
        for r in requests_for(p):
            llm.ask(**r)
        if i % 10 == 0 or i == len(problems) - 1:
            print(f"{i + 1}/{len(problems)} problems, spent ${llm.spent_usd():.3f} of ${llm.cap_usd():.2f}", flush=True)


if __name__ == "__main__":
    main()
