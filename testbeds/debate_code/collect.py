"""Generate the debate transcripts (run in the cloud workspace; needs ADAPTION_API_KEY).

    python -m testbeds.debate_code.collect [--limit N] [--workers W]

Per item the strong model writes four turns: an opening for each side, written without seeing
the other, then a rebuttal for each side after both openings and their verified evidence.
Each debater is told to argue its side whether or not it is true. Requested function calls
are executed on the hidden submission in the sandbox. Everything is cached, so a re-run is
free and resumes where it stopped. Writes data/transcripts.json when no call failed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import llm, prompts
from .evidence import MAX_RUNS, gather

HERE = Path(__file__).resolve().parent
MAX_TOKENS = 12000  # the strong model reasons before answering; reasoning counts against this.
# At 6000 the INCORRECT-side debater returned nothing on 2 of 7 correct pilot items after three draws.
FAILED: list = []
DONE: list = []


def ask(item, kind, side, transcript=""):
    tmpl = prompts.OPEN if kind == "open" else prompts.REBUT
    prompt = tmpl.format(problem=item["problem"], code=item["code"], side=prompts.SIDE[side], rules=prompts.RULES,
                         runs=MAX_RUNS[kind], transcript=transcript)
    text = ""
    for sample in range(3):  # an empty answer (all tokens spent on reasoning) is redrawn, at most twice
        try:
            text = llm.ask(prompt, model=llm.models()[0], system=prompts.SYSTEM, max_tokens=MAX_TOKENS,
                           sample=sample, tag=f"debater_{kind}_{side}")
        except RuntimeError as e:
            if "credit cap" in str(e):
                raise
            FAILED.append(str(e)[:120])
            return None
        if text.strip():
            break
    return gather(text, item["code"], item["entry_point"], kind)


def debate(item):
    turns = dict(zip(("pro_open", "con_open"), llm.pmap(lambda s: ask(item, "open", s), ["pro", "con"], workers=2)))
    if None not in turns.values():
        opening = "\n\n".join(prompts.render_turn(n, turns[k]) for n, k in
                              (("Debater A (argues CORRECT)", "pro_open"), ("Debater B (argues INCORRECT)", "con_open")))
        turns.update(zip(("pro_rebut", "con_rebut"),
                         llm.pmap(lambda s: ask(item, "rebut", s, opening), ["pro", "con"], workers=2)))
    DONE.append(1)
    if len(DONE) % 25 == 0:
        print(f"{len(DONE)} items done, spend {llm.spend():.2f}", flush=True)
    return {"id": item["id"], "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()
    items = json.loads((HERE / "data" / "items.json").read_text())
    if args.limit:
        items = [i for i in items if i["split"] == "dev"][:args.limit]
    rows = llm.pmap(debate, items, workers=args.workers)
    by = {i["id"]: i for i in items}
    stats = Counter()
    for r in rows:
        for k, t in r["turns"].items():
            if t is None:
                continue
            stats[f"{k}_turns"] += 1
            stats[f"{k}_empty"] += not t["argument"]
            stats[f"{k}_runs"] += len(t["runs"])
            stats[f"{k}_run_errors"] += sum("raised an error" in x["result"] for x in t["runs"])
            stats[f"{k}_unverified_quotes"] += sum(not q["verified"] for q in t["quotes"])
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    print("items", len(rows), "correct", sum(by[r["id"]]["correct"] for r in rows), "spend", round(llm.spend(), 2))
    if FAILED:
        print(f"{len(FAILED)} calls failed after retries (e.g. {FAILED[0]}); rerun to fill them. transcripts.json not written.")
    elif args.limit is None:
        (HERE / "data" / "transcripts.json").write_text(json.dumps(rows, indent=1))
    else:
        (HERE / "data" / "pilot_transcripts.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
