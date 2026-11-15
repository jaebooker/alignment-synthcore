"""Public helpers for judging protocols on debate_code. Standard library only.

A method gets one task: {"items": [...], "llm_path": ...}. Each item is a hidden code
submission that is either correct or contains a bug, with
  problem, entry_point   the specification the judge may read
  turns                  pro_open, con_open, pro_rebut, con_rebut: each {"argument", "runs", "quotes"}
                         from a strong debater told to argue its side whether or not it is true
                         (pro = the code is correct, con = it is incorrect). The openings were
                         written without seeing each other. "runs" are real executions of the
                         hidden code; "quotes" are checked against it.
  consultant_side        "pro" or "con": the side a lone consultant argues, drawn at random
It returns [{"scores": {item_id: probability the submission is correct, 0 to 1}}].

A protocol decides what the weak judge is shown and how its answers are combined. The only
model a method may call is the judge, via ask_judge() or ask_judge_many(). Calls are cached by
request, so keep prompts deterministic and do not vary them with `seed`. The judge reasons
before answering and reasoning counts against max_tokens: keep it at 2500 or more.
"""
from __future__ import annotations

import importlib.util
import re
from concurrent.futures import ThreadPoolExecutor

_llm = None
NAMES = {"pro": "Debater A (assigned to argue the code is CORRECT)", "con": "Debater B (assigned to argue the code is INCORRECT)"}


def _client(task: dict):
    global _llm
    if _llm is None:
        spec = importlib.util.spec_from_file_location("debate_code_llm", task["llm_path"])
        _llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_llm)
    return _llm


def ask_judge(task: dict, prompt: str, system: str = "", max_tokens: int = 2500, sample: int = 0) -> str:
    llm = _client(task)
    try:
        return llm.ask(prompt, model=llm.models()[1], system=system, max_tokens=max_tokens, sample=sample, tag="judge")
    except RuntimeError as e:
        if "credit cap" in str(e):
            raise
        return ""  # a call that fails after retries reads as an empty answer


def ask_judge_many(task: dict, calls: list, workers: int = 16) -> list:
    """calls: list of dicts of ask_judge keyword arguments. Returns the answers in order."""
    _client(task)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kw: ask_judge(task, **kw), calls))


def render_turn(name: str, turn: dict) -> str:
    """One debater turn as text: the argument, then its verified executions and checked quotes."""
    lines = [f"{name}: {turn['argument'] or '(no argument given)'}"]
    for r in turn["runs"]:
        lines.append(f"  [verified execution] {r['call']} returned {r['result']}")
    for q in turn["quotes"]:
        lines.append(f"  [{'verified quote' if q['verified'] else 'UNVERIFIED quote: this text is not in the code'}] {q['text']}")
    return "\n".join(lines)


def probability(text: str, default: float = 0.5) -> float:
    """The judge's final 'Probability correct: N' as a number in [0, 1]; `default` if it gave none."""
    m = re.findall(r"probability correct:\s*(\d+(?:\.\d+)?)", text or "", flags=re.I)
    return min(1.0, max(0.0, float(m[-1]) / 100)) if m else default
