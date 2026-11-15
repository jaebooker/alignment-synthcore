"""Public helpers for monitor methods on control_open. Standard library only.

A method gets one task: {"items": [{"id", "problem", "code"}, ...], "llm_path": ...}.
Some items are honest solutions, some contain a working backdoor; nothing says which.
It returns [{"scores": {item_id: suspicion float}}]: higher = more suspicious.

Each method is also run a second time on a probe task (the red team querying its own copy
of your monitor). ask_trusted() shifts sample numbers for that run automatically.

The only model a method may call is the trusted one, via ask_trusted() or ask_trusted_many().
Calls are cached by request, so keep prompts deterministic and do not vary them with `seed`.
For repeated draws, pass sample=0,1,2... The trusted model reasons before answering and
reasoning counts against max_tokens: keep it at 2000 or more.
"""
from __future__ import annotations

import importlib.util
import re
from concurrent.futures import ThreadPoolExecutor

_llm = None


def _client(task: dict):
    global _llm
    if _llm is None:
        spec = importlib.util.spec_from_file_location("control_open_llm", task["llm_path"])
        _llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_llm)
    return _llm


def ask_trusted(task: dict, prompt: str, system: str = "", max_tokens: int = 2000, sample: int = 0,
                effort: str = "low") -> str:
    llm = _client(task)
    sample += int(task.get("sample_offset", 0))
    try:
        return llm.ask(prompt, model=llm.models()[1], system=system, max_tokens=max_tokens, sample=sample, tag="monitor")
    except RuntimeError as e:
        if "credit cap" in str(e):
            raise
        return ""  # a call that fails after retries reads as an empty answer


def ask_trusted_many(task: dict, calls: list, workers: int = 16) -> list:
    """calls: list of dicts of ask_trusted keyword arguments (prompt, system, max_tokens, sample).
    Returns the answers in order. Same calls and cache as ask_trusted, run concurrently."""
    _client(task)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda kw: ask_trusted(task, **kw), calls))


def last_number(text: str, default: float = 0.0) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return float(nums[-1]) if nums else default
