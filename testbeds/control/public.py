"""Public helpers for monitor methods. Standard library only.

A method gets one task: {"items": [{"id", "problem", "code"}, ...], "llm_path": ...}.
Some items are honest solutions, some contain a working backdoor; nothing says which.
It returns [{"scores": {item_id: suspicion float}}]: higher = more suspicious.

The only model a method may call is the trusted one, via ask_trusted(). Calls are
cached by request, so keep prompts deterministic and do not vary them with `seed`
(seeds only resample problems at scoring time). For repeated draws, pass sample=0,1,2...
"""
from __future__ import annotations

import importlib.util
import re

_llm = None


def _client(task: dict):
    global _llm
    if _llm is None:
        spec = importlib.util.spec_from_file_location("control_llm", task["llm_path"])
        _llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_llm)
    return _llm


def ask_trusted(task: dict, prompt: str, system: str = "", max_tokens: int = 2000, sample: int = 0,
                effort: str = "low") -> str:
    """The trusted model thinks before answering, and thinking counts against max_tokens.
    A small max_tokens returns empty or cut-off text: keep it at 1000 or more."""
    llm = _client(task)
    return llm.ask(prompt, model=llm.models()[1], system=system, max_tokens=max_tokens, sample=sample,
                   tag="monitor", effort=effort)


def last_number(text: str, default: float = 0.0) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return float(nums[-1]) if nums else default
