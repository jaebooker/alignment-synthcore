"""Adaption-hosted model client for the control_hosted testbed. Standard library + `adaption`.

* Same call shape as testbeds/control/llm.py (ask(prompt, model, system, max_tokens, ...)), so
  the control prompts and monitor methods carry over unchanged.
* Every response is cached by a hash of the full request in results/_cache/control_hosted/,
  with its token usage and the credit cost the API reported. A gzipped snapshot ships in
  data/llm_cache.jsonl.gz, so scoring needs no key and costs nothing.
* Live calls only happen when CONTROL_LIVE=1; otherwise a cache miss raises CacheMiss.
* A ledger refuses live calls once spend reaches CONTROL_CREDIT_CAP (default 150 credits).
* Calls stream, which avoids the gateway's timeout on long reasoning outputs.

The key is read from ADAPTION_API_KEY or the file named by ADAPTION_KEY_FILE. Never logged.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE_DIR = ROOT / "results" / "_cache" / "control_hosted"
CACHE_FILE = CACHE_DIR / "llm_cache.jsonl"
LEDGER = CACHE_DIR / "spend.json"
SHIPPED = HERE / "data" / "llm_cache.jsonl.gz"
CAP = float(os.environ.get("CONTROL_CREDIT_CAP", "150"))

UNTRUSTED = "qwen3.5-397b"
TRUSTED = "gemma-4-31b-it"
# Trusted monitors a method may call. Fixed before any monitor was scored (prereg, 2026-11-10).
# Untrusted is Qwen 3.5 397B; Qwen 3.8 27B is the one same-family monitor. (GLM 5.2 was tried
# first: with Testbed 6's attack prompt it reasoned past 32k tokens on 5 of 6 attempts.)
POOL = ("gemma-4-31b-it", "gemma-4-26b-a4b-it", "qwen3.8-27b", "deepseek-v4-flash", "glm-53-flash")

STATS = {"live": 0, "truncated": 0}
_lock = threading.Lock()
_cache: dict | None = None
_client = None


class CacheMiss(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def models() -> tuple[str, str]:
    return UNTRUSTED, TRUSTED


def spent() -> float:
    try:
        return float(json.loads(LEDGER.read_text())["credits"])
    except (OSError, ValueError, KeyError):
        return 0.0


# names used by the control collector's progress lines
spent_usd = spent


def cap_usd() -> float:
    return CAP


def request_key(model, system, prompt, max_tokens, temperature, sample) -> str:
    blob = json.dumps({"model": model, "system": system, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temperature, "sample": sample}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        for path in (SHIPPED, CACHE_FILE):
            if path.exists():
                op = gzip.open if path.suffix == ".gz" else open
                with op(path, "rt") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        _cache[r["key"]] = r
    return _cache


def _client_():
    global _client
    if _client is None:
        if not os.environ.get("ADAPTION_API_KEY") and os.environ.get("ADAPTION_KEY_FILE"):
            os.environ["ADAPTION_API_KEY"] = Path(os.environ["ADAPTION_KEY_FILE"]).read_text().strip()
        from adaption import Adaption
        _client = Adaption()
    return _client


def _call(model, system, prompt, max_tokens, temperature):
    st = _client_().with_options(timeout=float(os.environ.get("CONTROL_CALL_TIMEOUT", "600"))).responses.create(
        model=model, instructions=system or None, input=prompt, max_output_tokens=max_tokens,
        temperature=temperature, store=False, stream=True)
    parts, resp = [], None
    for ev in st:
        et = getattr(ev, "type", "")
        if et == "response.output_text.delta":
            parts.append(ev.delta)
        elif et in ("response.completed", "response.incomplete"):
            resp = ev.response
    u = getattr(resp, "usage", None)
    usage = {"in": getattr(u, "input_tokens", 0) or 0, "out": getattr(u, "output_tokens", 0) or 0,
             "cost": float(getattr(u, "cost", 0) or 0)}
    status = getattr(resp, "status", None) or "unknown"
    return "".join(parts), usage, status


def lookup(prompt, model, system="", max_tokens=4000, temperature=1.0, sample=0) -> dict | None:
    return _load().get(request_key(model, system, prompt, max_tokens, temperature, sample))


def ask_full(prompt: str, model: str, system: str = "", max_tokens: int = 4000, temperature: float = 1.0,
             sample: int = 0, tag: str = "") -> dict:
    """Return the cache record {"text", "usage": {"in", "out", "cost"}, "status", ...}."""
    key = request_key(model, system, prompt, max_tokens, temperature, sample)
    with _lock:
        hit = _load().get(key)
    if hit is not None:
        return hit
    if os.environ.get("CONTROL_LIVE") != "1":
        raise CacheMiss(f"no cached response for request {key[:12]} ({tag}); collect with CONTROL_LIVE=1")
    with _lock:
        if spent() >= CAP:
            raise BudgetExceeded(f"credit cap {CAP} reached (spent {spent():.3f})")
    err = None
    for attempt in range(6):
        try:
            text, usage, status = _call(model, system, prompt, max_tokens, temperature)
            if not text.strip() and status != "incomplete" and attempt < 2:
                err = "empty output"
                time.sleep(2)
                continue
            err = None
            break
        except Exception as e:  # rate limits, gateway errors
            err = repr(e)[:300]
            time.sleep(min(60, 3 * 2 ** attempt))
    if err:
        raise RuntimeError(f"{tag}: {err}")
    rec = {"key": key, "model": model, "tag": tag, "sample": sample, "max_tokens": max_tokens,
           "text": text, "usage": usage, "status": status, "t": round(time.time(), 1)}
    with _lock:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _load()[key] = rec
        with open(CACHE_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        LEDGER.write_text(json.dumps({"credits": spent() + usage["cost"], "cap": CAP}))
        STATS["live"] += 1
        STATS["truncated"] += status == "incomplete"
    return rec


def ask(prompt: str, model: str, system: str = "", max_tokens: int = 4000, temperature: float = 1.0,
        sample: int = 0, tag: str = "", effort=None) -> str:
    """`effort` is accepted for call compatibility with the Anthropic client and ignored."""
    return ask_full(prompt, model, system, max_tokens, temperature, sample, tag)["text"]
