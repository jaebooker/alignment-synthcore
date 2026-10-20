"""Minimal Anthropic Messages API client. Standard library only (runs on Python 3.10+).

* Reads ANTHROPIC_API_KEY (and optional workspace / model / spend settings) from the repo's .env.
  The key is never printed or logged.
* Every response is cached in results/_cache/control/llm_cache.jsonl, keyed by a hash
  of the full request. Re-running anything costs nothing once cached. A gzipped snapshot
  ships in data/llm_cache.jsonl.gz so anyone can re-score without a key.
* Live calls only happen when CONTROL_LIVE=1. Otherwise a cache miss raises CacheMiss,
  so scoring runs in the cloud workspace without a key.
* A shared ledger (results/_cache/control/spend.json) refuses any call whose worst-case
  cost would push total spend past min(MAX_SPEND_USD, HARD_CAP_USD).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "_cache" / "control"
CACHE_FILE = CACHE_DIR / "llm_cache.jsonl"
LEDGER = CACHE_DIR / "spend.json"
HARD_CAP_USD = 7.0  # agreed budget for this testbed (raised from $5, 2026-10-20); MAX_SPEND_USD can only lower it

# USD per million tokens (input, output), from the public pricing page, 2026-10-18
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}


ATTEMPTS = 9


def _backoff(attempt: int) -> int:
    """2, 4, 8, 16, 32, 60, 60, 60 seconds: rides out roughly five minutes of outage."""
    return min(60, 2 ** (attempt + 1))


class CacheMiss(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def _env() -> dict:
    vals = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    vals.update({k: v for k, v in os.environ.items() if k in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID",
                                                               "UNTRUSTED_MODEL", "TRUSTED_MODEL",
                                                               "MAX_SPEND_USD")})
    return vals


def models() -> tuple[str, str]:
    e = _env()
    return e.get("UNTRUSTED_MODEL") or "claude-sonnet-5-5", e.get("TRUSTED_MODEL") or "claude-haiku-5-5"


def cap_usd() -> float:
    try:
        return min(HARD_CAP_USD, float(_env().get("MAX_SPEND_USD") or HARD_CAP_USD))
    except ValueError:
        return HARD_CAP_USD


def spent_usd() -> float:
    return json.loads(LEDGER.read_text())["total_usd"] if LEDGER.exists() else 0.0


def _record(model: str, usage: dict, cost: float, tag: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"total_usd": 0.0, "by_tag": {}, "calls": 0}
    led["total_usd"] += cost
    led["calls"] += 1
    led["by_tag"][tag] = led["by_tag"].get(tag, 0.0) + cost
    LEDGER.write_text(json.dumps(led, indent=2))


def _ssl_context():
    """Default SSL context, falling back to certifi's CA bundle when it is installed."""
    import ssl
    try:
        import certifi  # optional
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


STATS = {"live": 0, "truncated": 0}  # live calls this process, and how many hit max_tokens or came back empty

_cache = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        shipped = Path(__file__).resolve().parent / "data" / "llm_cache.jsonl.gz"
        if shipped.exists():  # responses committed with the repo: re-scoring needs no API key
            import gzip
            with gzip.open(shipped, "rt") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        _cache[r["key"]] = r["text"]
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    _cache[r["key"]] = r["text"]
    return _cache


def request_key(model, system, prompt, max_tokens, temperature, sample, effort=None) -> str:
    parts = [model, system, prompt, max_tokens, temperature, sample]
    if effort is not None:  # appended only when set, so keys of earlier requests stay valid
        parts.append(effort)
    blob = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def ask(prompt: str, model: str, system: str = "", max_tokens: int = 1024, temperature: float | None = None,
        sample: int = 0, tag: str = "misc", effort: str | None = None) -> str:
    """One completion. `sample` distinguishes repeated draws of the same request.

    temperature is not sent unless given: newer models reject it. Reproducibility comes
    from the cache, not from sampling settings.

    Newer models think before answering and thinking tokens count against max_tokens, so
    keep max_tokens generous. effort ("low"/"medium"/"high") sets how much they think."""
    key = request_key(model, system, prompt, max_tokens, temperature, sample, effort)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    if os.environ.get("CONTROL_LIVE") != "1":
        raise CacheMiss(f"no cached response for request {key[:12]} (run the collect step with CONTROL_LIVE=1)")
    pin, pout = PRICES.get(model, (10.0, 50.0))
    worst = (len(system) + len(prompt)) / 3 / 1e6 * pin + max_tokens / 1e6 * pout
    if spent_usd() + worst > cap_usd():
        raise BudgetExceeded(f"spent ${spent_usd():.3f}; next call could cost ${worst:.4f}; cap ${cap_usd():.2f}")
    api_key = _env().get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from .env")
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    if effort is not None:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": effort}
    if system:
        body["system"] = system
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    workspace = _env().get("ANTHROPIC_WORKSPACE_ID")
    if workspace:  # needed for keys not scoped to a single workspace
        headers["anthropic-workspace-id"] = workspace
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers=headers)
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < ATTEMPTS - 1:
                time.sleep(_backoff(attempt))
                continue
            body = e.read()[:300]
            hint = ""
            if e.code == 401 and not body.startswith(b"{"):
                hint = (" (a network proxy rejected the request before it reached Anthropic; "
                        "run this from a normal terminal, not a sandboxed shell)")
            raise RuntimeError(f"API error {e.code}: {body!r}{hint}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as e:  # incl. IncompleteRead
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise RuntimeError("SSL certificates missing. On a python.org Mac install, run "
                                   "'Install Certificates.command' from the Python folder in Applications, "
                                   "or 'pip3 install certifi'.") from None
            if attempt < ATTEMPTS - 1:  # transient network failure (reset, timeout): back off and retry
                print(f"  network error ({e}); retry {attempt + 1}/{ATTEMPTS - 1} in {_backoff(attempt)}s", flush=True)
                time.sleep(_backoff(attempt))
                continue
            raise RuntimeError(f"network error after {ATTEMPTS} attempts: {e}. Re-run to resume (cached calls are free). Check that "
                               "'curl -sI https://api.anthropic.com' works from this terminal.") from None
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage", {})
    cost = u.get("input_tokens", 0) / 1e6 * pin + u.get("output_tokens", 0) / 1e6 * pout
    _record(model, u, cost, tag)
    STATS["live"] += 1
    if data.get("stop_reason") == "max_tokens" or not text.strip():
        STATS["truncated"] += 1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps({"key": key, "model": model, "tag": tag, "effort": effort, "stop": data.get("stop_reason"),
                            "usage": u, "text": text}) + "\n")
    cache[key] = text
    return text
