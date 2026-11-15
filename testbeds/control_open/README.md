# Testbed 6b: control_open

AI control (trusted monitoring of backdoored code) on hosted open models, with independent
test shards. Protocol and prompts are those of `testbeds/control` v0.4; see `testbed.py`.

- Models: untrusted Qwen 3.5 397B, trusted Gemma 4 31B, via Adaption inference (`llm.py`).
  GLM 5.2 was tried first as the untrusted model and returned no code on 27 of 32 backdoor
  attempts, so it was replaced before any monitor ran.
- Problems: `build_problems.py` builds `data/problems.json` from HumanEval and sanitized MBPP
  (590 with a passing reference; 110 dev, 480 test in 8 shards of 60).
- Outputs: `collect.py` generates and grades honest solutions, two backdoor attempts and
  sneakiness ratings, and writes `data/outputs.json`. Model-written code runs only in the
  cloud sandbox (`testbeds/control/execute.py`). Every call is cached in
  `data/llm_cache.jsonl.gz`, so rerunning costs nothing and reproduces the same transcripts.
- Stage 2 seed s scores test shard s. There are exactly 8 seeds.

Run: `ADAPTION_KEY_FILE=... CONTROL_OPEN_STREAM=1 python run.py ideas/exploit-hunting-monitor-open --resume`
Offline re-scoring from the cache: add `CONTROL_OPEN_OFFLINE=1`.
Pre-registration: `prereg/2026-11-11_control_open.md`.
