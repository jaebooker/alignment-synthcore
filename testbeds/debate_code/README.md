# Testbed 8: debate_code

Debate about hidden code, judged by a weaker model. See `testbed.py` for the design.

- Items: `build_items.py` draws one submission per control_open problem, correct or a working
  backdoor, by a fixed coin (560 items; 102 dev, 458 test in 8 shards).
- Debaters: Qwen 3.5 397B. `collect.py` generates an opening and a rebuttal per side and runs
  the executions each debater requests on the hidden code (`evidence.py`, cloud sandbox only).
- Judge: Gemma 4 31B, called by protocol methods through `public.py`. It never sees the code.
- Baseline: consultancy (`baseline_consultancy.py`).
- `reference.py`: two arms outside the harness in which the judge does read the code.
- Every call is cached in `data/llm_cache.jsonl.gz`; `DEBATE_CODE_OFFLINE=1` re-scores without a key.

Run: `ADAPTION_KEY_FILE=... DEBATE_CODE_STREAM=1 python run.py ideas/debate-weak-judge --resume`
Pre-registration and outcomes: `prereg/2026-11-12_debate_code.md`.

Known limits: one debater model and one judge model; debate shows the judge more text and
evidence than two-sided openings, so that comparison is not budget-matched; debaters cannot see
execution results before the judge does; the adversarially persuasive debater is not built.
