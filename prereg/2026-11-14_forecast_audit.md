# Pre-registration: can models forecast which alignment ideas survive a planted-truth test?

Status: FROZEN 2026-11-14 03:08 UTC, approved by J. Hashes of this file's inputs are in
experiments/epistemics_sprint/frozen_sha256.txt, written before the first non-pilot call.
Sprint: Apart AI x Epistemics (Track 1). Everything in experiments/epistemics_sprint/ is new;
the catalogue outcomes (results/commons.json) and the Adaption client are pre-existing.

## Items
114 binary questions from 45 tested catalogue ideas in 11 testbed families (items.jsonl,
sha256 recorded at freeze). Outcomes were fixed by the harness between 2026-10-15 and 2026-11-12.
- gate (45, 53% yes): mechanism gate passed.
- win (45, 29% yes): verdict beats_baseline or safe_but_incomplete.
- sign (24, 79% yes): given Stage 2 ran, mean safety difference above zero.
Prompts carry the idea, source citation, testbed, metric, baseline and the registered gate
quantity and threshold. No verdicts,
no observed numbers.

## Forecasters and conditions
Seven hosted open models through Adaption (models.json: glm-5.2, glm-5.3-flash, qwen3.5-397b,
qwen3.8-27b, deepseek-v4-flash, gemma-4-31b-it, gemma-4-26b-a4b-it), 16,000 output-token cap, temperature 1,
3 samples per cell, item forecast = mean of parsed probabilities. No tools, no retrieval.
- blind: the prompt above.
- base_rate: plus the yes-rate for the same question on the other testbed families.
- no_source: the source citation removed.
Spend cap: 200 Adaption credits.

## Primary claims (question = win, pooled across models and per model)
- H1 optimism: mean forecast minus observed rate > 0 in blind.
- H2 discrimination: AUROC > 0.5 in blind.
- H3 no skill: Brier skill vs the leave-one-family-out base rate <= 0 in blind.
- H4 reference class helps: Brier(base_rate) < Brier(blind), paired by item.
- H5 citation cue: mean forecast(blind) > mean forecast(no_source), paired by item.
Each is judged by a 95% bootstrap interval that resamples testbed families (11 clusters, B = 2000).
gate and sign are secondary and reported the same way.

## Checks
- Controls: the two null-control ideas should get low win forecasts; reported separately.
- Leakage probe: each model is asked, with no context, what it knows of this repo's verdicts.
  Any model that reports a correct verdict is flagged and also analysed apart.
- Parse failures and API errors are reported per model; a model above 10% failures is dropped.
- Duplicate ideas (toy/LM, original/paraphrase testbed) sit in one cluster, so they do not
  narrow the intervals.

## Known limits, stated now
About 34 independent ideas and 11 families: intervals will be wide and reliability diagrams
coarse. One research group, one harness, toy scale. The repo is public, so leakage cannot be
ruled out, only probed. Verdicts are about these implementations.

## Seen before the freeze (disclosure)
A pilot ran before this freeze: 10 of the 114 items, blind condition only, one sample, six
models in full and two glm-5.2 calls (about 0.6 credits). We saw those forecasts. They were
mostly 0.6 to 0.9 on a subset with 40% yes, and the two Gemma models answered about 0.85 to
nearly every item. H1 was written before the pilot; the pilot agrees with it, so H1 is less
of a blind prediction than H2 to H5. Three things changed because of the pilot, none after
seeing any outcome-linked score: gate text now uses the registered quantity and threshold
for every item (eleven items had an empty rule), the token cap rose from 3,000 to 16,000
(answers were cut off), and samples per cell fell from 5 to 3 (cost). The pilot items stay
in the main set; results are also reported with them removed.

## Not yet built at freeze
The leakage probe is specified above but not coded. It will be written and run before the
analysis is read, and its prompt committed with the results.
