# Amendment 1: a second, secondary item set of pre-registered sub-claims

Status: FROZEN 2026-11-14 06:24 UTC, approved by J. Adds to prereg/2026-11-14_forecast_audit.md and
changes nothing in it. The five frozen hypotheses, the 114 catalogue items and the frozen scripts are untouched.
Hashes of this file, subclaims.jsonl, build_subclaims.py, forecast_sub.py and analyze_sub.py
are appended to frozen_sha256.txt, before the first sub-claim call.

## Why
The catalogue set has about 34 independent ideas and a low yes-rate on its primary question
(29%). A forecaster that answers about 0.8 to everything looks optimistic there for a reason
that has nothing to do with the items. A second set with a different yes-rate separates
"optimistic about alignment methods" from "says the same number whatever is asked".

## Items
74 claims that this repo's confirmatory studies registered before their confirmation runs
(subclaims.jsonl), from 19 study reports in 7 testbed families; 65% were met. Each item has a
design description with no results, the registered claim, and whether it was met.
Differences from the catalogue set, stated plainly:
- The claims are the research group's own hypotheses about its testbeds, not tests of
  published methods. Many are measurement or sanity claims.
- The design descriptions were written by Claude on 2026-11-14 with the outcomes in view.
  Wording could leak the answer. Thirteen outcomes were checked against a stored summary file;
  the rest were read from report tables.
- Where a report gives the registered threshold the claim uses it; otherwise the claim restates
  the hypothesis as the report gives it.
- Families are unequal: bounded_agency has 23 items (87% met), oversight_gridworld 9 (11% met).
- About 10 further studies were not extracted (time). Selection was by reading order, before
  any forecast on this set existed.

## Run
Same seven models, 16,000-token cap, temperature 1, 3 samples per cell. Two conditions: blind,
and base_rate (the met-rate of claims from the other families). Separate cache and spend
files (data/sub/), cap 150 credits. The leakage probe is not repeated; it covered the repo.

## Claims (all secondary to the five frozen hypotheses)
- A1 set-insensitivity: per model, mean blind forecast on this set minus mean blind forecast on
  the catalogue `win` items is smaller than half the 36-point gap in yes-rates (under 0.18).
- A2 discrimination: AUROC > 0.5 on this set in blind.
- A3 no skill: Brier skill vs the leave-one-family-out base rate <= 0 in blind.
- A4 reference class helps: Brier(base_rate) < Brier(blind), paired by item.
A2 to A4 use 95% bootstrap intervals over the 7 families (B = 2000). A1 is reported per model
with the pooled value and no interval across sets, since the sets share no items.

## What had been seen when this was written
The 10-item pilot on the catalogue set (disclosed in the main file), in which forecasts sat
mostly between 0.6 and 0.9. A1 was written with that in mind. No forecast on any sub-claim,
and no score from the main run, which was in progress and unread. One fact from the main run was known: glm-5.2 returned
empty answers on most calls and was dropped (DEVIATIONS.md). This set is run on the same six
remaining models; `seven models` above means the six that remain.
