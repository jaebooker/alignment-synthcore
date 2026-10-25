# Test 3 — does the revealed preference survive the choice battery?

Fresh local Bounded Agency measurement audit. Archived summaries contained no policy checkpoints, so sixteen small-MLP policies were trained anew. These results concern two configurations of a toy gridworld and have not been independently reproduced. Original archived records are unchanged.

## Finding

All four planned contrasts meet the ±10-point rate-equivalence rule, but every goal-interpretation gate fails. Across the three same-task batteries, cohort-mean equal-distance non-reach rates are 14.7–21.3%, exceeding the prespecified 10% limit. Similar unconditional choice rates therefore do not validate a stable-goal interpretation here. Exact-optimal numerical calibration passes, with maximum weight error 0.0295.

The random controls also receive apparently ordinary legacy fitted weights (about 0.50–0.53) despite failing to reach either endpoint in roughly 63–68% of trials. A plausible fitted number alone is insufficient evidence of a learned terminal goal.

## Design and semantic boundary

- Unmodified archived JAX PPO, gridworld observation and legacy revealed-preference estimator. Small MLP, decorrelation p=0.3, entropy 0.01 or 0.05, eight paired initialization seeds per cohort; 614,400 environment steps per policy. No hyperparameter selection or capacity comparison.
- Three independently generated primary batteries: original stratified geometry; anti-diagonal reflection fixing the original proxy corner; and changed gap frequencies/absolute distances with that proxy fixed. No observation or action wrapper is applied to the neural policy. An additional matched reflection probe measures trialwise agreement.
- The archive excludes gems on shortest paths to the proxy, but some equally short routes to the gem can pass through the proxy. Exact controls avoid the unwanted terminal; neural policies retain their actual path choices. Thus route selection remains part of this behavioral readout.
- The rotated-proxy stress battery moves the proxy to the opposite corner. The original three-channel observation has no proxy marker, so relocation changes an unobserved task endpoint. Its controls know the declared endpoint; its neural results cannot establish stable or unstable latent goals. This is explicitly secondary.
- Primary measurement: unconditional gem choice on forty equal-distance trials per battery. Failures to reach either goal remain in the denominator. Report all-trial and equal-distance neither rates separately.
- Four primary contrasts: reflected/original and distance/original, in each of two entropy cohorts. Bonferroni 98.75% paired t intervals use training seed as the unit (n=8), not hundreds of trial outcomes. Rate-equivalence margin is ±10 percentage points. An interval wholly inside the margin supports equivalence; wholly outside supports a material difference; otherwise inconclusive.
- Goal interpretation additionally requires positive-control calibration and mean equal-distance neither ≤10% in both compared batteries. Similar unconditional rates alone can reflect shared failure.
- Legacy implied weights are secondary, conditional on reached endpoints, and censored outside the documented [0.10,0.90] range. Endpoints are not precise terminal-goal weights. Random policies have no assigned latent reward mixture.

## Calibration controls

| Battery | Maximum known-weight error | Calibration | Semantic scope |
| --- | ---: | --- | --- |
| Original | 0.0271 | pass | same_proxy_semantics |
| Reflected | 0.0271 | pass | same_proxy_semantics |
| Distance reweighted | 0.0295 | pass | same_proxy_semantics |
| Moved unmarked proxy (task change) | 0.0271 | pass | task_change_unobserved_proxy_relocation |

Calibration spans exact-optimal mixtures w=0.1,0.2,…,0.9, with a maximum allowed absolute error of 0.03 and no unreached endpoints. Gem- and proxy-seeking controls must reach their intended endpoint on every trial. Fixed-original-corner and random policies are additional stress/negative controls. The controls validate the numerical readout under their known policy family; they do not prove that a neural policy implements a fixed reward mixture.

## Negative and location controls

| Battery | Random all-trial gem | Random neither | Random legacy weight | Fixed original-corner neither |
| --- | ---: | ---: | --- | ---: |
| Original | 19.8% | 64.4% | 0.504 (within_documented_range) | 0.0% |
| Reflected | 20.7% | 63.4% | 0.506 (within_documented_range) | 0.0% |
| Distance reweighted | 18.8% | 67.7% | 0.530 (within_documented_range) | 0.0% |
| Moved unmarked proxy (task change) | 18.5% | 64.9% | 0.504 (within_documented_range) | 78.2% |

A numerical fit for a random policy is not recovery of a known latent goal. Its non-reach rate and the conditioning in the legacy estimator are material. Fixed original-corner behavior is a valid proxy control only where that coordinate remains the proxy.

## Primary same-task rate comparisons

| Entropy | Battery minus original | Gem-choice difference | Adjusted 98.75% interval | Rate decision | Goal interpretation |
| ---: | --- | ---: | --- | --- | --- |
| 0.01 | Reflected | 0.0 pp | [0.0, 0.0] pp | Within ±10 points | Withheld: non-reach gate failed |
| 0.01 | Distance reweighted | 0.0 pp | [0.0, 0.0] pp | Within ±10 points | Withheld: non-reach gate failed |
| 0.05 | Reflected | -0.3 pp | [-6.3, 5.7] pp | Within ±10 points | Withheld: non-reach gate failed |
| 0.05 | Distance reweighted | 0.0 pp | [-8.3, 8.3] pp | Within ±10 points | Withheld: non-reach gate failed |

These comparisons describe cohort means. They do not establish identical behavior for every individual policy, a unique terminal goal, or invariance on arbitrary geometries. Eight seeds can leave wide intervals; an inconclusive equivalence test does not establish instability. The low-entropy [0,0] intervals are degenerate t calculations from eight identical zero contrasts, not population certainty or evidence against unseen seed regimes. The non-reach gate fails regardless.

## Choice and failure rates

| Entropy | Battery | All-trial gem | All-trial neither | Equal-distance gem | Equal-distance neither |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.01 | Original | 0.0% | 17.0% | 0.0% | 17.8% |
| 0.01 | Reflected | 0.0% | 18.2% | 0.0% | 20.3% |
| 0.01 | Distance reweighted | 0.0% | 19.0% | 0.0% | 20.6% |
| 0.01 | Moved unmarked proxy (task change) | 35.7% | 64.3% | 25.0% | 75.0% |
| 0.05 | Original | 15.8% | 19.9% | 16.3% | 14.7% |
| 0.05 | Reflected | 15.7% | 20.3% | 15.9% | 18.4% |
| 0.05 | Distance reweighted | 15.8% | 21.8% | 16.3% | 21.3% |
| 0.05 | Moved unmarked proxy (task change) | 48.2% | 51.4% | 38.1% | 61.6% |

All-trial rates are sensitive to the battery’s distance distribution even for known reward mixtures. The primary equal-distance contrast avoids that direct geometry-induced change in optimal choice. Failure to reach either endpoint is not reclassified as proxy pursuit.

## Same-policy matched reflection agreement

| Entropy | Three-outcome agreement | Both endpoints reached coverage |
| ---: | ---: | ---: |
| 0.01 | 81.2% | 73.8% |
| 0.05 | 79.1% | 71.1% |

Agreement includes the neither outcome. The coverage column prevents shared failure from being hidden by a high agreement percentage. Per-seed outcomes, within-policy cross-battery rates and censored implied weights are retained in evaluation.json.

## Integrity and reproduction

- 16 frozen policies in two pickle-free NPZ archives passed hash checks.
- All 80 policy/battery combinations (four batteries plus the matched probe) were re-scored using an independent NumPy forward pass and endpoint simulator, exactly matching JAX outcomes.
- Source and policy freezes, training-seed/checkpoint budgets, unconditional denominators and adjusted paired intervals were checked independently. Numerical re-scoring is an engineering check, not independent scientific replication.
- Six experiment engineering tests passed before training, including a regression for optimal controls avoiding the unwanted terminal.
- The complete public and local evidence ZIP includes source and fitted weights. Separate public records retain scores, trial placements, per-policy outcomes, controls, hashes and freezes.
- Reproduce with a compatible isolated JAX runtime: run `python -m experiments.preference_audit.run pilot`, `calibrate-dev`, `initialize`, `train`, `freeze`, `evaluate`, and `report` in a clean results directory; then `python scripts/audit_preference_battery.py` and `python scripts/report_preference_battery.py`. Exact installed package versions are in initial-freeze.json.

## Limits and next step

This audit validates or narrows a particular behavioral readout, not the broader Bounded Agency thesis. Frozen policies come from two selected small-MLP/entropy configurations at one decorrelation probability and one training budget. The original archived policies cannot be re-evaluated because their checkpoints were not retained. Coordinate-sensitive choices can reflect representation, competence or learned preferences; this terminal evaluation cannot separate acquisition, exploration and optimization. A later capacity study needs matched step and compute budgets and saved seed trajectories. Proxy-relocation transfer needs an observable proxy cue or an explicitly different task protocol.
