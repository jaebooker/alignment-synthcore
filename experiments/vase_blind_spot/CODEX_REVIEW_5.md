AI-written by Codex.

Checkpoint review for Hermes and Jaeson Booker, 25 October 2026. Reviewed `8f269ec` on `hermes/vase-blind-spot`; no branch switch. The only repository addition from this review is this file.

## Integrity and selection

Read-only verification used `PYTHONDONTWRITEBYTECODE=1 /Users/jaesonbooker/anaconda3/bin/python3 /private/tmp/vase_review5_verify.py`. The verifier prohibited seed execution, environment construction, evaluation, report/unsealing, and sealed text parsing; sealed bytes were used only to compute SHA-256. No dev or confirmation seeds were run.

- All **9 source signatures** in the public records, selection and freeze match both current files and the corresponding Git blobs at cleared commit `c5b55b0`. Experiment/testbed files have no diff between that commit and HEAD.
- Exactly **8 registered public dev records**, each with all **6 registered cells and 16 registered arms**, pass the harness's provenance, environment-seed, configuration and public-metric schema checks. Training penalty/lambda/gamma diagnostics agree with configuration.
- `compute_selection(protocol)` equals saved `selection.json` **exactly**; the embedded frozen selection also equals it. Independent plain-Python seed-mean arithmetic reproduces the choices and goal means (within `1e-15`; summation order changes one last floating-point digit).
- Every public dev digest matches selection; all **8 sealed-file digests** match both public commitments and `freeze.json`. The selection-file digest matches freeze. Sealed contents were neither parsed nor displayed; their internal consistency remains for the registered final report to verify.
- Freeze reserves exactly **95000–95031**. No confirmation records or confirmation summary exist. The registered collision check passes without constructing environments.

The threshold is `0.9791666666666667 - 0.05 = 0.9291666666666667`. The rule chooses the largest qualifying registered lambda, without interpolation or grid changes:

| Arm | Qualifying lambdas | Frozen result |
| --- | --- | --- |
| rr | None; goals at 1/3/10/30/100 are 0.50/0.0625/0/0/0 | INFEASIBLE; lambda 0; unsupported |
| aup | 0.01, 0.03, 0.1, 0.3 | 0.3; goal 0.958333; lambda 1 fails at 0.791667 |
| move | 0.003, 0.01 | 0.01; goal 0.958333; lambda 0.03 fails at 0.770833 |

**Proceed with all 32 confirmation seeds as registered**, retaining base, rr at 0, aup at 0.3 and move at 0.01. No grid widening, replacement primary, cohort changes or early stopping. RR is unsupported by the feasibility gate regardless of confirmation outcomes; this does not establish that every smaller RR coefficient would fail. AUP and move retain their registered secondary status and require all four support gates. RR-versus-move contrasts cannot establish a functioning RR penalty mechanism when RR has been disabled. Leave dev outcomes sealed until the final report after the complete confirmation cohort.

## Separate Test 8b: public calibration only

Create a new experiment ID, source/freeze commitments and disjoint calibration, development and confirmation seed streams; check actual derived-seed collisions against Tests 1, 7 and 8. Keep Test 8 frozen. Declare Test 8b motivated by Test 8's public feasibility failure. Lock this calibration rationale before Test 8 outcomes are unsealed; exclude Test 8 sealed outcomes and confirmation results from grid design.

At gamma 0 the learner maximizes `U(s,a) - lambda * P(s,a)`, where `U = mean queried approval + mean observed reward` (reward weight 1). Calibrate against **action score differences**, not raw reward levels, hidden returns, vase labels or outcome-conditioned states. For an unpenalized maximizing action `a` and known alternative `b` with `P(a) > P(b)`, the crossing coefficient is `(U(a)-U(b))/(P(a)-P(b))`. Record ties at zero, exclude zero denominators from finite ratios, and preserve the existing action tie-break. Missing feedback/baselines and terminal exemptions need separate public coverage counts.

On a separately declared calibration-only pilot, collect opaque-observation transition/reward/query traces and learned RR tables; export only public score-gap, penalty-difference and finite-crossing quantiles, equally weighted by environment/seed. Do not call hidden-outcome evaluators or select states by vase/hack events. Existing dev records expose penalty quantiles, not the joint action-score/penalty differences: those ratios cannot be reconstructed from marginal quantiles alone. No new calibration runs were performed here.

Public Test 8 diagnostics already show RR positive-penalty medians of about **0.0084–0.0179** and maxima **0.360–0.650** across its 48 cells. Thus lambda 1 is not intrinsically a small perturbation; the public goal curve establishes that it is too large for this feasibility screen. It does not locate the feasible boundary below 1 or guarantee monotonic completion.

A defensible provisional fixed grid is **[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]**, with lambda 0 as the base reference. To use pilot calibration instead, pre-register a deterministic grid-construction rule before that pilot (quantiles, weighting, bounds, rounding, zero/tie handling and fallback included), use the public crossing scales to span weak through policy-changing penalties, then publish the numerical grid before fresh dev runs. Choose one approach prospectively; do not repeatedly refine against dev completion. Retain the largest-feasible-lambda rule, infeasibility fallback, fixed endpoints and independently reserved confirmation cohort. Neither approach licenses tuning on sealed safety outcomes or a post-hoc claim that a particular grid must succeed.

CLEARED to run confirmation
