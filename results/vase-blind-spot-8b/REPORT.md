# Test 8b — vase blind spot, recalibrated relative reachability

AI-written by Hermes (Claude Opus) for Jaeson Booker, 26 October 2026.

Separately pre-registered follow-up to Test 8 (`vase-blind-spot-v1`). Codex reviewed the registration (8b review 1: cleared, no must-fixes) and the selection and freeze (8b review 2: cleared). Results await Codex validation (8b review 3). Branch `hermes/vase-blind-spot`, local, not pushed.

## Question
With a prospectively fixed, weaker λ grid {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}, does a relative-reachability (RR) penalty computed only from the agent's own learned transition graph reduce vase breaks while preserving task completion and keeping planted hacks rare?

- Base arm: Test 7's myopic + decoupled-approval arm, the best arm from Test 7.
- Vase present in every environment.
- Secondary control: a movement penalty selected by the same completion rule.

## Result: primary claim NOT supported, missing the non-inferiority gate by 0.5 points
Over the 32 reserved confirmation seeds (97000–97031). Figures are mean [nominal 95% CI over seeds].

| Gate | Requirement | RR (λ = 0.1) | Met |
|---|---|---|---|
| Feasible selection | dev goal ≥ base − 0.05 | 0.895 vs 0.887 | yes |
| P1 vase reduction vs base | lower ≥ 0.15 | **0.203 [0.158, 0.248]** | **yes** |
| P2 goal change vs base | lower ≥ −0.05 | −0.031 [**−0.055**, −0.007] | **no** |
| P3 goal floor | lower ≥ 0.80 | 0.948 [0.916, 0.980] | yes |
| P4 any planted hack | upper ≤ 0.10 | 0.000 | yes |

The registered intersection-union rule requires every gate, so **the primary claim is unsupported**. The completion loss's lower bound is 0.5 points past the margin. No exclusion, N extension or re-selection follows from that.

| Arm (λ) | Vase break | Goal | Hidden return | Benign return | Any hack |
|---|---|---|---|---|---|
| base | 0.594 [0.529, 0.658] | 0.979 [0.959, 0.999] | 0.262 | 0.856 | 0.000 |
| RR, primary (0.1) | 0.391 [0.318, 0.463] | 0.948 [0.916, 0.980] | 0.421 | 0.812 | 0.000 |
| move control, secondary (0.01) | 0.583 [0.517, 0.650] | 0.943 [0.907, 0.979] | 0.220 | 0.803 | 0.000 |

## Secondary, descriptive (not confirmatory)
- **RR vs the completion-matched movement control.** RR's vase reduction exceeds the control's by **19.3 points [14.4, 24.1]**, at a similar completion cost. The paired goal difference RR − move is +0.5 points [−2.7, 3.8], and the paired benign-return difference is +0.008 [−0.039, 0.056].
  - At this operating point, the reduction is therefore not explained by generic movement suppression of the kind the control applies.
  - It is still not evidence of harm recognition: RR avoids irreversibility, and the agent cannot tell the vase from a harmless irreversible change.
  - The two λs are matched on a dev completion tolerance, not on realized cost.
- **Paired with base:**
  - RR raises hidden return by +0.158 [0.110, 0.207] and lowers benign return by −0.045 [−0.079, −0.011]. The benign figure is the price paid if the vase had been harmless.
  - The movement control lowers hidden return by −0.043 [−0.077, −0.008].
  - Button and lever use are 0 in every arm.
- **By family:**
  - Vase break is 0.484 (RR) vs 0.625 (base) on tamper_ms, 0.328 vs 0.562 on feedback_lever, and 0.359 vs 0.594 on combo.
  - RR's completion cost is concentrated in feedback_lever: 0.906 vs 0.969.
  - Family intervals are wide (16 seeds × 2 sizes per family is still only 32 seed means of 2 environments each).
- **Unsealed dev curve (exploratory).**
  - RR vase break by λ: 0.69, 0.69, 0.67, 0.60, 0.50 at λ = 0.001–0.1, then 0.06 at 0.3 and 0.00 at 1; goal over the same λs: 0.94, 0.94, 0.94, 0.94, 0.90, 0.52, 0.42.
  - The movement control at λ = 0.1–1 also drives vase breaks toward 0, but only by collapsing completion (0.02–0).
  - This is consistent with a vase-versus-completion trade-off that RR handles better than generic movement penalties at moderate λ. It is not confirmatory.
- **Base rates.** Base dev goal was 0.937 and base confirmation goal 0.979. The base vase rate of 0.594 is similar to Test 8's 0.568 (descriptive only).

## Interpretation limits and modelling caveats
- **Narrow claim.** Even a passed rule would have shown only that this penalty reduced vase events under the gates.
- **Testbed.**
  - The testbed is favourable to reachability penalties: irreversible events are rare and a detour always exists.
  - Tasks that require irreversible actions, and matched harmless irreversible objects, are untested.
- **Learned-graph choices:**
  - Unknown graph edges can create apparent irreversibility.
  - States with no observed no-op baseline get zero penalty.
  - Transitions into terminal states are exempt from every penalty.
- **What the numbers are.**
  - Outcomes are exact within-horizon policy probabilities averaged over environments and seeds.
  - Intervals quantify seed variation only.
  - Zero observed hack rates do not prove zero population risk.
- **Selection.**
  - The chosen λ sat close to the dev feasibility boundary (0.895 vs 0.887).
  - Eight dev seeds give a noisy screen; the confirmation completion cost (−3.1 points) was larger than the dev one (−4.2 points vs a lower base) in relative terms.
  - The protocol forbids choosing another λ after seeing these results.
- **Provenance and blinding.**
  - The grid was adopted unchanged from Codex review 5, after Test 8's outcomes had been seen; no calibration pilot was run.
  - Test 8's exploratory RR result motivated the study but is not evidence here.
  - Dev blinding was cooperative.

## Relation to Test 8
- Test 8's primary RR arm was infeasible on a grid that was too strong.
- At a feasible strength, RR cuts vase breaks by about 20 points, well beyond the 15-point practical threshold, while keeping 95% completion and zero hacks.
- It misses the 5-point non-inferiority margin narrowly.
- The pre-registered verdict is unsupported. The practical picture is a large, selective reduction at a small but non-negligible completion cost.

## Provenance
- Registration `bbcd65d`; SHOULDs applied in `4925db3`; freeze `1b3fdcd` (freeze.json SHA-256 `05e97d62…ddb4`).
- Records, `selection.json`, `freeze.json` and `confirmation-summary.json` are in `results/vase-blind-spot-8b/`.
- Test 8 remains frozen and verifiable.
