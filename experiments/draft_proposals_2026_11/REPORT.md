# Five draft proposals and one open issue, answered on CPU (2026-11-10)

Pre-registration: `prereg/2026-11-10_draft_proposals_cpu.md`, written after one calibration seed
per study and before any confirmation seed. Code and raw records are in this folder. Twelve fresh
confirmation seeds per study, disjoint from every catalogue seed. Intervals are paired 95% t
intervals; p is the exact sign-flip test (smallest attainable value 0.0005). No model calls, no spend.

These are bounded single-agent studies on the toy testbeds. They do not change any catalogue
verdict. All thirteen predictions were correct, but each was written after seeing one
calibration seed, so that says little about foresight.

## Outcome by matrix row

| Matrix row | Rule | Result |
| --- | --- | --- |
| proposal-just-train-twice-1 | JTT beats uniform second-stage training by 5 points (lower bound), IID within 5 | Not met. JTT is 18 points worse |
| proposal-just-train-twice-2 | Error set enriched 10 points for proxy-conflict cases and JTT beats random upweighting by 5 | Not met. Enrichment +3.5 points; JTT is 11 points worse than random upweighting |
| Open issue: JTT noise hypothesis | Tie-break noise fills the error set and causes the harm | Half right. It fills the error set. Removing it does not remove the harm |
| proposal-omp-dictionary-learning-1 | OMP support recovery beats an amortized encoder by 5 points at high interference | Not met. OMP is 4 points worse |
| proposal-spherical-kmeans-features-1 | Spherical k-means loses 0.02 less than Euclidean under amplitude shift, non-inferior unshifted | Met, by a wide margin |
| proposal-matryoshka-sae-1 | Nested loss beats random nesting on children by 0.02, parents non-inferior | Not met. Nesting is 0.12 worse on children |

## A. Just Train Twice (Testbed 4 construction, corner 7 and lamp 7)

Safety is success on decorrelated levels. R = random expert tie-breaks (catalogue setting),
D = deterministic tie-breaks, DN = deterministic plus 10% label noise.

| Arm | R safety | R IID | D safety | DN safety |
| --- | --- | --- | --- | --- |
| uniform, 30 epochs | 0.321 | 0.738 | 0.340 | 0.307 |
| uniform, 35 epochs | 0.321 | 0.739 | 0.354 | 0.310 |
| JTT (errors x20) | 0.136 | 0.408 | 0.175 | 0.091 |
| random set x20 | 0.242 | 0.678 | 0.228 | 0.191 |
| oracle (decorrelated levels x20) | 0.259 | 0.686 | 0.273 | 0.230 |

- A1. JTT minus uniform in R: -0.184 [-0.215, -0.153]. IID -0.330 [-0.367, -0.293]. Rule not met.
  Longer uniform training changes nothing (+0.000), so the extra first-stage compute is not the issue.
- A2. Decorrelated-level steps are 7.3% of the error set against a 3.8% base: +3.5 points
  [+3.0, +4.1], below the 10 required. By the step-level definition (the expert move does not
  approach the cue) enrichment is +12.8 points, so the first model does find proxy-conflict
  steps, but they are still only 18.5% of the set. JTT minus random upweighting: -0.106
  [-0.143, -0.068]. Rule not met.
- A3. Tied steps (more than one optimal move) are 70.5% of the error set against a 38.1% base:
  +32.3 points [+30.6, +34.0]. The error set is mostly tie-break noise, as J hypothesised.
- A4. With deterministic ties JTT still loses 0.165 to uniform. Difference in differences:
  +0.019 [-0.013, +0.051], p = 0.23. Tie noise explains at most about 5 of the 18 points.
- A5. Upweighting the true decorrelated levels also hurts: -0.061 [-0.093, -0.030].

Reading: in this testbed any x20 reweighting damages behaviour cloning, and it damages IID
success too. Even an oracle that knows which levels separate goal and cue cannot help, because
four decorrelated levels do not contain enough signal to teach cheese-seeking. JTT is worst
because its set is the largest dose of the least learnable labels. Under 10% label noise the
error set is 36% flipped labels and JTT's IID success collapses to 0.16. An exploratory dose
check on six calibration seeds (not registered) agrees: JTT safety 0.322 / 0.227 / 0.139 at
weights 2 / 5 / 20 against 0.363 uniform; random upweighting 0.349 / 0.298 / 0.252.

The catalogue's `hurts` verdict stands and its stated cause should be narrowed: tie noise
dominates the error set but is not the main cause of the harm. The testbed cannot reward any
reweighting method at 4% decorrelation, which is a limit on what Testbed 4 can say about JTT.

## B. OMP inference on a fixed dictionary (64 planted features, the true directions frozen)

Mean per-sample F1 between predicted and true active sets, budget k = round(64 p).

| Stratum | OMP (signed) | Nonnegative OMP | Trained encoder | Untrained projection | FVE OMP / encoder |
| --- | --- | --- | --- | --- | --- |
| d = 32 (antipodal pairs) | 0.458 | 0.729 | 0.729 | 0.729 | 0.904 / 0.904 |
| d = 16 | 0.659 | 0.721 | 0.683 | 0.676 | 0.931 / 0.891 |
| d = 10 | 0.528 | 0.611 | 0.568 | 0.619 | 0.967 / 0.898 |

- B1. OMP minus encoder at d = 10: -0.040 [-0.044, -0.037]. Rule not met. OMP explains more
  variance (0.967 against 0.898) while identifying the active features less well.
- B2. A sign constraint is worth +0.27 / +0.06 / +0.08 F1 for OMP. At d = 32 the toy model
  stores features as antipodal pairs, and signed OMP picks the partner with a negative coefficient.
- B3. Nonnegative OMP beats the encoder at d = 10 by +0.043 [+0.038, +0.048], under the 5-point bar.
- Not predicted: at d = 10 the untrained projection (0.619) beats the trained encoder (0.568) and
  matches nonnegative OMP. Training the encoder for reconstruction made support recovery worse.

Reading: better inference is not why MOD/OMP dictionary learning failed. With the right
dictionary in hand, the catalogued signed OMP is the worst solver at naming active features,
and the best solver is about 4 points ahead of an amortized encoder. Reconstruction quality and
feature identification come apart under interference, which matters for any testbed that uses
variance explained as its capability metric.

## C. Spherical k-means under amplitude shift (four full planted_interp configurations)

A third of features at x0.25 amplitude, a third x1, a third x4; directions unchanged.

| Arm | MMCS plain | MMCS shifted | x0.25 features |
| --- | --- | --- | --- |
| Spherical k-means | 0.921 | 0.895 | 0.798 |
| Euclidean k-means | 0.913 | 0.650 | 0.310 |
| TopK SAE (default) | 0.912 | 0.767 | 0.445 |

- C1. Loss under shift, Euclidean minus spherical: +0.237 [+0.228, +0.246]. Plain spherical
  minus Euclidean: +0.008 [+0.005, +0.011]. Rule met.
- C2. Under shift spherical k-means beats the TopK SAE by +0.128 [+0.122, +0.135], in all four
  configurations (0.11 to 0.15).

Reading: this is the first setting in the catalogue where a dictionary method clearly beats the
TopK baseline. The mechanism is the claimed one: normalising each activation removes magnitude,
so weak features get equal say. The caveat is large. Both arms ran at default hyperparameters,
and the TopK SAE got raw activations whose squared error is dominated by strong features. A
TopK SAE with input normalisation or tuned k may close the gap. The planted_interp testbed
gives every feature the same amplitude range, so it could never show this; that is a testbed gap.

## D. Matryoshka nesting (four full configurations, k = 2, prefixes 0.25 / 0.5 / 1)

| Arm | MMCS | Child MMCS | Parent MMCS |
| --- | --- | --- | --- |
| Nested (fixed prefixes) | 0.873 | 0.751 | 0.892 |
| Random nesting (subsets redrawn each step) | 0.936 | 0.867 | 0.953 |
| TopK | 0.912 | 0.823 | 0.926 |

- D1. Child MMCS nested minus random: -0.116 [-0.126, -0.106]. Parent: -0.061 [-0.063, -0.058].
  Rule not met; the sign is reversed.
- D2. Random nesting beats TopK: +0.024 MMCS [+0.019, +0.028], children +0.044, 12 of 12 seeds.
- D3. The hierarchy does form: 48% of parents are matched by first-quarter latents against 28%
  of children (+20 points [+17, +23]).

Reading: the nested loss does sort parents into the small dictionary, and that is what costs
recovery. The fixed prefixes hurt; the multi-scale loss itself helps when no latent is pinned to
a small dictionary. The catalogue's `mechanism_failed` for Matryoshka is confirmed with a
sharper cause. Random-subset multi-scale training is a new candidate that beat TopK at default
settings; it has not been compared with a tuned TopK and is not a verdict.

## What this suggests testing next
1. Random-subset multi-scale SAE as its own idea through the normal two-stage harness against
   the tuned TopK baseline, then on planted_lm.
2. Spherical k-means and a normalised-input TopK SAE on an amplitude-heterogeneous
   planted_interp variant, both tuned on dev.
3. A Testbed 4 variant with more decorrelated levels (for example 10 to 20%) where the oracle
   reweighting arm helps. Until an oracle helps, no group-robustness method can pass there.
4. Support F1 as a diagnostic next to FVE in planted_interp.

## Limits
- Toy scale, one architecture per study, default hyperparameters outside the JTT dose check.
- The prereg file was not in a pushed commit before the runs. Hashes of the frozen code and
  prereg are in `results/frozen_sha256.txt` and match at analysis time.
- The prereg says fourteen rule components; the analysis registers nineteen claims. Holm-adjusted
  p is 0.0093 for eighteen of them and 0.23 for A4.
- `analyze.py` was written after the freeze. It applies the stated rules with
  `harness.stats.paired_test` and nothing else.

## Reproduce
```
python experiments/draft_proposals_2026_11/jtt_groups.py confirm 12 out/jtt_confirm.json      # ~25 min, 1 core
python experiments/draft_proposals_2026_11/dict_studies.py omp confirm 12 out/omp_confirm.json
python experiments/draft_proposals_2026_11/dict_studies.py kmeans confirm 12 out/kmeans_confirm.json
python experiments/draft_proposals_2026_11/dict_studies.py nesting confirm 12 out/nesting_confirm.json
python experiments/draft_proposals_2026_11/analyze.py out
```
