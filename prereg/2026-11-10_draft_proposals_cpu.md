# Pre-registration: five draft test-matrix proposals and one open issue, CPU only

Written 2026-11-10 (about 21:50 UTC), after one calibration seed per study and before any
confirmation seed was generated. Calibration and confirmation use disjoint seed ranges that
no catalogue run has used. Code: `experiments/draft_proposals_2026_11/`. The file was written
in a cloud workspace and is not in a pushed commit at the time of the runs, so the timestamp
is not externally verifiable.

Scope: bounded single-agent studies on the existing toy testbeds. They elaborate the draft
designs in the Alignment Commons test matrix (v3). Success rules are taken from the matrix
rows. A missed minimum-effect rule does not establish zero effect. No catalogue verdict changes.

Statistics for every claim: seed is the unit, n = 12 confirmation seeds, paired differences,
two-sided 95% t interval and exact sign-flip permutation p from `harness.stats.paired_test`.
"Lower bound" means the lower end of that interval. No exclusions; a crashed seed is rerun
with the same seed. Fourteen rule components are tested; they are reported individually
against the matrix's stated rules, with Holm-adjusted p-values alongside for the directional
claims.

## A. Just Train Twice (proposal-just-train-twice-1, -2; open issue on tie-break noise)
Testbed 4 construction (corner 7 and lamp 7, 4% decorrelated training levels), seed base
600,000 (calibration 500,000). Conditions: R random expert tie-breaks (catalogue setting,
primary), D deterministic tie-breaks, DN deterministic plus 10% random label noise. Arms:
uniform, uniform_long (35 epochs), jtt, random (same set size, x20), oracle (decorrelated-level
steps x20). Same architecture, learning rate, second-stage seed and 30 second-stage epochs.
Safety = success on decorrelated levels, mean of the two families.

- A1 (proposal 1). Rule: jtt - uniform safety lower bound >= +0.05 and IID success difference
  lower bound >= -0.05, in R. Prediction: not met; jtt is below uniform in R.
- A2 (proposal 2). Rule: decorrelated-level steps are enriched in the error set by >= 10
  percentage points (lower bound of share-in-set minus base share) and jtt - random safety
  lower bound >= +0.05, in R. Prediction: not met on both parts.
- A3 (open issue, the noise hypothesis). In R, steps where the expert had more than one optimal
  move are over-represented in the error set (lower bound > 0). Prediction: yes, by more
  than 20 points.
- A4 (open issue, causal part). If tie noise causes the harm, removing it should shrink it:
  (jtt - uniform in D) - (jtt - uniform in R) has lower bound > 0. Prediction: no. On the
  calibration seed jtt stayed far below uniform in D, and random upweighting also hurt, so
  I expect the harm to come from x20 upweighting itself rather than from which steps it picks.
- A5 (secondary). oracle - uniform safety in R: prediction not positive (lower bound <= 0).
Reported regardless: all arms in all conditions, spurious-zone rate, error-set composition
(decorrelated, proxy-conflict, tied, noisy shares) per family.

## B. OMP inference on a fixed dictionary (proposal-omp-dictionary-learning-1)
Toy superposition models, n = 64 planted features, d in {32, 16, 10}, p_active in {0.03, 0.06},
30% children, noise 0.03, seed base 60,000 (calibration 50,000). The dictionary is the planted
directions, frozen and identical for every solver. Budget k = round(64 p) nonzeros per sample.
Solvers: OMP (sklearn, signed, as in the catalogued MOD method); an amortized TopK encoder
trained for reconstruction with the decoder frozen; secondary: nonnegative OMP and untrained
projection + top-k. Metric: mean per-sample F1 between the predicted and true active sets on
4,000 held-out samples.

- B1. Rule: F1(OMP) - F1(encoder) lower bound >= +0.05 in the d = 10 stratum (mean of the two
  sparsities). Prediction: not met; OMP is below the encoder there despite higher variance
  explained.
- B2 (secondary). F1(nonnegative OMP) - F1(OMP) lower bound > 0 in every stratum. Prediction: yes.
- B3 (secondary). F1(nonnegative OMP) - F1(encoder) at d = 10: prediction positive but below 0.05.
Reported regardless: all strata, variance explained, active counts, coherence.

## C. Spherical k-means under amplitude shift (proposal-spherical-kmeans-features-1)
The four full planted_interp configurations, seed base 60,000. Shift: a random third of
features has amplitude x0.25, a third x1, a third x4; directions and noise unchanged. Arms at
equal dictionary size: spherical k-means (catalogued defaults), Euclidean k-means (sklearn,
3 restarts, 50 iterations, same centring), TopK SAE (catalogued defaults). Metric MMCS, mean
over configurations. Loss under shift = MMCS(plain) - MMCS(shift).

- C1. Rule: loss(Euclidean) - loss(spherical) lower bound >= 0.02, and plain-condition
  spherical - Euclidean lower bound >= -0.01. Prediction: met.
- C2 (secondary). Under shift spherical k-means beats the TopK SAE (lower bound > 0).
  Prediction: yes. This would be the first setting in the catalogue where a dictionary method
  beats the TopK baseline, though at default rather than tuned hyperparameters.
Reported regardless: recovery of the x0.25 features per arm.

## D. Matryoshka nesting (proposal-matryoshka-sae-1)
The four full configurations, seed base 60,000, catalogued defaults (k = 2, prefixes 0.25 /
0.5 / 1). Arms at equal decoder width and k: nested (fixed prefixes), random nesting (same
multi-scale loss, sub-dictionary membership redrawn every step), TopK. Child = planted feature
that fires only with its parent; parent = planted feature with at least one represented child.

- D1. Rule: child MMCS nested - random lower bound >= +0.02 and parent MMCS nested - random
  lower bound >= -0.01. Prediction: not met; nested is below random nesting on children.
- D2 (secondary). Random nesting versus TopK on overall MMCS: prediction positive (lower
  bound > 0), i.e. the multi-scale loss helps as a regularizer while the fixed hierarchy hurts.
- D3 (descriptive). Parents are matched by first-quarter latents more often than children.

## Stopping
Fixed n = 12 for every study. No interim looks at confirmation data change the design.
