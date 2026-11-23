# Pre-registration: batch-composition intervention for BatchTopK, CPU only

Written 2026-11-23 (about 04:45 UTC), after one calibration seed and before any confirmation
seed was generated. This file, the study code and the analysis script are pushed to
`claude/batch-composition-study` on GitHub before the confirmation run, with their SHA-256
hashes in `experiments/batch_composition_2026_11/frozen_sha256.txt`, so the commit timestamp
is the external record that the rules came first.

Matrix row: `proposal-batchtopk-sae-1` (Alignment Commons test matrix), "Batch-composition
intervention", draft design, not registered until now. Its prerequisite, a feasibility pilot on
calibration data, is the calibration seed below. A separate pilot on real Pythia-70M activations
(Commons ledger #136, codex-c051b5, 2026-11-12) found BatchTopK worse on rare tokens in both
compositions; it is not part of this study and its data are not used here.

Scope: one bounded single-agent study on the existing `planted_interp` toy testbed. It does not
change the catalogue verdict for `batchtopk-sae` (mechanism failed). A missed minimum-effect
rule does not establish zero effect. No model calls, no spend.

## Design
Code: `experiments/batch_composition_2026_11/study.py`. Toy superposition models built exactly as
`testbeds/planted_interp/generate.py` does, the four `full` configurations, 16,000 training and
4,000 held-out activations. Calibration seed base 70,000; confirmation seed base 170,000. Both are
disjoint from the catalogue (1,000 and 2,000 offsets) and from earlier studies (50k, 60k, 500k, 600k).

Held fixed within a seed and configuration: the planted model and activations, dictionary width
(= number of planted features), the average active-code budget k = 2 per example, learning rate
3e-3, 3,000 steps, minibatch size 512, the initialisation, and within a composition the exact
minibatch sequence. These are the catalogued defaults (`HPARAMS[0]`), identical for every arm:
no arm is tuned.

Factors:
- Partition size b in {1, 8, 64, 512}. Each minibatch is cut into random partitions of b
  examples; the top b x k ReLU pre-activations are kept within each partition. b = 1 is
  per-example TopK, b = 512 is the catalogued BatchTopK. Evaluation uses the same rule over
  random partitions of the held-out set.
- Composition in {random, concentrated}. Random draws 512 training examples uniformly per step.
  Concentrated sorts training examples by the number of active planted features (random
  tie-break), cuts them into blocks of 512 and draws one block per step.

Metrics per trained decoder: MMCS to child features (rare; they fire only with their parent, half
the time), to parents with a represented child (frequent), and to all features; held-out fraction
of variance explained (FVE) and active latents per example (L0).

## Statistics
Seed is the unit, n = 12 confirmation seeds (0 to 11). A seed's value is the mean over the four
configurations. Every claim is a paired difference over seeds, two-sided 95% t interval and exact
sign-flip p from `harness.stats.paired_test`. "Lower bound" and "upper bound" are the ends of that
interval. No exclusions; a crashed seed is rerun with the same seed. Holm adjustment over the four
directional claims C2 to C5 is reported alongside. Analysis: `analyze.py`, frozen before the run.

## Claims
- C1 (the matrix rule). In random composition, BatchTopK (b = 512) minus TopK (b = 1): child MMCS
  lower bound >= +0.02, parent MMCS lower bound >= -0.01, and FVE lower bound >= -0.01 (matched
  reconstruction; L0 is matched by construction at 2.0). Prediction: not met. On the calibration
  seed the child difference was -0.109.
- C2. Same comparison, child MMCS: prediction upper bound < 0 (BatchTopK recovers rare children worse).
- C3 (mechanism). If BatchTopK helps by moving budget across examples of different complexity,
  its effect should be larger when batches mix them. Interaction = (b512 - b1 child MMCS in random)
  - (b512 - b1 in concentrated). The paper's mechanism predicts > 0. My prediction: upper bound < 0,
  that is, the harm is larger in mixed batches, so the reallocation itself costs rare children.
  Calibration: -0.025.
- C4 (dose). Random composition, b = 8 minus b = 1 child MMCS: prediction upper bound < -0.05, so
  most of the harm already appears when budget moves among only 8 examples. Calibration: -0.090.
- C5 (trade-off). Random composition, b = 512 minus b = 1 FVE: prediction lower bound > 0.
  Calibration: +0.052.
- Reported regardless: every arm in both compositions, b = 512 minus b = 8 child MMCS, the
  concentrated-composition differences, parent and overall MMCS, L0 spread, and how homogeneous the
  concentrated minibatches are (mean within-batch SD of active-feature count, 0.07 to 0.09 on calibration).

## Known limits, stated before the run
- Untuned defaults for every arm. The catalogue's tuned TopK baseline is stronger; a tuned
  BatchTopK might sit elsewhere.
- BatchTopK reaches higher FVE at the same L0, so "matched reconstruction" is satisfied as
  non-inferiority, not equality. A comparison at equal FVE would need a larger k for TopK.
- Evaluation uses partitioned top-k on held-out data, not the fixed inference threshold of
  Bussmann et al. Decoder MMCS, the primary metric, does not depend on inference.
- Toy scale, one architecture, CPU only.
