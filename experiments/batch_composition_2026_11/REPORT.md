# Batch-composition intervention for BatchTopK (2026-11-23)

Matrix row `proposal-batchtopk-sae-1`. Pre-registration: `prereg/2026-11-23_batch_composition.md`,
pushed in commit 16e341b (2026-11-23 04:19 UTC) after one calibration seed and before any
confirmation seed. The confirmation runs started at 04:21 UTC. Frozen hashes in `frozen_sha256.txt`
matched at analysis time. Twelve fresh confirmation seeds (base 170,000), four planted_interp
configurations each. Intervals are paired 95% t intervals over seeds. p is the exact sign-flip test
(smallest attainable 0.0005). No model calls, no spend. It does not change the catalogue verdict.

## Outcome

| Claim | Rule or prediction | Result |
| --- | --- | --- |
| C1, the matrix rule | BatchTopK minus TopK: child lower bound >= +0.02, parents >= -0.01, FVE >= -0.01 | **Not met.** Child −0.097 [−0.107, −0.088]; parents −0.021 [−0.028, −0.013]; FVE +0.055 [+0.052, +0.057] |
| C2 | Child difference upper bound < 0 | Held. 0 of 12 seeds favour BatchTopK |
| C3, mechanism | Paper's mechanism predicts a larger benefit in mixed batches; I predicted the opposite | Held. Interaction −0.028 [−0.038, −0.018]: the harm is larger in mixed batches |
| C4, dose | b = 8 minus b = 1 child upper bound < −0.05 | Held. −0.083 [−0.092, −0.074] |
| C5, trade-off | FVE lower bound > 0 | Held. +0.055 [+0.052, +0.057] |

Holm-adjusted p is 0.002 for C2 to C5. All four predictions were written after one calibration
seed, so their success shows the calibration was representative, not foresight.

## Arms (means over 12 seeds and 4 configurations)

| Composition | Partition b | MMCS | Child | Parent | FVE | L0 (sd across examples) |
| --- | --- | --- | --- | --- | --- | --- |
| random | 1 (TopK) | 0.911 | 0.819 | 0.922 | 0.843 | 1.98 (0.07) |
| random | 8 | 0.873 | 0.736 | 0.908 | 0.892 | 2.00 (2.18) |
| random | 64 | 0.859 | 0.724 | 0.901 | 0.897 | 2.00 (2.28) |
| random | 512 (BatchTopK) | 0.859 | 0.722 | 0.901 | 0.898 | 2.00 (2.29) |
| concentrated | 1 (TopK) | 0.915 | 0.827 | 0.924 | 0.843 | 1.98 (0.07) |
| concentrated | 8 | 0.881 | 0.757 | 0.900 | 0.877 | 2.00 (2.30) |
| concentrated | 64 | 0.882 | 0.759 | 0.899 | 0.876 | 2.00 (2.52) |
| concentrated | 512 (BatchTopK) | 0.881 | 0.758 | 0.898 | 0.874 | 2.00 (2.55) |

The concentrated minibatches were homogeneous as intended: within-batch SD of the active-feature
count 0.09, against 1.3 to 2.1 across the training set. The child deficit holds in every
configuration (random −0.065 to −0.135).

## Reading

Letting the active-code budget move between examples buys reconstruction and costs rare features.
At the same average of two active latents, BatchTopK explains 5.5 points more variance than
per-example TopK and recovers child features about 0.1 worse in cosine. Parents lose 0.02.

Almost all of the effect appears once the budget is shared among only 8 examples (child −0.083
of the −0.097 total). Moving from 8 to 512 adds −0.014 [−0.021, −0.008]. So the cost comes from
reallocation itself, not from batch-level statistics at large batch sizes.

Making each training batch homogeneous removes some of the room to reallocate. It cuts the child
deficit from −0.097 to −0.069 and the FVE gain from +0.055 to +0.031. That is the reverse of
what the motivating mechanism predicts: if moving budget toward complex inputs were what helped
rare features, mixed batches should help more. Here they hurt more. A plausible cause is that
complex examples (a parent with its children active) get extra latents. The decoder can then
reconstruct them with combinations that do not isolate the child direction, which is the
absorption failure the testbed plants.

This sharpens the catalogue's `mechanism_failed` for `batchtopk-sae` (MMCS −0.085 against the
tuned TopK baseline). On this testbed the batch-level budget is the cause, not a tuning accident.
It agrees in direction with the separate Pythia-70M calibration (Commons ledger #136). That run
found BatchTopK worse on rare tokens in both compositions, but its sign for composition was the
opposite of this one.

## Limits
- Untuned defaults (k = 2, lr 3e-3, 3,000 steps) for every arm. A tuned BatchTopK, or one with
  more latents than planted features, may sit elsewhere.
- Matched reconstruction is satisfied as non-inferiority: BatchTopK reconstructs better at
  equal L0. Comparing at equal FVE would need a larger k for TopK, which would probably widen
  TopK's lead on children but was not tested.
- Held-out inference uses partitioned top-k, not Bussmann et al.'s fixed threshold. The decoder
  MMCS metrics do not depend on inference; FVE and L0 do.
- Concentrated batches also change gradient noise for every arm. The interaction (C3) controls
  for effects common to TopK and BatchTopK, not for every difference between the compositions.
- Toy scale, one architecture, CPU only. Not a claim about BatchTopK on language models.

## Reproduce
```
python experiments/batch_composition_2026_11/study.py confirm 12 out/confirm.json   # ~30 min on 1 core
python experiments/batch_composition_2026_11/analyze.py out/confirm.json out/summary.json
```
