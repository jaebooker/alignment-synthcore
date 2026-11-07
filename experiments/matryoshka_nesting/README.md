# Does nesting recover parent-child features? (matryoshka-nesting-v1)

Follow-up to `matryoshka-sae` (verdict `mechanism_failed`, triaged `weak_negative` with k at the grid edge). Proposal: `proposal-matryoshka-sae-1` in the Commons test matrix. Driver: `nesting.py`; engineering tests: `tests/test_matryoshka_nesting.py`.

Four arms share one trainer, initialisation, batch stream, optimiser, decoder width and k; only the loss groups differ: `topk`, `nested` (Matryoshka prefixes), `shuffled` (the same nested sizes over a fresh latent order every step) and `subsets` (the same sizes as fixed, non-nested random subsets). Parent and child labels are rebuilt from the generator's seeds and checked against the cached models.

## Status: calibrated, not frozen

Calibration (dev split, seeds 0 and 1, `results/matryoshka-nesting-v1/calibration.json`) is complete. Nothing has been frozen, registered or run on the test split.

Calibration is exploratory and selects hyperparameters; it is not evidence. At k=2 and k=3, consistently across both dev seeds, the fixed Matryoshka nesting recovered children 0.08-0.18 worse than shuffled nesting and 0.05-0.11 worse than TopK, with parents also worse; shuffled nesting beat TopK on children by 0.03-0.07; fixed random subsets behaved like Matryoshka. At k=1 all arms tie (child MMCS 0.81).

The selection rule as written picks k=1, where the arms cannot differ, so the primary k needs a decision before freezing. The freeze, registration (Commons document `matryoshka-nesting-v1-protocol`, revision 1) and confirmation (test split, 8 paired seeds) follow that decision.
