# Test 2 — does the predicted mechanism survive baseline tuning?

24 October 2026. Fresh equal-budget selection on current evaluators, followed by disjoint development gate checks and a locally frozen eight-seed paired confirmation per idea. This is an audit of the local implementations/testbeds, not an independent reproduction of the papers. Original result records and verdicts remain unchanged.

## Finding

All three fresh toy point gates passed at defaults and failed after safety-based tuning of both arms. In reserved full confirmation, all three simultaneous intervals had upper bounds below the declared 10-percentage-point mechanism advantage. This rules out that minimum in this bounded setup under the stated interval assumptions; it does not establish zero effect or refute the papers. Minimax-regret and ACCEL also had worse safety and capability than the tuned domain-randomization baseline.

## Design

- Ideas: causal-confusion interventions, minimax-regret levels and ACCEL.
- Fresh tuning: the existing four-candidate grid per method and per baseline, each evaluated on two fresh development/full seeds (12000–12001). Select highest mean safety; resolve exact ties by original candidate order. No selection on mechanism diagnostics.
- Gate audit: four separate development/toy seeds (22000–22003), comparing defaults with selected settings on the same instances. These comparisons isolate a tuning-setting change within the toy configuration.
- Confirmation: eight reserved test/full seeds (92000–92007). Retain the original diagnostic, orientation and minimum effect of 0.1. Do not attribute all toy-to-full changes to tuning.
- Primary decision: the selected method’s oriented diagnostic improvement relative to its selected baseline must clear the declared minimum. Report Bonferroni 98.333% paired-seed t intervals across three ideas: a lower bound ≥0.1 supports the declared minimum; an upper bound <0.1 rules it out in this setup; otherwise the result is inconclusive at that minimum. Original point-gate decisions are also shown separately.
- Safety/capability comparisons and 95% intervals are secondary/descriptive. Eight seeds can leave broad intervals; a failed significance test is not proof of no effect. The t intervals assume seed-level mean differences are adequately described by this approximation.
- Original per-task budgets are unchanged, including 300,000 UED environment steps per task. Equal candidate counts do not mean equal wall-clock compute, model size, or supervision access.
- Both UED methods share the same baseline fits when the configuration, seed and selected hyperparameters match. This saves compute without creating extra independent observations.
- Protocol/source were frozen before selection; selected hyperparameters were frozen before gate audit or confirmation. The audit always tunes both arms even if the default gate fails.

## Fresh toy gate: defaults versus selected settings

| Idea | Default diagnostic advantage | Default point gate | Selected diagnostic advantage | Selected point gate |
| --- | ---: | --- | ---: | --- |
| Causal confusion | 29.9 pp | pass | 0.8 pp | fail |
| Minimax regret | 29.9 pp | pass | 4.6 pp | fail |
| ACCEL | 32.4 pp | pass | 4.8 pp | fail |

Positive advantage means less copycat/proxy behavior than the baseline. The declared gate minimum is 10 percentage points. These four-seed development comparisons are descriptive, not a literature-wide gate failure rate.

## Reserved full confirmation

| Idea | Baseline diagnostic rate | Method diagnostic rate | Oriented improvement | Simultaneous 98.333% interval | Practical-effect decision |
| --- | ---: | ---: | ---: | --- | --- |
| Causal confusion | 7.2% | 7.8% | -0.6 pp | [-2.5, 1.2] pp | 10-point minimum ruled out here |
| Minimax regret | 23.2% | 34.7% | -11.6 pp | [-26.2, 3.1] pp | 10-point minimum ruled out here |
| ACCEL | 23.2% | 21.3% | 1.8 pp | [-3.0, 6.6] pp | 10-point minimum ruled out here |

## Secondary safety and capability

| Idea | Method safety | Baseline safety | Safety difference (95% interval) | Capability difference (95% interval) |
| --- | ---: | ---: | --- | --- |
| Causal confusion | 94.5% | 93.0% | 1.5 pp [-0.03, 3.0] | -0.8 pp [-1.0, -0.6] |
| Minimax regret | 51.8% | 85.8% | -34.1 pp [-40.9, -27.2] | -8.9 pp [-10.3, -7.6] |
| ACCEL | 57.7% | 85.8% | -28.1 pp [-34.6, -21.7] | -17.2 pp [-20.8, -13.5] |

The diagnostic is not the safety endpoint. A reduced proxy/copycat rate can coexist with worse task capability or worse intended-goal success. A safety advantage without the specified diagnostic improvement leaves the claimed mechanism unresolved.

## Selected settings

- confounded_imitation|ideas/causal-confusion-interventions/method.py: `{"epochs": 120, "hidden": [64, 64], "lr": 0.001, "mode": "queries"}`.
- confounded_imitation|testbeds/confounded_imitation/baseline_bc.py: `{"epochs": 60, "hidden": [128, 128], "lr": 0.003}`.
- ued_gridworld|ideas/minimax-regret-levels/method.py: `{"beta": 0.3, "capacity": 1000, "ent": 0.01, "lr": 0.001, "p_replay": 0.5, "rho": 0.3}`.
- ued_gridworld|testbeds/ued_gridworld/baseline_dr.py: `{"ent": 0.01, "lr": 0.001}`.
- ued_gridworld|ideas/accel-levels/method.py: `{"beta": 0.3, "capacity": 1000, "edits": 3, "ent": 0.01, "lr": 0.0005, "p_replay": 0.7, "rho": 0.3}`.

## Historical version boundary

The archived minimax-regret record uses an earlier UED evaluator hash. The new minimax and ACCEL fits both use current UED v0.2, with fresh tuning. Historical point gates are context, not same-version replications. The causal-confusion and ACCEL historical evaluator hashes match the current source; their old seeds are still not reused as fresh evidence.

## Integrity and reproduction

- 120 unique fitted-model archives passed SHA-256 checks.
- Candidate budgets, safety-based selection, disjoint stage seeds, selection-freeze ordering, source hashes and paired intervals were independently recalculated by the integrity script.
- 5 saved fitted models were re-scored against regenerated evaluation instances using the same evaluator. This tests reproducibility of saved evidence, not independent scientific replication.
- The public records ZIP contains scores, metadata and hashes; the complete local evidence bundle also includes fitted NPZ weights for re-scoring.
- Fourteen new-audit plus existing interactive engineering tests passed before fitting. Fifteen passed after the saved-score checker received a regression test for nested metrics and metadata.

Run `python -m experiments.tuning_audit.run initialize`, then `select`, `freeze`, `audit`, `confirm`, and `report` in a clean output directory. Keep the exact source/protocol hashes and runtime metadata. Model archives use pickle-free NPZ; no old baseline cache was used for fitting.

## Interpretation limits

This audit uses three selected ideas and small toy testbeds, a limited four-setting search, two selection seeds and eight confirmation seeds. It cannot estimate how common baseline sensitivity is across alignment research, prove the methods never work, or identify a unique causal explanation for a performance difference. Causal-confusion interventions have extra interaction/query access by design; the baseline remains the declared behavior-cloning comparator. Practical-regret curricula here do not implement an oracle-regret variant or the papers’ larger training budgets.

The next study should change one uncertainty at a time—such as a broader prespecified search, more independently reserved seeds, or a different task family—without rewriting this audit’s outcome.
