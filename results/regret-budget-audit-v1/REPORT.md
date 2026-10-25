# Test 5 — regret estimator and training budget audit

**Primary verdict: supported.** Oracle minus max-latest at 1M requested steps: 20.6 percentage points (95% paired-seed t interval 15.0 to 26.3). Oracle IID capability: 97.4%. The prespecified rule requires the interval lower bound above 10 points and IID capability at least 80%.

Four locally implemented curricula use the same student CNN/PPO: domain randomization, existing max-latest ACCEL, uniform random replay/editing, and privileged exact-start oracle regret. Two learning rates were tuned on two separate development seeds, independently at each budget. Confirmation uses eight reserved seeds. This is a new corner-family ablation of the local implementation, not a replication or a replacement for Test 2.

| Budget | Arm | Goal success | IID capability | Cue entry |
|---|---|---:|---:|---:|
| 301,056 environment steps | Domain randomization | 75.4% | 98.1% | 28.4% |
| 301,056 environment steps | Max-latest ACCEL | 44.5% | 85.9% | 39.8% |
| 301,056 environment steps | Random replay/edit | 73.1% | 95.8% | 24.5% |
| 301,056 environment steps | Oracle ACCEL | 80.4% | 80.8% | 16.3% |
| 1,001,472 environment steps | Domain randomization | 92.8% | 99.4% | 20.9% |
| 1,001,472 environment steps | Max-latest ACCEL | 75.8% | 93.2% | 25.0% |
| 1,001,472 environment steps | Random replay/edit | 93.1% | 99.3% | 24.9% |
| 1,001,472 environment steps | Oracle ACCEL | 96.4% | 97.4% | 21.3% |
| 180 training CPU seconds + boundary | Domain randomization | 67.4% | 97.6% | 33.6% |
| 180 training CPU seconds + boundary | Max-latest ACCEL | 51.9% | 90.1% | 38.1% |
| 180 training CPU seconds + boundary | Random replay/edit | 82.0% | 97.9% | 24.6% |
| 180 training CPU seconds + boundary | Oracle ACCEL | 79.6% | 83.2% | 17.9% |

## Interpretation

The primary oracle-replacement decision is supported. At1M requested steps, random replay/editing reaches 93.1% intended-goal success, compared with 75.8% for max-latest and 96.4% for oracle. At matched CPU, random reaches 82.0%, oracle 79.6%, and max-latest 51.9%. These are descriptive secondary comparisons. The random control is essential when assessing whether privileged information is necessary; random-versus-oracle superiority or equivalence was not a registered inferential contrast.

Selected max-latest performance is 44.5% at 300k and 75.8% at 1M steps, with independently tuned DR at 92.8% at 1M. Selected rates may differ between budgets, so this does not isolate extra training of one fixed configuration. A single CPU budget cannot establish how the methods converge with larger total compute. The local ranking/eviction scheme and remaining budget explanations must both be considered.

Oracle IID capability is 97.4% at the primary checkpoint, compared with the 80% gate. At early/matched-CPU budgets its capability is 80.8%/83.2%, versus 98.1%/97.6% for DR. Oracle-versus-DR superiority was not the registered primary contrast.


## Secondary comparisons (descriptive)

Positive differences favor the first named arm for intended-goal success. These intervals are not multiplicity-adjusted and do not replace the primary test.

| Budget | Comparison | Difference, percentage points | 95% paired-seed interval |
|---|---|---:|---:|
| steps_300000 | Oracle ACCEL minus Max-latest ACCEL | 35.9 | 22.5 to 49.3 |
| steps_300000 | Max-latest ACCEL minus Domain randomization | -30.9 | -48.2 to -13.5 |
| steps_300000 | Random replay/edit minus Max-latest ACCEL | 28.6 | 13.5 to 43.7 |
| steps_1000000 | Oracle ACCEL minus Max-latest ACCEL | 20.6 | 15.0 to 26.3 |
| steps_1000000 | Max-latest ACCEL minus Domain randomization | -17.0 | -22.7 to -11.3 |
| steps_1000000 | Random replay/edit minus Max-latest ACCEL | 17.3 | 11.8 to 22.7 |
| cpu | Oracle ACCEL minus Max-latest ACCEL | 27.7 | 12.7 to 42.7 |
| cpu | Max-latest ACCEL minus Domain randomization | -15.5 | -32.5 to 1.5 |
| cpu | Random replay/edit minus Max-latest ACCEL | 30.1 | 15.3 to 44.8 |

Learning rates may differ between budget checkpoints because they were selected independently. An early-to-late difference therefore describes selected performance at each budget; it does not isolate extra training of one fixed policy configuration. One matched CPU checkpoint cannot establish recovery across a CPU-budget curve.

| Arm at CPU budget | Mean actual CPU seconds | Mean env steps | Mean scoring steps | Mean oracle CPU seconds |
|---|---:|---:|---:|---:|
| Domain randomization | 180.77 | 246784 | 0 | 0.00 |
| Max-latest ACCEL | 180.39 | 410112 | 227866 | 0.00 |
| Random replay/edit | 180.38 | 407040 | 240263 | 0.00 |
| Oracle ACCEL | 180.59 | 359168 | 195921 | 25.75 |

## Instrumentation correction

Before development selection or any confirmation run, code inspection found that checkpoint scoring retained NumPy views backed by live network parameters. Saved weights were correct, but early metrics could use later weights. The entire interrupted attempt was excluded and archived separately; every valid study fit was restarted after an immutable on-disk snapshot loader and a mutation regression test were added. Scientific design, seeds, budgets and outcomes were unchanged. The old and corrected source freezes are preserved in instrumentation-correction.json. No invalid metrics were used for selection or interpretation.


## Accounting and oracle certification

Training CPU seconds include both training threads, environment draws, edits, ranking, score-only rollouts and finite-horizon oracle certification. Initialization is included; imports, snapshot serialization and held-out evaluation are excluded. Each checkpoint is saved after a complete 2,048-step PPO rollout. The 180-second checkpoint overshoots by at most 1.58 seconds in this run. This matches a local CPU resource, not FLOPs or energy.

Each oracle label is the exact finite-horizon attainable return from the actual training start minus the observed return. Integer Bellman dynamic programming is checked against independent shortest-path values for every permissible start on each queried level. Distances exceeding the 19-step horizon have value zero. Labels affect only curriculum scoring, eviction and ranking. Evaluation tables are generated only after training. The oracle changes both the max-ever reference and cross-start bias, so it does not isolate one estimator defect.

The independent geometry-based audit re-scored all 144 saved policies from 56 fresh fits, checked source freezes, model hashes, all job counts, tuning winners, seed separation and budget boundaries. Maximum metric discrepancy: 0.0. This is engineering verification, not independent scientific reproduction.

## Literature scope

[Abdel Sadek et al., Mitigating Goal Misgeneralization via Minimax Regret](https://arxiv.org/html/2507.03068v1) studies max-latest and graph-based oracle-latest estimators. Its reported corner experiments include 200M training steps (Appendix J), compared with this local 1M-step ceiling. These environments, curriculum details and evaluation protocols differ. This audit examines the local implementation, not the paper’s theoretical guarantees or full empirical claim.

## Limits

- Corner family only; 3M and lamp unrun.
- Two development seeds, two rates; tuning is limited and development outcomes noisy.
- The 2025 paper includes 200M-step corner experiments; this local 1M-step ceiling is substantially smaller and uses different mechanics.
- Eight seed groups yield wide intervals; no episode-level pseudoreplication.
- Oracle has privileged geometric information, and corrects both max-ever underestimation and cross-start reference bias.
- Only the attainable-return reference is exact; sampled policy returns and recent-window curriculum scores remain noisy.
- In-process PPO ablation; no new official benchmark verdict.
- 180 CPU-second checkpoint is local resource matching, with recorded rollout-boundary overshoot. No FLOP/energy claim.
- Secondary comparisons and budget changes are descriptive. One CPU cap cannot establish catch-up across a CPU-budget curve.
- Entering the cue zone is not exclusive decoy pursuit; successful trajectories can enter it.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run initialize
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run select
python -m experiments.regret_budget.run freeze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run confirm
python scripts/audit_regret_budget.py
python scripts/report_regret_budget.py
```

Start with an absent results/regret-budget-audit-v1 directory for a fresh independent run. CPU checkpoint weights and realized step counts may vary with hardware and process scheduling. Step checkpoints remain deterministic within the documented numerical runtime. All fitted weights and per-level certificate digest tables are in the complete local complete-evidence.zip. Source/public records archives omit these large artifacts; per-file SHA256 hashes and aggregate certification counts remain in the fit records. Independent re-scoring requires the complete local bundle.

Related method: [Parker-Holder et al., Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302). The published method motivates this audit; its broader claims are not tested here.
