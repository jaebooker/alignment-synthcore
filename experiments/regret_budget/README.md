# Test 5: regret estimator versus training budget

Read `protocol.json` for frozen design and precise claim boundaries. The original
sandboxed benchmark files are unchanged. `core.py` copies its PPO update mechanics
into an inspectable in-process adapter so total training process CPU can be counted.
It loads the original ACCEL curriculum and changes only the documented scores and
random control. Privileged labels are training-only completed-episode returns;
the learner never receives them as observations or loss inputs.

Run from repository root with the installed NumPy/SciPy/PyTorch runtime:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -p no:capture -q tests/test_regret_budget.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run initialize
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run select
python -m experiments.regret_budget.run freeze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run confirm
python scripts/audit_regret_budget.py
python scripts/report_regret_budget.py
```

Do not overwrite an existing freeze. Copy the existing results aside before a new
independent run. A completed fit can be reused only when its full job identity,
source hashes and checkpoint requests match. An interrupted partial fit is retrained.
All source, complete ledgers, certificate digests and policy hashes are retained;
large fitted weights and per-level certificate tables are excluded from Git and public records, and included in the complete local bundle. Public fit records retain per-file hashes and aggregate certification counts. Re-scoring requires that complete bundle.
