# Test 3: preference-battery audit

This is a new local measurement audit of the user-supplied Bounded Agency
implementation. The archived code is preserved byte-for-byte under `vendor/`.
The archive has outcome summaries but no saved trained policies. This study
therefore trains sixteen new policies, rather than claiming to re-evaluate the
original archived policies.

Use an isolated Python 3.11 environment and install `requirements.txt`. The
recorded CPU execution requests these environment settings:

```sh
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_preference_audit.py
python -m experiments.preference_audit.run pilot
python -m experiments.preference_audit.run calibrate-dev
python -m experiments.preference_audit.run initialize
python -m experiments.preference_audit.run train
python -m experiments.preference_audit.run freeze
python -m experiments.preference_audit.run evaluate
python scripts/audit_preference_battery.py
python scripts/report_preference_battery.py
```

Start a new execution in a clean results directory; preserve prior result
directories and their freezes. Training refuses a changed frozen source and
evaluation refuses to overwrite an existing evaluation record. Final weights use
pickle-free NPZ files, with eight policies per entropy-cohort archive.

The primary batteries keep the fixed proxy corner meaningful to the original
three-channel policy input. The anti-diagonal reflection fixes that corner;
the distance battery retains it while reweighting gaps and absolute distances.
The separate rotated-proxy stress test moves an endpoint that is not marked in
the observation. That task change cannot diagnose latent-goal instability.

The archive's placement exclusion is asymmetric. Exact controls find a shortest
route avoiding the unwanted terminal, including cases where another shortest
route would hit it. Neural policies retain their actual path choices.

Four primary paired-seed contrasts use adjusted 98.75% t intervals and a fixed
10-percentage-point equivalence margin. Neither outcomes stay in unconditional
choice-rate denominators. A separate non-reach gate limits goal interpretation.
Legacy implied weights retain censoring and are not precise terminal goals.
See `protocol.json` for the prospective design and `results/.../REPORT.md` for
the completed evidence and limits.
