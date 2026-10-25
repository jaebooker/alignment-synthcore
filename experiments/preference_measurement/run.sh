#!/usr/bin/env bash
# Reproduce preference-measurement-v2. The frozen scripts expect to run from a directory where
# src/ is the repository root, fresh/ holds the policies and protocol.json sits alongside, so
# this builds that layout in a scratch directory instead of editing the frozen files.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); WORK=${1:-$(mktemp -d)}
mkdir -p "$WORK/fresh"; cd "$WORK"; ln -sfn "$REPO" src
for f in measure.py policy.py train.py confirm.py protocol.json dev_controls.py dev_policies.py dev_detours.py; do
  cp "$REPO/experiments/preference_measurement/$f" .; done
sha256sum -c "$REPO/results/preference-measurement-v2/freeze.sha256"
if [ "${RETRAIN:-0}" = 1 ]; then  # needs jax 0.4.35; weights are platform-dependent
  for e in 0.01 0.05; do python3 train.py fresh/entropy_$e.npz $e 53000,53001,53002,53003,53004,53005,53006,53007; done
else cp "$REPO"/results/preference-measurement-v2/models/*.npz fresh/; fi
python3 confirm.py   # batteries.py imports the vendored gw.py, which needs jax
echo "outputs in $WORK"
