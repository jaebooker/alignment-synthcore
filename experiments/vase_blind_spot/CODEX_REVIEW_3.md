AI-written by Codex.

Third review for Hermes and Jaeson Booker, 25 October 2026. Reviewed `3d303e5` on `hermes/vase-blind-spot`, including `git diff HEAD~1..HEAD`; no branch switch. References below are to this revision.

## Verification

`PYTHONDONTWRITEBYTECODE=1 /Users/jaesonbooker/anaconda3/bin/python3 -m pytest -q -p no:cacheprovider tests/test_vase_blind_spot.py`: **16 passed in 2.01s**. These tests use replacement cohorts, not registered study seeds. Adapted `/private/tmp/vase_review2_probe.py` into `/private/tmp/vase_review3_probe.py`; all probe assertions passed, with expected rejections and the acceptance failures below recorded in `/private/tmp/vase_review3_probe.log`. Probe records are fabricated; the interrupted-write probe stubs environment construction and evaluation. Outputs are under the temporary directory printed in that log.

No registered dev/confirmation environments were run, and `results/vase-blind-spot-v1/dev-sealed` was not read. Registry arithmetic alone confirms 240 unique IDs and no Test 1/7 collision; a forced Test 1 collision is rejected. This review is the sole repository addition; existing untracked files were left alone.

## Blockers from review 2

| Item | Status | Evidence |
| --- | --- | --- |
| B1: validated selection at freeze | **Resolved** | `run.py:290–333` shares deterministic selection computation, validates current dev provenance, and compares the entire saved selection. Independent probes reject lambda 999, edited feasibility, changed source/protocol signature, and selection after confirmation starts. Existing tests retain write-once freeze and frozen-arm behavior. |
| B2: arm/schema provenance | **Resolved** | `run.py:219–274` checks top-level and row configs, exact arm sets, null dev freeze digest, exact public metric whitelist, cells, IDs and numeric bounds. Independent probes reject empty configs, missing/extra arms, hidden metrics, wrong stage/seed/environment, duplicate cells, missing metrics/configs, out-of-range values and confirmation lambda 888. NaN rejection is covered by the passing suite. |
| B3: committed, cache-validated sealed artifacts | **Partially resolved; still blocking** | `run.py:189–215,293–297,328–354` writes sealed first, commits its digest publicly and at freeze, checks missing/replaced sidecars on cache reuse/select, rejects post-freeze sealed edits, and checks overlapping metrics at report. Missing/replaced/disagreeing sidecars and interrupted paired writes were probed successfully. However, the overlap check ignores cell identity, and report never checks public records against their frozen hashes; see below. |

### Remaining B3 failure: positional comparison and unused public commitments

`report`, `run.py:349–354`, zips public and sealed rows by position, although `load_stage` validates their cell sets without requiring matching order. With two fabricated cells, sealed size-6 completion is **0.8** and size-7 is **1.0**. Swap only the public rows' size/environment-ID labels, retaining metric positions. Both records remain individually schema-valid, but public size-7 now says **0.8** and size-6 says **1.0**. Selection, freeze and complete-confirmation report all succeed despite per-environment disagreement. This succeeds with the disagreement present **before freeze**, so verifying file digests alone would not fix it.

A second probe makes that public edit **after freeze**: report still succeeds, although its public digest differs from `freeze['selection']['dev_record_sha256']['seed-101']`. Those frozen public commitments are never checked at report. Conversely, reversing complete public rows produces a positional false mismatch. Reproducers: `B3-committed-cell-mismatch`, `B3-cell-identity`, and `B3-row-order` in the temporary probe/output.

Fix before dev: verify every public dev digest against the frozen selection before unsealing; compare public/sealed records by `(family, size, environment_seed)`, not list position, and check every shared metric within the matching cell. Add regressions for committed cell disagreement and post-freeze public replacement. This is a dev provenance/consistency blocker; the demonstrated swaps preserve aggregate dev means and do **not** change the confirmation support calculation.

## Nonblocking items and retained review-2 checks

| Item | Status | Evidence / remaining limit |
| --- | --- | --- |
| Original M3: infeasibility/completion gates | **Resolved, retained** | `run.py:302–305,359–369` retains lambda-zero infeasible fallback, feasibility requirement, and the four registered interval gates. |
| Original M4 / movement interpretation | **Partially resolved** | `penalty.py:12` and protocol `mechanism_secondary` now call this descriptive and avoid mechanism identification; `run.py:374–376` adds paired RR-minus-move goal completion. Protocol `predictions.move` still categorically says a similar effect means RR is not specific to reachability; soften this to agree with the stated interpretation limit. The narrowed primary claim remains adequate. |
| Original M5 / mathematical validation | **Resolved, retained** | Passing suite retains learner equivalence, feedback isolation, hand-computed RR/AUP, graph validation and exemptions. Additional synthetic analytic AUP probe verifies terminal-zero continuation (expected penalty **0.5763853701**). |
| S1: incomplete-model diagnostics / contrasts | **Resolved, retained** | Coverage diagnostics and missing-baseline disclosure remain; separate button/lever, unknown-state, family and paired outcome contrasts remain (`penalty.py:174–181`; `run.py:371–380`). |
| S2 / auxiliary RNG / collision boundary | **Resolved** | Explicit Test 1 collision check added (`run.py:105–118`), independently forced and rejected. New reproducibility/stream-separation test passes; scoped `P_BLIND` test passes. |
| S3: terminal variant | **Resolved** | Variant remains disclosed; exemption tests and the added analytic terminal-value probe pass. |
| S4: harmless-vase counterfactual | **Resolved, retained** | `benign` remains a fixed-policy evaluator-only calculation (`run.py:121–131`), with adaptation/causal limits disclosed. |
| S5: fixed grids / cooperative blinding / release timing | **Partially resolved** | New `grid_endpoint_policy` explicitly forbids widening/refining qualifying or infeasible grids; `blinding` explicitly prohibits reading sealed outcomes until complete-confirmation report. B3 remains. Also update `selection_rule`'s older “sealed until freeze” wording to match the report-only release policy. |
| Precomputed `P` hardening | **Partially resolved; nonblocking for current caller** | `penalty.py:151–158` now rejects unknown penalty names and wrong-shaped, negative or nonfinite tables. Independent probes verify these and exact cached/fresh outputs plus no mutation for all four penalties at four lambdas. Arbitrary finite, nonnegative, same-shaped tables still lack penalty/graph/index identity binding. Current `run_env` builds and uses tables from the identical trace, so no current-call mismatch was found. |
| Statistical report / signs | **Resolved, retained** | Pairing, conjunction, feasibility and interval directions remain correct by inspection; synthetic complete reports succeed and include the new goal contrast. No additional support-calculation bug found. |

Low-priority diagnostic gap: removing public `sealed_sha256` still raises raw `KeyError` from `compute_selection` rather than `SchemaError`; a missing sealed file at report raises `FileNotFoundError`. Both fail closed, but clear recovery diagnostics would help. No other new production bug was found in this bounded review.

BLOCKED with remaining blocker B3: bind report to frozen public dev hashes and compare public/sealed metrics by environment cell.
