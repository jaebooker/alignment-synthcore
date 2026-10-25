# Test 6 — native Go continuation audit

This separate module extends four archived Go Concept Lab development cases.
It does not alter milestone 4, pool the manuscript study, or install a worker adapter.
Read protocol.json for frozen cases, decisions, strict endpoints and native rules.

Two source games yield four overlapping case/engine strata. The same associated
historical engine plays both colors at 128 root visits per generated move, after
forcing either its archived choice or the first annotated legal defensive move.
Eight paired seeds measure sensitivity within each case, not position prevalence.
All original stones retain permanent identities through captures and reoccupation.
Early terminal retention is not silently extrapolated to a fixed horizon.

Native binaries/models are external. input-provenance.json contains exact hashes;
locations JSON supplies machine paths without publishing them:

```json
{"leela":{"binary":"/path/to/leelaz","weights":"/path/to/leela-zero-57.txt.gz"},
 "katago":{"binary":"/path/to/katago-reference","weights":"/path/to/kata1-b6c96-s175395328-d26788732.txt.gz"}}
```

Run from this source checkout using Python with NumPy 2.0.2 and sgfmill 1.1.1
(original run used Python 3.9.10; matplotlib 3.9.4 generates the report figure):

```sh
python -m experiments.go_continuations.run initialize --locations /path/to/locations.json
python -m experiments.go_continuations.run execute --locations /path/to/locations.json
python scripts/audit_go_continuations.py
python scripts/report_go_continuations.py
```

Start with an absent results/go-continuation-audit-v1 for an independent run.
The fixed archived native binary hashes are platform/build-specific. A rebuilt
executable requires an explicitly versioned new input-provenance/protocol freeze;
do not substitute it into this existing run. Official engine sources are
https://github.com/leela-zero/leela-zero and https://github.com/lightvector/KataGo .
The pinned source commits and external binary/weight identities are recorded.

GTP transport, settings and sgfmill line checks are copied from the local Go
Concept Lab helpers, with original source hashes in input-provenance.json.
A separate NumPy audit replays all prefixes and continuations, checks every native
board and original-stone identity, raw visit logs, transcripts, SGFs, area scores,
all planned branches and source freezes. A formatting regression for Leela's
row-adjacent last-move marker was added to the independent audit after fitting;
no frozen continuation code, cases, seeds or outcomes changed.

Complete local evidence includes raw logs, which can contain machine paths.
Public records omit those logs and retain their hashes, full legal move sequences,
board snapshots, GTP transcripts and endpoint data. Native binaries/models are
not redistributed. Truncated lines have no invented game result.
