# Test 6 — native Go continuation audit

**Primary outcome:** 1/4 selected case/engine comparisons show a positive 64-ply original-stone preservation advantage under the frozen worst-case rule; 0/4 remain unresolved. These four comparisons overlap within two source games and are not a population sample.

Each archived case branches into its recorded 128-visit engine choice and the first independently legal safe alternative in its existing annotation. The historical source-game move can differ from the recorded engine move. A fresh process replays the complete original prefix; the associated engine then plays both colors at 128 root visits per generated move. Eight fixed seeds per branch and a 128-ply ceiling were frozen before outcomes. Ply 1 is the forced move.

Completed 64 planned branches and independently checked 6912 continuation plies. Engine errors: 0. Binaries and model hashes match the archived milestone-4 engines.

| Case / engine | Branch moves (recorded / defense) | Original stones | 64-ply retained / lost / unknown, recorded | 64-ply retained / lost / unknown, defense | Worst/best paired effect |
|---|---|---:|---:|---:|---:|
| LZ G1 m249 | N12 / R9 | 49 | 4 / 4 / 0 | 8 / 0 / 0 | 50.0 to 50.0 pp |
| LZ G1 m251 | N13 / Q12 | 49 | 8 / 0 / 0 | 8 / 0 / 0 | 0.0 to 0.0 pp |
| LZ G6 m221 | K1 / A8 | 34 | 7 / 1 / 0 | 6 / 2 / 0 | -12.5 to -12.5 pp |
| KataGo G6 m221 | J11 / A8 | 34 | 0 / 8 / 0 | 0 / 8 / 0 | 0.0 to 0.0 pp |

Unique move trajectories across the eight seeds (recorded / defense): LZ G1 m249: 8 / 8; LZ G1 m251: 8 / 8; LZ G6 m221: 8 / 8; KataGo G6 m221: 1 / 1.

## Interpretation

A legal immediate capture opportunity and a beneficial longer continuation are separate properties. The NumPy audit reconfirms every archived immediate loss/defense certificate, while the continuation tables report what the frozen native policies actually do. Those policies need not exploit every available capture. Only the first archived defensive alternative is tested; other legal defenses may yield different outcomes.

These data support case-specific material-retention conclusions under the tested policies. They do not establish an avoidable whole-game loss: most branches are horizon-truncated, and a terminal-board score remains a bounded self-play result rather than a forced-win certificate.

| Case | Recorded first original-stone capture | Defense first original-stone capture |
|---|---|---|
| LZ G1 m249 | ply2: 4/8; not observed: 4/8 | not observed: 8/8 |
| LZ G1 m251 | not observed: 8/8 | not observed: 8/8 |
| LZ G6 m221 | ply16: 1/8; not observed: 7/8 | ply10: 2/8; not observed: 6/8 |
| KataGo G6 m221 | ply4: 8/8 | ply4: 8/8 |

The effects above are deterministic sensitivity bounds over these eight selected paired seeds, not confidence intervals or population estimates. No pooling over four case/engine strata is used for inference.

## Strict endpoints and identity accounting

An original stone is permanently lost at its first capture. Merging with other stones does not add original targets, and later replacement stones do not resurrect captured identities. A loss before the deadline resolves all-original retention as failure. Reaching the deadline with every original stone retained resolves success. An early terminal or engine error with all original stones still present remains unresolved. Fractional retention bounds and first/complete capture times are retained in summary.json.

## Later horizon

| Case | 128-ply recorded retained / lost / unknown | 128-ply defense retained / lost / unknown | Worst/best paired effect |
|---|---:|---:|---:|
| LZ G1 m249 | 4 / 4 / 0 | 8 / 0 / 0 | 50.0 to 50.0 pp |
| LZ G1 m251 | 8 / 0 / 0 | 8 / 0 / 0 | 0.0 to 0.0 pp |
| LZ G6 m221 | 7 / 1 / 0 | 6 / 2 / 0 | -12.5 to -12.5 pp |
| KataGo G6 m221 | 0 / 8 / 0 | 0 / 8 / 0 | 0.0 to 0.0 pp |

## Game outcomes and remaining limits

There were 16 two-pass terminations, with 16 exact native/physical terminal-area score agreements. Both branches had verified terminal scores in 8 paired seeds; 0 of those pairs worsened the target-side win/loss result under defense. Horizon-truncated branches receive no winner. Terminal-board area is a bounded protocol outcome, not an independently adjudicated life/death or perfect-play result.

- Four outcome-selected case/engine strata overlap within only two source games; not four independent positions or a failure-rate sample.
- Eight search seeds measure sensitivity within a fixed case and policy, not an independent position population. Distinct seeds may produce identical trajectories; unique counts are reported.
- Both colors use the same associated historical engine; no optimal-adversary, forced-life or forced-win certificate. Absence of capture in self-play does not establish robustness against an opponent choosing a known legal capture witness.
- Only the first archived immediate defensive alternative is tested per case; other legal defenses may produce different later outcomes.
- Strict horizon retention is unresolved after early termination with all original stones still present. Captured identities never return when a point is reoccupied.
- Terminal-board occupancy/area is not independently adjudicated life and death. Only exact native/physical-area agreement receives a scored terminal record.
- Leela positional superko and native symmetries differ from Kata Chinese simple ko and forced identity; do not pool engines.
- Historical engineering checkpoints match archived hashes; the original published victim identities remain unresolved.
- No source-game result is reused as a continuation outcome. Truncated branches receive no invented winner.
- Reserved attack cohort and matched ordinary controls are later stages, not run here. The manuscript study remains separate.

## Verification and reproduction

A separate NumPy simulator replays every source prefix and continuation without using production sgfmill board transitions. It checks native boards after every move, permanent target removals, the original immediate tactical certificates, root visits, search/GTP response identities, output SGFs, score tallies, all branch counts and frozen file hashes. This is engineering verification, not independent scientific replication.

The native executables and models are external pinned dependencies; input-provenance.json lists hashes. Supply a private locations JSON mapping leela/katago to binary and weights paths. Run from the source checkout using an environment with NumPy and sgfmill:

```sh
python -m experiments.go_continuations.run initialize --locations /path/to/locations.json
python -m experiments.go_continuations.run execute --locations /path/to/locations.json
python scripts/audit_go_continuations.py
python scripts/report_go_continuations.py
```

Use an absent results/go-continuation-audit-v1 directory for an independent run. Existing error records are retained rather than retried. Native raw logs are included in the complete local evidence bundle; public records retain their hashes, move sequences, board snapshots, transcripts and full endpoint data. Absolute machine paths are not published. Source models/binaries are not redistributed.

The next ordered stage is a separately reserved attack-family cohort, followed by matched ordinary controls. No such cohort has been selected or evaluated here. The Loops Look Alive manuscript/native-engine check remains separate.


## Verified terminal-score details

| Case | Both terminal scores verified | Distinct target-relative area margins, recorded / defense |
|---|---:|---|
| LZ G1 m249 | 0/8 | [] / [] |
| LZ G1 m251 | 0/8 | [] / [] |
| LZ G6 m221 | 0/8 | [] / [] |
| KataGo G6 m221 | 8/8 | [-41.5] / [-41.5] |