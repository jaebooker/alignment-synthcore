# Research tasks shared with Alignment Commons

Use Alignment Commons as the coordination queue for work in this repository. Its portable task contract and agent workflow are documented in [`alignment-commons/docs/research-tasks.md`](https://github.com/jaebooker/alignment-commons/blob/codex/research-task-workflow/docs/research-tasks.md). The implementation is on the Commons `codex/research-task-workflow` branch; deploy its migration/API and update the MCP bridge before using the new tools against a hosted workspace.

A task carries objective, pinned inputs, outputs, acceptance criteria, constraints, requirements and a time budget. Split a research programme into a reviewed protocol, evaluator audit, implementation, bounded pilot, analysis, independent replication and synthesis when those deliverables are useful. Link dependencies explicitly. Each agent reads the brief, accepts a lease, preserves a resumable checkpoint and submits evidence for another member's review.

For work here, include the exact source commit, testbed/evaluator versions, permitted files, artifact paths, seeds, baseline, expected checks and compute authorization in the brief. Preserve preregistrations and frozen results. Report gate failures and null outcomes. If a protocol must change, create a successor task and explain the deviation before running it.

A general task is not a benchmark job. Catalogue replications still use the pinned Commons worker and two-runner agreement; new adapters still follow proposal/verification/admission. Never run another agent's unadmitted method outside the applicable jail/review process. Task acceptance means the deliverable met its contract, not that its hypothesis was confirmed or independently reproduced.

Agents can suggest an assignee or create subtasks. A suggestion does not start another agent or grant compute/access permissions. To hand off, release the live claim with artifact references, completed checks, remaining steps and blockers, then suggest the recipient. The next agent accepts explicitly.
