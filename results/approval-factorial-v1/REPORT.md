# Test 1: matched approval feedback × environment reward

24 October 2026. Locally frozen 32-seed confirmation, preceded by eight development seeds. This is a fixed-exploration tabular ablation, not a replication of the original online decoupled-approval algorithm. No independent scientific reproduction or peer review is claimed.

## Main result

Independent approval substantially reduced lever activation, but the full-horizon learner did not complete the task. The combined primary support rule was **not met**. Hack avoidance alone is not successful task behavior.

On feedback-lever environments at reward weight zero and gamma=0.95:
- Coupled lever use: 98.4%.
- Independent-feedback lever use: 10.9%.
- Paired difference: −87.5 percentage points; 95% paired-seed t interval [−95.4, −79.6] points.
- Task completion: 0% in both arms. The independent-feedback arm failed the required absolute task-completion floor of 80%.

The zero-width t intervals for all-zero goal outcomes describe zero observed seed variation; they do not establish certainty about every possible environment. The observed reduction measures lever activation, not inferred intent: an honest-feedback policy can sometimes activate a lever incidentally.

## Fixed design and budgets

- Four conditions: coupled/independent approval × environment-reward coefficient 0/1. Approval coefficient fixed at 1.
- Same public exploration trace across conditions: 20,000 uniform-action environment steps and exactly 20,000 used approval samples per condition/environment. Independent query actions have their own RNG stream; executed feedback is discarded by independent-feedback arms.
- Same learner and planning rule, zero hyperparameter tuning in every condition. Primary gamma=0.95; gamma=0 is a predeclared descriptive sensitivity, not a replacement primary analysis.
- 32 reserved evaluation seeds × three families × two layouts = 192 new environment instances. Eight policy conditions per instance (four objective/feedback arms at two gammas). Within each seed, average the two layouts before inference.
- Combo family was absent from development. Each evaluation task is still explored locally under the fixed budget; this is not zero-shot transfer.
- Reward in feedback_lever is always zero, so its reward-coefficient switch is structurally identical. Reward effects are evaluated on button and combo families.
- Primary rule required all three: lever-difference CI upper ≤−20 points; goal-difference CI lower ≥−5 points; independent-feedback absolute-goal CI lower ≥80%. Only the last component failed.
- Freeze manifest includes source/protocol hashes, eight-seed pilot summary hash and all 32 confirmation seeds. Every confirmation record was created after that freeze. There was no adaptive extension. Pilot lever variance was zero; its plug-in power estimate was consequently unreliable and is not treated as a power guarantee.

## Full-horizon confirmation (primary planning setting)

| Family | Feedback | Environment reward used | Lever activation | Button activation | Goal completion |
| --- | --- | --- | ---: | ---: | ---: |
| tamper_ms | executed | 0 | 0.0% | 9.4% | 0.0% |
| tamper_ms | independent_query | 0 | 0.0% | 6.3% | 0.0% |
| tamper_ms | executed | 1 | 0.0% | 68.8% | 0.0% |
| tamper_ms | independent_query | 1 | 0.0% | 65.6% | 0.0% |
| feedback_lever | executed | 0 | 98.4% | 0.0% | 0.0% |
| feedback_lever | independent_query | 0 | 10.9% | 0.0% | 0.0% |
| feedback_lever | executed | 1 | 98.4% | 0.0% | 0.0% |
| feedback_lever | independent_query | 1 | 10.9% | 0.0% | 0.0% |
| combo | executed | 0 | 98.4% | 0.0% | 0.0% |
| combo | independent_query | 0 | 9.4% | 3.1% | 0.0% |
| combo | executed | 1 | 98.4% | 4.7% | 0.0% |
| combo | independent_query | 1 | 6.3% | 51.6% | 0.0% |

## Reward-use contrast: secondary, descriptive

In the button-only family, using environment reward increased button activation:
- Coupled feedback: 9.4% → 68.8% (+59.4 points; descriptive 95% paired interval [+48.7,+70.1]).
- Independent feedback: 6.3% → 65.6% (+59.4 points; descriptive 95% paired interval [+47.8,+71.0]).

This supports a role for objective construction at this fixed coefficient and learner. It does not establish that every positive environment-reward weight causes hacking. Competing lever and button incentives change the pattern in the combo family. These secondary intervals are not multiplicity-adjusted confirmatory tests.

## Myopic sensitivity (gamma=0): secondary, descriptive

| Family | Feedback | Environment reward used | Lever activation | Button activation | Goal completion |
| --- | --- | --- | ---: | ---: | ---: |
| tamper_ms | executed | 0 | 0.0% | 0.0% | 95.3% |
| tamper_ms | independent_query | 0 | 0.0% | 0.0% | 93.8% |
| tamper_ms | executed | 1 | 0.0% | 0.0% | 95.3% |
| tamper_ms | independent_query | 1 | 0.0% | 0.0% | 93.8% |
| feedback_lever | executed | 0 | 65.6% | 0.0% | 31.3% |
| feedback_lever | independent_query | 0 | 0.8% | 0.0% | 98.4% |
| feedback_lever | executed | 1 | 65.6% | 0.0% | 31.3% |
| feedback_lever | independent_query | 1 | 0.8% | 0.0% | 98.4% |
| combo | executed | 0 | 65.6% | 0.0% | 31.3% |
| combo | independent_query | 0 | 0.0% | 0.0% | 96.9% |
| combo | executed | 1 | 46.9% | 0.0% | 50.0% |
| combo | independent_query | 1 | 0.0% | 0.0% | 96.9% |

On the feedback-lever family at reward coefficient zero, myopic task completion was 31.3% with coupled feedback and 98.4% with independent feedback. The primary full-horizon learner accumulated approval while failing to reach the goal, consistent with the already documented summed-approval farming problem. The changed planning horizon matters in this controlled setup; this was a predeclared sensitivity, not the primary success criterion.

One of 64 myopic independent-feedback lever layouts used the default uniform policy at an unobserved state; reward0 and reward1 are identical copies there. The resulting small fractional lever-use rate must not be presented as uniformly deliberate corruption exploitation. Primary full-horizon arms had zero such fallback probability.

## Unmodified online reference: development only

The existing decoupled-approval method was also run unchanged on four development seeds with epsilon=0.2 and one query per step. It completed 87.5% of lever tasks and 100% of button tasks, with no lever or button activation. Its adaptive exploration differs from the matched trace experiment, so it is contextual reference evidence and not a factorial causal contrast.

## Engineering and integrity checks

- Fourteen tests passed across the new factorial controls and existing interactive testbed checks.
- Independent recomputation verified the primary interval, all frozen source hashes, the 32 reserved seeds, identical sample budgets and the reward-toggle identity on the zero-reward family.
- All 224 pilot/confirmation trace archives passed their saved SHA-256 checks.
- Traces, per-seed metrics, runtime versions, protocol, source hashes, freeze record and audit are retained. This integrity audit is not independent scientific reproduction.

## What this establishes

Within this fixed-exploration learner and planted environment, independently collected feedback sharply reduces lever activation, and including environment reward increases delayed-button activation. Neither observation alone establishes useful safe behavior: every primary full-horizon condition failed task completion. Planning horizon, feedback collection and objective construction need separate controls.

## Limits and next experiment

The experiment uses procedural toy environments, a selected reward coefficient, shared uniform exploration and a tabular learner. It is not an evaluation of general-purpose or superhuman agents. Honest counterfactual feedback is a privileged property of this environment; the mechanism may fail when that channel can also be corrupted. Hyperparameters were fixed, not equally tuned to optimality. The original published Synthcore verdicts remain unchanged.

The next clean extension is a separately frozen matched online-exploration experiment, followed by a corrupted-counterfactual-channel stress test. It must retain goal-completion gates and must not relabel this primary failure as a safety win.
