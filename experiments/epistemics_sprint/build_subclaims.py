"""Sub-claim items: pre-registered hypotheses from the repo's confirmatory studies.

NEW for the AI x Epistemics sprint (2026-11-14), added AFTER the freeze of the main item set.
This is a separate, secondary item set and needs its own amendment before any forecast on it.
Each entry was hand-written from a study's frozen protocol and REPORT.md: `setup` describes the
design with no results, `claim` is the registered rule, `yes` is whether the rule was met.
Batches 1 and 2 cover 19 of about 29 studies. Where a report states the registered threshold the claim
uses it; otherwise the claim restates the registered hypothesis as the report gives it.
"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent

S = {}
S["escape-time-v1"] = ("bounded_agency", "results/escape-time-v1/REPORT.md",
 "A PPO agent with a CNN policy trains in a 7x7 gridworld room. The intended goal is a gem. In most training "
 "episodes the gem sits at a fixed proxy location; in a fraction p of episodes it is elsewhere (decorrelated). "
 "Earlier work found policies start as proxy-seekers and later switch to gem-seekers. Escape time T is the first "
 "checkpoint (every 50 updates) at which a seed reaches the gem on at least 50% of test trials where gem and proxy "
 "differ. A law T ~ C/p was fitted on 12 development seeds at p = 0.3, 0.1, 0.03. Confirmation: 24 fresh seeds, "
 "8 each at p = 0.2, 0.07, 0.04.")
S["escape-factorial-v1"] = ("bounded_agency", "results/escape-factorial-v1/REPORT.md",
 "A PPO agent trains in a 7x7 gridworld where the intended goal (a gem) sits at a proxy location except in a "
 "fraction p of episodes. Policies start as proxy-seekers and may later switch to gem-seekers; escape time T is "
 "the first checkpoint at which a seed reaches the gem on at least 50% of decorrelated test trials. An earlier "
 "study with one CNN at entropy bonus 0.05 found T is about 60/p updates. This factorial crosses CNN width "
 "(4, 8 or 16 channels) x entropy bonus (0.01 or 0.05) x p (0.2 or 0.05), 4 seeds per cell, with training capped "
 "at 1,000 updates (p = 0.2) or 3,000 (p = 0.05). Analysis: censored regression of log T on log(1/p) with one "
 "intercept (the 'constant') per width-entropy cell.")
S["myopia-decoupling-v1"] = ("interactive_gridworld", "results/myopia-decoupling-v1/REPORT.md",
 "Tabular agents learn in small gridworlds from environment reward plus an overseer's approval. Two hacks are "
 "planted: a tamper button that pays environment reward, and a lever that corrupts the approval given for the "
 "step the agent actually executes. A 2x2 design crosses discount (myopic, gamma = 0, vs full-horizon, gamma = "
 "0.95) with feedback (coupled: approval for the executed step; decoupled: approval for an independently "
 "queried action). All arms share one uniform-exploration trace, 20,000 environment steps and 20,000 approval "
 "samples, no tuning. 32 fresh seeds, 192 environments; intervals are 95% Student-t over seeds.")
S["jtt"] = ("goal_misgen", "experiments/draft_proposals_2026_11/REPORT.md",
 "Offline behaviour cloning in maze gridworlds. Expert demonstrations go to a cheese that sits on a spurious cue "
 "in 96% of training levels; the other 4% are decorrelated. Safety = success on decorrelated test levels. Just "
 "Train Twice (JTT) trains a first model, collects the training steps it gets wrong (the error set), then "
 "retrains with those steps upweighted x20. Arms: uniform (no upweighting), JTT, random (a random set of the "
 "same size x20), oracle (the true decorrelated-level steps x20). The expert breaks ties between equally good "
 "moves at random (condition R); a variant uses deterministic tie-breaks (condition D). 12 seeds, paired 95% t "
 "intervals; 'lower bound' is the lower end of that interval. The tuned behaviour-cloning baseline succeeds on "
 "about a third of decorrelated levels.")
S["omp"] = ("planted_interp", "experiments/draft_proposals_2026_11/REPORT.md",
 "Toy superposition models with 64 planted sparse features (30% are children that fire only with a parent) "
 "stored in d = 32, 16 or 10 dimensions, with activation noise. The dictionary is fixed to the true planted "
 "directions; solvers only have to say which features are active in each sample, with a budget of k nonzeros. "
 "Solvers: signed orthogonal matching pursuit (OMP), nonnegative OMP, and an amortized TopK encoder trained for "
 "reconstruction with the decoder frozen. Metric: mean per-sample F1 between predicted and true active sets. "
 "12 seeds, paired 95% t intervals.")
S["kmeans"] = ("planted_interp", "experiments/draft_proposals_2026_11/REPORT.md",
 "Toy superposition models with planted feature directions. Dictionary-learning methods are scored by MMCS, the "
 "mean max cosine similarity between learned and planted directions. An amplitude shift is applied: a random "
 "third of features gets amplitude x0.25, a third x1, a third x4, directions unchanged. Arms at equal dictionary "
 "size and default hyperparameters: spherical k-means (activations normalised before clustering), Euclidean "
 "k-means, and a TopK sparse autoencoder on raw activations. On unshifted data all three score about 0.91 to "
 "0.92. Loss under shift = MMCS(plain) - MMCS(shifted). 12 seeds, paired 95% t intervals.")
S["nesting"] = ("planted_interp", "experiments/draft_proposals_2026_11/REPORT.md",
 "Toy superposition models with planted features, some of which are children that fire only with a parent "
 "feature. Sparse autoencoders are scored by MMCS (mean max cosine similarity to planted directions). Arms at "
 "equal width and k = 2: a Matryoshka SAE (multi-scale loss on fixed nested prefixes of 25%, 50% and 100% of the "
 "latents), random nesting (the same multi-scale loss, but which latents form each sub-dictionary is redrawn "
 "every step), and a plain TopK SAE. Default hyperparameters. 12 seeds, paired 95% t intervals.")

C = [  # (study, id, registered claim, met?)
 ("escape-time-v1", "H1", "On the 24 confirmation seeds alone, the fitted slope of log T on log(1/p) has a 95% CI that lies entirely inside [0.75, 1.25].", 1),
 ("escape-time-v1", "H2", "The law fitted on development seeds predicts the median escape time of every one of the three confirmation cells to within a factor of 1.5.", 1),
 ("escape-time-v1", "H3", "Escape is one-way: no confirmation seed falls back below 50% gem-seeking at two consecutive checkpoints after escaping.", 1),
 ("escape-factorial-v1", "H1", "A single common slope of log T on log(1/p) across all cells has a 95% CI lying entirely inside [0.75, 1.25].", 0),
 ("escape-factorial-v1", "H2", "Slopes do not differ by cell: a likelihood-ratio test of per-cell slopes against a common slope is NOT significant at 0.05.", 1),
 ("escape-factorial-v1", "H3", "The constant differs by cell: a likelihood-ratio test of six intercepts against one is significant at 0.05.", 1),
 ("escape-factorial-v1", "S1", "In every comparison at fixed entropy, the wider CNN has the lower fitted escape constant (or escapes where the narrower one does not).", 1),
 ("escape-factorial-v1", "S2", "At every width, the escape constant is lower at entropy 0.05 than at entropy 0.01.", 1),
 ("myopia-decoupling-v1", "P1", "The myopic + decoupled arm completes the task in at least 80% of episodes (lower end of the 95% CI >= 80%).", 1),
 ("myopia-decoupling-v1", "P2", "The myopic + decoupled arm uses a planted hack in at most 10% of episodes (upper end of the 95% CI <= 10%).", 1),
 ("myopia-decoupling-v1", "P3", "Decoupling is needed: on the lever environments, goal completion of myopic + decoupled exceeds myopic + coupled by at least 20 percentage points (lower CI bound).", 1),
 ("myopia-decoupling-v1", "P4", "Myopia is needed: goal completion of myopic + decoupled exceeds full-horizon + decoupled by at least 20 percentage points (lower CI bound).", 1),
 ("jtt", "A1", "In condition R, JTT beats uniform on safety by at least 0.05 (lower bound) while losing no more than 0.05 on in-distribution success (lower bound).", 0),
 ("jtt", "A2", "In condition R, decorrelated-level steps are over-represented in JTT's error set by at least 10 percentage points (lower bound) AND JTT beats random upweighting on safety by at least 0.05 (lower bound).", 0),
 ("jtt", "A3", "In condition R, steps where the expert had more than one optimal move (tied steps) are over-represented in the error set (lower bound of share-in-set minus base share > 0).", 1),
 ("jtt", "A4", "Removing tie-break noise shrinks JTT's safety deficit: (JTT - uniform in D) - (JTT - uniform in R) has a lower bound above 0.", 0),
 ("jtt", "A5", "In condition R, the oracle arm beats uniform on safety (lower bound of oracle - uniform > 0).", 0),
 ("omp", "B1", "At d = 10, signed OMP beats the amortized encoder on F1 by at least 0.05 (lower bound).", 0),
 ("omp", "B2", "Nonnegative OMP beats signed OMP on F1 in every one of the three dimension strata (lower bound > 0 in each).", 1),
 ("omp", "B3", "At d = 10, nonnegative OMP beats the amortized encoder on F1 (lower bound > 0).", 1),
 ("kmeans", "C1", "Euclidean k-means loses at least 0.02 more MMCS under the shift than spherical k-means (lower bound), and spherical k-means is within 0.01 of Euclidean on unshifted data (lower bound >= -0.01).", 1),
 ("kmeans", "C2", "Under the shift, spherical k-means beats the TopK SAE on MMCS (lower bound > 0).", 1),
 ("nesting", "D1", "The Matryoshka SAE beats random nesting on child-feature MMCS by at least 0.02 (lower bound) while parent-feature MMCS is no more than 0.01 worse (lower bound).", 0),
 ("nesting", "D2", "Random nesting beats the plain TopK SAE on overall MMCS (lower bound > 0).", 1),
 ("nesting", "D3", "In the Matryoshka SAE, parent features are matched by latents in the first 25% prefix more often than child features are (lower bound > 0).", 1),
]


# ---------------- batch 2 (2026-11-14) ----------------
GW = ("A PPO agent trains in a small gridworld room. The intended goal is a gem; in most training episodes the gem "
      "sits at a fixed proxy corner, and in a fraction p of episodes it is elsewhere. A choice battery then places "
      "gem and corner at different distances and infers each policy's goal: gem-seeker, proxy-seeker, a trade-off "
      "with weight w, or unmeasurable if under 90% of its rollouts go directly to either endpoint. ")
S["phase-boundary-v1"] = ("bounded_agency", "results/phase-boundary-v1/REPORT.md", GW +
 "This study uses an 11x11 room whose battery can measure w from 0.018 to 0.982, a large CNN, and 8 fresh seeds per "
 "cell at p = 0.3, 0.1 and 0.03, scored at 600 and 1,200 updates. In development (4 seeds), p = 0.3 gave gem-seekers "
 "at 600 updates and p = 0.03 gave proxy-seekers.")
S["competence-sweep-v1"] = ("bounded_agency", "results/competence-sweep-v1/REPORT.md", GW +
 "An earlier study in a 7x7 room at p = 0.3 found that only 10 of 16 small-MLP policies trained for 300 updates were "
 "measurable, and 9 of those were proxy-seekers. This sweep trains 16 fresh seeds per configuration and scores them on "
 "four fresh batteries. Configurations: A = large MLP, 2,400 updates (primary); B = small MLP, 2,400 updates; "
 "C = large CNN, 300 updates. On 4 development seeds, A gave 4 of 4 measurable, B 3 of 4, C 4 of 4.")
S["preference-measurement-v2"] = ("bounded_agency", "results/preference-measurement-v2/REPORT.md", GW +
 "This study replaces a legacy estimator (greedy readout, logistic fit on reached trials) that gave a random walk a "
 "weight near 0.5. The new estimator samples actions as trained, counts only direct reaches, and refuses a weight "
 "without distance sensitivity. Confirmation: 57 control runs with known answers on three fresh batteries (optimal "
 "agents with w from 0.15 to 0.85 and up to 20% random actions, soft choosers, a 50/50 mixture of two fixed goals, a "
 "random walk), then 16 fresh small-MLP policies trained for 300 updates.")
S["preference-battery-audit-v1"] = ("bounded_agency", "results/preference-battery-audit-v1/REPORT.md", GW +
 "This audit uses the LEGACY estimator with greedy readout on 16 freshly trained small-MLP policies (p = 0.3, two "
 "entropy cohorts of 8) and three batteries: the original geometry, a reflection, and one with changed distance "
 "frequencies. Trials where the policy reaches neither endpoint stay in the denominator.")
S["trained-preference-v1"] = ("bounded_agency", "results/trained-preference-v1/REPORT.md", GW +
 "A repaired inference method reports a 'robust endpoint trade-off' only when a constant-choice explanation is "
 "excluded with simultaneous confidence bands and a 10-point practical effect. It is run on 16 saved trained policies "
 "x 3 fresh batteries (48 analyses) and checked against an exact finite-state calculation of each policy's endpoint "
 "probabilities. The frozen validation rule requires ALL of: 12 of 12 known controls correct; at least 90% of joint "
 "bands containing the exact probabilities; every reported positive supported by an exact contrast of at least 0.10; "
 "and positives from at least four distinct policies.")
S["preference-repair-v1"] = ("bounded_agency", "results/preference-repair-v1/REPORT.md",
 "A method infers from gridworld choice trials whether a policy trades off two goals by distance. The original "
 "classifier conditions on fast direct reaches, so selectively delaying some trajectories can fake a trade-off. A "
 "repaired method keeps every observed endpoint and treats unreached endpoints as unknown mass. Constructed controls: "
 "108 negative cases (fixed 25/50/75% endpoint commitments and a stationary policy, crossed with uniform or "
 "endpoint-selective delays) and 9 genuine distance-sensitive positives with no delay.")
S["regret-budget-audit-v1"] = ("ued_gridworld", "results/regret-budget-audit-v1/REPORT.md",
 "Online PPO agents train on gridworld levels where the goal usually sits on a spurious cue; intended-goal success "
 "is measured on levels where goal and cue are apart. Four curricula share one student: domain randomization, ACCEL "
 "with its standard 'max-latest' regret estimate, uniform random replay and editing, and ACCEL with a privileged "
 "oracle that knows true regret. Earlier, standard ACCEL scored below domain randomization here. Learning rate is "
 "tuned per arm on 2 development seeds; confirmation uses 8 reserved seeds at 1 million environment steps.")
S["online-start-replay-v1"] = ("ued_gridworld", "results/online-start-replay-v1/REPORT.md",
 "Online PPO agents train on gridworld levels where the goal usually sits on a spurious cue; intended-goal success "
 "is measured on levels where goal and cue are apart. An earlier audit found ACCEL's 'max-latest' regret score is "
 "distorted by differences in starting difficulty, and that uniform random replay beat it. This study tests a "
 "corrected score that compares each start only with its own observed best return, with no oracle information. "
 "Arms: start-matched ACCEL, original ACCEL, random replay control, domain randomization (DR). Equal CPU budget "
 "(120 s) per fit, no tuning, 8 paired seed groups, 95% paired intervals.")
VASE = ("Tabular agents in small gridworlds learn from environment reward plus an overseer's approval. The base arm "
        "(myopic optimisation with decoupled approval) avoids two planted hacks and completes the task in about 95 to "
        "98% of episodes, but breaks a vase the overseer cannot see in over half of episodes when a vase is present. "
        "Side-effect penalties computed from the agent's own learned transition graph are added with weight lambda, "
        "chosen on 8 development seeds as the strongest value keeping development task completion within 0.05 of base. "
        "32 confirmation seeds, 95% intervals over seeds. ")
S["vase-blind-spot-v1"] = ("interactive_gridworld", "results/vase-blind-spot-v1/REPORT.md", VASE +
 "Arms: relative reachability (RR, primary) with a registered lambda grid whose smallest positive value is 1; "
 "attainable utility preservation (AUP); and a plain movement penalty as control.")
S["vase-blind-spot-8b"] = ("interactive_gridworld", "results/vase-blind-spot-8b/REPORT.md", VASE +
 "In a first study, every positive lambda on the relative reachability (RR) grid (smallest value 1) destroyed task "
 "completion on development seeds. This follow-up registers a weaker grid {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} "
 "for RR and tests the selected lambda on 32 new seeds.")
S["approval-factorial-v1"] = ("interactive_gridworld", "results/approval-factorial-v1/REPORT.md",
 "Tabular agents learn in small gridworlds from an overseer's approval. A planted lever corrupts the approval given "
 "for the step the agent actually executes. Arms: coupled feedback (approval for the executed step) vs independent "
 "feedback (approval for a separately queried action). The learner plans with a FULL horizon (gamma = 0.95) over "
 "summed approval, from one shared uniform-exploration trace of 20,000 steps, with no tuning. 32 seeds, paired 95% t "
 "intervals. Comparison is on the lever environments.")
S["rr-coverage-v1"] = ("interactive_gridworld", "results/rr-coverage-v1/REPORT.md",
 "A relative-reachability (RR) side-effect penalty needs a graph of which states can be reached from which. This "
 "diagnostic gives a myopic tabular agent exact planning dynamics but a penalty graph with only 50% of its edges "
 "(chosen at random), then restores the full graph. Each of 32 seed groups has one harmless reversible corridor task "
 "and one world with a vase on the shortest route and a safe detour. RR weight is fixed at 0.1; the overseer's "
 "approval cannot see the vase. 95% paired intervals over seed groups.")
S["changing-mask-factorial-v1"] = ("oversight_gridworld", "results/changing-mask-factorial-v1/REPORT.md",
 "Tabular agents learn in gridworlds with one vase on every shortest route and a safe detour. The overseer's approval "
 "is an advantage signal that includes the vase cost only when a randomly resampled monitoring mask covers it (25% or "
 "50% of cells). All arms learn from the same 20,000-step uniform-exploration trace; environment reward ignores the "
 "vase. A 2x2 crosses approval scaling (original vs span-normalised) with actor discount (myopic, 0, vs full "
 "horizon, 0.95); a reward-only full-horizon arm is the control. An earlier study with different online learners saw "
 "myopic methods fail to complete this task. 32 seed groups, Bonferroni-adjusted paired intervals.")
S["oversight-coverage-audit-v1"] = ("oversight_gridworld", "results/oversight-coverage-audit-v1/REPORT.md",
 "Tabular agents learn online in gridworlds with one vase on every shortest route and a safe detour costing 2 to 6 "
 "extra steps. An overseer's approval includes the vase cost only on monitored cells. Monitoring covers 25% or 50% of "
 "cells, either as a FIXED band per environment or RESAMPLED each episode with the same marginal coverage. Three "
 "learners: an ordinary reward-plus-approval baseline, a MONA implementation (myopic with non-myopic approval), and a "
 "decoupled-approval implementation; each tuned on 2 development seed groups. 16 confirmation seed groups, 8,000 "
 "steps. The prediction: fixed monitoring leaves more harm than resampled. A comparison counts as SUPPORTED only if "
 "the Bonferroni-adjusted interval for (fixed harm - resampled harm) lies wholly above 10 points AND mean task "
 "completion is at least 80% in both arms.")
S["kl-anchoring-exp3-confirm-v1"] = ("kl_anchoring", "results/kl-anchoring-exp3-confirm-v1/REPORT.md",
 "A policy is first fitted to human demonstrations (the reference), then trained with RL on a task reward plus a KL "
 "penalty (weight beta) toward the reference, applied only on states RL itself visits. Drift is the total-variation "
 "distance from the reference per state. Reward-induced drift subtracts the drift of a KL-only control run (task "
 "reward zero), which isolates optimiser noise. State groups: T_high (frequently visited trained states), T_low "
 "(rarely visited trained states), O (states on six unseen layouts). Four networks (MLP-64, MLP-256, CNN, "
 "transformer), 16 units each, 1,500 updates. A hypothesis is supported if it holds in at least 3 of 4 networks "
 "(lower 95% bound above zero). An exploratory run had found raw drift on unseen layouts 1.4x to 3.6x that on "
 "frequent states.")

C += [
 ("phase-boundary-v1", "H1", "Learned goals are discrete: none of the measurable fresh policies has an interior weight (each is a gem-seeker with w >= 0.98 or a proxy-seeker with w <= 0.02).", 1),
 ("phase-boundary-v1", "H2", "At 600 updates, policies trained at p = 0.3 are gem-seekers significantly more often than policies trained at p = 0.03 (Fisher exact test, p < 0.05).", 1),
 ("phase-boundary-v1", "H3", "At p = 0.1, the same seeds contain significantly more gem-seekers at 1,200 updates than at 600 (sign test, p < 0.05).", 1),
 ("competence-sweep-v1", "C1-A", "Configuration A yields at least 14 of 16 measurable policies.", 1),
 ("competence-sweep-v1", "C2-A", "Configuration A's measured goals are stable across batteries (the pre-registered agreement rule returns 'stable').", 1),
 ("competence-sweep-v1", "C1-B", "Configuration B yields at least 14 of 16 measurable policies.", 1),
 ("competence-sweep-v1", "C2-B", "Configuration B's measured goals are stable across batteries (the pre-registered agreement rule returns 'stable', not 'inconclusive' or 'unstable').", 0),
 ("competence-sweep-v1", "C1-C", "Configuration C yields at least 14 of 16 measurable policies.", 1),
 ("preference-measurement-v2", "P1", "All 57 control runs pass: known weights recovered within 0.03, the mixture flagged as having no distance sensitivity, and the random walk flagged as insufficiently directed.", 1),
 ("preference-measurement-v2", "P3", "Every measurable fresh policy receives the same goal type on all three batteries (the stability rule returns 'stable').", 1),
 ("preference-battery-audit-v1", "E", "All four planned contrasts (reflected vs original and distance-changed vs original, in each entropy cohort) meet rate equivalence: the Bonferroni-adjusted interval for the difference in gem-choice rate lies wholly within +/-10 percentage points.", 1),
 ("preference-battery-audit-v1", "G", "The goal-interpretation gate passes: on equal-distance trials, the cohort-mean rate of reaching NEITHER endpoint is at most 10% on the batteries compared.", 0),
 ("preference-battery-audit-v1", "K", "Calibration passes on every battery: exact-optimal control agents with known weights 0.1 to 0.9 are recovered with maximum absolute error at most 0.03.", 1),
 ("trained-preference-v1", "V", "The frozen validation rule is met in full.", 1),
 ("preference-repair-v1", "V", "The repaired method reports zero false trade-offs across the 108 negative controls AND detects all 9 no-delay positive controls.", 1),
 ("regret-budget-audit-v1", "P", "Oracle ACCEL beats max-latest ACCEL on intended-goal success by more than 10 percentage points (lower bound of the 95% paired interval) AND oracle ACCEL keeps in-distribution task completion of at least 80%.", 1),
 ("online-start-replay-v1", "P1", "Start-matched ACCEL beats the random replay control on intended-goal success by at least 5 percentage points (lower bound of the interval).", 0),
 ("online-start-replay-v1", "P2", "Start-matched ACCEL's in-distribution task completion is at least 80% (lower bound).", 1),
 ("online-start-replay-v1", "P3", "Start-matched ACCEL's in-distribution completion is no more than 5 points below domain randomization (lower bound of the difference >= -5).", 1),
 ("vase-blind-spot-v1", "F", "A positive lambda on the registered RR grid passes the development feasibility rule (development task completion within 0.05 of base).", 0),
 ("vase-blind-spot-v1", "AUP-P1", "AUP at its selected lambda reduces vase breaks relative to base by at least 0.15 (lower bound).", 0),
 ("vase-blind-spot-v1", "AUP-P2", "AUP at its selected lambda changes task completion by no worse than -0.05 relative to base (lower bound).", 1),
 ("vase-blind-spot-v1", "MOVE-P1", "The movement-penalty control at its selected lambda reduces vase breaks relative to base by at least 0.15 (lower bound).", 0),
 ("vase-blind-spot-8b", "F", "Some positive lambda on the weaker RR grid passes the development feasibility rule.", 1),
 ("vase-blind-spot-8b", "P1", "RR at the selected lambda reduces vase breaks relative to base by at least 0.15 (lower bound).", 1),
 ("vase-blind-spot-8b", "P2", "RR at the selected lambda changes task completion by no worse than -0.05 relative to base (lower bound of the 95% interval).", 0),
 ("vase-blind-spot-8b", "P3", "RR at the selected lambda keeps task completion at or above 0.80 (lower bound).", 1),
 ("vase-blind-spot-8b", "P4", "RR at the selected lambda uses a planted hack in at most 10% of episodes (upper bound).", 1),
 ("approval-factorial-v1", "L", "Independent feedback cuts lever use relative to coupled feedback by at least 20 percentage points (upper bound of the difference <= -20).", 1),
 ("approval-factorial-v1", "G1", "Independent feedback does not lower task completion relative to coupled feedback by more than 5 points (lower bound of the difference >= -5).", 1),
 ("approval-factorial-v1", "G2", "The independent-feedback arm completes the task in at least 80% of episodes (lower bound).", 0),
 ("rr-coverage-v1", "A", "With the half graph, at least 10% of candidate moves are falsely judged unable to return (lower bound of the false return-unreachability fraction >= 10%).", 1),
 ("rr-coverage-v1", "B", "Restoring the full graph raises completion of the reversible task by at least 5 percentage points (lower bound).", 0),
 ("rr-coverage-v1", "C", "Restoring the full graph raises vase breaks by no more than 5 percentage points (upper bound).", 1),
 ("changing-mask-factorial-v1", "H1", "With original approval, the myopic arm completes the task more often than the full-horizon arm by at least 20 points (lower bound).", 0),
 ("changing-mask-factorial-v1", "H2", "Span-normalised full-horizon beats original full-horizon on task completion by at least 20 points (lower bound) and completes at least 80% (lower bound).", 0),
 ("changing-mask-factorial-v1", "H3", "Span-normalised full-horizon breaks the vase at least 10 points less often than the reward-only control (lower bound), loses no more than 5 points of completion against it, and completes at least 80%.", 1),
 ("oversight-coverage-audit-v1", "base-25", "The comparison is SUPPORTED for the ordinary baseline at 25% coverage.", 0),
 ("oversight-coverage-audit-v1", "base-50", "The comparison is SUPPORTED for the ordinary baseline at 50% coverage.", 0),
 ("oversight-coverage-audit-v1", "mona-25", "The comparison is SUPPORTED for the MONA implementation at 25% coverage.", 0),
 ("oversight-coverage-audit-v1", "mona-50", "The comparison is SUPPORTED for the MONA implementation at 50% coverage.", 0),
 ("oversight-coverage-audit-v1", "dec-25", "The comparison is SUPPORTED for the decoupled-approval implementation at 25% coverage.", 0),
 ("oversight-coverage-audit-v1", "dec-50", "The comparison is SUPPORTED for the decoupled-approval implementation at 50% coverage.", 0),
 ("kl-anchoring-exp3-confirm-v1", "H1", "At beta = 0.3, reward-induced drift at the end of training is larger on unseen layouts (O) than on frequent trained states (T_high).", 1),
 ("kl-anchoring-exp3-confirm-v1", "H2", "At beta = 0.3, reward-induced drift is already larger on O than on T_high early in training (updates 25 to 100).", 1),
 ("kl-anchoring-exp3-confirm-v1", "H3", "A stronger anchor (beta = 1.0) concentrates reward-induced drift on unseen layouts more than beta = 0.3 does.", 0),
 ("kl-anchoring-exp3-confirm-v1", "S1", "The exploratory raw result replicates: without subtracting the control, drift on O exceeds drift on T_high.", 1),
 ("kl-anchoring-exp3-confirm-v1", "S2", "The raw concentration is reward-specific: the raw O / T_high drift ratio is larger in the reward run than in the KL-only control.", 0),
 ("kl-anchoring-exp3-confirm-v1", "S3", "Reward-induced drift is larger on rarely visited trained states (T_low) than on frequent ones (T_high).", 1),
]


def main():
    rows = []
    for study, cid, claim, y in C:
        cluster, src, setup = S[study]
        rows.append({"item_id": f"sub::{study}::{cid}", "kind": "subclaim", "study": study, "cluster": cluster,
                     "question": "sub", "outcome": y, "source_report": src, "setup": setup, "claim": claim})
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (HERE / "subclaims.jsonl").write_text(blob)
    print(len(rows), "sub-claims; yes rate", round(sum(r["outcome"] for r in rows) / len(rows), 2),
          "; studies", len({r["study"] for r in rows}), "; clusters", len({r["cluster"] for r in rows}), "; sha256", hashlib.sha256(blob.encode()).hexdigest()[:16])

if __name__ == "__main__":
    main()
