# Bounded agency: goal phase diagrams and convergence to the optimum

Small-scale tests of whether the goals of trained agents can be predicted from
environment, reward, and optimizer, measured only by revealed preference.

## Files

- `gw.py`: 7x7 gridworld. True goal is a gem; proxy goal is the top-right corner.
  In training the gem sits on the corner except in a fraction `p` of episodes.
  Also the OOD choice test and exact-optimal choices for reward `w*gem + (1-w)*corner`.
- `revpref.py`: estimates the indifference distance gap and the implied weight `w`
  on the gem from choice outcomes. Identifiable range is w in [0.10, 0.90]
  (censored outside it). Validated on exact-optimal policies: recovers w within +/-0.03.
- `ppo.py`: PPO in plain JAX, vmapped over seeds. Architectures `mlp_s`, `mlp_l`,
  `cnn_s`, `cnn_l`. Entropy coefficient is the optimization axis.
- `sweep.py`: Experiments 1 and 2a. `python3 sweep.py {mlp_s|mlp_l|cnn_s|cnn_l}`.
  Writes one JSON per config to `results/` with outcomes at 6 training checkpoints.
- `power.py`: Experiment 2b. Turner-style power-seeking MDP, tabular Q-learning
  (zero vs optimistic init) on random reward functions, compared to the exact
  optimal prediction n/(n+1).
- `analyze.py`: summary table (`results/summary_final.json`) and figures.

## Requirements

`pip install jax optax numpy scipy matplotlib`. CPU only. Full sweep is about
3 hours on 2 cores; it scales across seeds via `vmap`, so a GPU makes it minutes.

## Key definitions

- **Revealed preference test:** gem placed away from the corner and off the
  agent's shortest path to the corner. The episode ends at whichever the agent
  reaches first. `delta = d_gem - d_corner`.
- **Implied weight:** indifference at `delta*` means `w/(1-w) = gamma^(-delta*)`,
  with gamma = 0.8.
- **Optimal-agent tile:** at p = 0 every w in (0, 1] is reward-equivalent on the
  training distribution. At any p > 0 only w = 1 is consistent.
