# Results

| Idea | Testbed | Verdict | Safety diff (95% CI) | p | Flags |
| --- | --- | --- | --- | --- | --- |
| Null control: the baseline against itself | backdoor | `no_effect` | +0.000 (+0.000 to +0.000) | 1.000 |  |
| Exact sparse inference instead of an amortized SAE encoder | planted_interp | `mechanism_failed` | not run |  |  |
| Spherical k-means as a feature dictionary | planted_interp | `no_effect` | +0.006 (-0.001 to +0.013) | 0.070 |  |
| Minimal universal perturbation as a backdoor detector | backdoor | `mech_ok_no_win` | -0.076 (-0.176 to +0.023) | 0.117 |  |
| Weight-spectrum outliers as a data-free backdoor signal | backdoor | `mechanism_failed` | not run |  |  |

# Direction signals

| Direction | Testbed | Tested | Mechanism rate (80% CI) | Win rate (80% CI) | Reading |
| --- | --- | --- | --- | --- | --- |
| interpretability/mechanistic-interpretability/sparse-dictionary-learning | planted_interp | 2 | 1/2: 0.50 (0.20-0.80) | 0/2: 0.25 (0.03-0.54) | real, not yet useful |
| robustness/backdoors/backdoor-detection | backdoor | 2 | 1/2: 0.50 (0.20-0.80) | 0/2: 0.25 (0.03-0.54) | real, not yet useful |
