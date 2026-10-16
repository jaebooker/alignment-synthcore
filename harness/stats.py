"""Paired-seed statistics. Deterministic: same inputs, same outputs."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as st


def paired_test(method, baseline, higher_is_better=True, alpha=0.05, power=0.8):
    """Compare method vs baseline over paired seeds.

    Positive `mean_diff` always means "method is better", whichever way the
    metric points. Uses an exact sign-flip permutation test for n <= 16
    (2^n flips), Monte Carlo with a fixed RNG above that.
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("method and baseline must be 1-D and the same length")
    d = (a - b) if higher_is_better else (b - a)
    n = len(d)
    mean = float(d.mean())

    if n <= 16:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([1.0, -1.0], size=(20000, n))
    null = (signs * d).mean(axis=1)
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(np.mean(np.abs(null) >= abs(mean) - 1e-12))

    if n > 1:
        se = float(d.std(ddof=1) / math.sqrt(n))
        tcrit = float(st.t.ppf(1 - alpha / 2, n - 1))
        mde = float((tcrit + st.t.ppf(power, n - 1)) * se)
    else:
        se, tcrit, mde = float("nan"), float("nan"), float("nan")

    return {
        "n": n,
        "method_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "mean_diff": mean,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
        "p_value": p,
        "mde": mde,
        "significant": bool(p < alpha),
        "per_seed_diff": [float(x) for x in d],
        "min_attainable_p": float(2.0 / len(signs)) if n <= 16 else None,
    }
