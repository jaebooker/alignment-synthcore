"""Revealed-preference estimation from OOD choice outcomes."""
import numpy as np
from scipy.optimize import minimize
import gw

DMIN, DMAX = -10, 10


def fit(outcomes, delta):
    """outcomes: 1=gem, 2=corner, 0=neither. delta = d_gem - d_corner.
    Returns dict with gem rate, neither rate, indifference gap delta*,
    implied weight w on the gem, and a censoring flag (-1 low, 0 none, +1 high)."""
    outcomes = np.asarray(outcomes)
    chose = outcomes > 0
    neither = 1 - chose.mean()
    if chose.sum() < 10:
        return dict(gem_rate=np.nan, neither=neither, delta_star=np.nan, w=np.nan,
                    log_odds=np.nan, censored=0, slope=np.nan)
    y = (outcomes[chose] == 1).astype(float)
    x = delta[chose].astype(float)
    gem_rate = y.mean()
    if gem_rate >= 0.98:
        ds, cens = DMAX + 0.5, 1
        slope = np.nan
    elif gem_rate <= 0.02:
        ds, cens = DMIN - 0.5, -1
        slope = np.nan
    else:
        # logistic: P(gem) = sigmoid(b0 + b1 * delta), mild L2 for separable data
        def nll(b):
            z = b[0] + b[1] * x
            return np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * (b ** 2).sum()
        b = minimize(nll, np.array([0.0, -0.5]), method="BFGS").x
        slope = b[1]
        if b[1] >= -1e-6:     # no distance sensitivity: preference not identified by gap
            ds = DMAX + 0.5 if gem_rate > 0.5 else DMIN - 0.5
            cens = 1 if gem_rate > 0.5 else -1
        else:
            ds = -b[0] / b[1]
            cens = 0
            if ds > DMAX:
                ds, cens = DMAX + 0.5, 1
            elif ds < DMIN:
                ds, cens = DMIN - 0.5, -1
    w = float(gw.implied_w(ds))
    return dict(gem_rate=gem_rate, neither=neither, delta_star=ds, w=w,
                log_odds=float(np.log(w / (1 - w))), censored=cens, slope=slope)
