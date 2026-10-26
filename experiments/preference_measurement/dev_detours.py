import sys, json
import numpy as np
sys.path[:0] = ["src/experiments/preference_audit/vendor", "src/experiments/preference_audit"]
import batteries as B
import measure as M, policy as PL
P = json.load(open("src/experiments/preference_audit/protocol.json"))
b = B.make_batteries(P["battery_seeds"]["development"])["original"]
def show(label, act, K=32):
    rng = np.random.default_rng(5)
    out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], K, rng)
    d = np.where(out == 1, b["dg"][:, None], np.where(out == 2, b["dc"][:, None], 0))
    ex = (st - d)[out > 0]
    q = np.percentile(ex, [50, 75, 90, 95])
    cov = {s: float(((out > 0) & (st <= d + s)).mean()) for s in (2, 4, 6, 8)}
    rel = float(((out > 0) & (st <= 1.5 * d + 2)).mean())
    print(f"{label:14s} reach {np.mean(out>0):.2f} gem {np.mean(out==1):.2f} extra q50/75/90/95 {q} cov@slack {cov} rel {rel:.2f}")
for tag, seeds in (("entropy_0.01", [0, 1, 3, 5]), ("entropy_0.05", [0, 1, 4, 5, 7])):
    Pm = PL.load(f"src/results/preference-battery-audit-v1/models/{tag}.npz")
    for s in seeds: show(f"{tag[-4:]} s{s}", PL.act_fn(Pm, s))
rng = np.random.default_rng(1)
for kind, kw in (("random", {}), ("optimal", dict(w=0.5, eps=0.2)), ("optimal", dict(w=0.5, eps=0.3))):
    act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=32, rng=rng, w=kw.get("w"), eps=kw.get("eps", 0))
    show(f"{kind} {kw.get('eps','')}", act)
