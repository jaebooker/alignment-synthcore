import sys, json
import numpy as np
sys.path[:0] = ["src/experiments/preference_audit/vendor", "src/experiments/preference_audit"]
import batteries as B, revpref
import measure as M
P = json.load(open("src/experiments/preference_audit/protocol.json"))
bats = B.make_batteries(P["battery_seeds"]["development"])
K = 32
cases = [("optimal", dict(w=w)) for w in (0.15, 0.3, 0.5, 0.7, 0.85)] + \
        [("optimal", dict(w=0.3, eps=0.05)), ("optimal", dict(w=0.7, eps=0.1)), ("optimal", dict(w=0.3, eps=0.2)),
         ("soft_optimal", dict(w=0.3, tau=2.0)), ("soft_optimal", dict(w=0.7, tau=2.0)),
         ("optimal", dict(w=0.02)), ("optimal", dict(w=0.98)),
         ("mixture", {}), ("random", {})]
for name in ("original", "reflected", "distance_reweighted"):
    b = bats[name]; print("==", name)
    for kind, kw in cases:
        rng = np.random.default_rng(7)
        act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=K, rng=rng,
                               w=kw.get("w"), eps=kw.get("eps", 0.0), tau=kw.get("tau"))
        out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], K, rng)
        f = M.fit(out, st, b["dg"], b["dc"], b["delta"])
        leg = revpref.fit(out[:, 0], b["delta"])  # legacy on one sample per trial
        wci = f["w_ci"] and [round(x, 3) for x in f["w_ci"]]
        tw=kw.get("w"); cov_ok = "" if (f["w_ci"] is None or tw is None) else ("IN" if f["w_ci"][0]<=tw<=f["w_ci"][1] else "OUT")
        print(f"  {cov_ok:3s} {kind:12s} {str(kw):24s} cov {f['directed_coverage']:.2f} type {f['type']:20s} "
              f"w {f['w'] if f['w'] is None else round(f['w'],3)} ci {wci} | legacy w {leg['w']:.3f}")
