import sys, json
import numpy as np
sys.path[:0] = ["src/experiments/preference_audit/vendor", "src/experiments/preference_audit"]
import batteries as B
import measure as M, policy as PL
P = json.load(open("src/experiments/preference_audit/protocol.json"))
bats = B.make_batteries(P["battery_seeds"]["development"])
for tag in ("entropy_0.01", "entropy_0.05"):
    Pm = PL.load(f"src/results/preference-battery-audit-v1/models/{tag}.npz"); print("==", tag)
    for s in range(8):
        row = []
        for name in ("original", "reflected", "distance_reweighted"):
            b = bats[name]
            for mode, K in (("sample", 32), ("greedy", 1)):
                rng = np.random.default_rng(1000 + s)
                out, st = M.rollouts(PL.act_fn(Pm, s, mode), b["agent"], b["gem"], b["proxy"], K, rng)
                f = M.fit(out, st, b["dg"], b["dc"], b["delta"])
                w = "" if f["w"] is None else f" w{f['w']:.2f}[{f['w_ci'][0]:.2f},{f['w_ci'][1]:.2f}]"
                row.append(f"{name[:4]}/{mode[0]}: {f['type'][:10]} cov{f['directed_coverage']:.2f}{w}")
        print(f" s{s} " + " | ".join(row[i] for i in (0, 1, 2, 4)))
