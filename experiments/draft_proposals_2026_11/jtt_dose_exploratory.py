"""Exploratory, not pre-registered: does the harm from upweighting scale with the weight?
Calibration seeds only (base 500,000), condition R. Usage: python jtt_dose_exploratory.py n_seeds out.json"""
import json
import sys

import numpy as np
import torch

import jtt_groups as J

torch.set_num_threads(1)
res = {}
for seed in range(int(sys.argv[1])):
    for i, (fam, n) in enumerate(J.FAMILIES):
        t = J.build(fam, n, J.BASE["calib"] + 1000 * seed + i, "random", 0.0)
        X, y = torch.tensor(t["obs"], dtype=torch.float32), torch.tensor(t["action"])
        ts = seed * 100 + i
        first = J.train(X, y, torch.ones(len(y)), J.ID_EPOCHS, ts, t["view"])
        with torch.no_grad():
            wrong = (first(X).argmax(1) != y).numpy()
        rand = np.zeros(len(y), bool)
        rand[np.random.default_rng(ts + 5).choice(len(y), int(wrong.sum()), replace=False)] = True
        r = {"uniform": J.score(J.train(X, y, torch.ones(len(y)), J.EPOCHS, ts + 1, t["view"]), t)["safety"]}
        for up in (2.0, 5.0, 20.0):
            for name, mask in (("jtt", wrong), ("random", rand)):
                w = torch.where(torch.tensor(mask), torch.tensor(up), torch.tensor(1.0))
                r[f"{name}_x{int(up)}"] = J.score(J.train(X, y, w, J.EPOCHS, ts + 1, t["view"]), t)["safety"]
        res[f"{seed}/{fam}"] = r
        print(seed, fam, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
keys = list(next(iter(res.values())))
print("MEAN", {k: round(float(np.mean([v[k] for v in res.values()])), 3) for k in keys})
