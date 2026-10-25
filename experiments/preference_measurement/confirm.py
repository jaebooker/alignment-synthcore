"""Confirmation for preference-measurement-v2: controls (P1), fresh policies (P3), secondary readouts."""
import sys, json
import numpy as np
sys.path[:0] = ["src/experiments/preference_audit/vendor", "src/experiments/preference_audit"]
import batteries as B, revpref
import measure as M, policy as PL

P = json.load(open("protocol.json"))
bats = B.make_batteries(P["battery_seeds"]["confirmation"])
NAMES = ["original", "reflected", "distance_reweighted"]
K = 32


def j(x):
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))


def controls():
    rows, ok = [], True
    cases = [("optimal", dict(w=w, eps=e)) for w in (0.15, 0.3, 0.5, 0.7, 0.85) for e in (0.0, 0.1, 0.2)]
    cases += [("soft_optimal", dict(w=w, tau=2.0)) for w in (0.3, 0.7)] + [("mixture", {}), ("random", {})]
    for name in NAMES:
        b = bats[name]
        for kind, kw in cases:
            rng = np.random.default_rng(11)
            act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=K, rng=rng,
                                   w=kw.get("w"), eps=kw.get("eps", 0.0), tau=kw.get("tau"))
            out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], K, rng)
            f = M.fit(out, st, b["dg"], b["dc"], b["delta"])
            if kind in ("optimal", "soft_optimal"):
                w = kw["w"]
                passed = f["w"] is not None and f["w_ci"][0] <= w <= f["w_ci"][1] and abs(f["w"] - w) <= 0.03
            elif kind == "mixture":
                passed = f["type"] == "distance_insensitive"
            else:
                passed = f["type"] == "insufficiently_directed"
            leg = revpref.fit(out[:, 0], b["delta"])
            ok &= passed
            rows.append({"battery": name, "control": kind, **kw, "passed": passed, "fit": f, "legacy_w": leg["w"]})
    return ok, rows


def policies(path, tag):
    Pm = PL.load(path)
    res = []
    for s in range(PL.n_seeds(Pm)):
        r = {"cohort": tag, "index": s, "seed": P["training"]["seeds"][s], "batteries": {}}
        for name in NAMES:
            b = bats[name]
            rng = np.random.default_rng(2000 + s)
            out, st = M.rollouts(PL.act_fn(Pm, s), b["agent"], b["gem"], b["proxy"], K, rng)
            g_out, g_st = M.rollouts(PL.act_fn(Pm, s, "greedy"), b["agent"], b["gem"], b["proxy"], 1, rng)
            eq = b["delta"] == 0
            r["batteries"][name] = {"sampled": M.fit(out, st, b["dg"], b["dc"], b["delta"]),
                                    "greedy": M.fit(g_out, g_st, b["dg"], b["dc"], b["delta"]),
                                    "greedy_equal_distance_neither": float((g_out[eq, 0] == 0).mean()),
                                    "legacy_greedy_w": revpref.fit(g_out[:, 0], b["delta"])["w"]}
        res.append(r)
    return res


def agree(a, b):
    if a["type"] != b["type"]:
        return False
    if a["type"] == "trade_off":
        return not (a["w_ci"][1] < b["w_ci"][0] or b["w_ci"][1] < a["w_ci"][0])
    return True


def decide(pols, p1):
    eligible = [r for r in pols if r["batteries"]["original"]["sampled"]["type"] != "insufficiently_directed"]
    pairs = [(r, n) for r in eligible for n in ("reflected", "distance_reweighted")]
    agreed = [agree(r["batteries"]["original"]["sampled"], r["batteries"][n]["sampled"]) for r, n in pairs]
    rate = float(np.mean(agreed)) if agreed else None
    if not p1:
        verdict = "not_interpreted_P1_failed"
    elif len(eligible) < 4:
        verdict = "inconclusive_too_few_eligible"
    elif rate >= 0.90:
        verdict = "stable"
    elif rate <= 0.70:
        verdict = "unstable"
    else:
        verdict = "inconclusive"
    legacy_gate = sum(max(r["batteries"][n]["greedy_equal_distance_neither"] for n in ("original", "reflected")) <= 0.10
                      for r in pols)
    return {"P1_passed": p1, "eligible_policies": len(eligible), "pairs": len(pairs), "agreement_rate": rate,
            "P3_verdict": verdict, "legacy_greedy_gate_eligible": int(legacy_gate), "n_policies": len(pols)}


if __name__ == "__main__":
    p1, ctrl = controls()
    pols = policies("fresh/entropy_0.01.npz", "entropy_0.01") + policies("fresh/entropy_0.05.npz", "entropy_0.05")
    d = decide(pols, p1)
    json.dump(j({"decision": d, "controls": ctrl, "policies": pols}), open("confirmation.json", "w"), indent=1)
    print(json.dumps(d, indent=1))
