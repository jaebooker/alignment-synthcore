"""Test 8 (vase blind spot): side-effect penalties on Test 7's best arm.

AI-written by Hermes (Claude Opus) for Jaeson Booker.

    python -m experiments.vase_blind_spot.run dev|select|freeze|confirm|report

Reuses Test 1's collector and evaluator and Test 7's best arm (gamma 0, independent-query
approval) unchanged. The only environment change: every environment has the vase
(P_BLIND = 1), on fresh seed strings disjoint from Tests 1 and 7.
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.approval_factorial.run import collect, evaluate
from experiments.myopia_decoupling.run import any_hack
from experiments.vase_blind_spot.penalty import fit_penalized
from testbeds.interactive_gridworld import generate as G

G.P_BLIND = 1.0  # documented environment change: the vase is always present

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "vase-blind-spot-v1"
PROTOCOL = HERE / "protocol.json"


def dump(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature():
    files = [*HERE.glob("*.py"), PROTOCOL, *(ROOT / "experiments/approval_factorial").glob("*.py"),
             ROOT / "experiments/myopia_decoupling/run.py",
             ROOT / "testbeds/interactive_gridworld/generate.py", ROOT / "testbeds/interactive_gridworld/testbed.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def env_seed(stage, seed, family, size):
    text = f"vase-blind-spot-v1/{stage}/{seed}/{family}/{size}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def benign(env, m_policy):
    """Hidden return if the vase were harmless (PENALTY 0): the cost of avoiding it anyway."""
    pi = m_policy
    T, S = env["T"], len(env["T"])
    d = np.zeros(S)
    d[0] = 1
    total = 0.0
    for _ in range(G.HORIZON):
        w = d[:, None] * pi
        total += float((w * env["R_base"]).sum())
        d = np.bincount(T.ravel(), weights=w.ravel(), minlength=S)
    return total


def as_pi(env, policy):
    pi = np.full(env["T"].shape, 0.2)
    for o, a in policy.items():
        i = env["id_to_state"].get(int(o))
        if i is not None:
            pi[i] = 0
            pi[i, int(a)] = 1
    return pi


def fit(trace, arm, p, es):
    return fit_penalized(trace["obs"], trace["action"], trace["next_obs"], trace["reward"],
                         trace["query_action"], trace["queried_approval"], trace["terminated"],
                         p["reward_weight"], p["gamma"], penalty=arm["penalty"], lam=arm["lam"],
                         penalty_seed=es)


def arms_for(stage, p):
    if stage == "dev":
        return [{"id": "base", "penalty": "none", "lam": 0.0}] + [
            {"id": f"{pen}_lam{lam:g}", "penalty": pen, "lam": lam} for pen in ("rr", "aup") for lam in p["lambda_grid"][pen]]
    sel = json.loads((OUT / "selection.json").read_text())
    return [{"id": "base", "penalty": "none", "lam": 0.0},
            {"id": "rr", "penalty": "rr", "lam": sel["rr"]}, {"id": "aup", "penalty": "aup", "lam": sel["aup"]}]


def run_seed(stage, seed, p):
    path = OUT / stage / f"seed-{seed}.json"
    sig = signature()
    if path.exists():
        if json.loads(path.read_text())["source_hashes"] != sig:
            raise RuntimeError("source changed since this seed was run; refusing to reuse it")
        return
    rows, t0 = [], time.monotonic()
    for family in p["families"]:
        for size in p["sizes"]:
            es = env_seed(stage, seed, family, size)
            env = G.make_env(family, size, es)
            assert env["vase"] is not None
            trace = collect(env, p["training_steps"], es)
            arms = {}
            for arm in arms_for(stage, p):
                learned = fit(trace, arm, p, es)
                m = evaluate(env, learned)
                m["any_hack"] = any_hack(env, learned["policy"])
                m["benign_return"] = benign(env, as_pi(env, learned["policy"]))
                arms[arm["id"]] = {"metrics": m, "training": learned["diagnostics"]}
            rows.append({"family": family, "size": size, "environment_seed": es, "arms": arms})
    dump(path, {"stage": stage, "seed": seed, "source_hashes": sig, "runtime_s": time.monotonic() - t0, "rows": rows})
    print(f"{stage} seed {seed}: {time.monotonic() - t0:.1f}s", flush=True)


def seed_means(stage, arm, key):
    recs = [json.loads(f.read_text()) for f in sorted((OUT / stage).glob("seed-*.json"))]
    return np.array([np.mean([r["arms"][arm]["metrics"][key] for r in rec["rows"]]) for rec in recs])


def ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return [float(m), float(m - h), float(m + h)]


def select(p):
    """Label-free rule: largest lambda whose dev goal completion stays within the tolerance of base."""
    base = seed_means("dev", "base", "goal_completion").mean()
    out = {"rule": p["selection_rule"], "base_dev_goal": float(base), "candidates": {}}
    for pen in ("rr", "aup"):
        ok = []
        for lam in p["lambda_grid"][pen]:
            g = seed_means("dev", f"{pen}_lam{lam:g}", "goal_completion").mean()
            out["candidates"][f"{pen}_lam{lam:g}"] = float(g)
            if g >= base - p["goal_tolerance"]:
                ok.append(lam)
        out[pen] = max(ok) if ok else min(p["lambda_grid"][pen])
    dump(OUT / "selection.json", out)
    print(json.dumps(out, indent=1))


def report(p):
    res = {}
    for arm in ("base", "rr", "aup"):
        res[arm] = {k: ci(seed_means("confirmation", arm, k))
                    for k in ("vase_break", "goal_completion", "any_hack", "hidden_return", "benign_return", "safety")}
    rule = p["primary_support_rule"]
    comps = {}
    for pen in ("rr", "aup"):
        vb = ci(seed_means("confirmation", "base", "vase_break") - seed_means("confirmation", pen, "vase_break"))
        gd = ci(seed_means("confirmation", pen, "goal_completion") - seed_means("confirmation", "base", "goal_completion"))
        ah = res[pen]["any_hack"]
        comps[pen] = {"P1_vase_reduction": vb, "P1_met": vb[1] >= rule["P1_vase_reduction_ci95_lower_at_least"],
                      "P2_goal_difference": gd, "P2_met": gd[1] >= rule["P2_goal_difference_ci95_lower_at_least"],
                      "P3_any_hack": ah, "P3_met": ah[2] <= rule["P3_any_hack_ci95_upper_at_most"]}
        comps[pen]["supported"] = comps[pen]["P1_met"] and comps[pen]["P2_met"] and comps[pen]["P3_met"]
    n = len(list((OUT / "confirmation").glob("seed-*.json")))
    dump(OUT / "confirmation-summary.json", {"n_seeds": n, "arms": res, "support": comps,
                                             "primary": "rr", "secondary": "aup"})
    print(json.dumps({"n": n, "support": comps, "arms": res}, indent=1))


def main():
    p = json.loads(PROTOCOL.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "select", "freeze", "confirm", "report"])
    a = ap.parse_args()
    if a.stage == "dev":
        for s in p["dev_seeds"]:
            run_seed("dev", s, p)
    elif a.stage == "select":
        select(p)
    elif a.stage == "freeze":
        dump(OUT / "freeze.json", {"frozen_at": datetime.datetime.now().astimezone().isoformat(),
                                   "source_hashes": signature(), "selection": json.loads((OUT / "selection.json").read_text()),
                                   "confirmation_seeds": p["confirmation_seeds"]})
    elif a.stage == "confirm":
        fr = json.loads((OUT / "freeze.json").read_text())
        if fr["source_hashes"] != signature():
            raise SystemExit("source changed after freeze; refusing to confirm")
        for s in p["confirmation_seeds"]:
            run_seed("confirmation", s, p)
    else:
        report(p)


if __name__ == "__main__":
    main()
