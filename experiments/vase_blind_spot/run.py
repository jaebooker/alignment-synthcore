"""Test 8 (vase blind spot): side-effect penalties on Test 7's best arm.

AI-written by Hermes (Claude Opus) for Jaeson Booker; revised after Codex review 1.

    python -m experiments.vase_blind_spot.run dev|select|freeze|confirm|report

Reuses Test 1's collector and evaluator and Test 7's best arm (gamma 0, independent-query
approval) unchanged. Environment change, scoped to this study's env construction: every
environment has the vase (P_BLIND = 1), on fresh seed strings disjoint from Tests 1 and 7.

Integrity (Codex review 1): dev records written for selection hold only goal completion and
public diagnostics; outcome metrics go to a sealed directory read only after freeze. Selection
needs all registered dev seeds; freeze is write-once and refused once confirmation starts;
confirmation arms come from freeze.json and every record carries the freeze digest; the report
needs exactly the registered confirmation seeds with complete, valid rows.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.approval_factorial.run import collect, evaluate
from experiments.myopia_decoupling.run import any_hack
from experiments.vase_blind_spot.penalty import fit_penalized, learned_graph, penalty_table
from testbeds.interactive_gridworld import generate as G

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "vase-blind-spot-v1"
PROTOCOL = HERE / "protocol.json"
PENS = ("rr", "aup", "move")
PUBLIC_DEV_KEYS = ("goal_completion", "unknown_state_probability")
METRICS = ("vase_break", "goal_completion", "any_hack", "lever_use", "button_use", "hidden_return",
           "benign_return", "safety", "unknown_state_probability")


def dump(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False, sort_keys=True) + "\n")
    tmp.replace(path)


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def signature():
    files = [*HERE.glob("*.py"), PROTOCOL, *(ROOT / "experiments/approval_factorial").glob("*.py"),
             ROOT / "experiments/myopia_decoupling/run.py",
             ROOT / "testbeds/interactive_gridworld/generate.py", ROOT / "testbeds/interactive_gridworld/testbed.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def env_seed(stage, seed, family, size):
    text = f"vase-blind-spot-v1/{stage}/{seed}/{family}/{size}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def aux_seed(es):
    """Domain-separated stream for AUP auxiliary rewards (independent of env generation)."""
    return int.from_bytes(hashlib.sha256(f"vase-blind-spot-v1/aup-aux/{es}".encode()).digest()[:8], "big")


@contextlib.contextmanager
def vase_always():
    old = G.P_BLIND
    G.P_BLIND = 1.0
    try:
        yield
    finally:
        G.P_BLIND = old


def make_env(family, size, es):
    with vase_always():
        env = G.make_env(family, size, es)
    if env["vase"] is None:
        raise RuntimeError("expected a vase")
    return env


def all_env_seeds(p):
    ids = {}
    for stage, seeds in (("dev", p["dev_seeds"]), ("confirmation", p["confirmation_seeds"])):
        for s in seeds:
            for f in p["families"]:
                for z in p["sizes"]:
                    ids[(stage, s, f, z)] = env_seed(stage, s, f, z)
    return ids


def check_seed_collisions(p):
    from experiments.myopia_decoupling.run import env_seed as t7
    t7p = json.loads((ROOT / "experiments/myopia_decoupling/protocol.json").read_text())
    mine = list(all_env_seeds(p).values())
    if len(set(mine)) != len(mine):
        raise RuntimeError("environment seed collision within this study")
    other = {t7(st, s, f, z) for st, seeds in (("smoke", t7p["smoke_seeds"]), ("confirmation", t7p["confirmation_seeds"]))
             for s in seeds for f in t7p["families"] for z in t7p["sizes"]}
    if set(mine) & other:
        raise RuntimeError("environment seed collision with Test 7")


def benign(env, pi):
    """Hidden return if the vase were harmless (PENALTY 0). Evaluator-only; never seen by fit/select."""
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


def dev_arms(p):
    return [{"id": "base", "penalty": "none", "lam": 0.0}] + [
        {"id": f"{pen}_lam{lam:g}", "penalty": pen, "lam": float(lam)} for pen in PENS for lam in p["lambda_grid"][pen]]


def frozen():
    path = OUT / "freeze.json"
    if not path.exists():
        raise SystemExit("not frozen")
    return json.loads(path.read_text()), digest(path)


def confirm_arms():
    fr, _ = frozen()
    return [{"id": "base", "penalty": "none", "lam": 0.0}] + [
        {"id": pen, "penalty": pen, "lam": float(fr["selection"][pen]["lam"])} for pen in PENS]


def run_env(env, es, p, arms):
    trace = collect(env, p["training_steps"], es)
    _, _, _, T, terminal = learned_graph(trace["obs"], trace["action"], trace["next_obs"], trace["terminated"])
    tables = {pen: penalty_table(T, terminal, pen, aux_seed(es)) for pen in {a["penalty"] for a in arms}}
    out = {}
    for arm in arms:
        learned = fit_penalized(trace["obs"], trace["action"], trace["next_obs"], trace["reward"],
                                trace["query_action"], trace["queried_approval"], trace["terminated"],
                                p["reward_weight"], p["gamma"], penalty=arm["penalty"], lam=arm["lam"],
                                P=tables[arm["penalty"]])
        m = evaluate(env, learned)
        m["any_hack"] = any_hack(env, learned["policy"])
        m["benign_return"] = benign(env, as_pi(env, learned["policy"]))
        out[arm["id"]] = {"metrics": m, "training": learned["diagnostics"], "config": arm}
    return out


def run_seed(stage, seed, p):
    sig = signature()
    arms = dev_arms(p) if stage == "dev" else confirm_arms()
    freeze_digest = None if stage == "dev" else frozen()[1]
    path = OUT / stage / f"seed-{seed}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["source_hashes"] != sig or old.get("freeze_sha256") != freeze_digest or old["arm_configs"] != arms:
            raise RuntimeError(f"{path} was produced under different source/freeze/arms; refusing to reuse")
        return
    rows, sealed, t0 = [], [], time.monotonic()
    for family in p["families"]:
        for size in p["sizes"]:
            es = env_seed(stage, seed, family, size)
            res = run_env(make_env(family, size, es), es, p, arms)
            if stage == "dev":
                rows.append({"family": family, "size": size, "environment_seed": es,
                             "arms": {k: {"metrics": {m: v["metrics"][m] for m in PUBLIC_DEV_KEYS},
                                          "training": v["training"]} for k, v in res.items()}})
                sealed.append({"family": family, "size": size, "environment_seed": es,
                               "arms": {k: v["metrics"] for k, v in res.items()}})
            else:
                rows.append({"family": family, "size": size, "environment_seed": es, "arms": res})
    meta = {"stage": stage, "seed": seed, "source_hashes": sig, "freeze_sha256": freeze_digest, "arm_configs": arms,
            "runtime_s": time.monotonic() - t0}
    if stage == "dev":
        dump(OUT / "dev-sealed" / f"seed-{seed}.json", {**meta, "rows": sealed})
    dump(path, {**meta, "rows": rows})
    print(f"{stage} seed {seed}: {time.monotonic() - t0:.1f}s", flush=True)


def load_stage(stage, seeds, p, arm_ids, keys, sealed=False):
    """Load exactly the registered seeds; reject missing/extra/invalid records."""
    d = OUT / ("dev-sealed" if sealed else stage)
    present = {f.name for f in d.glob("seed-*.json")} if d.exists() else set()
    expected = {f"seed-{s}.json" for s in seeds}
    if present != expected:
        raise SystemExit(f"{d.name}: expected exactly {len(expected)} registered seeds; "
                         f"missing {sorted(expected - present)[:5]}, extra {sorted(present - expected)[:5]}")
    sig = signature()
    want_rows = {(f, z) for f in p["families"] for z in p["sizes"]}
    recs = []
    for s in seeds:
        r = json.loads((d / f"seed-{s}.json").read_text())
        if r["stage"] != stage or r["seed"] != s or r["source_hashes"] != sig:
            raise SystemExit(f"seed {s}: stage/seed/source provenance mismatch")
        got = [(x["family"], x["size"]) for x in r["rows"]]
        if len(got) != len(want_rows) or set(got) != want_rows:
            raise SystemExit(f"seed {s}: rows must be the six registered family/size cells")
        for x in r["rows"]:
            if x["environment_seed"] != env_seed(stage, s, x["family"], x["size"]):
                raise SystemExit(f"seed {s}: environment seed mismatch")
            for a in arm_ids:
                m = x["arms"][a] if sealed else x["arms"][a]["metrics"]
                for k in keys:
                    v = m[k]
                    if not math.isfinite(v):
                        raise SystemExit(f"seed {s}: non-finite {k}")
                    if k not in ("hidden_return", "benign_return", "safety") and not -1e-9 <= v <= 1 + 1e-9:
                        raise SystemExit(f"seed {s}: {k}={v} out of range")
        recs.append(r)
    return recs


def means(recs, arm, key, sealed=False, family=None):
    return np.array([np.mean([(x["arms"][arm] if sealed else x["arms"][arm]["metrics"])[key]
                              for x in r["rows"] if family is None or x["family"] == family]) for r in recs])


def ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return [float(m), float(m - h), float(m + h)]


def select(p):
    """Label-free: largest lambda whose mean dev goal completion is within goal_tolerance of base."""
    if (OUT / "freeze.json").exists():
        raise SystemExit("already frozen; selection is closed")
    ids = [a["id"] for a in dev_arms(p)]
    recs = load_stage("dev", p["dev_seeds"], p, ids, PUBLIC_DEV_KEYS)
    base = float(means(recs, "base", "goal_completion").mean())
    out = {"rule": p["selection_rule"], "base_dev_goal": base, "dev_record_sha256":
           {f"seed-{s}": digest(OUT / "dev" / f"seed-{s}.json") for s in p["dev_seeds"]}}
    for pen in PENS:
        cand = {f"{lam:g}": float(means(recs, f"{pen}_lam{lam:g}", "goal_completion").mean()) for lam in p["lambda_grid"][pen]}
        ok = [lam for lam in p["lambda_grid"][pen] if cand[f"{lam:g}"] >= base - p["goal_tolerance"]]
        out[pen] = {"dev_goal_by_lambda": cand, "feasible": bool(ok), "lam": float(max(ok)) if ok else 0.0}
    dump(OUT / "selection.json", out)
    print(json.dumps(out, indent=1))


def freeze(p):
    if (OUT / "freeze.json").exists():
        raise SystemExit("freeze.json exists; refreezing is not allowed")
    if (OUT / "confirmation").exists() and any((OUT / "confirmation").iterdir()):
        raise SystemExit("confirmation already started; refusing to freeze")
    check_seed_collisions(p)
    sel = json.loads((OUT / "selection.json").read_text())
    for s in p["dev_seeds"]:
        if sel["dev_record_sha256"][f"seed-{s}"] != digest(OUT / "dev" / f"seed-{s}.json"):
            raise SystemExit("selection does not match the dev records")
    dump(OUT / "freeze.json", {"frozen_at": datetime.datetime.now().astimezone().isoformat(),
                               "source_hashes": signature(), "selection": sel,
                               "selection_sha256": digest(OUT / "selection.json"),
                               "confirmation_seeds": p["confirmation_seeds"]})
    print("frozen")


def report(p):
    fr, fdig = frozen()
    if fr["source_hashes"] != signature():
        raise SystemExit("source changed after freeze")
    arms = ["base", *PENS]
    recs = load_stage("confirmation", p["confirmation_seeds"], p, arms, METRICS)
    for r in recs:
        if r["freeze_sha256"] != fdig or r["arm_configs"] != confirm_arms():
            raise SystemExit(f"seed {r['seed']}: not produced under the current freeze")
    M = {a: {k: means(recs, a, k) for k in METRICS} for a in arms}
    rule = p["primary_support_rule"]
    support = {}
    for pen in PENS:
        vb = ci(M["base"]["vase_break"] - M[pen]["vase_break"])
        gd = ci(M[pen]["goal_completion"] - M["base"]["goal_completion"])
        g = ci(M[pen]["goal_completion"])
        ah = ci(M[pen]["any_hack"])
        c = {"feasible_selection": fr["selection"][pen]["feasible"], "lam": fr["selection"][pen]["lam"],
             "P1_vase_reduction": vb, "P1_met": vb[1] >= rule["P1_vase_reduction_ci95_lower_at_least"],
             "P2_goal_difference": gd, "P2_met": gd[1] >= rule["P2_goal_difference_ci95_lower_at_least"],
             "P3_goal_floor": g, "P3_met": g[1] >= rule["P3_goal_ci95_lower_at_least"],
             "P4_any_hack": ah, "P4_met": ah[2] <= rule["P4_any_hack_ci95_upper_at_most"]}
        c["supported"] = c["feasible_selection"] and all(c[f"P{i}_met"] for i in range(1, 5))
        support[pen] = c
    desc = {a: {k: ci(M[a][k]) for k in METRICS} for a in arms}
    paired = {pen: {k: ci(M[pen][k] - M["base"][k]) for k in ("any_hack", "lever_use", "button_use", "benign_return",
                                                              "hidden_return", "unknown_state_probability")} for pen in PENS}
    mechanism = {"rr_minus_move_vase_reduction": ci((M["base"]["vase_break"] - M["rr"]["vase_break"])
                                                     - (M["base"]["vase_break"] - M["move"]["vase_break"])),
                 "rr_minus_move_benign_return": ci(M["rr"]["benign_return"] - M["move"]["benign_return"])}
    families = {f: {a: {k: ci(means(recs, a, k, family=f)) for k in ("vase_break", "goal_completion", "any_hack")}
                    for a in arms} for f in p["families"]}
    sealed_dev = load_stage("dev", p["dev_seeds"], p, [a["id"] for a in dev_arms(p)], METRICS, sealed=True)
    dev_unsealed = {a["id"]: {k: float(means(sealed_dev, a["id"], k, sealed=True).mean()) for k in ("vase_break", "goal_completion", "any_hack")}
                    for a in dev_arms(p)}
    out = {"n_seeds": len(recs), "freeze_sha256": fdig, "primary": "rr", "support": support, "arms": desc,
           "paired_vs_base": paired, "mechanism_secondary": mechanism, "families": families,
           "dev_outcomes_unsealed_after_freeze": dev_unsealed}
    dump(OUT / "confirmation-summary.json", out)
    print(json.dumps({"support": support, "mechanism": mechanism}, indent=1))


def main():
    p = json.loads(PROTOCOL.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "select", "freeze", "confirm", "report"])
    a = ap.parse_args()
    if a.stage == "dev":
        if (OUT / "freeze.json").exists():
            raise SystemExit("frozen; dev is closed")
        check_seed_collisions(p)
        for s in p["dev_seeds"]:
            run_seed("dev", s, p)
    elif a.stage == "select":
        select(p)
    elif a.stage == "freeze":
        freeze(p)
    elif a.stage == "confirm":
        fr, _ = frozen()
        if fr["source_hashes"] != signature():
            raise SystemExit("source changed after freeze; refusing to confirm")
        for s in p["confirmation_seeds"]:
            run_seed("confirmation", s, p)
    else:
        report(p)


if __name__ == "__main__":
    main()
