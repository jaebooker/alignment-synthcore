"""Triage for ideas whose Stage 1 gate failed.

    python triage.py                    # static checks on every mechanism_failed record
    python triage.py --run              # plus a tuned re-run of the gate (dev split only)
    python triage.py ideas/<idea> --run # one or more ideas

A reviewed diagnosis can be recorded in ideas/<idea>/idea.json as
    "triage_review": {"label": "genuine_negative", "note": "...", "reviewed_at": "YYYY-MM-DD"}
It sets the label while the flags stay listed. Reviews are dropped automatically when the
method changes, since they described the old code.

A failed gate can mean three different things:

  gate_artifact     The idea's default hyperparameters were bad. With method and
                    baseline both tuned on dev, the gate passes. The idea deserves
                    a real Stage 2 run before anyone reads its verdict as evidence.
  suspected_bug     The method's outputs look broken: identical to the baseline,
                    collapsed (recovers nothing), inverted (AUROC well below 0.5),
                    or suspiciously fast. Fix the method before judging the idea.
  genuine_negative  No bug signs, and the gate still fails when both arms are tuned.
  fixed_pending_rerun  ideas/<idea>/method.py differs from the copy stored with the
                    verdict: the method was changed (usually fixed) and must be rerun.
  weak_negative     As genuine_negative, but with a caveat: the tuned run nearly
                    passed, or its best setting sat at the edge of the grid.

Ideas with no tuned re-run yet get `needs_tuned_gate` (or `untriaged` when no
static flag fires either). Flags that weaken a reading, like a noisy 3-seed
gate, are listed alongside the label.

Triage is advisory. It never edits record.json or changes a verdict. The tuned
re-run uses only the dev split, so the held-out test split stays clean. Output:
results/_triage/<idea>.json and results/TRIAGE.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
OUT = RESULTS / "_triage"

EPS = 1e-9
# Diagnostics where 0 on every seed (while the baseline is above 0) means the
# method recovered nothing at all: almost always a bug, not a weak idea.
COLLAPSE_DIAGNOSTICS = ("recovery_rate",)
# Safety metrics that are AUROCs: a mean well under 0.5 means the score is
# probably pointed the wrong way.
AUROC_METRICS = ("auroc_heldout",)
INVERTED_MARGIN = 0.05
FAST_RATIO = 0.05      # method runtime under 5% of the baseline's
TESTBEDS_NEED_GPU = {"planted_lm"}
# Errors that come from the machine, not the method. A tuned run that dies on
# one of these is retried later, never counted as a method bug.
ENV_ERRORS = ("ModuleNotFoundError", "No module named", "CUDA", "out of memory", "TimeoutExpired")


# ---------------------------------------------------------------- static checks

def _per_seed(record):
    return (record.get("stage1") or {}).get("per_seed") or []


def _oriented_gate_diffs(record):
    """Per-seed values of the quantity the gate averaged, oriented so + = method better."""
    pred = record.get("prediction") or {}
    mech = pred.get("mechanism")
    diffs = []
    for row in _per_seed(record):
        m, b = row["method"], row["baseline"]
        if not mech:
            diffs.append(m["safety"] - b["safety"])
        elif mech.get("mode", "relative") == "absolute":
            return None
        else:
            mv, bv = m["diagnostics"].get(mech["diagnostic"]), b["diagnostics"].get(mech["diagnostic"])
            if mv is None or bv is None:
                return None
            diffs.append((mv - bv) if mech["direction"] == "higher" else (bv - mv))
    return diffs


def check_identical(record):
    rows = _per_seed(record)
    if not rows:
        return None
    for row in rows:
        m, b = row["method"], row["baseline"]
        if abs((m.get("safety") or 0) - (b.get("safety") or 0)) > EPS:
            return None
        for k, v in (m.get("diagnostics") or {}).items():
            bv = (b.get("diagnostics") or {}).get(k)
            if isinstance(v, (int, float)) and isinstance(bv, (int, float)) and abs(v - bv) > EPS:
                return None
    return {"flag": "identical_to_baseline",
            "detail": f"method matched the baseline exactly on safety and every diagnostic, all {len(rows)} seeds"}


def check_collapsed(record):
    rows = _per_seed(record)
    out = []
    for key in COLLAPSE_DIAGNOSTICS:
        mv = [r["method"]["diagnostics"].get(key) for r in rows]
        bv = [r["baseline"]["diagnostics"].get(key) for r in rows]
        if rows and all(v is not None and v <= EPS for v in mv) and all(v is not None for v in bv) \
                and np.mean(bv) > EPS:
            out.append({"flag": "collapsed_output",
                        "detail": f"{key} = 0 on every seed (baseline mean {np.mean(bv):.3f})"})
    return out


def check_inverted(record):
    rows = _per_seed(record)
    pred_safety = [r["method"]["safety"] for r in rows if r["method"].get("safety") is not None]
    metric = (record.get("stage1") or {}).get("quantity", "")
    if not pred_safety or not any(a in metric for a in AUROC_METRICS):
        return None
    mean = float(np.mean(pred_safety))
    if mean < 0.5 - INVERTED_MARGIN:
        flipped = 1.0 - mean
        base = float(np.mean([r["baseline"]["safety"] for r in rows]))
        return {"flag": "inverted_score",
                "detail": f"mean AUROC {mean:.3f} < 0.5; flipped it would be {flipped:.3f} "
                          f"(baseline {base:.3f})"}
    return None


def check_fast(record):
    rows = _per_seed(record)
    ms = [r["method"].get("seconds") for r in rows]
    bs = [r["baseline"].get("seconds") for r in rows]
    if rows and all(isinstance(x, (int, float)) for x in ms + bs) and np.mean(bs) > 0:
        ratio = float(np.mean(ms) / np.mean(bs))
        if ratio < FAST_RATIO:
            return {"flag": "suspiciously_fast",
                    "detail": f"method ran in {np.mean(ms):.2f}s vs baseline {np.mean(bs):.2f}s ({ratio:.1%})"}
    return None


def check_noisy_gate(record):
    diffs = _oriented_gate_diffs(record)
    if not diffs or len(diffs) < 2:
        return None
    mech = (record.get("prediction") or {}).get("mechanism") or {}
    need = float(mech.get("min_effect", (record.get("prediction") or {}).get("safety_min_effect", 0.0)))
    d = np.asarray(diffs)
    sd = float(d.std(ddof=1))
    mixed = (d > 0).any() and (d < 0).any()
    if mixed and abs(d.mean() - need) < sd:
        return {"flag": "noisy_gate",
                "detail": f"per-seed gate values {', '.join(f'{x:+.3f}' for x in d)}: signs disagree and the "
                          f"miss ({d.mean() - need:+.3f}) is inside one SD ({sd:.3f})"}
    return None


def static_checks(record):
    flags = []
    for fn in (check_identical, check_inverted, check_fast, check_noisy_gate):
        r = fn(record)
        if r:
            flags.append(r)
    flags += check_collapsed(record)
    return flags


BUG_FLAGS = {"collapsed_output", "inverted_score", "suspiciously_fast"}


# ------------------------------------------------------------- tuned gate re-run

class OutOfTime(Exception):
    """Raised when --budget runs out. Finished runs are checkpointed; rerun to resume."""


class Checkpoint:
    """Per-idea cache of finished runs, keyed by evaluator hash, so a tuned re-run
    can stop at any point and resume. Baseline runs are shared across ideas."""

    def __init__(self, idea, ev_hash, deadline):
        self.dir = OUT / "_work"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{idea}.json"
        self.ev = ev_hash
        self.deadline = deadline
        try:
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        except json.JSONDecodeError:
            self.data = {}
        if self.data.get("evaluator_hash") != ev_hash:
            self.data = {"evaluator_hash": ev_hash, "runs": {}}

    def get(self, key, fn):
        runs = self.data["runs"]
        if key in runs:
            return runs[key]
        if self.deadline is not None and time.time() > self.deadline:
            raise OutOfTime(key)
        runs[key] = fn()
        # Atomic replace: parallel triage processes share baseline checkpoints.
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, default=float))
        os.replace(tmp, self.path)
        return runs[key]


def _tune_ckpt(tb, method, ckpt, tag, budget=8, seeds=(0, 1)):
    """Same rule as harness.runner.tune: best mean dev safety over the HPARAMS grid."""
    from harness.runner import load_hparams, run_arm
    table = []
    for i, hp in enumerate(load_hparams(method)[:budget]):
        scores = [ckpt.get(f"{tag}:tune:{i}:{s}", lambda: run_arm(tb, method, "dev", "full", s, hp)["safety"])
                  for s in seeds]
        table.append({"hparams": hp, "dev_safety": float(np.mean(scores))})
    best = max(table, key=lambda r: r["dev_safety"])
    return best["hparams"], table


def _tuned_baseline_ckpt(tb, ckpt, budget=8):
    """Reuse the harness's cached tuned baseline; build it resumably if missing."""
    from harness.runner import CACHE
    path = CACHE / f"baseline_{tb.name}_{tb.version}_{tb.evaluator_hash()}.json"
    if path.exists():
        return json.loads(path.read_text())
    bck = Checkpoint(f"_baseline_{tb.name}", tb.evaluator_hash(), ckpt.deadline)
    hp, table = _tune_ckpt(tb, tb.baseline_method, bck, "baseline", budget)
    out = {"hparams": hp, "table": table}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def tuned_gate(record, idea_dir, stage1_seeds=3, budget=8, deadline=None, log=print):
    """Tune both arms on dev (full config), then rerun the gate on dev (toy config).

    Mirrors Stage 1 exactly except for hyperparameters. Never touches the test split.
    Resumable: raises OutOfTime past the deadline, keeping every finished run.
    """
    from harness.runner import _gate, get_testbed, run_arm

    tb = get_testbed(record["testbed"])
    method = Path(idea_dir) / "method.py"
    import hashlib
    method_hash = hashlib.sha256(method.read_bytes()).hexdigest()[:16]
    # Keyed on evaluator AND method code: editing method.py invalidates old runs.
    ckpt = Checkpoint(record["id"], f"{tb.evaluator_hash()}:{method_hash}", deadline)
    base = _tuned_baseline_ckpt(tb, ckpt, budget)
    hp_m, table_m = _tune_ckpt(tb, method, ckpt, "method", budget)
    rows = []
    for s in range(stage1_seeds):
        m = ckpt.get(f"gate:method:{s}", lambda: run_arm(tb, method, "dev", "toy", s, hp_m))
        b = ckpt.get(f"gate:baseline:{s}", lambda: run_arm(tb, tb.baseline_method, "dev", "toy", s, base["hparams"]))
        rows.append({"seed": s, "method": m, "baseline": b})
    gate = _gate(tb, record["prediction"], [r["method"] for r in rows], [r["baseline"] for r in rows])
    default_hp = table_m[0]["hparams"] if table_m else None
    return {**gate, "method_hparams": hp_m, "method_default_hparams": default_hp,
            "picked_default": hp_m == default_hp, "baseline_hparams": base["hparams"],
            "method_tuning": table_m, "per_seed": rows, "evaluator_hash": tb.evaluator_hash()}


NEAR_MISS = 0.75  # tuned gate reached at least this share of the required effect


def tuned_flags(record, tuned):
    """Flags read off the tuned re-run: a near miss weakens a genuine_negative."""
    if not tuned or tuned.get("error") or tuned.get("passed"):
        return []
    mech = (record.get("prediction") or {}).get("mechanism") or {}
    if mech.get("mode", "relative") == "absolute":
        return []
    need = float(mech.get("min_effect", (record.get("prediction") or {}).get("safety_min_effect", 0.0)))
    obs = float(tuned["observed"])
    out = []
    if need > 0 and obs >= NEAR_MISS * need:
        out.append({"flag": "near_miss",
                    "detail": f"tuned gate reached {obs:+.3f} of the required {need:+.3f} ({obs / need:.0%})"})
    out += grid_edge(tuned)
    return out


def grid_edge(tuned):
    """Best value sits at the end of a swept range: the grid may not contain the optimum."""
    table = tuned.get("method_tuning") or []
    best = tuned.get("method_hparams") or {}
    out = []
    for k, v in best.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        vals = sorted({r["hparams"].get(k) for r in table
                       if isinstance(r["hparams"].get(k), (int, float)) and not isinstance(r["hparams"].get(k), bool)})
        if v == vals[0] and isinstance(v, int) and v <= 1:
            continue  # a count already at its floor cannot be extended downward
        if len(vals) >= 3 and v in (vals[0], vals[-1]):
            out.append({"flag": "grid_edge",
                        "detail": f"best {k}={v} is the {'lowest' if v == vals[0] else 'highest'} value tried "
                                  f"({vals[0]} to {vals[-1]}); extend the grid before trusting the negative"})
    return out


# ---------------------------------------------------------------------- labeling

def label(flags, tuned, review=None, method_changed=False):
    names = {f["flag"] for f in flags}
    if method_changed:
        return "fixed_pending_rerun"
    if review and review.get("label") in NEXT_STEP:
        return review["label"]
    if names & BUG_FLAGS:
        return "suspected_bug"
    if tuned is None:
        return "needs_tuned_gate" if "identical_to_baseline" in names or names else "untriaged"
    if tuned.get("error"):
        # Missing packages or similar: the run says nothing about the method.
        return "needs_tuned_gate" if tuned.get("env_error") else "suspected_bug"
    if tuned["passed"]:
        return "gate_artifact"
    if "identical_to_baseline" in names and tuned.get("picked_default"):
        # Every tuned config still did nothing beyond the baseline.
        return "suspected_bug"
    if names & WEAKENING_FLAGS:
        return "weak_negative"
    return "genuine_negative"


# Flags that leave a negative standing but make it less conclusive.
WEAKENING_FLAGS = {"near_miss", "grid_edge", "noisy_gate"}


NEXT_STEP = {
    "fixed_pending_rerun": "method.py changed since this verdict. Rerun the idea (run.py, or the GPU job for planted_lm).",
    "gate_artifact": "Run Stage 2. Consider moving the tuned hparams to the front of HPARAMS.",
    "suspected_bug": "Inspect method.py against the flags before reading the verdict as evidence.",
    "weak_negative": "Verdict probably stands. Resolve the flagged caveat (widen the grid, more seeds) to make it conclusive.",
    "genuine_negative": "Verdict stands.",
    "needs_tuned_gate": "Run `python triage.py ideas/<idea> --run`.",
    "untriaged": "Run `python triage.py ideas/<idea> --run`.",
}


def triage_one(record_path, run=False, deadline=None, log=print):
    record = json.loads(Path(record_path).read_text())
    idea = record["id"]
    idea_dir = ROOT / "ideas" / idea
    flags = static_checks(record)
    prev = OUT / f"{idea}.json"
    tuned = None
    if prev.exists():  # keep an earlier tuned re-run if the evaluator has not changed
        old = json.loads(prev.read_text())
        if old.get("tuned_gate") and old["tuned_gate"].get("evaluator_hash") == record.get("evaluator_hash"):
            tuned = old["tuned_gate"]
    skip_reason = None
    if run:
        if record["testbed"] in TESTBEDS_NEED_GPU:
            skip_reason = f"{record['testbed']} runs on a rented GPU; tuned re-run not available locally"
        elif not (idea_dir / "method.py").exists():
            skip_reason = "ideas/<idea>/method.py not found"
        else:
            log(f"[{idea}] tuned gate re-run")
            try:
                tuned = tuned_gate(record, idea_dir, deadline=deadline, log=log)
            except OutOfTime:
                log(f"[{idea}] out of time; progress saved, rerun to resume")
                return None
            except Exception as e:  # noqa: BLE001  a crash here is itself a finding
                msg = f"{type(e).__name__}: {str(e)[-500:]}"
                tuned = {"error": msg, "env_error": any(m in msg for m in ENV_ERRORS),
                         "evaluator_hash": record.get("evaluator_hash")}
    spec_path = idea_dir / "idea.json"
    review = json.loads(spec_path.read_text()).get("triage_review") if spec_path.exists() else None
    stored, current = RESULTS / idea / "method.py", idea_dir / "method.py"
    method_changed = stored.exists() and current.exists() and stored.read_bytes() != current.read_bytes()
    if method_changed and tuned is not None and not run:
        tuned = None  # an earlier tuned re-run tested the old code
    flags += tuned_flags(record, tuned)
    lab = label(flags, tuned, None if method_changed else review, method_changed)
    out = {"id": idea, "title": record.get("title"), "testbed": record.get("testbed"),
           "verdict": record.get("verdict"), "label": lab, "next_step": NEXT_STEP[lab],
           "flags": flags, "default_gate": {k: (record.get("stage1") or {}).get(k)
                                            for k in ("quantity", "observed", "rule", "passed")},
           "tuned_gate": tuned, "tuned_skip_reason": skip_reason,
           "review": None if method_changed else review, "method_changed_since_verdict": method_changed,
           "advisory": "Triage never changes a verdict. Labels say how far to trust it.",
           "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    OUT.mkdir(parents=True, exist_ok=True)
    prev.write_text(json.dumps(out, indent=2, default=float))
    log(f"[{idea}] {lab}" + (f" ({'; '.join(f['flag'] for f in flags)})" if flags else ""))
    return out


# ----------------------------------------------------------------------- report

def write_report():
    rows, resolved = [], []
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text())
        rec = RESULTS / r["id"] / "record.json"
        now = json.loads(rec.read_text()).get("verdict") if rec.exists() else None
        if now and now != "mechanism_failed":
            resolved.append((r, now))  # rerun since triage and no longer fails the gate
        else:
            rows.append(r)
    order = ["fixed_pending_rerun", "suspected_bug", "gate_artifact", "needs_tuned_gate", "untriaged", "weak_negative", "genuine_negative"]
    rows.sort(key=lambda r: (order.index(r["label"]), r["testbed"] or "", r["id"]))
    counts = {k: sum(r["label"] == k for r in rows) for k in order}
    lines = ["# Triage of failed Stage 1 gates", "",
             "Advisory only: verdicts in SUMMARY.md are unchanged. Generated by `python triage.py`.", "",
             " · ".join(f"{k}: {v}" for k, v in counts.items() if v), "",
             "| Idea | Testbed | Label | Default gate | Tuned gate (dev) | Flags |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        dg = r["default_gate"]
        dg_s = f"{dg['observed']:+.4f} vs {dg['rule']}" if dg.get("observed") is not None else "-"
        tg = r.get("tuned_gate")
        if tg is None:
            tg_s = r.get("tuned_skip_reason") and "skipped (GPU)" or "not run"
        elif tg.get("error"):
            tg_s = "env error, retry" if tg.get("env_error") else "crashed"
        else:
            tg_s = f"{tg['observed']:+.4f} {'PASS' if tg['passed'] else 'FAIL'}"
        fl = "; ".join(f"{f['flag']}: {f['detail']}" for f in r["flags"]) or ""
        if r.get("review"):
            fl = f"**Reviewed:** {r['review'].get('note', '')} " + (f"(was: {fl})" if fl else "")
        lines.append(f"| {r['title']} | {r['testbed']} | `{r['label']}` | {dg_s} | {tg_s} | {fl} |")
    if resolved:
        lines += ["", "## Resolved since triage", "",
                  "Rerun after a fix and no longer failing the gate. See SUMMARY.md for the current verdict.", "",
                  "| Idea | Testbed | Triage label | Verdict now |", "| --- | --- | --- | --- |"]
        lines += [f"| {r['title']} | {r['testbed']} | `{r['label']}` | `{now}` |" for r, now in resolved]
    lines += ["", "## Labels", ""]
    lines += [f"- `{k}`: {NEXT_STEP[k].replace('<idea>', 'IDEA')}" for k in order]
    (RESULTS / "TRIAGE.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ideas", nargs="*", help="ideas/<idea> dirs; default: every mechanism_failed record")
    ap.add_argument("--run", action="store_true", help="also rerun the gate with both arms tuned (dev only)")
    ap.add_argument("--budget", type=float, default=None,
                    help="stop after this many seconds, saving progress; rerun to resume")
    args = ap.parse_args()
    if args.ideas:
        paths = [RESULTS / Path(i).name / "record.json" for i in args.ideas]
    else:
        paths = [p for p in sorted(RESULTS.glob("*/record.json"))
                 if json.loads(p.read_text()).get("verdict") == "mechanism_failed"]
    deadline = time.time() + args.budget if args.budget else None
    pending = 0
    for p in paths:
        if triage_one(p, run=args.run, deadline=deadline) is None:
            pending += 1
    if pending:
        print(f"{pending} idea(s) unfinished; rerun the same command to resume")
    write_report()
    print(f"wrote {RESULTS / 'TRIAGE.md'}")


if __name__ == "__main__":
    main()
