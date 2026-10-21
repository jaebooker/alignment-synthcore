"""Turn a planted_lm GPU job log into standard idea records.

    python -m testbeds.planted_lm.import_results path/to/job.log

Verdicts come from the same harness code as every other testbed (_gate, paired_test,
compute_verdict). The job also runs Stage 2 for ideas that fail the gate; those numbers
are kept in the record as "exploratory_stage2" and never affect the verdict.
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.runner import HARNESS_VERSION, MethodError, _gate  # noqa: E402
from harness.stats import paired_test  # noqa: E402
from harness.verdict import compute_verdict  # noqa: E402
from testbeds.planted_lm.testbed import PlantedLM  # noqa: E402


def parse(log: str) -> dict:
    out = {"stage1": {}, "full": {}}
    for line in log.splitlines():
        i = line.find("RESULT {")
        if i < 0:
            continue
        r = json.loads(line[i + 7:])
        if r["kind"] in ("stage1", "full"):
            out[r["kind"]][r["id"]] = r
        else:
            out[r["kind"]] = r
    return out


def errors(rows):
    return [x[arm]["error"] for x in rows for arm in ("method", "baseline") if "error" in x[arm]]


def record_for(idea_id: str, R: dict, tb, log_name: str) -> dict:
    idea_dir = ROOT / "ideas" / idea_id
    spec = json.loads((idea_dir / "idea.json").read_text())
    pred = spec["prediction"]
    mech = pred.get("mechanism")
    s1rows, full = R["stage1"].get(idea_id), R["full"].get(idea_id)
    record = {"id": idea_id, "title": spec["title"], "description": spec["description"],
              "source": spec.get("source"), "direction": spec.get("direction"), "testbed": tb.name,
              "testbed_version": tb.version, "harness_version": HARNESS_VERSION,
              "evaluator_hash": tb.evaluator_hash(), "prediction": pred, "stage1": None, "stage2": None,
              "tuning": None, "attempts": 1, "calibration": R.get("calibration"),
              "run": {"where": "Runpod GPU pod", "log": log_name}}
    stage1 = stage2 = None
    error = None
    try:
        if not s1rows or not full:
            raise MethodError("job did not finish this idea")
        errs = errors(s1rows["per_seed"]) + errors(full["per_seed"])
        if errs:
            raise MethodError(errs[0])
        rows = s1rows["per_seed"]
        gate = _gate(tb, pred, [r["method"] for r in rows], [r["baseline"] for r in rows])
        stage1 = {**gate, "per_seed": rows}
        record["stage1"] = stage1
        ms = [r["method"] for r in full["per_seed"]]
        bs = [r["baseline"] for r in full["per_seed"]]
        s2 = {"safety": paired_test([m["safety"] for m in ms], [b["safety"] for b in bs])}
        if mech:
            g2 = _gate(tb, pred, ms, bs)
            s2["mechanism"] = paired_test([m["diagnostics"][mech["diagnostic"]] for m in ms],
                                          [b["diagnostics"][mech["diagnostic"]] for b in bs],
                                          higher_is_better=(mech["direction"] == "higher"))
            s2["mechanism_confirmed"] = g2["passed"]
        s2["per_seed"] = full["per_seed"]
        record["tuning"] = {"method": full["tuning"], "baseline": R.get("baseline")}
        if stage1["passed"]:
            stage2 = s2
            record["stage2"] = stage2
        else:
            record["exploratory_stage2"] = {k: v for k, v in s2.items() if k != "per_seed"}
    except MethodError as e:
        error = str(e)
        record["error"] = error
    v = compute_verdict(stage1, stage2, error)
    record.update(v)
    record["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    record["verdict_source"] = "deterministic test code (paired-seed exact sign-flip permutation test)"
    out_dir = ROOT / "results" / idea_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "record.json").write_text(json.dumps(record, indent=2))
    shutil.copy(idea_dir / "method.py", out_dir / "method.py")
    shutil.copy(idea_dir / "idea.json", out_dir / "idea.json")
    return record


def main(log_path: str):
    text = Path(log_path).read_text()
    R = parse(text)
    tb = PlantedLM()
    dest = ROOT / "results" / "_planted_lm"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(log_path, dest / "job.log")
    print("calibration:", R.get("calibration"))
    print("baseline:", (R.get("baseline") or {}).get("hparams"))
    for idea_id in sorted(set(R["stage1"]) | set(R["full"])):
        r = record_for(idea_id, R, tb, "results/_planted_lm/job.log")
        print(f"{idea_id}: {r['verdict']} {r['flags']} {r.get('reason', '')}")


if __name__ == "__main__":
    main(sys.argv[1])
