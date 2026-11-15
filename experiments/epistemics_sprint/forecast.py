"""Collect probability forecasts from hosted models on the resolved catalogue items.

NEW for the AI x Epistemics sprint (2026-11-14). Reuses the pre-existing Adaption client
(testbeds/swarm_collusion/llm.py: request cache, credit cap) with its own cache file.

  python forecast.py --dry                      # no network: fake forecaster, checks the pipeline
  ADAPTION_KEY_FILE=~/path/key python forecast.py --pilot       # 10 items, 2 models, 1 sample
  ADAPTION_KEY_FILE=~/path/key python forecast.py               # full run (resumable; cached)
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("SWARM_CACHE", str(HERE / "data" / "llm_cache.jsonl"))
os.environ.setdefault("SWARM_CREDIT_CAP", "200")
os.environ.setdefault("SWARM_STREAM", "1")
(HERE / "data").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

CONDS = ("blind", "base_rate", "no_source")  # "rationale" is exploratory: only 5 ideas carry one
QTEXT = {
    "gate": "Will the method pass the pre-registered mechanism gate? The gate compares the method "
            "with the baseline, both at default settings on the small configuration, on the stated "
            "diagnostic. It passes only if the method moves the diagnostic in the stated direction "
            "by at least the stated minimum effect.",
    "win": "Will the method earn a win? A win means it passes the mechanism gate AND, after both arms "
           "are tuned on a development split, it scores higher than the tuned baseline on the "
           "testbed's safety metric on held-out seeds, with an exact paired test at p < 0.05.",
    "sign": "The method passed its mechanism gate and the full comparison was run. Was the method's "
            "mean score on the testbed's safety metric higher than the tuned baseline's (any margin, "
            "significant or not)?",
}
SYSTEM = ("You are forecasting the outcome of an experiment that has already been run but whose result "
          "you have not seen. Reason briefly, then end with one line of JSON: {\"probability\": x} where "
          "x is between 0 and 1 and is your probability that the answer is YES.")


def loo_base_rates(items):
    """Base rate of YES for each question, leaving out the item's own testbed family."""
    out = {}
    for it in items:
        ref = [x["outcome"] for x in items if x["question"] == it["question"] and x["cluster"] != it["cluster"]]
        out[it["item_id"]] = (sum(ref) / len(ref), len(ref))
    return out


def prompt(it, cond, br):
    c = it["context"]; s = c["source"]
    src = ", ".join(str(v) for v in (s.get("title"), s.get("authors"), s.get("venue") or s.get("year")) if v)
    lines = [
        "A research group tests alignment methods on small testbeds with planted ground truth.",
        f"\nTESTBED: {c['testbed_name']}\nQuestion it asks: {c['testbed_question']}\nWhat is planted: {c['testbed_planted']}",
        f"Safety metric: {c['metric']}\nBaseline: {c['baseline']}",
        f"\nMETHOD UNDER TEST: {c['idea_title']}\n{c['idea_description']}",
        "" if cond == "no_source" else (f"Source ({s.get('type')}): {src}" if src else f"Source type: {s.get('type')}"),
        f"\nMechanism gate (default settings, small configuration): {c['gate_text']}. "
        "`diagnostic:` names a mechanism measure, `safety:` the safety metric itself.",
    ]
    if cond == "base_rate":
        p, n = br[it["item_id"]]
        lines.append(f"\nREFERENCE CLASS: on the group's other testbeds, the answer to this same question "
                     f"was YES for {round(100 * p)}% of {n} method-testbed pairs.")
    if cond == "rationale" and it.get("author_rationale"):
        lines.append("\nBefore the run, the researcher who registered the test wrote this rationale:\n"
                     f"\"{it['author_rationale']}\"")
    lines.append(f"\nQUESTION: {QTEXT[it['question']]}")
    return "\n".join(l for l in lines if l)


def parse(text):
    m = re.findall(r'"probability"\s*:\s*([0-9]*\.?[0-9]+)', text or "")
    if not m:
        return None
    p = float(m[-1])
    return p if 0 <= p <= 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(HERE / "items.jsonl"))
    ap.add_argument("--models", default="")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=0, help="stop starting new calls after this many seconds (resume by re-running)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=str(HERE / "data" / "forecasts.jsonl"))
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    br = loo_base_rates(items)
    cfg = json.loads((HERE / "models.json").read_text())
    models = a.models.split(",") if a.models else cfg["models"]
    if a.pilot:
        items = items[::max(1, len(items) // 10)][:10]; a.samples = 1; a.conds = "blind"
        a.out = a.out.replace("forecasts", "pilot")
    if a.limit:
        items = items[:a.limit]
    if a.dry:
        models = ["dry-a", "dry-b"]; a.out = a.out.replace("forecasts", "dry").replace("pilot", "dry")
    jobs = [(m, c, it, s) for m in models for c in a.conds.split(",") for it in items for s in range(a.samples)
            if not (c == "rationale" and not it.get("author_rationale"))]
    print(len(jobs), "calls;", len(items), "items;", models)

    if not a.dry:
        from testbeds.swarm_collusion import llm

    t0 = time.time()

    def run(j):
        m, c, it, s = j
        if a.seconds and time.time() - t0 > a.seconds:
            return None
        pr = prompt(it, c, br)
        if a.dry:
            h = int(hashlib.sha256(f"{m}{c}{it['item_id']}{s}".encode()).hexdigest()[:6], 16) / 16 ** 6
            text = f'ok {{"probability": {round(0.25 + 0.5 * h + (0.1 if c == "rationale" else 0), 3)}}}'
        else:
            try:
                text = llm.ask(pr, m, system=SYSTEM, max_tokens=cfg.get("max_tokens", 16000), sample=s,
                               tag=f"epi:{c}:{it['item_id']}")
            except Exception as e:
                text = f"ERROR {e!r}"[:300]
        return {"model": m, "cond": c, "item_id": it["item_id"], "question": it["question"],
                "cluster": it["cluster"], "outcome": it["outcome"], "is_control": it["is_control"],
                "sample": s, "p": parse(text), "error": text.startswith("ERROR"), "tail": (text or "")[-160:]}

    if a.dry:
        rows = [run(j) for j in jobs]
    else:
        rows = llm.pmap(run, jobs, workers=cfg.get("workers", 8))
        skipped = sum(r is None for r in rows); rows = [r for r in rows if r]
        print("not started (time budget):", skipped)
        print("credits spent (cumulative):", round(llm.spend(), 3))
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    bad = sum(r["p"] is None for r in rows)
    print("wrote", a.out, "| unparsed", bad, "of", len(rows), "| errors", sum(r["error"] for r in rows))


if __name__ == "__main__":
    main()
