"""Forecasts on the sub-claim item set (Amendment 1). Separate run, separate cache and spend files.

NEW for the AI x Epistemics sprint, written after the main freeze. Imports the frozen forecast.py
only for its system prompt and parser, and does not modify any frozen file.

  python forecast_sub.py --dry
  export ADAPTION_API_KEY="$(cat ~/Downloads/adaption_key/adaptionkey.txt)"
  SWARM_CALL_TIMEOUT=600 python forecast_sub.py          # run only after forecast.py has finished
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
(HERE / "data" / "sub").mkdir(parents=True, exist_ok=True)
os.environ["SWARM_CACHE"] = str(HERE / "data" / "sub" / "llm_cache.jsonl")   # before importing forecast
os.environ.setdefault("SWARM_CREDIT_CAP", "150")
import forecast as F

CONDS = ("blind", "base_rate")

def loo(items):
    out = {}
    for it in items:
        ref = [x["outcome"] for x in items if x["cluster"] != it["cluster"]]
        out[it["item_id"]] = (sum(ref) / len(ref), len(ref))
    return out

def prompt(it, cond, br):
    lines = ["A research group runs small pre-registered experiments on alignment testbeds. Before each confirmation "
             "run it freezes a claim with a decision rule. Intervals are 95% unless stated.",
             f"\nSTUDY DESIGN: {it['setup']}", f"\nREGISTERED CLAIM: {it['claim']}"]
    if cond == "base_rate":
        p, n = br[it["item_id"]]
        lines.append(f"\nREFERENCE CLASS: of {n} registered claims from this group's studies on other testbed "
                     f"families, {round(100 * p)}% were met.")
    lines.append("\nQUESTION: On the confirmation data, was this registered claim met?")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=""); ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=0); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(HERE / "subclaims.jsonl")]
    br = loo(items)
    cfg = json.loads((HERE / "models.json").read_text())
    models = ["dry-a", "dry-b"] if a.dry else (a.models.split(",") if a.models else cfg["models"])
    out = HERE / "data" / "sub" / ("dry.jsonl" if a.dry else "sub_forecasts.jsonl")
    jobs = [(m, c, it, s) for m in models for c in CONDS for it in items for s in range(a.samples)]
    print(len(jobs), "calls;", len(items), "sub-claims;", models)
    if not a.dry:
        from testbeds.swarm_collusion import llm
    t0 = time.time()
    def run(j):
        m, c, it, s = j
        if a.seconds and time.time() - t0 > a.seconds:
            return None
        if a.dry:
            h = int(hashlib.sha256(f"{m}{c}{it['item_id']}{s}".encode()).hexdigest()[:6], 16) / 16 ** 6
            text = f'{{"probability": {round(0.3 + 0.5 * h, 3)}}}'
        else:
            try:
                text = llm.ask(prompt(it, c, br), m, system=F.SYSTEM, max_tokens=cfg.get("max_tokens", 16000),
                               sample=s, tag=f"episub:{c}:{it['item_id']}")
            except Exception as e:
                text = f"ERROR {e!r}"[:300]
        return {"model": m, "cond": c, "item_id": it["item_id"], "question": "sub", "cluster": it["cluster"],
                "outcome": it["outcome"], "is_control": False, "sample": s, "p": F.parse(text),
                "error": text.startswith("ERROR"), "tail": (text or "")[-160:]}
    rows = [run(j) for j in jobs] if a.dry else llm.pmap(run, jobs, workers=cfg.get("workers", 8))
    rows = [r for r in rows if r]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    err = sum(r["error"] for r in rows)
    print("wrote", out, "| rows", len(rows), "of", len(jobs), "| unparsed", sum(r["p"] is None for r in rows), "| errors", err)
    if not a.dry:
        print("credits spent on this set:", round(llm.spend(), 3))
    if err > 0.5 * max(1, len(rows)):
        print("WARNING: most calls failed. Check ADAPTION_API_KEY before trusting this file."); sys.exit(1)

if __name__ == "__main__":
    main()
