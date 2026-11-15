"""Build forecasting items from the resolved catalogue (results/commons.json).

NEW for the AI x Epistemics sprint (2026-11-14). Reads pre-existing results; changes none.
Each tested idea yields up to three binary questions whose answers were fixed by the
harness before any forecaster saw them:
  gate : did the pre-registered mechanism gate pass on the toy configuration?
  win  : did the idea get a significant safety gain over the tuned baseline
         (verdict beats_baseline or safe_but_incomplete)?
  sign : given Stage 2 ran, was the mean safety difference above zero?
The prompt fields never include verdicts, observed numbers, or the author's rationale.
"""
import json, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIN = {"beats_baseline", "safe_but_incomplete"}

def main(out):
    d = json.loads((ROOT / "results" / "commons.json").read_text())
    tb = {t["id"]: t for t in d["testbeds"]}
    items = []
    for i in d["ideas"]:
        if i["verdict"] == "untestable" or not i.get("testbed"):
            continue
        t = tb[i["testbed"]]
        pred = i.get("prediction") or {}
        mech = pred.get("mechanism") or {}
        ctx = {
            "idea_title": i["title"], "idea_description": i["description"],
            "source": {k: (i.get("source") or {}).get(k) for k in ("type", "title", "authors", "year", "venue")},
            "testbed_name": t["name"], "testbed_question": t["question"], "testbed_planted": t["planted"],
            "metric": t["metric"], "baseline": t["baseline"],
            "gate_rule": {k: mech.get(k) for k in ("diagnostic", "direction", "mode", "min_effect")},
            "safety_min_effect": pred.get("safety_min_effect"),
            # rule as registered (quantity and threshold only; the observed value is never included)
            "gate_text": f"{(i.get('gate') or {}).get('quantity')}; passes if the difference is {(i.get('gate') or {}).get('rule')}",
        }
        gate = (i.get("gate") or {}).get("passed")
        saf = i.get("safety") or {}
        qs = []
        if gate is not None:
            qs.append(("gate", bool(gate)))
            qs.append(("win", i["verdict"] in WIN))
        if saf.get("mean_diff") is not None:
            qs.append(("sign", saf["mean_diff"] > 0))
        for q, y in qs:
            items.append({
                "item_id": f"{i['id']}::{q}", "idea_id": i["id"], "question": q, "outcome": int(y),
                "cluster": i["testbed"].replace("_para", ""), "testbed": i["testbed"],
                "is_control": bool(i.get("control")), "source_type": ctx["source"]["type"],
                "source_year": ctx["source"]["year"], "resolved": i.get("updated"),
                "author_rationale": pred.get("rationale"), "context": ctx,
            })
    blob = "\n".join(json.dumps(x, sort_keys=True) for x in items) + "\n"
    Path(out).write_text(blob)
    print(len(items), "items; sha256", hashlib.sha256(blob.encode()).hexdigest()[:16])
    for q in ("gate", "win", "sign"):
        s = [x for x in items if x["question"] == q]
        print(q, len(s), "base rate", round(sum(x["outcome"] for x in s) / len(s), 3),
              "clusters", len({x["cluster"] for x in s}))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "items.jsonl"))
