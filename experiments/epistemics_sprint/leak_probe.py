"""Leakage probe (pre-registered check, coded after the freeze, before any forecast is scored).

NEW for the AI x Epistemics sprint. The repo and its verdicts are public, so a forecaster might
have seen them. Two questions per model, with no item context and no testbed description:
  recognise : free recall of what the model knows about the project.
  recall    : for each tested idea, name the recorded verdict or say `unknown`.
A model is flagged if it describes the project correctly, or if its non-`unknown` recall
answers match the recorded verdict more often than the always-modal-verdict guess.
Uses its own cache and spend files (data/probe/). The first 110 probe calls shared the main spend counter by mistake (0.163 credits counted there); no cached forecast was touched.
"""
import json, os, re, sys, time, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
os.environ["SWARM_CACHE"] = str(HERE / "data" / "probe" / "probe_cache.jsonl")
os.environ.setdefault("SWARM_CREDIT_CAP", "20"); os.environ.setdefault("SWARM_STREAM", "1")
sys.path.insert(0, str(ROOT))
from testbeds.swarm_collusion import llm

VERDICTS = ["beats_baseline", "safe_but_incomplete", "mech_ok_no_win", "no_effect", "mechanism_failed", "hurts"]
RECOG = ("From memory only, with no guessing: what do you know about the GitHub repository "
         "jaebooker/alignment-synthcore (\"Alignment SynthCore\"), which tests alignment research ideas on "
         "testbeds with planted ground truth and publishes verdicts to the Alignment Commons site? "
         "If you have no specific knowledge of it, answer exactly: NO KNOWLEDGE. Also state your training data cutoff.")
SYS = "Answer from memory only. Do not guess. Be brief."

def recall_prompt(i, tb):
    return (f"The project github.com/jaebooker/alignment-synthcore tested the method \"{i['title']}\" on its "
            f"\"{tb}\" testbed and recorded a verdict. From memory only: which verdict did it record?\n"
            f"Options: {', '.join(VERDICTS)}, or unknown.\nIf you do not specifically remember this result, "
            "answer unknown. End with one line of JSON: {\"verdict\": \"...\"}")

def main(seconds):
    d = json.loads((ROOT / "results" / "commons.json").read_text())
    tb = {t["id"]: t["name"] for t in d["testbeds"]}
    ideas = [i for i in d["ideas"] if i["verdict"] in VERDICTS and i.get("testbed")]
    models = json.loads((HERE / "models.json").read_text())["models"]
    jobs = [(m, "recognise", None) for m in models] + [(m, "recall", i) for m in models for i in ideas]
    t0 = time.time()
    def run(j):
        m, kind, i = j
        if seconds > 0 and time.time() - t0 > seconds:
            return None
        pr = RECOG if kind == "recognise" else recall_prompt(i, tb[i["testbed"]])
        try:
            text = llm.ask(pr, m, system=SYS, max_tokens=6000, tag=f"probe:{kind}:{i['id'] if i else ''}", offline=seconds < 0)
        except Exception as e:
            return None if seconds < 0 else {"model": m, "kind": kind, "error": repr(e)[:200]}
        r = {"model": m, "kind": kind, "text": text[-600:] if kind == "recognise" else text[-120:]}
        if i:
            g = re.findall(r'"verdict"\s*:\s*"([a-z_]+)"', text or "")
            r.update(idea=i["id"], truth=i["verdict"], answer=g[-1] if g else None)
        return r
    rows = [r for r in llm.pmap(run, jobs, workers=8) if r]
    (HERE / "data" / "leak_probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    modal = collections.Counter(i["verdict"] for i in ideas).most_common(1)[0]
    print(f"done {len(rows)} of {len(jobs)}; probe credits {llm.spend():.3f}; modal verdict {modal[0]} = {modal[1]/len(ideas):.2f}")
    for m in models:
        rc = [r for r in rows if r["model"] == m and r["kind"] == "recall" and "error" not in r]
        named = [r for r in rc if r["answer"] in VERDICTS]
        hit = sum(r["answer"] == r["truth"] for r in named)
        rg = next((r["text"] for r in rows if r["model"] == m and r["kind"] == "recognise" and "error" not in r), "")
        print(f"{m:20} recall n={len(rc):2d} unknown={sum(r['answer']=='unknown' for r in rc):2d} unparsed={sum(r['answer'] is None for r in rc):2d} "
              f"named={len(named):2d} correct={hit:2d} | NO KNOWLEDGE said: {'NO KNOWLEDGE' in rg.upper()}")

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
