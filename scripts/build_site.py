"""Write site/data.json for the public SynthCore site from results/.

Reads results/commons.json (the published verdicts) and each results/<idea>/record.json
(for when it ran and how long it took). Run after summarize.py:

    python summarize.py && python scripts/build_site.py

The site only reads data.json; nothing here changes a verdict.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "site" / "data.json"

# Testbeds grouped into the areas the live view draws as clusters.
AREAS = [
    ("interp", "Interpretability", ["planted_interp", "planted_lm"]),
    ("backdoor", "Backdoors", ["backdoor"]),
    ("hacking", "Reward hacking", ["reward_hacking", "interactive_gridworld"]),
    ("misgen", "Goal misgeneralization", ["goal_misgen", "ued_gridworld", "confounded_imitation"]),
    ("oversight", "Scalable oversight", ["weak_to_strong", "debate_code"]),
    ("control", "AI control", ["control", "control_hosted", "control_open"]),
    ("swarm", "Multi-agent collusion", ["swarm_collusion", "swarm_collusion_para"]),
]


def short(text, n=280):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    pub = json.loads((RES / "commons.json").read_text())
    known = {t for _, _, ts in AREAS for t in ts}
    extra = [t["id"] for t in pub["testbeds"] if t["id"] not in known]
    if extra:
        raise SystemExit(f"Add these testbeds to AREAS in scripts/build_site.py: {extra}")

    ideas = []
    for x in pub["ideas"]:
        rec_path = RES / x["id"] / "record.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        s = x.get("safety") or {}
        src = x.get("source") or {}
        ideas.append({
            "id": x["id"],
            "title": x["title"],
            "description": short(x.get("description")),
            "testbed": x.get("testbed"),
            "nearest_testbed": x.get("nearest_testbed"),
            "topic": x.get("topic"),
            "control": bool(x.get("control")),
            "verdict": x["verdict"],
            "verdict_v1": x.get("verdict_v1"),
            "flags": x.get("flags") or [],
            "mechanism_confirmed": x.get("mechanism_confirmed"),
            "gate": x.get("gate"),
            "safety": {"diff": s.get("mean_diff"), "ci95": s.get("ci95"), "p": s.get("p_value")} if s else None,
            "triage": (x.get("triage") or {}).get("label") if isinstance(x.get("triage"), dict) else x.get("triage"),
            "untestable_reason": short(x.get("untestable_reason"), 220) if x.get("untestable_reason") else None,
            "source": {k: src.get(k) for k in ("title", "authors", "year", "venue", "url") if src.get(k)},
            "updated": rec.get("updated_at") or x.get("updated"),
            "runtime_s": rec.get("runtime_s"),
        })

    runs = sorted((i for i in ideas if i["updated"]), key=lambda i: i["updated"], reverse=True)
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit(),
        "repo": pub.get("repo"),
        "min_ideas": pub.get("min_ideas"),
        "areas": [{"id": a, "name": n, "testbeds": ts} for a, n, ts in AREAS],
        "testbeds": [{k: t.get(k) for k in ("id", "name", "question", "planted", "metric", "baseline", "version")} for t in pub["testbeds"]],
        "ideas": ideas,
        "runs": [i["id"] for i in runs],
        "signals": pub.get("signals", []),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ideas)} ideas, {len(data['testbeds'])} testbeds")


if __name__ == "__main__":
    main()
