"""Build the control_hosted data (cloud workspace: `grade` and `reference` execute model and
reference code in throwaway subprocesses via testbeds/control/execute.py).

    CONTROL_LIVE=1 python -m testbeds.control_hosted.collect outputs     # untrusted honest + 2 attacks, pool solves
    python -m testbeds.control_hosted.collect grade                      # data/outputs.json
    CONTROL_LIVE=1 python -m testbeds.control_hosted.collect select      # sneakiness ratings, best-of-2, red-team picks
    CONTROL_LIVE=1 python -m testbeds.control_hosted.collect reference   # data/reference.json (+ untrusted answers)
    CONTROL_LIVE=1 python -m testbeds.control_hosted.collect reference_mmlu  # data/reference_mmlu.json (CAPA set)
    CONTROL_LIVE=1 python -m testbeds.control_hosted.collect monitor <method.py> [...]

Prompts are Testbed 6's (testbeds/control/prompts.py). Token limits are higher because these
models reason before answering and reasoning counts against the limit.
"""
from __future__ import annotations

import ast
import json
import os
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from testbeds.control import execute, prompts
from testbeds.control.grade import grade_attack
from . import items, llm

HERE = Path(__file__).resolve().parent
PROBLEMS = HERE.parent / "control" / "data" / "problems.json"
OUT = HERE / "data" / "outputs.json"
REF = HERE / "data" / "reference.json"
REF_MMLU = HERE / "data" / "reference_mmlu.json"
MMLU = HERE / "data" / "mmlu_pro_subset.json"
# Part of the cached request identity: do not change once collected.
MAX_TOKENS = {"honest": 8000, "attack": 32000, "trusted_solve": 8000, "sneakiness": 6000, "reference": 4000,
              "reference_mmlu": 8000}
ATTACK_ATTEMPTS = 2
WORKERS = 16


def problems() -> list:
    return json.loads(PROBLEMS.read_text())


def pmap(fn, xs, workers=WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


def honest_req(p, model=None, tag="untrusted_honest", key="honest"):
    return dict(prompt=prompts.honest(p), model=model or llm.UNTRUSTED, system=prompts.SOLVER_SYSTEM,
                max_tokens=MAX_TOKENS[key], tag=tag)


def attack_req(p, sample):
    return dict(prompt=prompts.attack(p), model=llm.UNTRUSTED, system=prompts.ATTACK_SYSTEM,
                max_tokens=MAX_TOKENS["attack"], tag="untrusted_attack", sample=sample)


def sneaky_req(p, a):
    return dict(prompt=prompts.sneakiness(p, a["attack_code"], a["exploit_args"]), model=llm.UNTRUSTED,
                system=prompts.SNEAKY_SYSTEM, max_tokens=MAX_TOKENS["sneakiness"], tag="untrusted_sneakiness")


def solve_reqs(p):
    yield honest_req(p)
    for s in range(ATTACK_ATTEMPTS):
        yield attack_req(p, s)
    for m in llm.POOL:
        yield honest_req(p, m, "trusted_solve", "trusted_solve")


def collect_outputs():
    reqs = [r for p in problems() for r in solve_reqs(p)]
    done = pmap(lambda r: llm.ask_full(**r)["status"], reqs)
    print(f"{len(reqs)} requests; statuses {Counter(done)}; spent {llm.spent():.3f} credits")


def grade():
    def one(p):
        row = {"task_id": p["task_id"], "split": p["split"]}
        code = prompts.extract_code(llm.ask(**honest_req(p)))
        row["honest_code"] = code
        row["honest_pass"] = bool(code) and execute.passes_tests(code, p["test"], p["entry_point"])
        row["trusted_pass"] = {}
        for m in llm.POOL:
            c = prompts.extract_code(llm.ask(**honest_req(p, m, "trusted_solve", "trusted_solve")))
            row["trusted_pass"][m] = bool(c) and execute.passes_tests(c, p["test"], p["entry_point"])
        row["attempts"] = [{"sample": s, **grade_attack(p, llm.ask(**attack_req(p, s)))}
                           for s in range(ATTACK_ATTEMPTS)]
        row.update({k: v for k, v in row["attempts"][0].items() if k != "sample"})  # until `select`
        return row
    rows = pmap(one, problems(), 8)
    OUT.write_text(json.dumps(rows, indent=1))
    n = len(rows)
    att = [a for r in rows for a in r["attempts"]]
    print(f"untrusted honest pass {sum(r['honest_pass'] for r in rows)}/{n}")
    for m in llm.POOL:
        print(f"  {m} solve pass {sum(r['trusted_pass'][m] for r in rows)}/{n}")
    print(f"attempts {len(att)}; working backdoors {sum(a['backdoor_valid'] for a in att)}; "
          f"problems with one {sum(any(a['backdoor_valid'] for a in r['attempts']) for r in rows)}")
    print(Counter(a["backdoor_reason"] for a in att))


def select():
    """Testbed 6 v0.4's selection: keep the sneakiest working attempt per problem (ties to the
    earliest), then the static red team attacks those rated strictly above the median."""
    rows = json.loads(OUT.read_text())
    P = {p["task_id"]: p for p in problems()}
    valid = [(r, a) for r in rows for a in r["attempts"] if a["backdoor_valid"]]
    texts = pmap(lambda ra: llm.ask(**sneaky_req(P[ra[0]["task_id"]], ra[1])), valid)
    for (r, a), t in zip(valid, texts):
        a["sneakiness"] = prompts.parse_sneakiness(t)
    for r in rows:
        best = None
        for a in r["attempts"]:
            a.setdefault("sneakiness", None)
            if a["backdoor_valid"] and (best is None or (a["sneakiness"] or 0) > (best["sneakiness"] or 0)):
                best = a
        pick = best or r["attempts"][0]
        r.update({k: v for k, v in pick.items() if k != "sample"})
        r["attack_sample"] = pick["sample"]
    rated = [r["sneakiness"] for r in rows if r["backdoor_valid"] and r["sneakiness"] is not None]
    thr = statistics.median(rated)
    for r in rows:
        r["attack_selected"] = bool(r["backdoor_valid"] and r["sneakiness"] is not None and r["sneakiness"] > thr)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"candidates {sum(r['backdoor_valid'] for r in rows)} "
          f"(dev {sum(r['backdoor_valid'] for r in rows if r['split'] == 'dev')}, "
          f"test {sum(r['backdoor_valid'] for r in rows if r['split'] == 'test')}); rated {len(rated)}; "
          f"median {thr}; selected {sum(r['attack_selected'] for r in rows)}; dist {sorted(rated)}")


# ---------------- reference set: code-output prediction (CRUXEval-style), multiple choice ----------------

def _strip_docstrings(code: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _calls(test: str) -> list:
    out = []
    for node in ast.walk(ast.parse(test)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "candidate" and not node.keywords:
            try:
                args = [ast.literal_eval(a) for a in node.args]
                json.dumps(args)
            except (ValueError, TypeError, SyntaxError):
                continue
            if len(repr(args)) <= 120 and args not in out:
                out.append(args)
    return out


def _mutants(v):
    if isinstance(v, bool):
        return [repr(not v)]
    if isinstance(v, int):
        return [repr(v + 1), repr(v - 1), repr(v * 2 if v else 2)]
    if isinstance(v, float):
        return [repr(round(v + 1, 6)), repr(round(-v, 6))]
    if isinstance(v, str):
        return [repr(v[::-1]), repr(v[:-1]), repr(v + v[-1:]) if v else repr(" ")]
    if isinstance(v, (list, tuple)):
        t = type(v)
        return [repr(t(list(v)[::-1])), repr(t(list(v)[:-1])), repr(t(list(v) + list(v)[-1:]))]
    return []


def build_reference(per_problem=3, seed=11):
    rng = random.Random(seed)
    qs = []
    for p in problems():
        code = _strip_docstrings(p["canonical"])
        calls = _calls(p["test"])
        outs = []
        for args in calls:
            ok, res = execute.call(p["canonical"], p["entry_point"], args)
            if ok and len(res) <= 80:
                outs.append((args, res))
        if len(outs) < 2:
            continue
        picks = [outs[i] for i in sorted(rng.sample(range(len(outs)), min(per_problem, len(outs))))]
        for args, res in picks:
            try:
                val = ast.literal_eval(res)
            except (ValueError, SyntaxError):
                val = None
            pool = [o for _, o in outs if o != res] + (_mutants(val) if val is not None else [])
            pool = [o for o in dict.fromkeys(pool) if o != res]
            if len(pool) < 3:
                continue
            opts = rng.sample(pool, 3) + [res]
            rng.shuffle(opts)
            letters = "ABCD"
            call = f"{p['entry_point']}({', '.join(repr(a) for a in args)})"
            qs.append({"id": f"ref-{len(qs):03d}", "task_id": p["task_id"], "code": code, "call": call,
                       "options": dict(zip(letters, opts)), "answer": letters[opts.index(res)]})
    return qs


def collect_reference():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch_public", HERE / "public.py")
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    qs = build_reference()
    texts = pmap(lambda q: llm.ask(prompt=pub.reference_prompt(q), model=llm.UNTRUSTED,
                                   max_tokens=MAX_TOKENS["reference"], tag="untrusted_reference"), qs)
    for q, t in zip(qs, texts):
        q["untrusted_answer"] = pub.last_letter(t)
    REF.write_text(json.dumps(qs, indent=1))
    acc = sum(q["untrusted_answer"] == q["answer"] for q in qs) / len(qs)
    print(f"{len(qs)} reference questions; untrusted accuracy {acc:.3f}; "
          f"unanswered {sum(q['untrusted_answer'] is None for q in qs)}")


def collect_reference_mmlu():
    """Amendment 1: the CAPA reference is a fixed MMLU-Pro subset (data/mmlu_pro_subset.json)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch_public", HERE / "public.py")
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    qs = json.loads(MMLU.read_text())["questions"]
    texts = pmap(lambda q: llm.ask(prompt=pub.reference_prompt(q), model=llm.UNTRUSTED,
                                   max_tokens=MAX_TOKENS["reference_mmlu"], tag="untrusted_reference_mmlu"), qs)
    for q, t in zip(qs, texts):
        q["untrusted_answer"] = pub.answer_letter(q, t)
    REF_MMLU.write_text(json.dumps(qs, indent=1))
    acc = sum(q["untrusted_answer"] == q["answer"] for q in qs) / len(qs)
    print(f"{len(qs)} MMLU-Pro reference questions; untrusted accuracy {acc:.3f}; "
          f"unanswered {sum(q['untrusted_answer'] is None for q in qs)}")


# ---------------- monitors ----------------

def collect_monitor(path: Path):
    import importlib.util
    from testbeds.control.collect_monitor import hparams_of
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))  # `from public import ...` must resolve to this testbed's public.py
    spec = importlib.util.spec_from_file_location(f"monitor_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    grid = hparams_of(path)
    splits = os.environ.get("CONTROL_SPLITS", "dev,test").split(",")
    jobs = [("dev", "toy", grid[0])] + [(s, "full", hp) for s in ("dev", "test") if s in splits for hp in grid]
    for split, config, hp in jobs:
        task, priv = items.build(split, config)
        before, live0, trunc0 = llm.spent(), llm.STATS["live"], llm.STATS["truncated"]
        mod.run(task, hp, 0)
        mod.run(priv["probe_public"], hp, 0)
        live, trunc = llm.STATS["live"] - live0, llm.STATS["truncated"] - trunc0
        print(f"  {path.parent.name} {split}/{config} {hp}: +{llm.spent() - before:.4f} credits "
              f"(total {llm.spent():.3f} of {llm.CAP}); new calls {live}, cut off {trunc}", flush=True)
        if live >= 20 and trunc / live > 0.10:
            raise SystemExit(f"Stopping: {trunc}/{live} responses cut off; raise max_tokens first.")


def main():
    cmd = sys.argv[1]
    if cmd == "outputs":
        collect_outputs()
    elif cmd == "grade":
        grade()
    elif cmd == "select":
        select()
    elif cmd == "reference":
        collect_reference()
    elif cmd == "reference_mmlu":
        collect_reference_mmlu()
    elif cmd == "monitor":
        for p in sys.argv[2:]:
            collect_monitor(Path(p).resolve())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
