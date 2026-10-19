"""Collect step 2 (run on the machine with the API key). Standard library only.

    CONTROL_LIVE=1 python3 -m testbeds.control.collect_monitor <path/to/method.py> [more ...]

Runs a monitor over every item the harness will score (toy, dev and test splits, every
HPARAMS entry), plus the red team's probe copy of it on the backdoors, so its
trusted-model calls are cached. Scoring then runs anywhere,
offline. Nothing here executes model-written code. Spend is capped by the shared ledger.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from . import items, llm

HERE = Path(__file__).resolve().parent


def hparams_of(path: Path) -> list:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "HPARAMS" for t in node.targets):
            return ast.literal_eval(node.value)
    return [{}]


def load(path: Path):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))  # so `from public import ...` resolves
    spec = importlib.util.spec_from_file_location(f"monitor_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(path: Path):
    mod, grid = load(path), hparams_of(path)
    jobs = [("dev", "toy", grid[0])] + [(s, "full", hp) for s in ("dev", "test") for hp in grid]
    lib = mod.ask_trusted.__globals__["_client"]({"llm_path": str(HERE / "llm.py")})  # the instance methods use
    for split, config, hp in jobs:
        task, priv = items.build(split, config)
        before, live0, trunc0 = llm.spent_usd(), lib.STATS["live"], lib.STATS["truncated"]
        mod.run(task, hp, 0)
        mod.run(priv["probe_public"], hp, 0)
        live, trunc = lib.STATS["live"] - live0, lib.STATS["truncated"] - trunc0
        print(f"  {path.parent.name or path.stem} {split}/{config} {hp}: +${llm.spent_usd() - before:.4f} "
              f"(total ${llm.spent_usd():.3f} of ${llm.cap_usd():.2f}); new calls {live}, cut off {trunc}", flush=True)
        if live >= 20 and trunc / live > 0.10:
            raise SystemExit(f"Stopping: {trunc}/{live} responses were cut off at max_tokens. "
                             "Raise max_tokens in this monitor before spending more.")


def main():
    for p in sys.argv[1:]:
        collect(Path(p).resolve())


if __name__ == "__main__":
    main()
