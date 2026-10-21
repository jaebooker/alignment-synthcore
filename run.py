"""Run ideas through the pipeline.

    python run.py ideas/<idea-slug> [ideas/<another> ...]
    python run.py --all
    python run.py ideas/<idea> --resume [--budget SECONDS]

--resume saves every finished arm under results/_cache/runs/ and reuses it, so a run cut
off by a crash, a sleeping laptop or --budget picks up where it stopped. Rerun the same
command until it reports a verdict. The verdict code and record are unchanged.
"""
import time
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import runner  # noqa: E402
from harness.runner import OutOfTime, run_idea  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ideas", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage2-seeds", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="cache finished arms on disk and reuse them")
    ap.add_argument("--budget", type=float, default=None, help="with --resume: stop after this many seconds")
    args = ap.parse_args()
    if args.budget is not None and not args.resume:
        ap.error("--budget needs --resume, or the work done before stopping is lost")
    runner.RESUME["on"] = args.resume
    runner.RESUME["deadline"] = time.time() + args.budget if args.budget else None
    unfinished = []
    dirs = sorted((ROOT / "ideas").iterdir()) if args.all else [Path(p) for p in args.ideas]
    for d in dirs:
        if (d / "idea.json").exists():
            try:
                run_idea(d, stage2_seeds=args.stage2_seeds, log=lambda m: print(m, flush=True))
            except OutOfTime as e:
                print(f"[{d.name}] stopped: {e}. Progress saved; rerun to resume.", flush=True)
                unfinished.append(d.name)
    if unfinished:
        print(f"unfinished: {', '.join(unfinished)}")
        sys.exit(3)


if __name__ == "__main__":
    main()
