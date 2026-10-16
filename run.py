"""Run ideas through the pipeline.

    python run.py ideas/<idea-slug> [ideas/<another> ...]
    python run.py --all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.runner import run_idea  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ideas", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage2-seeds", type=int, default=8)
    args = ap.parse_args()
    dirs = sorted((ROOT / "ideas").iterdir()) if args.all else [Path(p) for p in args.ideas]
    for d in dirs:
        if (d / "idea.json").exists():
            run_idea(d, stage2_seeds=args.stage2_seeds, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
