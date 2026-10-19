"""Run model-written Python in a throwaway subprocess (evaluator side, cloud workspace only).

Each call gets a fresh temp dir, `python -I`, a wall-clock timeout and CPU/memory
limits. Never run this on a personal machine: attack-mode code is written to
misbehave on purpose (wrong answers on rare inputs), and isolation is the reason
grading happens here.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile

TIMEOUT_S = 10


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


def _run(script: str) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory(prefix="ctl_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(script)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=TIMEOUT_S + 2, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            return p.returncode, p.stdout, p.stderr[-1000:]
        except subprocess.TimeoutExpired:
            return -9, "", "timeout"


def passes_tests(code: str, test: str, entry_point: str) -> bool:
    rc, _, _ = _run(f"{code}\n\n{test}\n\ncheck({entry_point})\n")
    return rc == 0


def call(code: str, entry_point: str, args: list) -> tuple[bool, str]:
    """Call entry_point(*args). Returns (ok, repr of result or error)."""
    script = (f"{code}\n\nimport json as _j\n_args = _j.loads({json.dumps(json.dumps(args))})\n"
              f"print('__RESULT__' + repr({entry_point}(*_args)))\n")
    rc, out, err = _run(script)
    if rc != 0 or "__RESULT__" not in out:
        return False, err or f"rc={rc}"
    return True, out.split("__RESULT__", 1)[1].strip()
