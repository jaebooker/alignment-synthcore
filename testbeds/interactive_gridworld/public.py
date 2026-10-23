"""Client for the interactive gridworld testbed. Methods import this as `public`.

    import public
    e = public.env(i)              # task i
    obs = e.reset()                # opaque int observation
    obs, reward, done, approval = e.step(a)   # a in 0..4; approval: the overseer's view of this executed step
    v = e.query_approval(a)        # overseer's approval of action a at the current state, without executing it
    e.budget()                     # {"steps": remaining, "queries": remaining}

reward is always 0.0 for tasks whose public spec says env_reward=False.
Exceeding a budget raises public.BudgetExceeded. Each step and each query costs one unit.
The environment itself runs in the harness; this module only forwards requests.
"""
import json
import os

_CONN = None


class BudgetExceeded(Exception):
    pass


class EnvError(Exception):
    pass


def _connect(rfd, wfd):
    """Called by the sandbox before the method runs."""
    global _CONN
    _CONN = (os.fdopen(rfd, "r", buffering=1), os.fdopen(wfd, "w", buffering=1))


def _call(req):
    if _CONN is None:
        raise EnvError("not connected to the harness (run the method through run.py)")
    r, w = _CONN
    w.write(json.dumps(req) + "\n")
    w.flush()
    line = r.readline()
    if not line:
        raise EnvError("harness closed the connection")
    resp = json.loads(line)
    if "error" in resp:
        if resp["error"] == "BudgetExceeded":
            raise BudgetExceeded(resp.get("message", ""))
        raise EnvError(f"{resp['error']}: {resp.get('message', '')}")
    return resp


class Env:
    def __init__(self, task):
        self.task = int(task)

    def reset(self):
        return _call({"op": "reset", "task": self.task})["obs"]

    def step(self, a):
        r = _call({"op": "step", "task": self.task, "a": int(a)})
        return r["obs"], r["reward"], r["done"], r["approval"]

    def query_approval(self, a):
        return _call({"op": "query", "task": self.task, "a": int(a)})["approval"]

    def budget(self):
        return _call({"op": "budget", "task": self.task})


def env(i):
    return Env(i)
