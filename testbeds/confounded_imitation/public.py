"""Public module for the confounded-imitation testbed. Methods import this as `public`.

    e = public.env(i)
    obs = e.reset()                       # list of 5 floats; dims permuted, unnamed
    obs, reward, done = e.step(a)         # a in 0..2; reward -1 per step until the goal
    a_star = e.query_expert()             # the expert's action at the current state
    e.budget()                            # {"episodes", "steps", "queries"} remaining
Exceeding a budget raises BudgetExceeded.

Policies: return {"arch": {"type": "mlp", "hidden": [int, ...], "mask_input": bool},
"state_dict": {name: numpy array}, "mask": [0/1 per dim]}. The evaluator zeroes masked dims;
with mask_input the mask is also appended to the input (a graph-conditioned policy).
"""
import json
import os

import numpy as np
import torch
from torch import nn

N_ACTIONS = 3
_CONN = None


class BudgetExceeded(Exception):
    pass


class EnvError(Exception):
    pass


def _connect(rfd, wfd):
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
        return r["obs"], r["reward"], r["done"]

    def query_expert(self):
        return _call({"op": "query_expert", "task": self.task})["action"]

    def budget(self):
        return _call({"op": "budget", "task": self.task})


def env(i):
    return Env(i)


def make_policy(cfg: dict, d_in: int) -> nn.Module:
    if cfg["type"] != "mlp":
        raise ValueError(f"unknown arch {cfg!r}")
    layers, d = [], d_in
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, N_ACTIONS))
    return nn.Sequential(*layers)


def load_policy(out: dict, d_in: int) -> nn.Module:
    m = make_policy(out["arch"], d_in)
    m.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in out["state_dict"].items()})
    m.eval()
    return m


def export(model: nn.Module, cfg: dict, mask=None) -> dict:
    out = {"arch": cfg, "state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}
    if mask is not None:
        out["mask"] = [float(x) for x in mask]
    return out
