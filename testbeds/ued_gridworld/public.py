"""Public module for the UED gridworld testbed. Methods import this as `public`.

Environment client (episodes run in the harness):
    v = public.venv(task_i)
    ids = v.new_levels(k)                  # k new levels from the default distribution
    kids = v.edit_levels(ids, k=3)         # a child of each level after k random edits (harness picks them)
    obs = v.reset(slots, level_ids)        # start episodes on chosen levels -> uint8 (n, C, V, V)
    obs, reward, done = v.step(slots, actions)   # one step for each listed slot
    v.budget()                             # {"steps": remaining, "levels": remaining}
A slot that is done must be reset (on any drawn level) before it is stepped again.
Actions: 0 up, 1 down, 2 left, 3 right. Exceeding a budget raises BudgetExceeded.

Policies: a method returns {"arch": cfg, "state_dict": {name: numpy array}} with cfg one of
    {"type": "mlp", "hidden": [int, ...]}
    {"type": "cnn", "channels": [int, ...], "hidden": int}

Shared trainer: train_ppo(task_i, spec, hp, curriculum, seed) runs PPO and asks the
curriculum which level each new episode uses and whether to learn from it. Methods may use
it or write their own learner. A curriculum implements
    next_level(levels) -> (level_id, train: bool)   # levels: a LevelSource for drawing new ids
    episode_done(level_id, ret)                     # called with each finished episode's return
"""
import base64
import json
import os

import numpy as np
import torch
from torch import nn

N_ACTIONS = 4
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


def _arr(d):
    return np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["shape"])


class Venv:
    def __init__(self, task):
        self.task = int(task)

    def new_levels(self, k=1):
        return _call({"op": "new_levels", "task": self.task, "n": int(k)})["levels"]

    def edit_levels(self, levels, k=1):
        return _call({"op": "edit_levels", "task": self.task, "levels": [int(x) for x in levels], "k": int(k)})["levels"]

    def reset(self, slots, levels):
        return _arr(_call({"op": "reset", "task": self.task, "slots": [int(s) for s in slots],
                           "levels": [int(x) for x in levels]})["obs"])

    def step(self, slots, actions):
        r = _call({"op": "step", "task": self.task, "slots": [int(s) for s in slots],
                   "actions": [int(a) for a in actions]})
        return _arr(r["obs"]), np.array(r["reward"], np.float32), np.array(r["done"], bool)

    def budget(self):
        return _call({"op": "budget", "task": self.task})


def venv(i):
    return Venv(i)


class LevelSource:
    """Draws new level ids in batches, so a curriculum can ask for one at a time cheaply."""

    def __init__(self, v, batch=256):
        self.v, self.batch, self.buf = v, batch, []

    def new(self):
        if not self.buf:
            self.buf = list(self.v.new_levels(self.batch))
        return self.buf.pop(0)


# ---------------------------------------------------------------- policies

def make_policy(cfg: dict, in_channels: int, view: int) -> nn.Module:
    if cfg["type"] == "mlp":
        layers, d = [nn.Flatten()], in_channels * view * view
        for h in cfg["hidden"]:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, N_ACTIONS))
        return nn.Sequential(*layers)
    if cfg["type"] == "cnn":
        layers, c = [], in_channels
        for k in cfg["channels"]:
            layers += [nn.Conv2d(c, k, 3, padding=1), nn.ReLU()]
            c = k
        layers += [nn.Flatten(), nn.Linear(c * view * view, cfg["hidden"]), nn.ReLU(),
                   nn.Linear(cfg["hidden"], N_ACTIONS)]
        return nn.Sequential(*layers)
    raise ValueError(f"unknown arch {cfg!r}")


def load_policy(out: dict, in_channels: int, view: int) -> nn.Module:
    m = make_policy(out["arch"], in_channels, view)
    m.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in out["state_dict"].items()})
    m.eval()
    return m


def export(model: nn.Module, cfg: dict) -> dict:
    return {"arch": cfg, "state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}


# ---------------------------------------------------------------- shared PPO trainer

class ActorCritic(nn.Module):
    def __init__(self, cfg, c, v):
        super().__init__()
        self.cfg = cfg
        self.policy = make_policy(cfg, c, v)
        self.value = nn.Linear(cfg["hidden"], 1)

    def forward(self, x):
        h = self.policy[:-1](x)
        return self.policy[-1](h), self.value(h).squeeze(-1)


class Uniform:
    """Domain randomization: every episode on a fresh level from the default distribution."""

    def next_level(self, levels):
        return levels.new(), True

    def episode_done(self, level, ret):
        pass


def train_ppo(task_i, spec, hp, curriculum, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = venv(task_i)
    levels = LevelSource(v)
    cfg = {"type": "cnn", "channels": [32, 32], "hidden": hp.get("hidden", 128)}
    net = ActorCritic(cfg, spec["n_channels"], spec["view"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    E, T = hp.get("n_envs", 32), hp.get("rollout", 64)
    gamma, lam = hp.get("gamma", 0.99), hp.get("gae_lambda", 0.95)
    slots = list(range(E))
    lv, train, ret = [0] * E, [True] * E, np.zeros(E)
    for k in slots:
        lv[k], train[k] = curriculum.next_level(levels)
    obs = v.reset(slots, lv)
    try:
        while True:
            O, A, LP, R, D, V, M = [], [], [], [], [], [], []
            for _ in range(T):
                o = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    lg, val = net(o)
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample()
                mask = torch.tensor(train, dtype=torch.float32)
                obs2, r, d = v.step(slots, a.tolist())
                ret += r
                O.append(o), A.append(a), LP.append(dist.log_prob(a)), V.append(val), M.append(mask)
                R.append(torch.tensor(r)), D.append(torch.tensor(d, dtype=torch.float32))
                done_slots = [k for k in slots if d[k]]
                if done_slots:
                    for k in done_slots:
                        curriculum.episode_done(lv[k], float(ret[k]))
                        ret[k] = 0.0
                        lv[k], train[k] = curriculum.next_level(levels)
                    fresh = v.reset(done_slots, [lv[k] for k in done_slots])
                    obs2 = obs2.copy()
                    obs2[done_slots] = fresh
                obs = obs2
            with torch.no_grad():
                _, nv = net(torch.tensor(obs, dtype=torch.float32))
            adv = torch.zeros(T, E)
            last = torch.zeros(E)
            for t in reversed(range(T)):
                nxt = nv if t == T - 1 else V[t + 1]
                delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]
                last = delta + gamma * lam * (1 - D[t]) * last
                adv[t] = last
            rets = adv + torch.stack(V)
            m = torch.stack(M).flatten() > 0
            if m.sum() < 64:
                continue
            Ob, Ab, LPb = torch.cat(O)[m], torch.cat(A)[m], torch.cat(LP)[m]
            ADb, RTb = adv.flatten()[m], rets.flatten()[m]
            ADb = (ADb - ADb.mean()) / (ADb.std() + 1e-8)
            for _ in range(hp.get("epochs", 3)):
                for mb in torch.randperm(len(Ob)).split(hp.get("minibatch", 512)):
                    lg, val = net(Ob[mb])
                    dist = torch.distributions.Categorical(logits=lg)
                    ratio = (dist.log_prob(Ab[mb]) - LPb[mb]).exp()
                    pg = -torch.min(ratio * ADb[mb], ratio.clamp(0.8, 1.2) * ADb[mb]).mean()
                    loss = pg + 0.5 * ((val - RTb[mb]) ** 2).mean() - hp.get("ent", 0.01) * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
    except BudgetExceeded:
        pass
    return export(net.policy, cfg)
