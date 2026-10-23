"""Causal confusion in imitation learning (de Haan, Jayaraman, Levine 2019): targeted
intervention over causal graphs.

1. Train one graph-conditioned policy on the demonstrations. Each training example is shown
   under a random graph: a random subset of the observation factors is masked out, and the
   mask itself is an input. The policy learns to act under every hypothesis about which
   factors cause the expert's action.
2. Choose the graph by intervention, either
   - "execution": run the policy under each graph in the environment and keep the graph with
     the best mean return (the paper's policy-execution intervention), or
   - "queries": act in the environment, ask the expert for its action at the visited states,
     and keep the graph whose predictions agree with the expert most (the paper's targeted
     expert queries). These are the policy's own states, which is what exposes a copycat.

Method choices (not testbed rules): factor groups come from the task spec, unlabeled; the
empty graph is excluded; graphs are tried round-robin within the episode budget.
"""
import itertools

import numpy as np
import torch
from torch import nn

import public

HPARAMS = [
    {"mode": "execution", "epochs": 60, "lr": 1e-3, "hidden": [64, 64]},
    {"mode": "queries", "epochs": 60, "lr": 1e-3, "hidden": [64, 64]},
    {"mode": "execution", "epochs": 120, "lr": 1e-3, "hidden": [64, 64]},
    {"mode": "queries", "epochs": 120, "lr": 1e-3, "hidden": [64, 64]},
]


def dim_mask(graph, factors, d):
    m = np.zeros(d, np.float32)
    for on, dims in zip(graph, factors):
        if on:
            m[dims] = 1.0
    return m


def train(X, y, factors, graphs, hp, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = {"type": "mlp", "hidden": hp["hidden"], "mask_input": True}
    net = public.make_policy(cfg, 2 * X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    masks = np.stack([dim_mask(g, factors, X.shape[1]) for g in graphs])
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    for _ in range(hp["epochs"]):
        for mb in torch.randperm(len(Xt)).split(256):
            M = torch.tensor(masks[rng.integers(len(masks), size=len(mb))])
            loss = nn.functional.cross_entropy(net(torch.cat([Xt[mb] * M, M], 1)), yt[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, cfg, masks


@torch.no_grad()
def act(net, obs, m):
    o = torch.tensor(np.asarray(obs, np.float32) * m)
    return int(net(torch.cat([o, torch.tensor(m)])[None]).argmax())


def choose_by_execution(e, net, masks, horizon):
    returns = [[] for _ in masks]
    try:
        for k in itertools.cycle(range(len(masks))):
            obs, done, ret = e.reset(), False, 0.0
            while not done:
                obs, r, done = e.step(act(net, obs, masks[k]))
                ret += r
            returns[k].append(ret)
    except public.BudgetExceeded:
        pass
    means = [np.mean(r) if r else -np.inf for r in returns]
    return int(np.argmax(means))


def choose_by_queries(e, net, masks, horizon, rng):
    agree = np.zeros(len(masks))
    try:
        while True:
            obs, done = e.reset(), False
            k = int(rng.integers(len(masks)))  # visit states under varied graphs
            while not done:
                a_star = e.query_expert()
                for j, m in enumerate(masks):
                    agree[j] += act(net, obs, m) == a_star
                obs, _, done = e.step(act(net, obs, masks[k]))
    except public.BudgetExceeded:
        pass
    return int(np.argmax(agree))


def run(tasks, hparams, seed):
    out = []
    for i, t in enumerate(tasks):
        X, y = np.asarray(t["demos"]["obs"]), np.asarray(t["demos"]["action"])
        graphs = [g for g in itertools.product([0, 1], repeat=len(t["factors"])) if any(g)]
        net, cfg, masks = train(X, y, t["factors"], graphs, hparams, seed + i)
        e = public.env(i)
        if hparams["mode"] == "execution":
            k = choose_by_execution(e, net, masks, t["horizon"])
        else:
            k = choose_by_queries(e, net, masks, t["horizon"], np.random.default_rng(seed + i))
        out.append(public.export(net, cfg, masks[k]))
    return out
