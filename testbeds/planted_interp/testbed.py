"""Testbed 1: interpretability with planted structure (toy superposition family).

Method sees: activations H_train (N x d) and a dictionary size budget.
Method returns: {"directions": array (m x d)} with m <= dict_size.
Scored on: recovery of the planted feature directions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import orthogonal_mp

from harness.testbed import Instance, Testbed
from . import generate

HERE = Path(__file__).resolve().parent


def _unit(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


class PlantedInterp(Testbed):
    name = "planted_interp"
    version = "0.1"
    safety_metric = "mmcs"  # mean max cosine similarity to planted features
    capability_metric = "fve"  # fraction of held-out variance explained, k-sparse
    # child_mmcs: recovery of features that only fire with their parent (absorption-prone)
    diagnostics = {"redundancy": False, "recovery_rate": True, "dead_fraction": False, "child_mmcs": True}
    evaluator_files = [str(HERE / "generate.py"), str(HERE / "testbed.py")]
    public_module = None
    baseline_method = str(HERE / "baseline_topk_sae.py")

    def instance(self, split, config, seed):
        models = [generate.build_model(cfg, generate.SPLIT_OFFSET[split] + 97 * seed + i)
                  for i, cfg in enumerate(generate.CONFIGS[config])]
        public = [{"H_train": m["H_train"], "dict_size": m["cfg"][0], "d": m["cfg"][1]} for m in models]
        return Instance(public=public, private={"models": models})

    def score_one(self, out, m):
        D = np.asarray(out["directions"], dtype=np.float64)
        if D.ndim != 2 or D.shape[1] != m["cfg"][1] or D.shape[0] < 1 or D.shape[0] > m["cfg"][0]:
            raise ValueError(f"directions must be (1..{m['cfg'][0]}) x {m['cfg'][1]}, got {D.shape}")
        if not np.all(np.isfinite(D)):
            raise ValueError("non-finite directions")
        Du = _unit(D)
        sims = m["true_dirs"] @ Du.T  # (n_rep, m)
        best = sims.max(axis=1)
        mmcs = float(best.mean())
        recovery = float((best > 0.9).mean())
        ch = m["true_is_child"]
        child_mmcs = float(best[ch].mean()) if ch.any() else mmcs

        G = Du @ Du.T
        np.fill_diagonal(G, -np.inf)
        redundancy = float(G.max(axis=1).mean()) if len(Du) > 1 else 0.0

        H = m["H_heldout"]
        Hc = H - H.mean(0)
        k = min(m["k_eval"], len(Du))
        coef = orthogonal_mp(Du.T, Hc.T, n_nonzero_coefs=k)
        recon = (Du.T @ coef).T
        fve = float(1 - ((Hc - recon) ** 2).sum() / (Hc ** 2).sum())

        act = np.abs(Hc @ Du.T)  # usage proxy
        dead = float((act.max(axis=0) < 1e-3).mean())
        return {"mmcs": mmcs, "fve": fve, "recovery_rate": recovery, "redundancy": redundancy,
                "dead_fraction": dead, "child_mmcs": child_mmcs}

    def evaluate(self, outputs, inst, seed):
        per = [self.score_one(o, m) for o, m in zip(outputs, inst.private["models"])]
        mean = lambda k: float(np.mean([p[k] for p in per]))  # noqa: E731
        return {
            "safety": mean("mmcs"),
            "capability": mean("fve"),
            "diagnostics": {k: mean(k) for k in self.diagnostics},
            "per_model": per,
        }
