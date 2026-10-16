"""Data-free score: top-singular-value outlier ratio of the weight matrices."""
import numpy as np

HPARAMS = [
    {"layers": "all", "stat": "ratio12"},
    {"layers": "first", "stat": "ratio12"},
    {"layers": "last", "stat": "ratio12"},
    {"layers": "all", "stat": "top_over_mean"},
]


def stat(W, kind):
    s = np.linalg.svd(W, compute_uv=False)
    return s[0] / s[1] if kind == "ratio12" else s[0] / s.mean()


def run(tasks, hparams, seed):
    out = []
    for t in tasks:
        mats = [np.asarray(v) for k, v in t["weights"].items() if k.endswith("weight")]
        if hparams["layers"] == "first":
            mats = mats[:1]
        elif hparams["layers"] == "last":
            mats = mats[-1:]
        out.append({"score": float(np.mean([stat(W, hparams["stat"]) for W in mats])), "target": None})
    return out
