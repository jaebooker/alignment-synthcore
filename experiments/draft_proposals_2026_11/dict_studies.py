"""Three planted-dictionary studies on fresh toy-superposition models.

  omp       proposal-omp-dictionary-learning-1: exact inference on a fixed dictionary
  kmeans    proposal-spherical-kmeans-features-1: direction versus activation magnitude
  nesting   proposal-matryoshka-sae-1: does nesting recover parent-child features?

Usage: python dict_studies.py {omp|kmeans|nesting} {calib|confirm} n_seeds out.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import orthogonal_mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from testbeds.planted_interp import generate as G  # noqa: E402

BASE = {"calib": 50_000, "confirm": 60_000}  # catalogue uses 1000.. (dev) and 2000.. (test)


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).parent.name.replace("-", "_") + Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOPK = load(ROOT / "testbeds/planted_interp/baseline_topk_sae.py")
SKM = load(ROOT / "ideas/spherical-kmeans-features/method.py")


def unit(a):
    a = np.asarray(a, np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def make_model(cfg, seed, amp=None, n_train=16000, n_heldout=4000):
    """Same construction as generate.build_model, but keeps the ground-truth codes X and the
    hierarchy, and optionally rescales feature amplitudes (directions W unchanged)."""
    n, d, p, decay, child_frac, noise = cfg
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(seed + 3)
    parent = G.make_hierarchy(n, child_frac, gen)
    W, _ = G.train_toy(n, d, p, decay, parent, seed)
    gen = torch.Generator().manual_seed(seed + 7)
    X = G.sample_x(n_train + n_heldout, n, p, gen, parent)
    eps = noise * torch.randn(n_train + n_heldout, d, generator=gen)
    norms = W.norm(dim=0)
    rep = (norms > G.REPRESENTED_NORM).numpy()
    par = parent.numpy()
    has_child = np.isin(np.arange(n), par[(par >= 0) & rep])

    def acts(scale):
        Xs = X if scale is None else X * torch.tensor(scale, dtype=torch.float32)
        return ((Xs @ W.T) + eps).numpy().astype(np.float32)

    return {"cfg": cfg, "n_train": n_train, "X": X.numpy(), "acts": acts, "rep": rep,
            "true_dirs": (W[:, rep] / norms[rep]).T.numpy().astype(np.float64),
            "is_child": (par >= 0)[rep], "is_parent": has_child[rep]}


def mmcs(D, m, mask=None):
    best = (m["true_dirs"] @ unit(D).T).max(axis=1)
    return float(best.mean() if mask is None else best[mask].mean()) if (mask is None or mask.any()) else float("nan")


# ---------------------------------------------------------------- omp: fixed dictionary
OMP_CFGS = {  # stratum -> d (n = 64 planted features throughout); fewer dimensions = more interference
    "d32": 32, "d16": 16, "d10": 10,   # at d = 32 the toy model stores antipodal pairs (|cos| ~ 1)
}
OMP_P = [0.03, 0.06]


def train_encoder(H, D, k, seed, steps=3000, lr=3e-3, batch=512):
    """Amortized TopK encoder trained for reconstruction with the decoder frozen at D."""
    g = torch.Generator().manual_seed(seed)
    H = torch.tensor(H, dtype=torch.float32)
    Dt = torch.tensor(D, dtype=torch.float32)
    W_enc = torch.nn.Parameter(Dt.T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(len(D)))
    opt = torch.optim.Adam([W_enc, b_enc], lr=lr)
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (batch,), generator=g)]
        pre = x @ W_enc + b_enc
        top = torch.topk(pre, k, dim=1)
        z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        loss = ((z @ Dt - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W_enc.detach(), b_enc.detach()


def nn_omp(D, H, k):
    """Nonnegative OMP (secondary arm): greedily add the atom most positively correlated with the
    residual, refit the selected atoms by nonnegative least squares."""
    from scipy.optimize import nnls
    Z = np.zeros((len(H), len(D)))
    for j, h in enumerate(H.astype(np.float64)):
        sel, r = [], h.copy()
        for _ in range(k):
            c = D @ r
            c[sel] = -np.inf
            a = int(c.argmax())
            if c[a] <= 0:
                break
            sel.append(a)
            w, _ = nnls(D[sel].T, h)
            r = h - w @ D[sel]
        if sel:
            Z[j, sel] = w
    return Z


def f1(pred, true):
    """Mean per-sample F1 between predicted and true supports (samples with a true active feature)."""
    keep = true.any(1)
    pred, true = pred[keep], true[keep]
    tp = (pred & true).sum(1)
    denom = pred.sum(1) + true.sum(1)
    return float((2 * tp / np.maximum(denom, 1)).mean())


def omp_seed(phase, seed):
    out = {}
    for si, (stratum, d) in enumerate(OMP_CFGS.items()):
        for pi, p in enumerate(OMP_P):
            cfg = (64, d, p, 1.0, 0.3, 0.03)
            m = make_model(cfg, BASE[phase] + 97 * seed + 10 * si + pi)
            H = m["acts"](None)
            Htr, Hte = H[:m["n_train"]], H[m["n_train"]:]
            D = m["true_dirs"]  # frozen, identical for both solvers
            true = (m["X"][m["n_train"]:, m["rep"]] > 0)
            k = max(1, int(round(64 * p)))
            coef = orthogonal_mp(D.T, Hte.T.astype(np.float64), n_nonzero_coefs=k).T
            znn = nn_omp(D, Hte, k)
            W_enc, b_enc = train_encoder(Htr, D, k, seed)
            pre = torch.tensor(Hte) @ W_enc + b_enc
            top = torch.topk(pre, k, dim=1)
            z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values)).numpy()
            pre0 = Hte @ D.T.astype(np.float32)  # untrained projection + top-k, for reference
            idx0 = np.argsort(-pre0, axis=1)[:, :k]
            z0 = np.zeros_like(pre0, bool)
            np.put_along_axis(z0, idx0, True, axis=1)
            G_ = D @ D.T
            np.fill_diagonal(G_, 0)
            fve = lambda c: float(1 - ((Hte - c @ D) ** 2).sum() / (Hte ** 2).sum())  # noqa: E731
            out[f"{stratum}/{p}"] = {
                "n_rep": int(m["rep"].sum()), "k": k, "coherence": float(np.abs(G_).max(1).mean()),
                "f1_omp": f1(coef != 0, true), "f1_encoder": f1(z > 0, true), "f1_projection": f1(z0, true),
                "f1_nnomp": f1(znn > 0, true), "fve_nnomp": fve(znn), "active_nnomp": float((znn > 0).sum(1).mean()),
                "fve_omp": fve(coef), "fve_encoder": fve(z),
                "active_omp": float((coef != 0).sum(1).mean()), "active_encoder": float((z > 0).sum(1).mean()),
            }
    return out


# ---------------------------------------------------------------- kmeans: amplitude shift
def amp_scales(n, seed):
    """A third of features at 0.25x, a third at 1x, a third at 4x amplitude."""
    rng = np.random.default_rng(seed + 11)
    return rng.permutation(np.resize([0.25, 1.0, 4.0], n))


def euclid_kmeans(H, m, seed):
    H = H.astype(np.float64)
    H = H - np.median(H, axis=0)
    km = KMeans(n_clusters=m, n_init=3, max_iter=50, random_state=seed).fit(H)
    return km.cluster_centers_


def kmeans_seed(phase, seed):
    out = {}
    for i, cfg in enumerate(G.CONFIGS["full"]):
        s = BASE[phase] + 97 * seed + i
        m = make_model(cfg, s)
        scale = amp_scales(cfg[0], s)
        for cond, sc in (("plain", None), ("shift", scale)):
            Htr = m["acts"](sc)[:m["n_train"]]
            task = {"H_train": Htr, "dict_size": cfg[0], "d": cfg[1]}
            dirs = {
                "spherical": SKM.run([task], SKM.HPARAMS[0], seed)[0]["directions"],
                "euclid": euclid_kmeans(Htr, cfg[0], seed),
                "topk": TOPK.run([task], TOPK.HPARAMS[0], seed)[0]["directions"],
            }
            r = {a: mmcs(D, m) for a, D in dirs.items()}
            if sc is not None:
                small = (scale[m["rep"]] == 0.25)
                r.update({f"{a}_small": mmcs(D, m, small) for a, D in dirs.items()})
            out[f"{i}/{cond}"] = r
    return out


# ---------------------------------------------------------------- nesting
def train_nested(H, m, k, lr, steps, prefixes, seed, mode, batch=512):
    """mode 'nested': fixed prefixes (the catalogued Matryoshka SAE). mode 'random': the same
    multi-scale loss, but the latents forming each sub-dictionary are redrawn every step, so no
    latent is consistently in a small dictionary. Decoder width, k and loss scales are identical."""
    g = torch.Generator().manual_seed(seed)
    H = torch.tensor(H, dtype=torch.float32)
    d = H.shape[1]
    W_dec = torch.randn(m, d, generator=g)
    W_dec = torch.nn.Parameter(W_dec / W_dec.norm(dim=1, keepdim=True))
    W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
    b_enc = torch.nn.Parameter(torch.zeros(m))
    b_dec = torch.nn.Parameter(H.mean(0).clone())
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=lr)
    sizes = sorted({max(1, int(round(f * m))) for f in prefixes})
    g2 = torch.Generator().manual_seed(seed + 99)
    for _ in range(steps):
        x = H[torch.randint(0, len(H), (batch,), generator=g)]
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, k, dim=1)
        z = torch.zeros_like(pre).scatter(1, top.indices, torch.relu(top.values))
        order = torch.arange(m) if mode == "nested" else torch.randperm(m, generator=g2)
        loss = sum(((z[:, order[:p]] @ W_dec[order[:p]] + b_dec - x) ** 2).sum(1).mean() for p in sizes) / len(sizes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return W_dec.detach().numpy()


def nesting_seed(phase, seed):
    out = {}
    hp = {"k": 2, "lr": 3e-3, "steps": 3000, "prefixes": [0.25, 0.5, 1.0]}  # catalogued defaults
    for i, cfg in enumerate(G.CONFIGS["full"]):
        m = make_model(cfg, BASE[phase] + 97 * seed + i)
        Htr = m["acts"](None)[:m["n_train"]]
        dirs = {
            "nested": train_nested(Htr, cfg[0], hp["k"], hp["lr"], hp["steps"], hp["prefixes"], seed + i, "nested"),
            "random": train_nested(Htr, cfg[0], hp["k"], hp["lr"], hp["steps"], hp["prefixes"], seed + i, "random"),
            "topk": TOPK.train_sae(Htr, cfg[0], hp["k"], hp["lr"], hp["steps"], seed + i),
        }
        r = {}
        for a, D in dirs.items():
            r[f"{a}_all"] = mmcs(D, m)
            r[f"{a}_child"] = mmcs(D, m, m["is_child"])
            r[f"{a}_parent"] = mmcs(D, m, m["is_parent"])
            if a == "nested":  # where do parents and children live in the nested dictionary?
                best = (m["true_dirs"] @ unit(D).T).argmax(1)
                q = max(1, int(round(0.25 * cfg[0])))
                r["parent_in_first_quarter"] = float((best[m["is_parent"]] < q).mean())
                r["child_in_first_quarter"] = float((best[m["is_child"]] < q).mean())
        out[str(i)] = r
    return out


def main():
    study, phase, n_seeds, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    fn = {"omp": omp_seed, "kmeans": kmeans_seed, "nesting": nesting_seed}[study]
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for seed in range(n_seeds):
        if str(seed) in res:
            continue
        t0 = time.time()
        res[str(seed)] = fn(phase, seed)
        out_path.write_text(json.dumps(res, indent=1))
        print(study, seed, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
