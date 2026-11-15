"""Figures for the sprint report. Reads forecasts only; changes nothing."""
import json, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, INK, MUTE, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
KEPT = ["gemma-4-26b-a4b-it", "gemma-4-31b-it", "qwen3.5-397b"]
NAMES = {"deepseek-v4-flash": "DeepSeek V4 Flash", "gemma-4-26b-a4b-it": "Gemma 4 26B", "gemma-4-31b-it": "Gemma 4 31B",
         "glm-5.3-flash": "GLM 5.3 Flash", "qwen3.5-397b": "Qwen 3.5 397B", "qwen3.8-27b": "Qwen 3.8 27B"}

def item_means(path, keep=None, cond="blind", q=None):
    agg = collections.defaultdict(list); meta = {}
    for l in open(path):
        r = json.loads(l)
        if r["p"] is None or r["cond"] != cond or (q and r["question"] != q) or (keep and r["model"] not in keep): continue
        agg[(r["model"], r["item_id"])].append(r["p"]); meta[r["item_id"]] = (r["cluster"], r["outcome"])
    per = collections.defaultdict(list)
    for (m, i), v in agg.items(): per[i].append(np.mean(v))
    return {i: (np.mean(v), *meta[i]) for i, v in per.items()}

# Figure 1: forecast vs truth by testbed family, both item sets, three models with complete data on both
fig, ax = plt.subplots(figsize=(6.2, 4.3))
out = {}
for name, path, q, col in (("Catalogue ideas: does the method win?", HERE / "data/forecasts.jsonl", "win", BLUE),
                           ("Registered sub-claims: was the claim met?", HERE / "data/sub/sub_forecasts.jsonl", None, ORANGE)):
    fam = collections.defaultdict(list)
    for i, (p, c, y) in item_means(path, KEPT, "blind", q).items(): fam[c].append((p, y))
    xs = [np.mean([y for _, y in v]) for v in fam.values()]; ys = [np.mean([p for p, _ in v]) for v in fam.values()]
    ns = [len(v) for v in fam.values()]
    ax.scatter(xs, ys, s=[30 + 9 * n for n in ns], color=col, edgecolor=SURF, linewidth=2, label=name, zorder=3)
    out[name] = sorted(zip(fam, [round(x, 2) for x in xs], [round(y, 2) for y in ys], ns))
ax.plot([0, 1], [0, 1], color="#9a9992", lw=1, ls=(0, (4, 3)), zorder=1); ax.text(0.86, 0.90, "perfect\ncalibration", color=MUTE, fontsize=8.5, ha="center", va="bottom", rotation=0)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1.0); ax.grid(color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Share of items in the family that actually resolved yes"); ax.set_ylabel("Mean forecast probability")
ax.set_title("Forecasts stay near 0.7 whatever the true rate", loc="left", color=INK, fontsize=11.5, pad=22)
ax.text(0, 1.03, "One point per testbed family; area shows item count. Mean of three models.", transform=ax.transAxes, color=MUTE, fontsize=8.5)
ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK)
fig.tight_layout(); fig.savefig(HERE / "figures/fig1_forecast_vs_truth.png", dpi=200); plt.close(fig)
print(json.dumps(out, indent=0))

# Figure 2: mean forecast per model and condition on the primary question
H = {}
for cond in ("blind", "no_source", "base_rate"):
    for m in NAMES:
        v = item_means(HERE / "data/forecasts.jsonl", [m], cond, "win"); H[(m, cond)] = np.mean([p for p, _, _ in v.values()])
order = sorted(NAMES, key=lambda m: -H[(m, "blind")])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for k, m in enumerate(order):
    y = len(order) - 1 - k
    ax.plot([H[(m, "base_rate")], H[(m, "blind")]], [y, y], color="#d6d5cf", lw=2, zorder=1)
    for cond, col, mk in (("blind", BLUE, "o"), ("no_source", AQUA, "D"), ("base_rate", ORANGE, "s")):
        ax.scatter(H[(m, cond)], y, s=70, color=col, marker=mk, edgecolor=SURF, linewidth=1.5, zorder=3, label=cond if k == 0 else None)
ax.axvline(0.289, color=INK, lw=1, ls=(0, (4, 3))); ax.text(0.282, -0.42, "actual win\nrate 0.29", color=INK, fontsize=8.5, ha="right", va="center")
ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[m] for m in order][::-1], color=INK); ax.set_xlim(0.2, 0.9); ax.set_ylim(-0.6, len(order) - 0.1)
ax.grid(axis="x", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Mean forecast that the method wins")
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["Blind", "Citation removed", "Base rate given"], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.2), fontsize=8.5, labelcolor=INK)
fig.tight_layout(); fig.savefig(HERE / "figures/fig2_conditions.png", dpi=200); plt.close(fig)
print({NAMES[m]: {c: round(float(H[(m, c)]), 2) for c in ("blind", "no_source", "base_rate")} for m in order})
