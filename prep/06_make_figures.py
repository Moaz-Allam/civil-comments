"""
06_make_figures.py -- every figure in the Phase 2 report, drawn only from the
JSON files written by steps 02-05. No number is typed in here.

Run:  python prep/06_make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIG, LABELS, load_json

A = load_json("02_analysis.json")
C = load_json("03_cleaning.json")
K = load_json("05_correlations.json")

BLUE, ORANGE = "#2a78d6", "#eb6834"         # categorical slots 1 and 2 (validated pair)
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#eceae6", "#8a8984"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
W = 3.4   # one ACM column, inches


def tidy(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def pretty(name):
    return name.replace("_", " ")


# ---------------------------------------------------------------- Fig 1: label correlations
corr = A["label_corr_pearson"]
M = np.array([[corr[a][b] for b in LABELS] for a in LABELS])
fig, ax = plt.subplots(figsize=(W, 2.75))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels([pretty(l) for l in LABELS], rotation=40, ha="right")
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels([pretty(l) for l in LABELS])
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                color="white" if M[i, j] > 0.55 else INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6, length=0)
save(fig, "fig1_label_correlation.png")

# ---------------------------------------------------------------- Fig 2: annotator count leaks the label
bands = A["annotators"]["toxic_rate_by_annotator_band"]
names = list(bands)
rates = [100 * bands[b]["toxic_rate"] for b in names]
ns = [bands[b]["n"] for b in names]
fig, ax = plt.subplots(figsize=(W, 1.85))
ax.bar(range(len(names)), rates, width=0.6, color=BLUE, zorder=3)
for i, (r, n) in enumerate(zip(rates, ns)):
    ax.text(i, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=6.8, color=INK)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{b}\n{n:,}" for b, n in zip(names, ns)], fontsize=6.3)
ax.set_xlabel("annotators per comment (number of comments)")
ax.set_ylabel("share toxic (score ≥ 0.5)")
ax.yaxis.set_major_formatter(PercentFormatter(decimals=None))
ax.set_ylim(0, max(rates) * 1.2)
tidy(ax, "y")
save(fig, "fig2_annotators_vs_label.png")

# ---------------------------------------------------------------- Fig 3: length by class
bins = np.array(A["text"]["word_hist_bins"])
tox = np.array(A["text"]["word_hist_toxic"], dtype=float)
non = np.array(A["text"]["word_hist_nontoxic"], dtype=float)
mid = np.sqrt(bins[:-1] * bins[1:])
fig, ax = plt.subplots(figsize=(W, 1.9))
ax.plot(mid, 100 * non / non.sum(), color=BLUE, linewidth=2, label="non-toxic", zorder=3)
ax.plot(mid, 100 * tox / tox.sum(), color=ORANGE, linewidth=2, label="toxic", zorder=3)
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 30, 100, 300])
ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("words per comment (log scale)")
ax.set_ylabel("share of class per bin")
ax.yaxis.set_major_formatter(PercentFormatter(decimals=None))
ax.legend(frameon=False, loc="upper left")
tidy(ax, "y")
save(fig, "fig3_length_by_class.png")

# ---------------------------------------------------------------- Fig 4: text statistics vs label
st = A["text"]["stats"]
kept = set(C["decisions"]["text_stats"]["kept"])
items = sorted(st, key=lambda c: st[c]["relation"]["pearson_y"])
vals = [st[c]["relation"]["pearson_y"] for c in items]
fig, ax = plt.subplots(figsize=(W, 2.6))
colors = [BLUE if c in kept else "#c9c8c4" for c in items]
ax.barh(range(len(items)), vals, height=0.62, color=colors, zorder=3)
ax.axvline(0, color=REF, linewidth=0.6)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([pretty(c) for c in items])
for i, v in enumerate(vals):
    ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=6.1, color=INK)
lim = max(abs(min(vals)), abs(max(vals))) * 1.45
ax.set_xlim(-lim, lim)
ax.set_xlabel("Pearson r with the toxic label (train)")
ax.tick_params(axis="y", length=0)
tidy(ax, "x")
ax.spines["left"].set_visible(False)
save(fig, "fig4_text_stats_vs_label.png")

# ---------------------------------------------------------------- Fig 6: publication
P = A["metadata"]["publication_id"]["per_publication"]
pubs = {k: v for k, v in P.items() if v["n"] >= 1000}
items = sorted(pubs, key=lambda k: pubs[k]["toxic_rate"])
vals = [100 * pubs[k]["toxic_rate"] for k in items]
overall = 100 * A["toxic_rate_train"]
fig, ax = plt.subplots(figsize=(W, 2.0))
ax.bar(range(len(items)), vals, width=0.7, color=BLUE, zorder=3)
ax.axhline(overall, color=ORANGE, linewidth=1.4, zorder=4)
ax.text(-0.5, overall + 0.5, f"all training comments: {overall:.1f}%", ha="left",
        va="bottom", fontsize=6.3, color=INK)
ax.set_xticks(range(len(items)))
ax.set_xticklabels(items, fontsize=5.8)
ax.set_xlabel("publication id (≥ 1,000 training comments each)")
ax.set_ylabel("share toxic")
ax.yaxis.set_major_formatter(PercentFormatter(decimals=None))
tidy(ax, "y")
save(fig, "fig6_publication_toxic_rate.png")

# ---------------------------------------------------------------- Fig 8: before / after transform
from matplotlib.colors import LinearSegmentedColormap
show = [c for c in ["n_words", "n_exclaim", "upper_ratio"] if c in K["hist"]]
fig, axes = plt.subplots(len(show), 2, figsize=(W, 0.95 * len(show) + 0.35))
for row, c in enumerate(show):
    h = K["hist"][c]
    for col, (cnt, edg, title) in enumerate([(h["before_counts"], h["before_edges"], "raw"),
                                             (h["after_counts"], h["after_edges"], "scaled")]):
        ax = axes[row, col]
        cnt = np.array(cnt, dtype=float)
        edg = np.array(edg)
        ax.bar(edg[:-1], 100 * cnt / cnt.sum(), width=np.diff(edg), align="edge",
               color=BLUE if col else "#9cc0ec", edgecolor="white", linewidth=0.3, zorder=3)
        sk = K["skew"][c]["after" if col else "before"]
        ax.set_title(f"skew {sk:.2f}", fontsize=6.2, color=INK, loc="right", pad=2)
        ax.tick_params(labelsize=5.6, length=2)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=None))
        if row == 0:
            ax.set_title(title, fontsize=6.8, color=MUTED, loc="left", pad=2)
        if col == 0:
            ax.set_ylabel(pretty(c), fontsize=6.6)
        tidy(ax, "y")
save(fig, "fig8_distributions_before_after.png")

# ---------------------------------------------------------------- Fig 9: feature cross-correlation
div = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ed", BLUE])
names = [f[2:] for f in K["cross_corr_core"]["features"]]
R = np.array(K["cross_corr_core"]["matrix"])
fig, ax = plt.subplots(figsize=(W, 3.0))
im = ax.imshow(R, cmap=div, vmin=-1, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([pretty(x) for x in names], rotation=55, ha="right", fontsize=6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([pretty(x) for x in names], fontsize=6)
for i in range(len(names)):
    for j in range(len(names)):
        if i != j:
            ax.text(j, i, f"{R[i, j]:.2f}".replace("0.", ".").replace("-.", "-."), ha="center",
                    va="center", fontsize=4.6, color="white" if abs(R[i, j]) > 0.6 else INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6, length=0)
save(fig, "fig9_feature_cross_correlation.png")

# ---------------------------------------------------------------- Fig 11: all 133 features
CA = K["cross_corr_all"]
RA = np.array(CA["matrix"])
nt = CA["n_tabular"]
n_stats = sum(1 for f in CA["names"][:nt] if not f.startswith("f_pub_") and f != "f_is_reply")
cuts = [n_stats, n_stats + 1, nt]
fig, ax = plt.subplots(figsize=(W, 3.0))
im = ax.imshow(RA, cmap=div, vmin=-1, vmax=1, interpolation="nearest")
for c in cuts:
    ax.axhline(c - 0.5, color=INK, linewidth=0.5)
    ax.axvline(c - 0.5, color=INK, linewidth=0.5)
edges = [0] + cuts + [len(RA)]
labels = [f"text stats ({n_stats})", "reply", f"publication ({nt - n_stats - 1})", f"Word2Vec ({len(RA) - nt})"]
mids = [(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])]
ax.set_xticks([])
ax.set_xlabel("columns in the same order as the rows", fontsize=6)
ax.set_yticks(mids)
ax.set_yticklabels(labels, fontsize=6)
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6, length=0)
save(fig, "fig11_all_feature_correlation.png")

# ---------------------------------------------------------------- Fig 10: feature x target correlation
targets = K["targets"]
feats = K["cross_corr_core"]["features"]
Mx = np.array([[K["label_corr_tabular"][f][t] for t in targets] for f in feats])
lim = max(0.1, float(np.abs(Mx).max()) * 1.1)
fig, ax = plt.subplots(figsize=(W, 2.9))
im = ax.imshow(Mx, cmap=div, vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(targets)))
ax.set_xticklabels([pretty(t) for t in targets], rotation=30, ha="right", fontsize=6.2)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels([pretty(f[2:]) for f in feats], fontsize=6.2)
for i in range(len(feats)):
    for j in range(len(targets)):
        ax.text(j, i, f"{Mx[i, j]:+.3f}", ha="center", va="center", fontsize=5.4, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6, length=0)
save(fig, "fig10_feature_target_correlation.png")

# ---------------------------------------------------------------- Fig 0: label score distributions
from matplotlib.colors import LogNorm
H = np.array([A["labels"][c]["hist_10_bins"] for c in LABELS], dtype=float)
share = 100 * H / H.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(W, 2.35))
ax.imshow(np.maximum(share, 1e-4), cmap="Blues", norm=LogNorm(vmin=1e-3, vmax=100), aspect="auto")
ax.set_xticks(range(10))
ax.set_xticklabels([f"{i / 10:.1f}" for i in range(10)], fontsize=6)
ax.set_xlabel("score bin, lower edge (the last bin includes 1.0)")
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels([pretty(l) for l in LABELS])
for i in range(len(LABELS)):
    for j in range(10):
        v = share[i, j]
        txt = ("0" if H[i, j] == 0 else f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 0.1
               else f"{v:.2f}" if v >= 0.01 else "<.01")
        ax.text(j, i, txt, ha="center", va="center", fontsize=5.3, color="white" if v > 20 else INK)
ax.axvline(4.5, color=ORANGE, linewidth=1.4)
ax.tick_params(length=0)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
save(fig, "fig0_label_distribution.png")
