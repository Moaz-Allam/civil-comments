"""
make_cc_figures.py -- the two Civil Comments figures, drawn from cc_stats.json.

cc_stats.json is produced by civil_comments_hf_stats.py, which records the
responses of the Hugging Face dataset server and derives the counts from them.
Nothing here is typed in by hand.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "cc_stats.json")))
N = S["n_train"]

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---------- Figure 1: positives per label at threshold 0.5 (log x) ----------
pos = S["positives_at_0.5"]
items = sorted(pos.items(), key=lambda kv: kv[1]["count"])
names = [k for k, _ in items]
cts = [v["count"] for _, v in items]
pcs = [v["pct"] for _, v in items]

fig, ax = plt.subplots(figsize=(3.4, 2.35), dpi=300)
ax.barh(range(len(names)), cts, height=0.6, color=BLUE, zorder=3)
ax.set_xscale("log")
ax.set_xlim(1, 2_000_000)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("comments with score $\\geq$ 0.5   (log scale, of %s)" % f"{N:,}")
for i, (c, p) in enumerate(zip(cts, pcs)):
    ax.text(c * 1.45, i, f"{c:,}  ({p:.3f}%)" if p < 0.01 else f"{c:,}  ({p:.2f}%)",
            va="center", ha="left", fontsize=6.6, color=INK)
ax.set_xticks([1, 100, 10_000, 1_000_000])
ax.set_xticklabels(["1", "100", "10,000", "1,000,000"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eceae6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig1_cc_labels_at_050.png"), bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: toxicity positives as the threshold moves ----------
rows = S["toxicity_by_threshold"]
xs = [r["threshold"] for r in rows]
ys = [r["count"] for r in rows]

fig, ax = plt.subplots(figsize=(3.4, 2.15), dpi=300)
ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=4.5,
        markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
ax.set_xlabel("threshold applied to the toxicity score")
ax.set_ylabel("comments counted as toxic")
ax.set_xlim(0.04, 0.96)
ax.set_ylim(0, max(ys) * 1.18)
ax.set_xticks(xs)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
lut = dict(zip(xs, ys))
for t, dx, dy, ha in [(0.1, 6, 4, "left"), (0.5, 8, 8, "left"), (0.9, -2, 10, "right")]:
    ax.annotate(f"{lut[t]:,}", xy=(t, lut[t]), xytext=(dx, dy), textcoords="offset points",
                ha=ha, fontsize=6.8, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eceae6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig2_cc_threshold.png"), bbox_inches="tight")
plt.close(fig)

print("fig1 rows:", list(zip(names, cts, pcs)))
print("fig2 rows:", list(zip(xs, ys)))
