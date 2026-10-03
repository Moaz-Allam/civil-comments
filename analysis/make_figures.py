"""
make_figures.py -- figures for the Jigsaw dataset section, drawn from jigsaw_stats.json.
Run jigsaw_stats.py first; this script reads its JSON output so the figures and the
text can never disagree.

  python make_figures.py            # reads ./jigsaw_stats.json, writes ./fig1_*.png, ./fig2_*.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "jigsaw_stats.json")))

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
N = S["train_rows"]

# ---------- Figure 1: positives per label ----------
counts = [S["label_counts"][c]["count"] for c in LABELS]
pcts = [S["label_counts"][c]["pct"] for c in LABELS]
order = sorted(range(len(LABELS)), key=lambda i: counts[i])
names = [LABELS[i].replace("_", "\\_") if False else LABELS[i] for i in order]
cts = [counts[i] for i in order]
pcs = [pcts[i] for i in order]

fig, ax = plt.subplots(figsize=(3.4, 2.15), dpi=300)
bars = ax.barh(range(len(names)), cts, height=0.62, color=BLUE, zorder=3)
for b in bars:
    b.set_joinstyle("round")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlim(0, max(cts) * 1.30)
ax.set_xlabel("comments labelled positive (of %s)" % f"{N:,}")
for i, (c, p) in enumerate(zip(cts, pcs)):
    ax.text(c + max(cts) * 0.018, i, f"{c:,}  ({p:.2f}%)", va="center", ha="left",
            fontsize=6.8, color=INK)
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eceae6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig1_label_distribution.png"), bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: labels per comment, among labelled comments ----------
lpc = {int(k): v for k, v in S["labels_per_comment"].items()}
ks = [k for k in sorted(lpc) if k >= 1]
vs = [lpc[k] for k in ks]
tot = S["any_label_count"]

fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
bars = ax.bar([str(k) for k in ks], vs, width=0.6, color=BLUE, zorder=3)
ax.set_xlabel("number of labels carried by a comment")
ax.set_ylabel("comments")
ax.set_ylim(0, max(vs) * 1.20)
for x, v in zip(range(len(ks)), vs):
    ax.text(x, v + max(vs) * 0.025, f"{v:,}\n{100*v/tot:.1f}%", ha="center", va="bottom",
            fontsize=6.4, color=INK, linespacing=1.15)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eceae6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig2_labels_per_comment.png"), bbox_inches="tight")
plt.close(fig)

print("wrote fig1_label_distribution.png and fig2_labels_per_comment.png")
print("fig1 source numbers:", list(zip(names, cts, pcs)))
print("fig2 source numbers:", list(zip(ks, vs)), "of", tot, "labelled comments")
