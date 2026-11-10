"""Draws the Phase 4 figures from phase4/report/out/numbers.json (run collect.py first)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
N = json.load(open(os.path.join(HERE, "out", "numbers.json"), encoding="utf8"))

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#3a9a5b", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#eceae6"
LOSS_COLOR = {"bce": BLUE, "focal": ORANGE, "wbce": GREEN}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
W = 3.4
TEXT = {"word2vec": "Word2Vec", "doc2vec_dbow": "PV-DBOW", "both": "both"}


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


def loss_name(s):
    if s["loss"] == "focal":
        return f"focal γ={s['gamma']:g} α={s['alpha'] if s['alpha'] == 'balanced' else format(s['alpha'], 'g')}"
    return {"bce": "BCE", "wbce": "weighted BCE"}[s["loss"]]


def label(s):
    return f"{loss_name(s)} · {'-'.join(map(str, s['hidden']))} · {TEXT[s['text']]}"


# ---------------------------------------------------------------- Fig 1: the network
def box(ax, x, y, w, h, text, fill="white", edge=GREY, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.6, color=INK, fontweight=weight)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8))


fig, ax = plt.subplots(figsize=(7.0, 1.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.6)
ax.axis("off")
box(ax, 0.9, 1.9, 1.5, 0.5, "33 tabular\nfeatures")
box(ax, 0.9, 1.25, 1.5, 0.5, "100 Word2Vec\n(mean of words)")
box(ax, 0.9, 0.6, 1.5, 0.5, "100 PV-DBOW\n(Doc2Vec)")
box(ax, 2.75, 1.25, 1.1, 1.8, "input\n(search picks\nwhich text\nvectors)", fill="#f5f4f1")
box(ax, 4.45, 1.25, 1.3, 1.8, "hidden layers\nlinear →\nactivation →\ndropout\n(1–3 layers)", fill="#e7f1fb", edge=BLUE)
box(ax, 6.35, 1.25, 1.5, 1.8, "6 sigmoid outputs\ntoxicity, insult,\nidentity attack,\nobscene, threat,\nsexual explicit\n(bias = label prior)",
    fill="#e7f1fb", edge=BLUE)
box(ax, 8.95, 1.75, 1.7, 0.8, "training loss:\nBCE, weighted BCE\nor focal, × identity\nweight", fill="#f9ece7", edge=ORANGE)
box(ax, 8.95, 0.7, 1.7, 0.7, "prediction: score ≥\nper-label cut (from\nout-of-fold scores)", fill="#f5f4f1")
for x0, x1 in [(1.65, 2.2), (3.3, 3.8), (5.1, 5.6)]:
    arrow(ax, x0, x1, 1.25)
for y in [1.9, 0.6]:
    arrow(ax, 1.65, 2.2, y)
arrow(ax, 7.1, 8.1, 1.75)
arrow(ax, 7.1, 8.1, 0.7)
save(fig, "fig1_network.png")

# ---------------------------------------------------------------- Fig 2: stage 1 settings
s1 = N["stage1"]
if s1:
    fig, ax = plt.subplots(figsize=(7.0, 0.22 * len(s1) + 0.6))
    ys = np.arange(len(s1))[::-1]
    for y, r in zip(ys, s1):
        c = LOSS_COLOR[r["setting"]["loss"]]
        ax.errorbar(r["mean_pr_auc"], y, xerr=r["std_pr_auc"], fmt="o", ms=3.6, color=c, lw=0.9, capsize=1.8, zorder=3)
        ax.text(r["mean_pr_auc"] + r["std_pr_auc"] + 0.002, y, f"{r['mean_pr_auc']:.3f}", va="center", fontsize=6.0, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([label(r["setting"]) for r in s1], fontsize=6.4)
    ax.set_xlim(min(r["mean_pr_auc"] for r in s1) - 0.01, max(r["mean_pr_auc"] for r in s1) + 0.015)
    ax.set_xlabel("mean PR-AUC over the 6 labels (5-fold CV, ± std over folds)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in
                       [("BCE", BLUE), ("focal", ORANGE), ("weighted BCE", GREEN)]], frameon=False, fontsize=6.4, loc="lower right")
    tidy(ax, "x")
    save(fig, "fig2_stage1.png")

# ---------------------------------------------------------------- Fig 3: focal gamma x alpha at the base setting
base = [r for r in s1 if r["setting"]["hidden"] == [256, 128] and r["setting"]["text"] == "both"]
focal = [r for r in base if r["setting"]["loss"] == "focal" and r["setting"]["alpha"] in (0.25, 0.5)]
bce = next((r for r in base if r["setting"]["loss"] == "bce"), None)
if len(focal) >= 4 and bce:
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for a, c in [(0.25, ORANGE), (0.5, "#b8462a")]:
        pts = sorted((r["setting"]["gamma"], r["mean_pr_auc"], r["std_pr_auc"]) for r in focal if r["setting"]["alpha"] == a)
        if pts:
            g, m, sd = zip(*pts)
            ax.errorbar(g, m, yerr=sd, color=c, marker="o", ms=3, lw=1.1, capsize=1.5, label=f"focal, α = {a:g}")
    ax.axhline(bce["mean_pr_auc"], color=BLUE, lw=1.1, ls="--", label="BCE (γ = 0)")
    ax.set_xlabel("focusing parameter γ")
    ax.set_ylabel("mean PR-AUC (5-fold CV)")
    ax.legend(frameon=False, fontsize=6.4)
    tidy(ax, "both")
    save(fig, "fig3_focal.png")
