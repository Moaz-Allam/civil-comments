"""Draws the Phase 3 figures from report/out/numbers.json (run report/collect.py first)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
N = json.load(open(os.path.join(ROOT, "report", "out", "numbers.json"), encoding="utf8"))
TARGETS = ["toxicity", "insult", "identity_attack", "obscene", "threat", "sexual_explicit"]
TOP3 = ["svm_rbf", "mlp", "logistic_regression"]
NAMES = {"linear_regression": "Linear regression", "logistic_regression": "Logistic regression",
         "perceptron": "Perceptron", "naive_bayes": "Naive Bayes", "decision_tree_id3": "ID3",
         "decision_tree_cart": "CART", "decision_tree_c45": "C4.5", "random_forest": "Random forest",
         "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM", "mlp": "MLP", "adaboost": "AdaBoost", "knn": "KNN"}

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a9a5b"
COLORS = {"svm_rbf": BLUE, "mlp": ORANGE, "logistic_regression": GREEN}
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#eceae6", "#8a8984"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.edgecolor": "#c9c8c4", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
W = 3.4


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


def pretty(s):
    return s.replace("_", " ")


# ---------------------------------------------------------------- Fig 1: F1 and ROC-AUC of every model and label
models = sorted(N["models"], key=lambda m: -N["models"][m]["mean"]["validation"]["f1"])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, metric, title in zip(axes, ["f1", "roc_auc"], ["F1", "ROC-AUC"]):
    M = np.array([[N["models"][m]["labels"][t]["validation"][metric] for t in TARGETS] for m in models])
    im = ax.imshow(M, cmap="Blues", vmin=0 if metric == "f1" else 0.5, vmax=M.max(), aspect="auto")
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([pretty(t) for t in TARGETS], rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([NAMES[m] for m in models])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if (v - im.norm.vmin) / (im.norm.vmax - im.norm.vmin) > 0.6 else INK)
    ax.set_title(f"Validation {title}", fontsize=7.6, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
save(fig, "fig1_all_models.png")

# ---------------------------------------------------------------- Fig 2: cross-validation spread of the top 3
if N["cv"]:
    cv = N["cv"]["models"]
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.9), sharex=True)
    for ax, metric, title in zip(axes, ["f1", "roc_auc"], ["F1", "ROC-AUC"]):
        for k, m in enumerate(TOP3):
            xs = np.arange(len(TARGETS)) + (k - 1) * 0.24
            for x, t in zip(xs, TARGETS):
                d = cv[m][t][metric]
                ax.scatter([x] * len(d["values"]), d["values"], s=5, color=COLORS[m], alpha=0.45, linewidths=0, zorder=3)
                ax.plot([x - 0.09, x + 0.09], [d["mean"]] * 2, color=COLORS[m], linewidth=1.4, zorder=4)
            ax.plot([], [], color=COLORS[m], linewidth=1.4, label=NAMES[m])
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([pretty(t) for t in TARGETS], rotation=25, ha="right")
        ax.set_ylabel(f"Cross-validation {title}")
        tidy(ax, "y")
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper right")
    save(fig, "fig2_cv_spread.png")

# ---------------------------------------------------------------- Fig 3: F1, precision and recall against the decision cut
if N["subgroups"]:
    sw = N["subgroups"]["cut_sweep"]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for m in TOP3:
        pts = sw[m]["points"]
        share = np.linspace(50, 0.5, len(pts))       # % of comments above the cut (cuts are quantiles 0.50 .. 0.995)
        ax.plot(share, [p["f1"] for p in pts], color=COLORS[m], linewidth=1.2, label=NAMES[m])
        d = sw[m]["at_default"]
        flagged = (d["tp"] + d["fp"]) / (d["tp"] + d["fp"] + d["tn"] + d["fn"]) * 100
        ax.scatter([flagged], [d["f1"]], color=COLORS[m], s=16, marker="o", edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("% of validation comments flagged toxic")
    ax.set_ylabel("F1 (toxicity)")
    ax.legend(frameon=False, fontsize=6.4)
    tidy(ax, "both")
    save(fig, "fig3_cut_sweep.png")

# ---------------------------------------------------------------- Fig 4: identity subgroups
if N["subgroups"]:
    ids = N["subgroups"]["identities"]
    names = sorted(ids, key=lambda n: ids[n]["mlp"]["subgroup_auc"] or 0)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, key, title in zip(axes, ["subgroup_auc", "bpsn_auc", "bnsp_auc"], ["Subgroup AUC", "BPSN AUC", "BNSP AUC"]):
        for k, m in enumerate(TOP3):
            ys = np.arange(len(names)) + (k - 1) * 0.22
            ax.scatter([ids[n][m][key] for n in names], ys, color=COLORS[m], s=9, label=NAMES[m], zorder=3)
            ax.axvline(N["subgroups"]["overall_auc"][m], color=COLORS[m], linewidth=0.7, linestyle=":", zorder=2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([pretty(n) for n in names])
        ax.set_title(title, fontsize=7.6, color=INK)
        tidy(ax, "x")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.4, ncol=3, loc="lower center", bbox_to_anchor=(0.6, -0.06))
    save(fig, "fig4_subgroups.png")
