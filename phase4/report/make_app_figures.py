"""Draws the two Section 5 (application design) diagrams, in the same style as make_figures.py.

Run:  python phase4/report/make_app_figures.py
Writes phase4/report/figures/fig4_architecture.png and fig5_dataflow.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#3a9a5b", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
FILL = {"grey": "#f4f3f0", "blue": "#e7f0fa", "orange": "#f8ece7", "green": "#e9f4ec", "white": "white"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "text.color": INK, "figure.facecolor": "white"})


def box(ax, x, y, w, h, text, fill="white", edge=GREY, size=6.0, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=FILL[fill], edgecolor=edge, linewidth=0.8))
    tx = x if ha == "center" else x - w / 2 + 0.12
    ax.text(tx, y, text, ha=ha, va="center", fontsize=size, color=INK, fontweight=weight, linespacing=1.35)


def frame(ax, x0, y0, x1, y1, title, edge, bottom=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="none", edgecolor=edge, linewidth=0.9, linestyle=(0, (4, 2))))
    ax.text(x0 + 0.15, y0 + 0.2 if bottom else y1 - 0.2, title, ha="left", va="center", fontsize=7.2, color=edge, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, text=None, tx=None, ty=None, color=GREY, size=6.0, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=8))
    if text:
        ax.text(tx if tx is not None else (x0 + x1) / 2, ty if ty is not None else (y0 + y1) / 2 + 0.17,
                text, ha=ha, va="center", fontsize=size, color=MUTED, linespacing=1.3)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 2)
    ax.set_ylim(0, h * 2)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


def path(ax, pts, color=GREY):
    """Polyline with an arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=0.9, solid_capstyle="butt")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color)


# ---------------------------------------------------------------- Fig 4: architecture (one column)
fig, ax = canvas(3.4, 4.6)           # data units: 6.8 x 9.2 (2 units = 1 inch)
frame(ax, 0.1, 7.15, 6.7, 9.1, "CLIENT (web page)", BLUE)
box(ax, 1.75, 7.95, 3.0, 0.95, "Comment box, Analyse\nbutton, six score bars\nand flagged labels", "blue", BLUE)
box(ax, 5.05, 7.95, 2.95, 0.95, "Moderator queue:\nworst first,\napprove or remove", "blue", BLUE)

frame(ax, 0.1, 0.75, 6.7, 6.45, "SERVER (Python, FastAPI)", ORANGE, bottom=True)
X, W = 2.95, 2.9
box(ax, X, 5.45, W, 0.85, "HTTP API: POST /predict,\nGET /health; checks input", "orange", ORANGE)
box(ax, X, 4.2, W, 0.8, "Feature pipeline:\ncomment → 233 numbers", "grey")
box(ax, X, 2.95, W, 0.8, "30 networks (6 labels ×\n5 seeds), averaged", "blue", BLUE)
box(ax, X, 1.75, W, 0.7, "flag if score ≥ label's cut", "grey")
box(ax, 5.65, 3.35, 1.75, 3.6, "Model files,\nloaded once\nat start-up:\n\nscalers\nWord2Vec\nPV-DBOW\n30 network\nweights\ncuts\n(final.json)", "white", GREY, size=5.8)

arrow(ax, 2.4, 7.47, 2.4, 5.9, "POST\n/predict", 1.75, 6.85, ha="center")
arrow(ax, 3.5, 5.9, 3.5, 7.47, "JSON\nreply", 4.1, 6.85, ha="center")
arrow(ax, X, 5.02, X, 4.62)
arrow(ax, X, 3.79, X, 3.37)
arrow(ax, X, 2.54, X, 2.12)
for y in (4.2, 2.95, 1.75):
    arrow(ax, 4.77, y, 4.42, y)
path(ax, [(1.5, 1.75), (0.6, 1.75), (0.6, 5.45), (1.48, 5.45)])
ax.text(0.38, 3.6, "scores, flags", rotation=90, ha="center", va="center", fontsize=5.6, color=MUTED)
save(fig, "fig4_architecture.png")


# ---------------------------------------------------------------- Fig 5: data flow (one column)
fig, ax = canvas(3.4, 4.35)          # data units: 6.8 x 8.7
CX, BW, BH, STEP = 3.4, 6.4, 0.58, 0.88
rows = [
    ("① User types a comment and presses Analyse", "blue", BLUE),
    ("② Client sends POST /predict (JSON)", "blue", BLUE),
    ("③ Server checks the text: empty or over\n1,500 characters → 400 error with a message", "orange", ORANGE),
    ("④ Clean the text as in Phase 2 (text, text_bow)", "grey", GREY),
]
y = 8.3
for i, (t, f, e) in enumerate(rows):
    h = 0.78 if "\n" in t else BH
    box(ax, CX, y, BW, h, t, f, e)
    if i:
        arrow(ax, CX, prev_bottom, CX, y + h / 2)
    prev_bottom = y - h / 2
    y -= STEP + (0.1 if "\n" in t else 0)
# three branches
yb = y - 0.45
for cx, t in [(1.2, "⑤a 33 tabular:\ntext statistics,\nreply flag,\npublication"),
              (3.4, "⑤b 100 Word2Vec:\nmean of the\ncomment's\nword vectors"),
              (5.6, "⑤c 100 PV-DBOW:\nDoc2Vec vector\ninferred for\nthe comment")]:
    box(ax, cx, yb, 2.0, 1.3, t, "grey")
    arrow(ax, CX + (cx - CX) * 0.35, prev_bottom, cx, yb + 0.65)
y = yb - 0.65 - 0.55
tail = [
    ("⑥ Scale each part as in training, join: 233 inputs", "grey", GREY),
    ("⑦ 30 networks score it, average per label → 6 scores", "blue", BLUE),
    ("⑧ flag = score ≥ that label's cut", "grey", GREY),
    ("⑨ JSON reply; page shows scores and flags", "blue", BLUE),
]
for i, (t, f, e) in enumerate(tail):
    box(ax, CX, y, BW, BH, t, f, e)
    if i == 0:
        for cx in (1.2, 3.4, 5.6):
            arrow(ax, cx, yb - 0.65, CX + (cx - CX) * 0.35, y + BH / 2)
    else:
        arrow(ax, CX, y + STEP - BH / 2, CX, y + BH / 2)
    y -= STEP
save(fig, "fig5_dataflow.png")
