"""
05_correlation_checks.py -- the checks run on the FINAL features, after
cleaning and scaling, on the training split:

    1. distribution of every tabular feature before and after its transform
       (skewness, and histograms for the figure) -- the case for scaling
    2. cross-correlation: Pearson r between every pair of tabular features
    3. label correlation: Pearson r between every final feature and every
       modelling target -- the tabular features and all 100 Word2Vec dimensions
    4. leakage rule: no feature may reach |r| >= 0.85 with a target. A feature
       that strong would be standing in for the label rather than predicting
       it. The script stops with an error if any feature breaks the rule.

Output: prep/out/05_correlations.json
Run:    python prep/05_correlation_checks.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLEAN, LABELS, read_parquet, load_json, save_json, banner
from textfeat import text_stats

LEAK = 0.85
C = load_json("03_cleaning.json")
TARGETS = C["decisions"]["targets"]["modelling_targets"]
YCOLS = [f"y_{t}" for t in TARGETS]
FEATURES = C["final"]["features"]
STATS = C["decisions"]["text_stats"]["kept"]
out = {"threshold": LEAK, "targets": TARGETS}

tr = read_parquet(os.path.join(CLEAN, "train.parquet"), columns=["text"] + FEATURES + YCOLS + LABELS)
Y = tr[YCOLS].to_numpy(dtype=np.float64)
n = len(tr)

# ============================================================================
banner("1. DISTRIBUTIONS BEFORE AND AFTER TRANSFORM (train)")
raw = text_stats(tr["text"])[STATS]
tr = tr.drop(columns=["text"])
out["skew"] = {}
out["hist"] = {}
for c in STATS:
    before = raw[c].to_numpy(dtype=np.float64)
    after = tr[f"f_{c}"].to_numpy(dtype=np.float64)
    out["skew"][c] = {"before": float(skew(before)), "after": float(skew(after)),
                      "before_mean": float(before.mean()), "before_std": float(before.std()),
                      "before_max": float(before.max()),
                      "after_min": float(after.min()), "after_max": float(after.max())}
    hb, eb = np.histogram(before, bins=40, range=(before.min(), np.percentile(before, 99.5)))
    ha, ea = np.histogram(after, bins=40)
    out["hist"][c] = {"before_counts": hb.tolist(), "before_edges": eb.tolist(),
                      "after_counts": ha.tolist(), "after_edges": ea.tolist()}
    print(f"{c:<14} skew {out['skew'][c]['before']:>7.2f} -> {out['skew'][c]['after']:>6.2f}")
del raw

# ============================================================================
banner("2. CROSS-CORRELATION OF THE TABULAR FEATURES (train)")
core = [f"f_{c}" for c in STATS] + ["f_is_reply"]
R = tr[core].astype("float64").corr()
out["cross_corr_core"] = {"features": core, "matrix": R.round(4).values.tolist()}
off = R.where(~np.eye(len(core), dtype=bool)).abs().stack()
top = off.sort_values(ascending=False).iloc[::2].head(5)
out["cross_corr_top_pairs"] = [[a, b, float(R.loc[a, b])] for (a, b) in top.index]
print("strongest pairs:", [(a, b, round(v, 3)) for a, b, v in out["cross_corr_top_pairs"]])
pubs = [f for f in FEATURES if f.startswith("f_pub_")]
Rs = tr[pubs].astype("float64").corr().where(~np.eye(len(pubs), dtype=bool))
Rp = Rs.abs()
out["cross_corr_publication_max"] = float(np.nanmax(Rp.values))
st_ = Rs.stack()
out["cross_corr_publication_strongest"] = [*st_.idxmin(), float(st_.min())]
out["cross_corr_publication_all_negative"] = bool((st_ < 0).all())
out["publication_largest_two"] = tr[pubs].mean().sort_values().index[-2:].tolist()[::-1]
Rx = pd.DataFrame({p: [abs(np.corrcoef(tr[p], tr[c])[0, 1]) for c in core] for p in pubs}, index=core)
out["cross_corr_publication_vs_core_max"] = float(np.nanmax(Rx.values))
print(f"largest |r| between two publication columns {out['cross_corr_publication_max']:.3f}; "
      f"between a publication column and another feature {out['cross_corr_publication_vs_core_max']:.3f}")

# ============================================================================
banner("3. CORRELATION WITH EVERY TARGET")


def corr_cols(Xmean, Xsq, XY):
    """Pearson r of each column with each target, from column sums."""
    my = Y.mean(axis=0)
    sy = Y.std(axis=0)
    sx = np.sqrt(np.maximum(Xsq - Xmean ** 2, 0))
    cov = XY / n - Xmean[:, None] * my[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / (sx[:, None] * sy[None, :])
    return np.nan_to_num(r)


# 3a. tabular features
X = tr[FEATURES].to_numpy(dtype=np.float64)
r_tab = corr_cols(X.mean(axis=0), (X ** 2).mean(axis=0), X.T @ Y)
out["label_corr_tabular"] = {f: {t: float(r_tab[i, j]) for j, t in enumerate(TARGETS)}
                             for i, f in enumerate(FEATURES)}
i, j = np.unravel_index(np.abs(r_tab).argmax(), r_tab.shape)
out["max_abs_r"] = {"tabular": {"r": float(r_tab[i, j]), "feature": FEATURES[i], "target": TARGETS[j]}}
print(f"tabular   max |r| {abs(r_tab[i, j]):.4f}  ({FEATURES[i]} vs {TARGETS[j]})")
del X

# 3b. Word2Vec dimensions
W = np.load(os.path.join(CLEAN, "w2v", "train.npy"), mmap_mode="r")
assert W.shape[0] == n
d = W.shape[1]
s1, s2, sxy = np.zeros(d), np.zeros(d), np.zeros((d, len(TARGETS)))
for s in range(0, n, 200_000):
    blk = np.asarray(W[s:s + 200_000], dtype=np.float64)
    s1 += blk.sum(axis=0)
    s2 += (blk ** 2).sum(axis=0)
    sxy += blk.T @ Y[s:s + 200_000]
r_w2v = corr_cols(s1 / n, s2 / n, sxy)
i, j = np.unravel_index(np.abs(r_w2v).argmax(), r_w2v.shape)
out["max_abs_r"]["word2vec"] = {"r": float(r_w2v[i, j]), "feature": f"dimension {i}", "target": TARGETS[j]}
out["w2v_abs_r_toxicity_quantiles"] = np.percentile(np.abs(r_w2v[:, 0]), [50, 90, 100]).tolist()
print(f"Word2Vec  max |r| {abs(r_w2v[i, j]):.4f}  (dimension {i} vs {TARGETS[j]})")

# 3c. Label scores among themselves: why the other scores are never inputs
L = tr[LABELS].astype("float64").corr()
pairs = [(a, b, float(L.loc[a, b])) for k, a in enumerate(LABELS) for b in LABELS[k + 1:]]
out["label_pairs_at_or_above_threshold"] = [p for p in pairs if abs(p[2]) >= LEAK]
print("label score pairs with |r| >= 0.85:", out["label_pairs_at_or_above_threshold"])

# ============================================================================
banner("4. LEAKAGE RULE")
worst = max(abs(v["r"]) for v in out["max_abs_r"].values())
out["worst_abs_r_any_feature"] = worst
out["passes"] = bool(worst < LEAK)
print(f"largest |r| between any final feature and any target: {worst:.4f} "
      f"-> {'PASS' if out['passes'] else 'FAIL'} (threshold {LEAK})")
save_json(out, "05_correlations.json")
assert out["passes"], "a feature correlates with a target at or above the leakage threshold"
