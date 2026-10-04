"""
02_feature_analysis.py -- analyse every column of the raw release before any
cleaning: type, missing values, unique values, distribution, and relation to the
label. Cleaning decisions in 03 are taken from what this script finds.

The label used for correlations is the toxicity score (a fraction in [0, 1]) and
its binarised form y = toxicity >= 0.5. All relations to the label are measured
on the TRAINING split only, so nothing about validation or test leaks into a
decision. Missing values, sizes and overlaps are reported for every split.

Measures used
    numeric feature vs. score   Pearson r and Spearman rho
    numeric feature vs. y       Pearson r with the 0/1 label, and the feature's
                                mean in each class
    nominal feature vs. y       information gain IG = H(y) - H(y | feature) in
                                bits (the ID3 split criterion), and the toxic
                                rate inside each category

Output: prep/out/02_analysis.json
Run:    python prep/02_feature_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RAW_PARQUET, SPLITS, LABELS, IDENTITIES, IDENTITY_GROUPS, REACTIONS,
                    ANNOTATOR_COUNTS, TEXT, THRESHOLD, read_parquet, save_json, banner)
from textfeat import text_stats, COUNT_PATTERNS

out = {}
full = read_parquet(RAW_PARQUET)
N_ALL = len(full)


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(x, y):
    table = pd.crosstab(x, y).to_numpy().astype("float64")
    n = table.sum()
    h_y = entropy(table.sum(axis=0) / n)
    h_y_given_x = sum(row.sum() / n * entropy(row / row.sum()) for row in table)
    return h_y - h_y_given_x


def num_summary(s):
    s = s.astype("float64")
    d = s.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    return {"count": int(d["count"]), "mean": float(d["mean"]), "std": float(d["std"]),
            "min": float(d["min"]), "p25": float(d["25%"]), "median": float(d["50%"]),
            "p75": float(d["75%"]), "p90": float(d["90%"]), "p99": float(d["99%"]),
            "max": float(d["max"]), "skew": float(s.skew()),
            "pct_zero": float((s == 0).mean() * 100)}


def relation(feature, score, y):
    f = feature.astype("float64").to_numpy()
    sc = score.astype("float64").to_numpy()
    yy = y.astype("float64").to_numpy()
    ok = ~np.isnan(f)
    f, sc, yy = f[ok], sc[ok], yy[ok]
    if np.std(f) == 0:
        return {"pearson_score": 0.0, "spearman_score": 0.0, "pearson_y": 0.0,
                "mean_if_toxic": float(f[yy == 1].mean()), "mean_if_not": float(f[yy == 0].mean())}
    return {"pearson_score": float(np.corrcoef(f, sc)[0, 1]),
            "spearman_score": float(spearmanr(f, sc).statistic),
            "pearson_y": float(np.corrcoef(f, yy)[0, 1]),
            "mean_if_toxic": float(f[yy == 1].mean()),
            "mean_if_not": float(f[yy == 0].mean())}


# ============================================================================
banner("1. COLUMN INVENTORY (all splits)")
inv = {}
for c in full.columns:
    if c == "split":
        continue
    s = full[c]
    rec = {"dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
           "pct_missing": round(100 * float(s.isna().mean()), 4),
           "n_unique": int(s.nunique(dropna=True))}
    rec["missing_by_split"] = {sp: int(s[full["split"] == sp].isna().sum()) for sp in SPLITS}
    inv[c] = rec
    print(f"{c:<38}{rec['dtype']:<22}missing {rec['n_missing']:>10,} ({rec['pct_missing']:6.2f}%)"
          f"  unique {rec['n_unique']:>10,}")
out["inventory"] = inv
out["n_rows"] = {sp: int((full["split"] == sp).sum()) for sp in SPLITS}
out["n_rows"]["all"] = N_ALL

# ============================================================================
# From here on, relations to the label use the training split only.
train = full[full["split"] == "train"].reset_index(drop=True)
val_test = full[full["split"] != "train"].reset_index(drop=True)
del full
N = len(train)
score = train["toxicity"]
y = (score >= THRESHOLD).astype("int8")
out["n_train"] = N
out["toxic_rate_train"] = float(y.mean())

# ============================================================================
banner("2. LABELS (train)")
lab = {}
for c in LABELS:
    s = train[c].astype("float64")
    hist, _ = np.histogram(s, bins=np.linspace(0, 1, 11))
    lab[c] = {**num_summary(s),
              "n_distinct_values": int(s.nunique()),
              "positives_at_0.5": int((s >= THRESHOLD).sum()),
              "pct_positive_at_0.5": float((s >= THRESHOLD).mean() * 100),
              "hist_10_bins": hist.tolist()}
    print(f"{c:<18} mean {lab[c]['mean']:.4f}  zero {lab[c]['pct_zero']:6.2f}%  "
          f">=0.5 {lab[c]['positives_at_0.5']:>9,} ({lab[c]['pct_positive_at_0.5']:.4f}%)  "
          f"distinct values {lab[c]['n_distinct_values']:,}")
out["labels"] = lab
out["label_corr_pearson"] = train[LABELS].astype("float64").corr().round(4).to_dict()
pos = pd.DataFrame({c: train[c] >= THRESHOLD for c in LABELS})
out["labels_per_comment_at_0.5"] = {int(k): int(v) for k, v in pos.sum(axis=1).value_counts().sort_index().items()}
out["subtype_also_toxic_pct"] = {c: float(pos.loc[pos[c], "toxicity"].mean() * 100)
                                 for c in LABELS[1:] if pos[c].any()}
any_sub = pos[LABELS[1:]].any(axis=1)
out["toxic_without_any_subtype_pct"] = float((~any_sub[pos["toxicity"]]).mean() * 100)
out["toxicity_by_threshold"] = {f"{t:.1f}": int((score >= t).sum()) for t in np.arange(0.1, 1.0, 0.1)}
print("labels per comment:", out["labels_per_comment_at_0.5"])
print("toxic comments carrying no subtype >= 0.5: %.2f%%" % out["toxic_without_any_subtype_pct"])

# ============================================================================
banner("3. ANNOTATOR COUNTS (train)")
ann = {}
for c in ANNOTATOR_COUNTS:
    ann[c] = num_summary(train[c])
    ann[c]["relation"] = relation(train[c], score, y)
    vc = train[c].value_counts()
    ann[c]["top_values"] = {int(k): int(v) for k, v in vc.head(8).items()}
    print(c, {k: round(v, 3) for k, v in ann[c].items() if isinstance(v, float)})
    print("   top values", ann[c]["top_values"])
tac = train["toxicity_annotator_count"]
ann["toxicity_annotator_count"]["n_below_4"] = int((tac < 4).sum())
ann["toxicity_annotator_count"]["n_at_least_10"] = int((tac >= 10).sum())
ann["toxic_rate_by_annotator_band"] = {
    band: {"n": int(m.sum()), "toxic_rate": float(y[m].mean())}
    for band, m in {"3-5": tac.between(3, 5), "6-9": tac.between(6, 9),
                    "10-99": tac.between(10, 99), "100+": tac >= 100}.items() if m.any()}
print("toxic rate by annotator band", ann["toxic_rate_by_annotator_band"])
out["annotators"] = ann

# ============================================================================
banner("4. TEXT (train, plus overlaps across splits)")
txt = {}
t = train[TEXT].fillna("")
txt["n_null"] = int(train[TEXT].isna().sum())
txt["n_empty_or_whitespace"] = int((t.str.strip() == "").sum())
stats = text_stats(t)
txt["stats"] = {}
for c in stats.columns:
    txt["stats"][c] = {**num_summary(stats[c]), "relation": relation(stats[c], score, y)}
    r = txt["stats"][c]["relation"]
    print(f"{c:<14} median {txt['stats'][c]['median']:>8.2f}  p99 {txt['stats'][c]['p99']:>8.2f}  "
          f"rho {r['spearman_score']:+.3f}  r_y {r['pearson_y']:+.3f}  "
          f"mean toxic {r['mean_if_toxic']:.3f} vs {r['mean_if_not']:.3f}")
txt["patterns"] = {k: v[1] for k, v in COUNT_PATTERNS.items()}
# length distribution by class, for the figure
bins = np.unique(np.round(np.logspace(0, np.log10(400), 31)))
txt["word_hist_bins"] = bins.tolist()
txt["word_hist_toxic"] = np.histogram(stats.loc[y == 1, "n_words"], bins=bins)[0].tolist()
txt["word_hist_nontoxic"] = np.histogram(stats.loc[y == 0, "n_words"], bins=bins)[0].tolist()
txt["n_chars_max_by_split"] = {"train": int(stats["n_chars"].max())}

# Exact duplicates, after trimming outer whitespace only.
key = t.str.strip()
vc = key.value_counts()
dup_texts = vc[vc > 1]
txt["dup_distinct_texts_repeated"] = int(len(dup_texts))
txt["dup_rows_involved"] = int(dup_texts.sum())
txt["dup_rows_redundant"] = int(dup_texts.sum() - len(dup_texts))
txt["dup_top"] = [{"text": str(k)[:60], "count": int(v)} for k, v in dup_texts.head(10).items()]
grp = pd.DataFrame({"k": key, "y": y}).loc[key.isin(dup_texts.index)]
yspread = grp.groupby("k", observed=True)["y"].agg(["min", "max"])
txt["dup_texts_with_conflicting_y"] = int((yspread["min"] != yspread["max"]).sum())
print(f"duplicate texts: {txt['dup_distinct_texts_repeated']:,} distinct strings cover "
      f"{txt['dup_rows_involved']:,} rows ({txt['dup_rows_redundant']:,} redundant); "
      f"{txt['dup_texts_with_conflicting_y']:,} of them disagree on y")
print("most repeated:", txt["dup_top"][:5])

vt_key = val_test[TEXT].fillna("").str.strip()
train_set = set(key.unique().tolist())
overlap = vt_key.isin(train_set)
txt["val_test_rows_also_in_train"] = {sp: int(overlap[val_test["split"] == sp].sum()) for sp in ["validation", "test"]}
txt["train_rows_whose_text_is_in_val_test"] = int(key.isin(set(vt_key.unique().tolist())).sum())
print("validation/test rows whose exact text is also in train:", txt["val_test_rows_also_in_train"])
print("train rows whose exact text appears in validation/test:", txt["train_rows_whose_text_is_in_val_test"])
del train_set
txt["html_entity_rows"] = int(t.str.contains(r"&(?:amp|lt|gt|quot|#\d+);", regex=True).sum())
txt["html_tag_rows"] = int(t.str.contains(r"</?[a-zA-Z][^>]{0,20}>", regex=True).sum())
print("rows with HTML entities:", txt["html_entity_rows"], " rows with HTML-like tags:", txt["html_tag_rows"])
out["text"] = txt

# ============================================================================
banner("5. IDENTITY COLUMNS (train)")
idn = {}
annotated = train[IDENTITIES].notna().all(axis=1)
idn["n_annotated"] = int(annotated.sum())
idn["pct_annotated"] = float(annotated.mean() * 100)
idn["partially_missing_rows"] = int((train[IDENTITIES].notna().any(axis=1) & ~annotated).sum())
idn["annotated_iff_identity_annotator_count_gt_0"] = bool(
    (annotated == (train["identity_annotator_count"] > 0)).all())
sub = train.loc[annotated]
ysub = y[annotated]
base = float(ysub.mean())
idn["toxic_rate_in_annotated_subset"] = base
idn["per_identity"] = {}
for c in IDENTITIES:
    s = sub[c].astype("float64")
    m = s >= THRESHOLD
    idn["per_identity"][c] = {
        "mean": float(s.mean()), "pct_zero": float((s == 0).mean() * 100),
        "n_mentioned": int(m.sum()), "pct_mentioned": float(m.mean() * 100),
        "toxic_rate_if_mentioned": float(ysub[m].mean()) if m.any() else None,
        "pearson_with_score": float(np.corrcoef(s, sub["toxicity"].astype("float64"))[0, 1]),
    }
    r = idn["per_identity"][c]
    tr = r["toxic_rate_if_mentioned"]
    print(f"{c:<38} mentioned {r['n_mentioned']:>7,} ({r['pct_mentioned']:5.2f}%)  "
          f"toxic if mentioned {tr if tr is None else round(100 * tr, 2)}%  r={r['pearson_with_score']:+.3f}")
any_id = (sub[IDENTITIES] >= THRESHOLD).any(axis=1)
idn["n_any_identity_mentioned"] = int(any_id.sum())
idn["toxic_rate_any_identity"] = float(ysub[any_id].mean())
idn["toxic_rate_no_identity"] = float(ysub[~any_id].mean())
idn["groups"] = {g: float((sub[cols] >= THRESHOLD).any(axis=1).mean() * 100)
                 for g, cols in IDENTITY_GROUPS.items()}
print(f"annotated rows {idn['n_annotated']:,} ({idn['pct_annotated']:.2f}%); toxic rate "
      f"{100 * base:.2f}% overall, {100 * idn['toxic_rate_any_identity']:.2f}% when any identity "
      f"is mentioned, {100 * idn['toxic_rate_no_identity']:.2f}% when none is")
out["identity"] = idn
del sub

# ============================================================================
banner("6. METADATA (train)")
meta = {"label_entropy_bits": entropy(np.bincount(y.astype(int)) / len(y))}

# rating -------------------------------------------------------------------
r = train["rating"].astype(str)
meta["rating"] = {"values": {k: int(v) for k, v in r.value_counts().items()},
                  "toxic_rate": {k: float(y[r == k].mean()) for k in r.unique()},
                  "mean_score": {k: float(score[r == k].mean()) for k in r.unique()},
                  "info_gain": info_gain(r, y)}
print("rating", meta["rating"])

# publication_id -------------------------------------------------------------
p = train["publication_id"]
pc = p.value_counts()
meta["publication_id"] = {
    "n_unique": int(p.nunique()),
    "top10_share_pct": float(pc.head(10).sum() / N * 100),
    "n_with_under_1000_rows": int((pc < 1000).sum()),
    "rows_in_publications_under_1000": int(pc[pc < 1000].sum()),
    "info_gain": info_gain(p, y),
    "per_publication": {int(k): {"n": int(pc[k]), "toxic_rate": float(y[p == k].mean())} for k in pc.index},
    "val_test_publications_not_in_train": sorted(set(val_test["publication_id"].unique().tolist()) - set(pc.index.tolist())),
}
rates = pd.Series({k: v["toxic_rate"] for k, v in meta["publication_id"]["per_publication"].items()
                   if v["n"] >= 1000})
meta["publication_id"]["toxic_rate_range_n_ge_1000"] = [float(rates.min()), float(rates.max())]
print(f"publication_id: {meta['publication_id']['n_unique']} values, information gain "
      f"{meta['publication_id']['info_gain']:.4f} bits, toxic rate {rates.min():.3%} - {rates.max():.3%} "
      f"(publications with >= 1000 comments)")

# created_date -----------------------------------------------------------------
d = train["created_date"]
meta["created_date"] = {"min": str(d.min()), "max": str(d.max()),
                        "range_by_split": {"train": [str(d.min()), str(d.max())],
                                           **{sp: [str(val_test.loc[val_test['split'] == sp, 'created_date'].min()),
                                                   str(val_test.loc[val_test['split'] == sp, 'created_date'].max())]
                                              for sp in ["validation", "test"]}}}
ym = d.dt.strftime("%Y-%m")
by_month = pd.DataFrame({"ym": ym, "y": y}).groupby("ym")["y"].agg(["size", "mean"])
meta["created_date"]["by_month"] = {k: {"n": int(v["size"]), "toxic_rate": float(v["mean"])}
                                    for k, v in by_month.iterrows()}
yr = d.dt.year
meta["created_date"]["by_year"] = {int(k): {"n": int((yr == k).sum()), "toxic_rate": float(y[yr == k].mean())}
                                   for k in sorted(yr.unique())}
hr = d.dt.hour
wd = d.dt.dayofweek
meta["created_date"]["toxic_rate_by_hour_utc"] = {int(k): float(y[hr == k].mean()) for k in range(24)}
meta["created_date"]["toxic_rate_by_weekday"] = {int(k): float(y[wd == k].mean()) for k in range(7)}
meta["created_date"]["info_gain"] = {"year": info_gain(yr, y), "month": info_gain(ym, y),
                                     "hour_utc": info_gain(hr, y), "weekday": info_gain(wd, y)}
print("created_date", meta["created_date"]["min"], "->", meta["created_date"]["max"])
print("   by year", meta["created_date"]["by_year"])
print("   information gain (bits)", meta["created_date"]["info_gain"])
print("   split ranges", meta["created_date"]["range_by_split"])

# parent_id -> is_reply ------------------------------------------------------
is_reply = train["parent_id"].notna()
meta["parent_id"] = {"pct_missing": float((~is_reply).mean() * 100),
                     "toxic_rate_reply": float(y[is_reply].mean()),
                     "toxic_rate_top_level": float(y[~is_reply].mean()),
                     "info_gain_is_reply": info_gain(is_reply, y)}
all_ids = set(train["id"].tolist()) | set(val_test["id"].tolist())
parents = train.loc[is_reply, "parent_id"].astype("int64")
meta["parent_id"]["pct_parents_present_in_data"] = float(parents.isin(all_ids).mean() * 100)
del all_ids
print("parent_id", meta["parent_id"])

# article_id -------------------------------------------------------------------
a = train["article_id"]
ac = a.value_counts()
size = a.map(ac)
meta["article_id"] = {"n_unique": int(a.nunique()),
                      "comments_per_article": num_summary(ac),
                      "spearman_thread_size_vs_score": float(spearmanr(size, score).statistic),
                      "articles_shared_with_val_test": int(len(set(ac.index) & set(val_test["article_id"].unique())))}
print("article_id", {k: v for k, v in meta["article_id"].items() if k != "comments_per_article"})

# reactions --------------------------------------------------------------------
meta["reactions"] = {}
for c in REACTIONS:
    s = train[c]
    meta["reactions"][c] = {**num_summary(s), "relation": relation(s, score, y),
                            "toxic_rate_if_zero": float(y[s == 0].mean()),
                            "toxic_rate_if_positive": float(y[s > 0].mean())}
    rr = meta["reactions"][c]
    print(f"{c:<9} zero {rr['pct_zero']:6.2f}%  p99 {rr['p99']:>6.1f}  max {rr['max']:>7.0f}  "
          f"skew {rr['skew']:6.1f}  rho {rr['relation']['spearman_score']:+.3f}")
out["metadata"] = meta

save_json(out, "02_analysis.json")
