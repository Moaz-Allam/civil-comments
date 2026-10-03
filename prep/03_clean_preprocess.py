"""
03_clean_preprocess.py -- clean the raw release and turn it into model-ready
features. Every rule below is justified by a number in 02_analysis.json; the
report quotes those numbers.

Order of operations
    A. Text cleaning, applied identically to every split (row-level, no fitting)
    B. Row filtering, applied to the TRAINING split only, so validation and test
       stay the official benchmark rows
    C. Labels: keep the seven scores, add 0/1 targets at 0.5
    D. Features: keep only what is known when a comment is posted; encode,
       transform and scale them with parameters fitted on train only
    E. Write the cleaned dump plus every fitted parameter

Output
    data/clean/{train,validation,test}.parquet   the cleaned dataset
    data/clean/sample_train_1000.csv             first 1,000 training rows, for a quick look
    data/clean/preprocessing_params.json         fitted scaler / encoder parameters
    prep/out/03_cleaning.json                    row counts at every step, decisions, final metrics

Run:  python prep/03_clean_preprocess.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RAW_PARQUET, CLEAN, SPLITS, LABELS, IDENTITIES, TEXT, THRESHOLD,
                    read_parquet, save_json, load_json, banner)
from textfeat import text_stats

A = load_json("02_analysis.json")
log = {"funnel": [], "decisions": {}}


def step(name, df, note=""):
    counts = {sp: int((df["split"] == sp).sum()) for sp in SPLITS}
    counts["all"] = int(len(df))
    log["funnel"].append({"step": name, **counts, "note": note})
    print(f"{name:<52} train {counts['train']:>10,}  val {counts['validation']:>7,}  "
          f"test {counts['test']:>7,}")


full = read_parquet(RAW_PARQUET)
banner("ROW COUNTS")
step("raw release", full)

# ============================================================================
# A. TEXT CLEANING (all splits, row by row)
# ============================================================================
CONTROL = "[\x01-\x08\x0b\x0c\x0e-\x1f​‌‍⁠﻿]"


def clean_text(s: pd.Series) -> pd.Series:
    """Light cleaning that keeps case and punctuation, for models that read raw
    text (the Stage 2 transformer)."""
    arr = pa.array(s.fillna(""), type=pa.string())
    arr = pc.replace_substring(arr, "\r\n", "\n")                 # Windows line endings
    arr = pc.replace_substring(arr, "\r", "\n")
    arr = pc.utf8_normalize(arr, form="NFKC")                       # ... -> ..., nbsp -> space, styled letters -> plain
    arr = pc.replace_substring_regex(arr, CONTROL, "")              # control and zero-width characters
    arr = pc.replace_substring_regex(arr, "[\t  ]+", " ")      # runs of spaces / tabs
    arr = pc.replace_substring_regex(arr, " *\n *", "\n")           # spaces around line breaks
    arr = pc.replace_substring_regex(arr, "\n{3,}", "\n\n")         # at most one blank line
    arr = pc.utf8_trim_whitespace(arr)
    out = pd.Series(arr.to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get),
                    index=s.index)
    # HTML entities (&amp; &#147; ...) occur in a handful of rows only; unescape them there.
    import html
    m = out.str.contains(r"&(?:[a-zA-Z]+|#\d+);", regex=True)
    out.loc[m] = out.loc[m].map(html.unescape)
    return out


def bow_text(s: pd.Series) -> pd.Series:
    """Normalisation for bag-of-words models (Stage 1, TF-IDF): lower case,
    placeholders for links / masked swear words / numbers, letters and digits only."""
    arr = pa.array(s, type=pa.string())
    arr = pc.utf8_lower(arr)
    arr = pc.replace_substring_regex(arr, r"https?://\S+|www\.\S+", " xxurl ")
    arr = pc.replace_substring_regex(arr, r"\w*\w\*+\w*|\w\*{2,}\w*", " xxmasked ")
    # Expand contractions before dropping apostrophes; dropping them blindly
    # turns "he'll" into "hell" and "we'll" into "well".
    arr = pc.replace_substring_regex(arr, "[‘’ʼ`]", "'")
    for pat, rep in [(r"\bwon't\b", "will not"), (r"\bcan't\b", "cannot"), (r"n't\b", " not"),
                     (r"'ll\b", " will"), (r"'re\b", " are"), (r"'ve\b", " have"),
                     (r"'m\b", " am"), (r"'d\b", " would"), (r"'s\b", "")]:
        arr = pc.replace_substring_regex(arr, pat, rep)
    arr = pc.replace_substring(arr, "'", "")
    arr = pc.replace_substring_regex(arr, r"[0-9]+(?:[.,][0-9]+)*", " xxnum ")
    arr = pc.replace_substring_regex(arr, r"[^a-z\s]+", " ")
    arr = pc.replace_substring_regex(arr, r"\s+", " ")
    arr = pc.utf8_trim_whitespace(arr)
    return pd.Series(arr.to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get),
                     index=s.index)


raw_len = full[TEXT].str.len()
full["text"] = clean_text(full[TEXT])
changed = (full["text"] != full[TEXT])
log["text_cleaning"] = {
    "rows_changed": int(changed.sum()),
    "pct_rows_changed": float(changed.mean() * 100),
    "chars_removed": int(raw_len.sum() - full["text"].str.len().sum()),
}
full = full.drop(columns=[TEXT])
empty = full["text"].str.len() == 0
log["text_cleaning"]["rows_empty_after_cleaning"] = int(empty.sum())
full = full.loc[~empty].reset_index(drop=True)
step("A. text cleaned; empty texts removed", full,
     f"{int(empty.sum())} comments were empty after cleaning")

# ============================================================================
# B. ROW FILTERING (training split only)
# ============================================================================
is_train = full["split"] == "train"

# B1. Exact duplicates inside train. Keep the copy with the most annotators,
# because its score is the most reliable; ties go to the lowest id.
tr = full.loc[is_train, ["id", "text", "toxicity", "toxicity_annotator_count"]].copy()
tr["y"] = tr["toxicity"] >= THRESHOLD
grp = tr.groupby("text", observed=True)["y"]
nuniq = grp.transform("nunique")
dup_mask = tr["text"].duplicated(keep=False)
log["decisions"]["duplicates"] = {
    "train_rows_in_duplicate_groups": int(dup_mask.sum()),
    "duplicate_groups": int(tr.loc[dup_mask, "text"].nunique()),
    "groups_with_conflicting_y": int(tr.loc[dup_mask & (nuniq > 1), "text"].nunique()),
}
order = tr.sort_values(["toxicity_annotator_count", "id"], ascending=[False, True])
keep_ids = set(order.drop_duplicates("text", keep="first")["id"].tolist())
drop_dup = is_train & ~full["id"].isin(keep_ids)
log["decisions"]["duplicates"]["train_rows_removed"] = int(drop_dup.sum())
full = full.loc[~drop_dup].reset_index(drop=True)
step("B1. duplicate texts inside train collapsed to one row", full,
     "kept the copy with the most annotators")
del tr, order, keep_ids, grp, nuniq

# B2. Leakage: a training comment whose exact text is also in validation or
# test would let a model memorise an evaluation answer. Remove it from train.
is_train = full["split"] == "train"
eval_texts = set(full.loc[~is_train, "text"].unique().tolist())
leak = is_train & full["text"].isin(eval_texts)
log["decisions"]["cross_split_leakage"] = {"train_rows_removed": int(leak.sum())}
full = full.loc[~leak].reset_index(drop=True)
step("B2. train texts that also occur in val/test removed", full)
del eval_texts

# B3. No other filtering. Records considered and rejected:
log["decisions"]["not_filtered"] = {
    "low_annotator_count": "only %d training rows have fewer than 4 annotators; score = votes / "
                           "annotators holds for every row, so they are valid labels"
                           % A["annotators"]["toxicity_annotator_count"]["n_below_4"],
    "very_short_texts": "texts such as 'No.' or 'BS' are real comments and some are toxic; kept",
    "long_texts": "the longest comment has %d characters (99th percentile %d), so there are no "
                  "runaway lengths to cap" % (int(A["text"]["stats"]["n_chars"]["max"]),
                                              int(A["text"]["stats"]["n_chars"]["p99"])),
    "validation_test": "left untouched except for row-level text cleaning, so results stay "
                       "comparable with published numbers on the same split",
}

# ============================================================================
# C. LABELS
# ============================================================================
for c in LABELS:
    full[f"y_{c}"] = (full[c] >= THRESHOLD).astype("int8")
TARGETS = ["toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
log["decisions"]["targets"] = {
    "threshold": THRESHOLD,
    "modelling_targets": TARGETS,
    "dropped_target": {"severe_toxicity": "13 positives in 1,804,874 training comments at 0.5; "
                                          "score kept, not used as a target"},
}

# ============================================================================
# D. FEATURES
# ============================================================================
is_train = full["split"] == "train"
stats = text_stats(full["text"])

# D1. Which text statistics to keep. Two rules, both computed on train:
#     redundant  -> |Pearson r| > 0.9 with a statistic already kept
#     no signal  -> |Spearman rho| < 0.01 AND |point-biserial r| < 0.01
#                   with the toxicity label (values from 02_analysis.json)
CANDIDATES = ["n_words", "mean_word_len", "upper_ratio", "n_caps_words", "n_exclaim",
              "n_question", "n_punct_runs", "n_masked", "n_urls", "n_digits", "n_newlines",
              "n_non_ascii", "n_chars", "n_letters", "n_upper"]
corr = stats.loc[is_train, CANDIDATES].astype("float64").corr()
kept, dropped = [], {}
for c in CANDIDATES:
    rel = A["text"]["stats"][c]["relation"]
    if abs(rel["spearman_score"]) < 0.01 and abs(rel["pointbiserial_y"]) < 0.01:
        dropped[c] = (f"no association with the label (rho {rel['spearman_score']:+.3f}, "
                      f"r_pb {rel['pointbiserial_y']:+.3f})")
        continue
    twin = next((k for k in kept if abs(corr.loc[c, k]) > 0.9), None)
    if twin:
        dropped[c] = f"redundant with {twin} (r = {corr.loc[c, twin]:.3f})"
        continue
    kept.append(c)
log["decisions"]["text_stats"] = {"kept": kept, "dropped": dropped,
                                  "pairwise_r_train": corr.round(3).to_dict()}
print("\ntext statistics kept:", kept)
for k, v in dropped.items():
    print(f"   dropped {k:<12} {v}")

# D2. Transform. Counts are right-skewed (skew in 02_analysis.json), so log1p
# them; then standardise every statistic with the TRAIN mean and std. Ratios
# are not counts, so instead of a log they are capped at the training 1st / 99th
# percentiles (comments made only of a link give mean word lengths above 40).
RATIOS = {"mean_word_len", "upper_ratio"}
params = {"text_stats": {}}
for c in kept:
    x = stats[c].astype("float64")
    p = {}
    if c in RATIOS:
        lo, hi = np.percentile(x[is_train], [1, 99])
        x = x.clip(lo, hi)
        p.update(transform="clip then z-score", clip=[float(lo), float(hi)])
    else:
        x = np.log1p(x)
        p.update(transform="log1p then z-score")
    mu, sd = float(x[is_train].mean()), float(x[is_train].std())
    p.update(mean=mu, std=sd)
    full[f"f_{c}"] = ((x - mu) / sd).astype("float32")
    params["text_stats"][c] = p
del stats

# D3. parent_id -> is_reply. The 43% "missing" parent ids are not missing data:
# a top-level comment has no parent. The binary flag is the information.
full["f_is_reply"] = full["parent_id"].notna().astype("int8")

# D4. publication_id -> one-hot. Publications with fewer than 1,000 training
# comments, and any publication never seen in training, share one "other"
# column, so no column is nearly empty.
pc_train = full.loc[is_train, "publication_id"].value_counts()
major = sorted(int(k) for k in pc_train[pc_train >= 1000].index)
pub = full["publication_id"].where(full["publication_id"].isin(major), -1)
for k in major:
    full[f"f_pub_{k}"] = (pub == k).astype("int8")
full["f_pub_other"] = (pub == -1).astype("int8")
params["publication_one_hot"] = {"categories": major, "other": "publications with < 1000 training "
                                 "comments or unseen in training"}

# D5. Identity columns: kept for the bias audit, never as model inputs. Not
# imputed: 77.6% of comments were never shown to identity annotators, and a 0
# would claim "no identity mentioned" for comments nobody checked.
full["identity_annotated"] = full[IDENTITIES[0]].notna().astype("int8")

# D6. Columns removed and why (figures from 02_analysis.json)
M = A["metadata"]
log["decisions"]["removed_columns"] = {
    "toxicity_annotator_count": "artefact of how comments were sampled for annotation: toxic rate "
                                "%.2f%% with 3-5 annotators vs %.2f%% with 10-99; unknown when a comment "
                                "is posted. Kept as n_annotators for sample weighting, never as a feature"
                                % (100 * A["annotators"]["toxic_rate_by_annotator_band"]["3-5"]["toxic_rate"],
                                   100 * A["annotators"]["toxic_rate_by_annotator_band"]["10-99"]["toxic_rate"]),
    "identity_annotator_count": "duplicates the identity_annotated flag",
    "rating": "peer-review verdict given after posting, i.e. a second human judgement of the same "
              "comment (Cramer's V %.3f); kept in the dump as metadata, not a feature" % M["rating"]["cramers_v"],
    "funny, wow, sad, likes, disagree": "reader reactions accumulate after posting; |rho| <= %.3f"
                                        % max(abs(M["reactions"][c]["relation"]["spearman_score"])
                                              for c in M["reactions"]),
    "created_date": "Cramer's V with the label: year %.3f, month %.3f, hour %.3f, weekday %.3f; all splits "
                    "span the same dates, and a 2015-2017 calendar does not transfer to new data"
                    % tuple(M["created_date"]["cramers_v"][k] for k in ["year", "month", "hour_utc", "weekday"]),
    "parent_id": "replaced by is_reply",
    "article_id": "identifier with %d values and no article shared between train and val/test, so "
                  "a value seen in training never occurs again; dropped" % M["article_id"]["n_unique"],
    "publication_id": "replaced by its one-hot encoding",
}

# D7. Text for bag-of-words models
full["text_bow"] = bow_text(full["text"])
log["text_cleaning"]["rows_with_empty_bow_text"] = int((full["text_bow"].str.len() == 0).sum())

# ============================================================================
# E. WRITE
# ============================================================================
banner("FINAL DATASET")
FEATURES = [c for c in full.columns if c.startswith("f_")]
cols = (["id", "split", "text", "text_bow"] + LABELS + [f"y_{c}" for c in LABELS]
        + ["toxicity_annotator_count"] + FEATURES + ["identity_annotated"] + IDENTITIES
        + ["publication_id", "created_date", "rating"])
full = full[cols].rename(columns={"toxicity_annotator_count": "n_annotators"})

final = {"rows": {}, "positives": {}, "columns": list(full.columns), "n_columns": full.shape[1],
         "features": FEATURES, "n_features_tabular": len(FEATURES)}
for sp in SPLITS:
    d = full[full["split"] == sp]
    final["rows"][sp] = int(len(d))
    final["positives"][sp] = {c: {"count": int(d[f"y_{c}"].sum()),
                                  "pct": float(d[f"y_{c}"].mean() * 100)} for c in LABELS}
    final.setdefault("identity_annotated", {})[sp] = int(d["identity_annotated"].sum())
    final.setdefault("text_chars", {})[sp] = {"mean": float(d["text"].str.len().mean()),
                                               "median": float(d["text"].str.len().median())}
    path = os.path.join(CLEAN, f"{sp}.parquet")
    d.drop(columns=["split"]).to_parquet(path, index=False, compression="zstd")
    final.setdefault("files", {})[f"{sp}.parquet"] = os.path.getsize(path)
    print(f"{sp:<11} rows {len(d):>10,}  toxic {final['positives'][sp]['toxicity']['count']:>8,} "
          f"({final['positives'][sp]['toxicity']['pct']:.3f}%)  file {os.path.getsize(path):>12,} bytes")
final["rows"]["all"] = int(len(full))
tr = full.loc[full["split"] == "train", [f"y_{c}" for c in LABELS] + FEATURES]   # columns needed only
final["majority_baseline_acc_train"] = {c: float(100 * (1 - tr[f"y_{c}"].mean())) for c in LABELS}
final["feature_check_train"] = {c: {"mean": float(tr[c].mean()), "std": float(tr[c].std())}
                                for c in FEATURES}
del tr
# rows are in split order, so these are the first 1,000 training rows
full.head(1000).to_csv(os.path.join(CLEAN, "sample_train_1000.csv"), index=False)
log["final"] = final

params["feature_columns"] = FEATURES
params["threshold"] = THRESHOLD
params["targets"] = TARGETS
params["text_cleaning"] = {"text": "CRLF->LF, NFKC, strip control/zero-width chars, collapse spaces, "
                                   "max one blank line, trim, unescape HTML entities",
                           "text_bow": "lower case, links->xxurl, masked words->xxmasked, contractions "
                                       "expanded, numbers->xxnum, other non-letters->space"}
import json
with open(os.path.join(CLEAN, "preprocessing_params.json"), "w") as f:
    json.dump(params, f, indent=2)
print(f"\nwrote data/clean/preprocessing_params.json")
save_json(log, "03_cleaning.json")
