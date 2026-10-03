"""
jigsaw_stats.py  -- dataset statistics for the Jigsaw Toxic Comment Classification Challenge.

Reproduce:
  1. Get train.csv / test.csv / test_labels.csv from the Kaggle competition page
     (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
     after accepting the competition rules.
  2. python jigsaw_stats.py <folder containing the three csv files>

Every number printed here is computed from the files, nothing is hard-coded.
File integrity (SHA-256) is printed first so results can be tied to exact files.
"""
import sys, os, json, hashlib
import pandas as pd
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
out = {}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("=" * 64)
print("FILE INTEGRITY")
print("=" * 64)
out["files"] = {}
for name in ["train.csv", "test.csv", "test_labels.csv"]:
    p = os.path.join(DATA, name)
    if os.path.exists(p):
        rec = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        out["files"][name] = rec
        print(f"{name:16s} {rec['bytes']:>12,} bytes  sha256={rec['sha256']}")

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
test_labels = pd.read_csv(os.path.join(DATA, "test_labels.csv"))

print()
print("=" * 64)
print("SHAPES AND COLUMNS")
print("=" * 64)
out["train_rows"], out["train_cols"] = map(int, train.shape)
out["test_rows"] = int(test.shape[0])
out["test_labels_rows"] = int(test_labels.shape[0])
out["train_columns"] = list(train.columns)
print("train.csv        rows=%d  cols=%d  %s" % (train.shape[0], train.shape[1], list(train.columns)))
print("test.csv         rows=%d  cols=%d  %s" % (test.shape[0], test.shape[1], list(test.columns)))
print("test_labels.csv  rows=%d  cols=%d  %s" % (test_labels.shape[0], test_labels.shape[1], list(test_labels.columns)))
print("duplicate ids in train:", int(train["id"].duplicated().sum()))
print("null comment_text in train:", int(train["comment_text"].isna().sum()))

print()
print("=" * 64)
print("LABEL COUNTS (train)")
print("=" * 64)
n = len(train)
out["label_counts"] = {}
print(f"{'label':<16}{'positives':>12}{'% of train':>14}")
for c in LABELS:
    k = int(train[c].sum())
    out["label_counts"][c] = {"count": k, "pct": round(100 * k / n, 4)}
    print(f"{c:<16}{k:>12,}{100*k/n:>13.3f}%")

any_lab = (train[LABELS].sum(axis=1) > 0)
out["any_label_count"] = int(any_lab.sum())
out["any_label_pct"] = round(100 * float(any_lab.mean()), 4)
out["clean_count"] = int((~any_lab).sum())
out["clean_pct"] = round(100 * float((~any_lab).mean()), 4)
print()
print("comments with >=1 label : %6d  (%.3f%%)" % (out["any_label_count"], out["any_label_pct"]))
print("comments with 0 labels  : %6d  (%.3f%%)" % (out["clean_count"], out["clean_pct"]))

print()
print("=" * 64)
print("LABELS PER COMMENT (train)")
print("=" * 64)
per = train[LABELS].sum(axis=1).value_counts().sort_index()
out["labels_per_comment"] = {int(k): int(v) for k, v in per.items()}
for k, v in per.items():
    print(f"{int(k)} label(s): {v:>7,}  ({100*v/n:6.3f}%)")

print()
print("=" * 64)
print("CO-OCCURRENCE WITH 'toxic' (train)")
print("=" * 64)
out["cooccurrence_with_toxic"] = {}
for c in LABELS:
    if c == "toxic":
        continue
    sub = train[train[c] == 1]
    share = float(sub["toxic"].mean()) if len(sub) else float("nan")
    out["cooccurrence_with_toxic"][c] = round(100 * share, 3)
    print(f"of {len(sub):>6,} '{c}' comments, {100*share:6.2f}% are also labelled toxic")

print()
print("=" * 64)
print("COMMENT LENGTH (train)")
print("=" * 64)
words = train["comment_text"].str.split().str.len()
chars = train["comment_text"].str.len()
out["length_words"] = {k: float(v) for k, v in words.describe().items()}
out["length_chars"] = {k: float(v) for k, v in chars.describe().items()}
out["length_words_p90"] = float(words.quantile(0.90))
out["length_words_p95"] = float(words.quantile(0.95))
out["length_words_p99"] = float(words.quantile(0.99))
print("words  :", words.describe().round(2).to_dict())
print("  p90=%.0f  p95=%.0f  p99=%.0f" % (out["length_words_p90"], out["length_words_p95"], out["length_words_p99"]))
print("chars  :", chars.describe().round(2).to_dict())
pct_over_128 = 100 * float((words > 128).mean())
pct_over_256 = 100 * float((words > 256).mean())
out["pct_words_over_128"] = round(pct_over_128, 3)
out["pct_words_over_256"] = round(pct_over_256, 3)
print("share longer than 128 words: %.2f%%   longer than 256 words: %.2f%%" % (pct_over_128, pct_over_256))

print()
print("=" * 64)
print("TEST SPLIT SCORING MASK")
print("=" * 64)
scored = (test_labels[LABELS] != -1).all(axis=1)
out["test_scored_rows"] = int(scored.sum())
out["test_unscored_rows"] = int((~scored).sum())
out["test_scored_pct"] = round(100 * float(scored.mean()), 4)
print("rows in test_labels.csv       : %d" % len(test_labels))
print("scored rows (labels != -1)    : %d  (%.2f%%)" % (out["test_scored_rows"], out["test_scored_pct"]))
print("unscored rows (labels == -1)  : %d" % out["test_unscored_rows"])
sc = test_labels[scored]
out["test_label_counts"] = {}
print()
print(f"{'label':<16}{'positives':>12}{'% of scored test':>20}")
for c in LABELS:
    k = int(sc[c].sum())
    out["test_label_counts"][c] = {"count": k, "pct": round(100 * k / len(sc), 4)}
    print(f"{c:<16}{k:>12,}{100*k/len(sc):>19.3f}%")

print()
print("=" * 64)
print("NAIVE MAJORITY BASELINE (predict 0 for everything, train split)")
print("=" * 64)
accs = []
for c in LABELS:
    a = 100 * float((train[c] == 0).mean())
    accs.append(a)
    print(f"{c:<16} accuracy = {a:6.3f}%   F1 = 0.000 (no positive predictions)")
out["majority_baseline_acc"] = {c: round(a, 4) for c, a in zip(LABELS, accs)}
out["majority_baseline_mean_acc"] = round(float(np.mean(accs)), 4)
out["all_clean_exact_match"] = round(out["clean_pct"], 4)
print(f"mean per-label accuracy = {np.mean(accs):.3f}%")
print(f"exact-match accuracy of the all-zero predictor = {out['clean_pct']:.3f}%")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jigsaw_stats.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("wrote jigsaw_stats.json")
