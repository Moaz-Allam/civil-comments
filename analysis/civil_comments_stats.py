"""
civil_comments_stats.py -- statistics for google/civil_comments, computed from
the Parquet files rather than quoted from anywhere.

Run download_civil_comments.py first, then:
    pip install pandas pyarrow
    python civil_comments_stats.py                 # uses ../data/civil_comments
    python civil_comments_stats.py <folder>        # or point it somewhere else

It prints each file's SHA-256 first, so any number below can be tied back to the
exact files it came from, and writes civil_comments_stats.json next to itself.

Note on thresholds: the labels are fractions of annotators, not 0/1. The Kaggle
competition treated a comment as positive when the score was >= 0.5, so that is
the threshold used here; the script also reports counts at 0.25 and 0.75 so you
can see how much the choice of threshold moves the numbers. That sensitivity is
worth a sentence in the report -- it is the main practical difference between
this dataset and the binary-labelled Jigsaw one.
"""
import sys, os, json, glob, hashlib
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "data", "civil_comments")

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat",
          "insult", "identity_attack", "sexual_explicit"]
THRESHOLDS = [0.25, 0.5, 0.75]
out = {"thresholds": THRESHOLDS, "splits": {}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(split):
    pats = [os.path.join(DATA, "data", f"{split}-*.parquet"),
            os.path.join(DATA, f"{split}-*.parquet")]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits
    return []


print("=" * 66)
print("FILES")
print("=" * 66)
files = {}
for split in ["train", "validation", "test"]:
    hits = find(split)
    if not hits:
        print(f"!! no parquet files found for split '{split}' under {DATA}")
        print("   run download_civil_comments.py first")
        sys.exit(1)
    files[split] = hits
    for p in hits:
        rec = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        out.setdefault("files", {})[os.path.basename(p)] = rec
        print(f"{os.path.basename(p):<40} {rec['bytes']:>12,} bytes")
        print(f"{'':<40} sha256={rec['sha256']}")

frames = {}
for split, hits in files.items():
    frames[split] = pd.concat([pd.read_parquet(p) for p in hits], ignore_index=True)

print()
print("=" * 66)
print("SPLIT SIZES AND COLUMNS")
print("=" * 66)
for split, df in frames.items():
    out["splits"][split] = {"rows": int(len(df))}
    print(f"{split:<12} rows={len(df):>10,}  cols={len(df.columns)}")
out["columns"] = list(frames["train"].columns)
out["total_rows"] = int(sum(len(d) for d in frames.values()))
print(f"{'TOTAL':<12} rows={out['total_rows']:>10,}")
print("columns:", out["columns"])

train = frames["train"]

print()
print("=" * 66)
print("LABEL SCORES (train) -- these are fractions of annotators, not 0/1")
print("=" * 66)
out["score_stats"] = {}
print(f"{'label':<18}{'mean':>10}{'median':>10}{'p99':>10}{'max':>8}{'exactly 0':>12}")
for c in LABELS:
    s = train[c].astype("float64")
    rec = {"mean": float(s.mean()), "median": float(s.median()),
           "p99": float(s.quantile(0.99)), "max": float(s.max()),
           "pct_exactly_zero": float((s == 0).mean() * 100)}
    out["score_stats"][c] = rec
    print(f"{c:<18}{rec['mean']:>10.4f}{rec['median']:>10.4f}"
          f"{rec['p99']:>10.4f}{rec['max']:>8.2f}{rec['pct_exactly_zero']:>11.2f}%")

print()
print("=" * 66)
print("POSITIVES PER LABEL (train) AT THREE THRESHOLDS")
print("=" * 66)
n = len(train)
out["positives"] = {}
header = f"{'label':<18}" + "".join(f"{'>=' + str(t):>20}" for t in THRESHOLDS)
print(header)
for c in LABELS:
    s = train[c].astype("float64")
    row = {}
    line = f"{c:<18}"
    for t in THRESHOLDS:
        k = int((s >= t).sum())
        row[str(t)] = {"count": k, "pct": round(100 * k / n, 4)}
        line += f"{k:>11,} ({100*k/n:5.2f}%)"
    out["positives"][c] = row
    print(line)

print()
print("=" * 66)
print("MULTI-LABEL STRUCTURE AT THRESHOLD 0.5 (train)")
print("=" * 66)
pos = pd.DataFrame({c: (train[c].astype("float64") >= 0.5) for c in LABELS})
per = pos.sum(axis=1).value_counts().sort_index()
out["labels_per_comment_at_0.5"] = {int(k): int(v) for k, v in per.items()}
any_lab = int((pos.sum(axis=1) > 0).sum())
out["any_label_count_at_0.5"] = any_lab
out["any_label_pct_at_0.5"] = round(100 * any_lab / n, 4)
out["clean_pct_at_0.5"] = round(100 * (n - any_lab) / n, 4)
for k, v in per.items():
    print(f"{int(k)} label(s): {v:>9,}  ({100*v/n:6.3f}% of train)")
print()
print(f"comments with >=1 label : {any_lab:>9,}  ({out['any_label_pct_at_0.5']:.3f}%)")
print(f"comments with 0 labels  : {n-any_lab:>9,}  ({out['clean_pct_at_0.5']:.3f}%)")

tox = pos["toxicity"]
print()
for c in LABELS:
    if c == "toxicity":
        continue
    sub = pos[pos[c]]
    if len(sub):
        share = float(sub["toxicity"].mean()) * 100
        out.setdefault("cooccurrence_with_toxicity", {})[c] = round(share, 3)
        print(f"of {len(sub):>8,} '{c}' comments, {share:6.2f}% are also toxicity>=0.5")

print()
print("=" * 66)
print("COMMENT LENGTH (train)")
print("=" * 66)
words = train["text"].str.split().str.len()
chars = train["text"].str.len()
out["length_words"] = {k: float(v) for k, v in words.describe().items()}
out["length_chars"] = {k: float(v) for k, v in chars.describe().items()}
print("words:", {k: round(float(v), 2) for k, v in words.describe().items()})
print("chars:", {k: round(float(v), 2) for k, v in chars.describe().items()})

print()
print("=" * 66)
print("WHY ACCURACY IS THE WRONG HEADLINE METRIC (train, threshold 0.5)")
print("=" * 66)
accs = []
for c in LABELS:
    a = 100 * float((~pos[c]).mean())
    accs.append(a)
    print(f"{c:<18} always-negative accuracy = {a:6.3f}%   F1 = 0.000")
out["majority_baseline_acc"] = {c: round(a, 4) for c, a in zip(LABELS, accs)}
out["majority_baseline_mean_acc"] = round(float(np.mean(accs)), 4)
print(f"mean across the seven labels = {np.mean(accs):.3f}%")

with open(os.path.join(HERE, "civil_comments_stats.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("wrote civil_comments_stats.json")
print()
print("NOTE: this release has no identity columns, so it cannot support a bias")
print("audit on its own. For that you need the Kaggle 'Jigsaw Unintended Bias'")
print("files or the CivilCommentsIdentities config in TensorFlow Datasets.")
