"""
civil_comments_hf_stats.py -- Civil Comments statistics derived from the
Hugging Face dataset server, with the raw responses recorded inline.

WHY THIS FILE EXISTS
    The Parquet files had not been downloaded when the report was written, and
    huggingface.co is not reachable from the machine the analysis ran on. The
    dataset server publishes per-column histograms computed over the FULL split
    ("partial": false), which is enough to derive exact counts. The raw
    responses are pasted in below exactly as returned, so every number in the
    report can be traced to them.

    Once you have run download_civil_comments.py, civil_comments_stats.py
    recomputes all of this from the Parquet files directly and goes further
    (per-comment label combinations, which histograms cannot give). The two
    should agree; if they do not, trust the local one and tell me.

ENDPOINTS USED (retrieved 20 September 2026)
    https://datasets-server.huggingface.co/size?dataset=google/civil_comments&config=default
    https://datasets-server.huggingface.co/statistics?dataset=google/civil_comments&config=default&split=train
    https://datasets-server.huggingface.co/filter?dataset=google/civil_comments&config=default&split=train&where="toxicity">=0.5

HOW THE COUNTS ARE DERIVED
    The histogram bin edges are 0, 0.1, ..., 1.0, and bins are half-open
    [lo, hi) with the last one closed. So the number of comments scoring >= 0.5
    on a column is the sum of the last five bins.

    That reading is not assumed, it is checked: the filter endpoint counts rows
    server-side and independently returns 144,334 for toxicity >= 0.5, which is
    exactly what summing the last five toxicity bins gives. The script asserts
    this, and also asserts that every histogram sums to the split size.

Run:  python civil_comments_hf_stats.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- raw responses
SIZE = {
    "num_rows_total": 1999514,
    "num_columns": 8,
    "num_bytes_parquet_files": 422061071,
    "splits": {"train": 1804874, "validation": 97320, "test": 97320},
}

TRAIN = {
    "num_examples": 1804874,
    "partial": False,
    "bin_edges": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "histograms": {
        "toxicity":        [1266229, 160959, 111597, 64613, 57142, 47924, 50959, 14620, 23314, 7517],
        "severe_toxicity": [1753896,  44540,   5944,   419,    62,    10,     2,     0,     0,    1],
        "obscene":         [1691692,  71189,  21020,  7055,  4315,  3103,  3681,  1230,  1276,  313],
        "threat":          [1721699,  51777,  17712,  5937,  3469,  1876,  1483,   192,   509,  220],
        "insult":          [1354169, 145470,  99558, 54366, 44777, 36497, 36345, 11903, 16339, 5450],
        "identity_attack": [1622772,  95852,  42701, 18517, 11622,  6926,  4892,   351,   999,  242],
        "sexual_explicit": [1753413,  28798,   9832,  4878,  3267,  2206,  1722,   295,   352,  111],
    },
    "means": {"toxicity": 0.10302, "severe_toxicity": 0.00458, "obscene": 0.01388,
              "threat": 0.00931, "insult": 0.08115, "identity_attack": 0.02264,
              "sexual_explicit": 0.00661},
    "text_chars": {"mean": 297.2343, "median": 202, "min": 1, "max": 1906, "std": 269.1966},
}

FILTER_CROSSCHECK = {"where": '"toxicity">=0.5', "num_rows_total": 144334}

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat",
          "insult", "identity_attack", "sexual_explicit"]

# ---------------------------------------------------------------- integrity
N = TRAIN["num_examples"]
print("=" * 68)
print("INTEGRITY CHECKS")
print("=" * 68)

assert sum(SIZE["splits"].values()) == SIZE["num_rows_total"], "splits do not sum to the total"
print(f"splits sum to the stated total          {sum(SIZE['splits'].values()):>12,}  OK")

for c in LABELS:
    s = sum(TRAIN["histograms"][c])
    assert s == N, f"{c} histogram sums to {s}, expected {N}"
print(f"every histogram sums to the split size  {N:>12,}  OK")

derived_tox = sum(TRAIN["histograms"]["toxicity"][5:])
assert derived_tox == FILTER_CROSSCHECK["num_rows_total"], "bin-edge reading disagrees with the filter endpoint"
print(f"histogram vs. independent row count     {derived_tox:>12,}  OK")
print()

# ---------------------------------------------------------------- derived
out = {"source": "Hugging Face dataset server", "retrieved": "2026-09-20",
       "n_train": N, "splits": SIZE["splits"], "total_rows": SIZE["num_rows_total"],
       "parquet_bytes": SIZE["num_bytes_parquet_files"],
       "means": TRAIN["means"], "text_chars": TRAIN["text_chars"]}

print("=" * 68)
print("POSITIVES PER LABEL AT THRESHOLD 0.5 (train split)")
print("=" * 68)
out["positives_at_0.5"] = {}
print(f"{'label':<18}{'comments':>12}{'% of split':>14}")
for c in LABELS:
    k = sum(TRAIN["histograms"][c][5:])
    out["positives_at_0.5"][c] = {"count": k, "pct": round(100 * k / N, 4)}
    print(f"{c:<18}{k:>12,}{100*k/N:>13.4f}%")

print()
print("=" * 68)
print("TOXICITY POSITIVES AS THE THRESHOLD MOVES (train split)")
print("=" * 68)
tox = TRAIN["histograms"]["toxicity"]
out["toxicity_by_threshold"] = []
for i in range(1, 10):
    k = sum(tox[i:])
    out["toxicity_by_threshold"].append({"threshold": round(i / 10, 1), "count": k,
                                         "pct": round(100 * k / N, 4)})
    print(f">= {i/10:.1f}   {k:>10,}   {100*k/N:6.3f}%")

lo = out["toxicity_by_threshold"][3]["count"]   # 0.4
hi = out["toxicity_by_threshold"][5]["count"]   # 0.6
print()
print(f"moving the threshold from 0.4 to 0.6 changes the positive set from "
      f"{lo:,} to {hi:,}, a factor of {lo/hi:.2f}")
out["threshold_sensitivity_0.4_to_0.6"] = {"at_0.4": lo, "at_0.6": hi, "ratio": round(lo / hi, 4)}

print()
print("=" * 68)
print("WHY ACCURACY IS THE WRONG HEADLINE METRIC (train, threshold 0.5)")
print("=" * 68)
accs = []
for c in LABELS:
    k = out["positives_at_0.5"][c]["count"]
    a = 100 * (N - k) / N
    accs.append(a)
    print(f"{c:<18} always-negative accuracy = {a:7.4f}%   F1 = 0.000")
out["majority_baseline_acc"] = {c: round(a, 4) for c, a in zip(LABELS, accs)}
out["majority_baseline_mean_acc"] = round(sum(accs) / len(accs), 4)
print(f"mean across the seven labels = {out['majority_baseline_mean_acc']:.4f}%")

with open(os.path.join(HERE, "cc_stats.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("wrote cc_stats.json")
