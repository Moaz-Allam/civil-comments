"""
Merges every outputs/*.json into outputs/summary.csv, one row per model, feature set and
label, with the validation and test metrics. Rank models on validation only.

Run:  python models/summarize.py
"""
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["accuracy", "error_rate", "precision", "recall", "specificity", "fpr", "f1", "roc_auc"]


def load():
    rows = []
    for path in glob.glob(os.path.join(ROOT, "outputs", "*.json")):
        with open(path, encoding="utf8") as f:
            r = json.load(f)
        row = {k: r[k] for k in ["model", "features", "label", "n_train", "sampled", "seconds"]}
        for split in ["validation", "test"]:
            row.update({f"{split}_{m}": r[split][m] for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "features", "label"], ignore_index=True)


if __name__ == "__main__":
    table = load()
    table.to_csv(os.path.join(ROOT, "outputs", "summary.csv"), index=False)
    print(f"wrote outputs/summary.csv: {len(table)} results")
