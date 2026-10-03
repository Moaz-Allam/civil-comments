"""
01_load.py -- load the three files of the full Civil Comments release into one
table with a split column, compact dtypes and one name for the main score.

    train.csv.gz                  -> train       (1,804,874 comments)
    test_public_expanded.csv.gz   -> validation  (   97,320)
    test_private_expanded.csv.gz  -> test        (   97,320)

Output
    data/interim/civil_comments_full.parquet   all 1,999,514 rows, 46 columns
    prep/out/01_load.json                      file hashes, shapes

Run:  python prep/01_load.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (KAGGLE_DIR, KAGGLE_FILES, SPLITS, RAW_PARQUET, LABELS,
                    IDENTITIES, REACTIONS, ANNOTATOR_COUNTS, TEXT, sha256, save_json, banner)

out = {"files": {}, "splits": {}, "checks": {}}

banner("SOURCE FILES (SHA-256)")
for name in KAGGLE_FILES:
    p = os.path.join(KAGGLE_DIR, name)
    rec = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    out["files"]["kaggle/" + name] = rec
    print(f"{name:<32}{rec['bytes']:>14,}  {rec['sha256']}")

# ----------------------------------------------------------------------------
banner("LOAD THE FULL RELEASE")
frames = []
for name, split in KAGGLE_FILES.items():
    df = pd.read_csv(os.path.join(KAGGLE_DIR, name), dtype={TEXT: "string"},
                     keep_default_na=False, na_values=[""])
    print(f"{name:<32} rows={len(df):>10,}  cols={df.shape[1]}")
    # The training file calls the main score "target"; the test files call it
    # "toxicity". Same quantity, so one name.
    df = df.rename(columns={"target": "toxicity"})
    df.insert(1, "split", split)
    out["splits"][split] = {"rows": int(len(df)), "source_file": name,
                            "source_columns": int(df.shape[1] - 1)}
    frames.append(df)

cols = frames[0].columns
for f in frames[1:]:
    assert set(f.columns) == set(cols), "the three files do not share one schema"
full = pd.concat([f[cols] for f in frames], ignore_index=True)
del frames

# Compact dtypes. Scores fit in float32 exactly enough (they are fractions with
# denominators of at most a few thousand), counts in int32.
for c in LABELS + IDENTITIES:
    full[c] = full[c].astype("float32")
for c in REACTIONS + ANNOTATOR_COUNTS + ["article_id"]:
    full[c] = full[c].astype("int32")
full["publication_id"] = full["publication_id"].astype("int16")
full["parent_id"] = full["parent_id"].astype("float64")      # NaN for top-level comments
full["rating"] = full["rating"].astype("category")
full["created_date"] = pd.to_datetime(full["created_date"], utc=True, format="ISO8601")
full["split"] = pd.Categorical(full["split"], categories=SPLITS)

out["total_rows"] = int(len(full))
out["columns"] = [c for c in full.columns if c != "split"]
out["n_columns"] = len(out["columns"])
print(f"TOTAL rows={len(full):,}  columns (excluding our split tag)={out['n_columns']}")

# ----------------------------------------------------------------------------
banner("CHECKS")
checks = out["checks"]

n_dup_ids = int(full["id"].duplicated().sum())
checks["duplicate_ids"] = n_dup_ids
print(f"duplicate ids across all splits: {n_dup_ids}")
assert n_dup_ids == 0

# ----------------------------------------------------------------------------
full.to_parquet(RAW_PARQUET, index=False, compression="zstd")
out["interim_parquet"] = {"path": "data/interim/civil_comments_full.parquet",
                          "bytes": os.path.getsize(RAW_PARQUET)}
print(f"\nwrote {RAW_PARQUET}  ({os.path.getsize(RAW_PARQUET):,} bytes)")
save_json(out, "01_load.json")
