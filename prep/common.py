"""
common.py -- paths and column groups shared by every Phase 2 script.

Every script in this folder reads from or writes to the locations below, so the
pipeline can be re-run end to end with:

    python prep/00_download.py
    python prep/01_load.py
    python prep/02_feature_analysis.py
    python prep/03_clean_preprocess.py
    python prep/04_word2vec.py
    python prep/05_correlation_checks.py
    python prep/06_make_figures.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA = os.path.join(PROJECT, "data")

# Raw sources
KAGGLE_DIR = os.path.join(DATA, "jigsaw_unintended_bias", "data")
KAGGLE_FILES = {
    # Kaggle file name -> split name used everywhere in this project.
    # The public test set becomes our validation split and the private test set
    # our test split.
    "train.csv.gz": "train",
    "test_public_expanded.csv.gz": "validation",
    "test_private_expanded.csv.gz": "test",
}
SPLITS = ["train", "validation", "test"]

# Outputs
INTERIM = os.path.join(DATA, "interim")
CLEAN = os.path.join(DATA, "clean")
OUT = os.path.join(HERE, "out")          # JSON statistics read by the report
FIG = os.path.join(HERE, "figures")      # PNG figures used in the report
for d in (INTERIM, CLEAN, OUT, FIG):
    os.makedirs(d, exist_ok=True)

RAW_PARQUET = os.path.join(INTERIM, "civil_comments_full.parquet")

# Column groups
ID = "id"
TEXT = "comment_text"
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat",
          "insult", "identity_attack", "sexual_explicit"]
IDENTITIES = [
    "male", "female", "transgender", "other_gender",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]
IDENTITY_GROUPS = {
    "gender": IDENTITIES[0:4],
    "sexual orientation": IDENTITIES[4:8],
    "religion": IDENTITIES[8:15],
    "race / ethnicity": IDENTITIES[15:20],
    "disability": IDENTITIES[20:24],
}
REACTIONS = ["funny", "wow", "sad", "likes", "disagree"]
ANNOTATOR_COUNTS = ["toxicity_annotator_count", "identity_annotator_count"]
META = ["created_date", "publication_id", "parent_id", "article_id", "rating"]

THRESHOLD = 0.5   # score >= 0.5 counts as positive, as in the Kaggle competition
SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    print(f"wrote {os.path.relpath(path, PROJECT)}")


def load_json(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_parquet(path, columns=None, split=None):
    """Read a parquet file with text kept in Arrow memory (about half the RAM of
    Python string objects, which matters with 1.8 million comments)."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    filters = [("split", "==", split)] if split else None
    table = pq.read_table(path, columns=columns, filters=filters)
    mapper = {pa.string(): pd.StringDtype("pyarrow"),
              pa.large_string(): pd.StringDtype("pyarrow")}.get
    return table.to_pandas(types_mapper=mapper)


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
