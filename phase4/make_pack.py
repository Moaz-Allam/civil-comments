"""
Builds the data pack for the Phase 4 MLP search on Colab (upload data/colab_pack/ to Drive).

Train and validation are merged into one pool. Each pool row gets a fold (0-4) from
StratifiedGroupKFold: comments of one article stay in one fold, and the folds are stratified
on the rarest positive label of each comment. Test is packed separately and used once.

Features per row: 33 tabular | 100 Word2Vec | 100 PV-DBOW, float16.

    python phase4/make_pack.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import CLEAN, ROOT, TARGETS, features, labels, tabular_columns

OUT = os.path.join(ROOT, "data", "colab_pack")
FOLDS = 5
IDENTITIES = ["male", "female", "transgender", "other_gender", "heterosexual", "homosexual_gay_or_lesbian",
              "bisexual", "other_sexual_orientation", "christian", "jewish", "muslim", "hindu", "buddhist",
              "atheist", "other_religion", "black", "white", "asian", "latino", "other_race_or_ethnicity",
              "physical_disability", "intellectual_or_learning_disability", "psychiatric_or_mental_illness",
              "other_disability"]


def pack_features(split):
    w2v = features(split, "word2vec")
    dbow = features(split, "doc2vec_dbow")
    n_tab = len(tabular_columns())
    X = np.empty((w2v.shape[0], n_tab + 200), dtype=np.float16)
    for s in range(0, len(X), 200_000):
        X[s:s + 200_000, :n_tab + 100] = w2v[s:s + 200_000]
        X[s:s + 200_000, n_tab + 100:] = dbow[s:s + 200_000, n_tab:]
    return X


def identities(split):
    t = pq.read_table(os.path.join(CLEAN, f"{split}.parquet"), columns=["id", "identity_annotated"] + IDENTITIES)
    return t.to_pandas()


def rarest_label_key(Y, rates):
    """Index of the rarest label a comment carries (len(TARGETS) = no label)."""
    key = np.full(len(Y), len(TARGETS), dtype=np.int8)
    for j in np.argsort(rates)[::-1]:          # common first, so rarer labels overwrite
        key[Y[:, j] == 1] = j
    return key


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = ["train", "validation"]
    Y = np.concatenate([np.asarray(labels(s)) for s in pool])
    ident = pd.concat([identities(s) for s in pool], ignore_index=True)
    article = pq.read_table(os.path.join(ROOT, "data", "interim", "civil_comments_full.parquet"),
                            columns=["id", "article_id"]).to_pandas().set_index("id").article_id
    groups = ident.id.map(article).to_numpy()
    assert not np.isnan(groups).any()

    rates = Y.mean(axis=0)
    key = rarest_label_key(Y, rates)
    fold = np.empty(len(Y), dtype=np.int8)
    for f, (_, te) in enumerate(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0).split(Y, key, groups)):
        fold[te] = f

    np.save(os.path.join(OUT, "X_pool.npy"), np.concatenate([pack_features(s) for s in pool]))
    np.save(os.path.join(OUT, "y_pool.npy"), Y.astype(np.uint8))
    np.save(os.path.join(OUT, "fold_pool.npy"), fold)
    np.save(os.path.join(OUT, "ident_pool.npy"), ident[IDENTITIES].to_numpy(np.float32, na_value=np.nan).astype(np.float16))
    np.save(os.path.join(OUT, "annotated_pool.npy"), ident.identity_annotated.to_numpy().astype(np.uint8))

    test_ident = identities("test")
    np.save(os.path.join(OUT, "X_test.npy"), pack_features("test"))
    np.save(os.path.join(OUT, "y_test.npy"), np.asarray(labels("test")).astype(np.uint8))
    np.save(os.path.join(OUT, "ident_test.npy"), test_ident[IDENTITIES].to_numpy(np.float32, na_value=np.nan).astype(np.float16))

    n_tab = len(tabular_columns())
    meta = {
        "targets": TARGETS, "identities": IDENTITIES, "folds": FOLDS,
        "rows": {"pool": int(len(Y)), "test": int(len(test_ident))},
        "articles": int(pd.Series(groups).nunique()),
        "columns": {"tabular": [0, n_tab], "word2vec": [n_tab, n_tab + 100], "doc2vec_dbow": [n_tab + 100, n_tab + 200]},
        "positive_rate": dict(zip(TARGETS, rates.round(6).tolist())),
        "fold_rows": np.bincount(fold).tolist(),
        "fold_positives": {t: np.bincount(fold, weights=Y[:, j]).astype(int).tolist() for j, t in enumerate(TARGETS)},
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
