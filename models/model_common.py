"""Helpers shared by decision_tree_c45, random_forest, svm, mlp, adaboost and knn."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (CHUNK, EVAL_TRAIN_ROWS, FEATURE_SETS, SPLITS, TARGETS, Timer, done, evaluate,  # noqa: F401
                         features, label_column, save_result)
from common.sample import stratified_indices  # noqa: F401
import common.data as _data

# use clean/ at the project root when data/clean/ is not set up
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOP = os.path.join(_ROOT, "clean")
if not os.path.exists(os.path.join(_data.CLEAN, "preprocessing_params.json")) and os.path.isdir(_TOP):
    _data.CLEAN = _TOP
    _data.CACHE = os.path.join(_TOP, "cache")
    os.makedirs(_data.CACHE, exist_ok=True)


def missing_files(texts):
    """Files still needed for these feature sets (a built cache file counts as present)."""
    clean, cache = _data.CLEAN, _data.CACHE
    have = lambda *p: os.path.exists(os.path.join(*p))
    out = []
    for split in SPLITS:
        if not have(cache, f"y_{split}.npy") and not have(clean, f"{split}.parquet"):
            out.append(f"{split}.parquet  (labels)")
        for text in texts:
            if have(cache, f"X_{text}_{split}.npy"):
                continue
            if not have(clean, f"{split}.parquet"):
                out.append(f"{split}.parquet  (tabular features)")
            if not have(clean, _data.TEXT_DIR[text], f"{split}.npy"):
                out.append(f"{_data.TEXT_DIR[text]}/{split}.npy")
    return list(dict.fromkeys(out))


def require(texts):
    miss = missing_files(texts)
    if miss:
        print(f"Data folder: {_data.CLEAN}\nMissing files for {', '.join(texts)}:")
        for m in miss:
            print("  -", m)
        sys.exit(1)


def load_train(text, target, n, seed=0):
    """Training rows in RAM as float32. n = 0 means all rows, otherwise a stratified sample."""
    y = label_column("train", target)
    X = features("train", text)
    if not n or n >= len(y):
        return np.array(X, dtype=np.float32), y, np.arange(len(y))
    idx = stratified_indices(y, n, seed)
    return np.asarray(X[idx], dtype=np.float32), y[idx], idx


def load_split(split, text):
    return np.array(features(split, text), dtype=np.float32)


class _Rows:
    def __init__(self, X, rows):
        self.X, self.rows = X, np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, sl):
        return np.asarray(self.X[self.rows[sl]])


def predict_chunks(predict, X):
    preds, ranks = [], []
    for s in range(0, len(X), CHUNK):
        p, r = predict(np.asarray(X[s:s + CHUNK]))
        preds.append(np.asarray(p).astype(np.uint8))
        if r is not None:
            ranks.append(np.asarray(r))
    return np.concatenate(preds), (np.concatenate(ranks) if ranks else None)


def score_all(predict, text, target, train_idx=None):
    """Metrics on train, validation and test.
    Train metrics use train_idx if given, otherwise the usual 300k stratified rows."""
    out = {}
    for split in SPLITS:
        y = label_column(split, target)
        X = features(split, text)
        if split == "train":
            rows = train_idx if train_idx is not None else stratified_indices(y, EVAL_TRAIN_ROWS, 0)
            p, r = predict_chunks(predict, _Rows(X, rows))
            out[split] = evaluate(y[rows], p, r)
        else:
            p, r = predict_chunks(predict, X)
            out[split] = evaluate(y, p, r)
    return out


def proba(model):
    return lambda b: (model.predict(b), model.predict_proba(b)[:, 1])


def decision(model):
    return lambda b: (model.predict(b), model.decision_function(b))


class _Parser(argparse.ArgumentParser):
    def parse_args(self, *args, **kw):
        a = super().parse_args(*args, **kw)
        require(a.features)
        return a


def parser(doc):
    ap = _Parser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", default=FEATURE_SETS)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    return ap


if __name__ == "__main__":
    # python models/model_common.py  -> lists what is missing for each feature set
    for text in FEATURE_SETS:
        miss = missing_files([text])
        print(f"{text}: " + ("ready" if not miss else "missing " + ", ".join(miss)))
    print("data folder:", _data.CLEAN)
