"""
Loading the Phase 2/3 features for every model.

X = 33 scaled tabular features + 100 text dimensions (Word2Vec or Doc2Vec), float32.
The first call builds one file per feature set and split under data/clean/cache; after
that X opens as a memory map, so nothing is held in RAM until it is read.

    python -m common.data        builds every cache file that is possible so far
"""
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from common.evaluate import evaluate
from common.sample import stratified_indices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")
CACHE = os.path.join(CLEAN, "cache")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

TARGETS = ["toxicity", "insult", "identity_attack", "obscene", "threat", "sexual_explicit"]
SPLITS = ["train", "validation", "test"]
TEXT_DIR = {"word2vec": "w2v", "doc2vec": "doc2vec", "doc2vec_dbow": "doc2vec_dbow"}
FEATURE_SETS = ["word2vec", "doc2vec"]      # the default runs; doc2vec_dbow is asked for by name
CHUNK = 100_000
EVAL_TRAIN_ROWS = 300_000          # training metrics are read on a sample of this size


def tabular_columns():
    params = json.load(open(os.path.join(CLEAN, "preprocessing_params.json"), encoding="utf8"))
    return params["feature_columns"]


def _build_labels(split):
    path = os.path.join(CACHE, f"y_{split}.npy")
    if not os.path.exists(path):
        t = pq.read_table(os.path.join(CLEAN, f"{split}.parquet"), columns=[f"y_{c}" for c in TARGETS])
        Y = np.column_stack([t.column(i).to_numpy() for i in range(len(TARGETS))]).astype(np.uint8)
        np.save(path, Y)
    return path


def _build_features(split, text):
    path = os.path.join(CACHE, f"X_{text}_{split}.npy")
    if os.path.exists(path):
        return path
    columns = tabular_columns()
    T = np.load(os.path.join(CLEAN, TEXT_DIR[text], f"{split}.npy"), mmap_mode="r")
    tmp = path + ".tmp.npy"
    X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(T.shape[0], len(columns) + T.shape[1]))
    row = 0
    source = pq.ParquetFile(os.path.join(CLEAN, f"{split}.parquet"))
    for batch in source.iter_batches(batch_size=CHUNK, columns=columns):
        n = batch.num_rows
        for j, c in enumerate(columns):
            X[row:row + n, j] = batch.column(batch.schema.get_field_index(c)).to_numpy().astype(np.float32)
        X[row:row + n, len(columns):] = T[row:row + n]
        row += n
    assert row == T.shape[0]
    X.flush()
    del X
    os.replace(tmp, path)
    return path


def features(split, text):
    return np.load(_build_features(split, text), mmap_mode="r")


def labels(split):
    return np.load(_build_labels(split), mmap_mode="r")


def label_column(split, target):
    return np.asarray(labels(split)[:, TARGETS.index(target)])


def result_path(model, text, target):
    return os.path.join(OUTPUTS, f"{model}_{text}_{target}.json")


def save_result(model, target, text, params, scores, seconds, seed=0, n_train=None, sampled=False, notes=None):
    """scores = {"train": evaluate(...), "validation": ..., "test": ...}"""
    record = {"model": model, "label": target, "features": text, "params": params, **scores,
              "seed": seed, "n_train": n_train, "sampled": sampled, "seconds": round(seconds, 1), "notes": notes}
    with open(result_path(model, text, target), "w", encoding="utf8") as f:
        json.dump(record, f, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
    v = scores["validation"]
    print(f"{model:<22} {text:<9} {target:<16} val F1 {v['f1']:.4f}  P {v['precision']:.3f}  R {v['recall']:.3f}  "
          f"AUC {v['roc_auc'] if v['roc_auc'] is None else round(v['roc_auc'], 4)}  acc {v['accuracy']:.4f}  "
          f"{seconds:.0f}s  {params}", flush=True)


def score_split(predict, split, text, target):
    """predict(X_block) -> (0/1 predictions, ranking score); runs in chunks to save memory."""
    X = features(split, text)
    y = label_column(split, target)
    rows = stratified_indices(y, EVAL_TRAIN_ROWS, 0) if split == "train" else np.arange(len(y))
    preds, ranks = [], []
    for s in range(0, len(rows), CHUNK):
        p, r = predict(np.asarray(X[rows[s:s + CHUNK]]))
        preds.append(p)
        if r is not None:
            ranks.append(r)
    return evaluate(y[rows], np.concatenate(preds), np.concatenate(ranks) if ranks else None)


def score_all(predict, text, target):
    return {split: score_split(predict, split, text, target) for split in SPLITS}


def done(model, text, target):
    return os.path.exists(result_path(model, text, target))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0


if __name__ == "__main__":
    for split in SPLITS:
        _build_labels(split)
        for text in TEXT_DIR:
            if os.path.exists(os.path.join(CLEAN, TEXT_DIR[text], f"{split}.npy")):
                print("built", _build_features(split, text), flush=True)
