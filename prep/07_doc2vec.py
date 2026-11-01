"""
07_doc2vec.py -- Phase 3 text features: one Doc2Vec (PV-DM) vector per comment,
as suggested in the Phase 2 feedback, to compare with the mean of Word2Vec vectors.

The model is trained on the training comments only. All three splits then get
their vectors the same way, through infer_vector. The vectors stored for training
documents during training are not used as features, because they come from a
different process than the vectors a new comment gets, and a classifier fitted on
one would be tested on the other. The script measures that gap and reports it.

The vectors are standardised with training statistics, and every dimension is
checked against the |r| >= 0.85 leakage rule.

Output
    data/clean/doc2vec/{split}.npy       float32, 100 dimensions, rows in parquet order
    data/clean/doc2vec/doc2vec.model     the fitted model
    data/clean/doc2vec/scaler_*.npy      training mean and std
    prep/out/07_doc2vec.json             settings, timings, checks

Run:  python prep/07_doc2vec.py                     (full run)
      python prep/07_doc2vec.py --limit 50000 --epochs 2 --out doc2vec_smoke
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLEAN, INTERIM, SEED, SPLITS, save_json, banner

DIM, WINDOW, MIN_COUNT, NEG = 100, 5, 5, 5
TARGETS = ["toxicity", "insult", "identity_attack", "obscene", "threat", "sexual_explicit"]
EMPTY = "xxempty"                  # a comment with no words still needs its own line
BLOCK = 20_000                     # comments per inference task
LEAK = 0.85
_model = None


def write_corpus(split, path, limit=None):
    """One comment per line; returns the row count and the byte offset of each block."""
    rows, offsets, pos = 0, [0], 0
    source = pq.ParquetFile(os.path.join(CLEAN, f"{split}.parquet"))
    with open(path, "wb") as f:
        for batch in source.iter_batches(batch_size=50_000, columns=["text_bow"]):
            for s in batch.column(0).to_pylist():
                if limit is not None and rows >= limit:
                    return rows, offsets
                b = ((s.strip() or EMPTY) + "\n").encode("utf8")
                f.write(b)
                pos += len(b)
                rows += 1
                if rows % BLOCK == 0:
                    offsets.append(pos)
    return rows, offsets


def init_worker(model_path):
    global _model
    from gensim.models.doc2vec import Doc2Vec
    _model = Doc2Vec.load(model_path, mmap="r")


def infer_block(task):
    corpus, start_byte, row0, nrows, dest, epochs = task
    out = np.lib.format.open_memmap(dest, mode="r+")
    with open(corpus, "rb") as f:
        f.seek(start_byte)
        for i in range(nrows):
            words = f.readline().decode("utf8").split()
            out[row0 + i] = _model.infer_vector(words, epochs=epochs)
    out.flush()
    return nrows


def max_label_corr(path, limit):
    """Largest |r| between any dimension and any target, computed in chunks."""
    table = pq.read_table(os.path.join(CLEAN, "train.parquet"), columns=[f"y_{t}" for t in TARGETS])
    Y = np.column_stack([table.column(i).to_numpy() for i in range(len(TARGETS))])[:limit].astype(np.float64)
    del table
    M = np.load(path, mmap_mode="r")
    n, d = M.shape
    s1, s2, sxy = np.zeros(d), np.zeros(d), np.zeros((d, Y.shape[1]))
    for s in range(0, n, 200_000):
        X = np.asarray(M[s:s + 200_000], dtype=np.float64)
        s1 += X.sum(0)
        s2 += (X ** 2).sum(0)
        sxy += X.T @ Y[s:s + 200_000]
    mx, sx = s1 / n, np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12))
    r = (sxy / n - mx[:, None] * Y.mean(0)[None, :]) / (sx[:, None] * Y.std(0)[None, :])
    i, j = np.unravel_index(np.abs(r).argmax(), r.shape)
    return {"max_abs_r": float(abs(r[i, j])), "dimension": int(i), "target": TARGETS[j]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows of each split")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dm", type=int, default=1, help="1 = PV-DM, 0 = PV-DBOW")
    ap.add_argument("--out", default="doc2vec", help="folder under data/clean")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--infer-procs", type=int, default=6)
    a = ap.parse_args()

    from gensim.models.doc2vec import Doc2Vec
    OUT = os.path.join(CLEAN, a.out)
    os.makedirs(OUT, exist_ok=True)
    res = {"settings": {"vector_size": DIM, "window": WINDOW, "min_count": MIN_COUNT, "negative": NEG,
                        "epochs": a.epochs, "dm": a.dm, "seed": SEED, "workers": a.workers,
                        "limit": a.limit, "scaling": "z-score per dimension, train statistics"}}

    banner("1. CORPUS FILES")
    t0 = time.time()
    corpora = {}
    for split in SPLITS:
        path = os.path.join(INTERIM, f"d2v_{a.out}_{split}.txt")
        n, offsets = write_corpus(split, path, a.limit)
        corpora[split] = (path, n, offsets)
        print(f"{split:<11} {n:>10,} comments  {os.path.getsize(path) / 1e6:,.0f} MB", flush=True)
    res["corpus_seconds"] = round(time.time() - t0)

    banner("2. TRAIN DOC2VEC (training split only)")
    t0 = time.time()
    model = Doc2Vec(corpus_file=corpora["train"][0], vector_size=DIM, window=WINDOW, min_count=MIN_COUNT,
                    negative=NEG, dm=a.dm, epochs=a.epochs, workers=a.workers, seed=SEED)
    assert model.corpus_count == corpora["train"][1], (model.corpus_count, corpora["train"][1])
    res["train_seconds"] = round(time.time() - t0)
    res["vocabulary_size"] = len(model.wv)
    res["corpus_words"] = int(model.corpus_total_words)
    print(f"vocabulary {len(model.wv):,} words, {model.corpus_total_words:,} tokens, {res['train_seconds']}s", flush=True)
    res["neighbours"] = {w: [[n, round(float(s), 3)] for n, s in model.wv.most_similar(w, topn=5)]
                         for w in ["idiot", "stupid", "thanks"] if w in model.wv.key_to_index}
    print(res["neighbours"], flush=True)
    stored_sample = np.array(model.dv.vectors[:2000], dtype=np.float32)
    model_path = os.path.join(OUT, "doc2vec.model")
    model.save(model_path, separately=["vectors", "syn1neg"])
    del model

    banner("3. INFER EVERY SPLIT THE SAME WAY")
    t0 = time.time()
    dests, tasks = {}, []
    for split in SPLITS:
        corpus, n, offsets = corpora[split]
        dest = os.path.join(OUT, f"{split}.npy")
        m = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32, shape=(n, DIM))
        m.flush()
        del m
        dests[split] = dest
        for k, byte in enumerate(offsets):
            row0 = k * BLOCK
            if row0 < n:
                tasks.append((corpus, byte, row0, min(BLOCK, n - row0), dest, a.epochs))
    done, total = 0, sum(t[3] for t in tasks)
    with mp.get_context("spawn").Pool(a.infer_procs, initializer=init_worker, initargs=(model_path,)) as pool:
        for r in pool.imap_unordered(infer_block, tasks):
            done += r
            print(f"   inferred {done:,} / {total:,}  ({time.time() - t0:.0f}s)", flush=True)
    res["infer_seconds"] = round(time.time() - t0)
    res["infer_comments_per_second"] = round(total / max(time.time() - t0, 1))

    banner("4. GAP BETWEEN STORED AND INFERRED TRAINING VECTORS")
    inferred = np.array(np.load(dests["train"], mmap_mode="r")[:2000])
    na, nb = np.linalg.norm(stored_sample, axis=1), np.linalg.norm(inferred, axis=1)
    res["stored_vs_inferred"] = {"mean_cosine": float(((stored_sample * inferred).sum(1) / (na * nb + 1e-9)).mean()),
                                 "mean_norm_stored": float(na.mean()), "mean_norm_inferred": float(nb.mean())}
    print(res["stored_vs_inferred"], flush=True)

    banner("5. STANDARDISE WITH TRAINING STATISTICS")
    M = np.load(dests["train"], mmap_mode="r")
    n = M.shape[0]
    s1, s2 = np.zeros(DIM), np.zeros(DIM)
    for s in range(0, n, 200_000):
        b = np.asarray(M[s:s + 200_000], dtype=np.float64)
        s1 += b.sum(0)
        s2 += (b ** 2).sum(0)
    del M
    mu, sd = s1 / n, np.sqrt(s2 / n - (s1 / n) ** 2)
    np.save(os.path.join(OUT, "scaler_mean.npy"), mu.astype(np.float32))
    np.save(os.path.join(OUT, "scaler_std.npy"), sd.astype(np.float32))
    for split in SPLITS:
        M = np.load(dests[split], mmap_mode="r+")
        for s in range(0, M.shape[0], 200_000):
            M[s:s + 200_000] = ((M[s:s + 200_000] - mu) / sd).astype(np.float32)
        M.flush()
        del M

    banner("6. LEAKAGE RULE")
    res["leakage"] = max_label_corr(dests["train"], corpora["train"][1])
    res["leakage"]["passes"] = bool(res["leakage"]["max_abs_r"] < LEAK)
    print(res["leakage"], flush=True)
    for split in SPLITS:
        os.remove(corpora[split][0])
    save_json(res, f"07_{a.out}.json")
    assert res["leakage"]["passes"], "a Doc2Vec dimension reaches the leakage threshold"


if __name__ == "__main__":
    main()
