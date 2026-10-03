"""
04_word2vec.py -- encode the comment text as dense document embeddings from a
Word2Vec model trained on the training comments (Lecture 8: word embedding,
skip-gram).

Why Word2Vec
    Text must be either one-hot encoded or embedded. A one-hot encoding over a
    vocabulary of tens of thousands of words is very wide and sparse, and treats
    "idiot" and "moron" as unrelated columns. Word2Vec places words that occur
    in similar contexts close together, so a comment is summarised in 100 dense
    numbers that logistic regression, KNN, an SVM or a small network can use
    directly.

Design (all fitted on the TRAINING split only)
    corpus      text_bow of every training comment (lower case, placeholders for
                links / masked words / numbers, letters only)
    model       skip-gram (sg=1), 100 dimensions, window 5, words seen at least
                5 times, 5 negative samples, 5 epochs. Skip-gram rather than CBOW
                because it learns better vectors for rare words, and most slurs
                and insults are rare.
    document    mean of the vectors of the comment's in-vocabulary words; a
                comment with no known word gets the zero vector
    scaling     every dimension standardised with the training mean and std

Output
    data/clean/w2v/{train,validation,test}.npy   float32, rows in parquet order
    data/clean/w2v/word2vec.kv                   the word vectors
    prep/out/04_word2vec.json                     settings, coverage, sanity checks

Run:  python prep/04_word2vec.py
"""
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLEAN, INTERIM, SPLITS, SEED, read_parquet, save_json, banner

DIM, WINDOW, MIN_COUNT, NEG, EPOCHS = 100, 5, 5, 5, 5
WORKERS = max(1, min(14, (os.cpu_count() or 2) - 2))
TOKEN = r"(?u)\b[a-z]+\b"
CHUNK = 200_000
W2V_DIR = os.path.join(CLEAN, "w2v")
os.makedirs(W2V_DIR, exist_ok=True)
CORPUS = os.path.join(INTERIM, "w2v_corpus_train.txt")
out = {"settings": {"vector_size": DIM, "window": WINDOW, "min_count": MIN_COUNT, "sg": 1,
                    "negative": NEG, "epochs": EPOCHS, "seed": SEED, "pooling": "mean",
                    "scaling": "z-score per dimension, train statistics"}}
t0 = time.time()

banner("1. TRAIN WORD2VEC (training split only)")
train = read_parquet(os.path.join(CLEAN, "train.parquet"), columns=["text_bow"])["text_bow"]
with open(CORPUS, "w", encoding="utf-8", newline="\n") as f:
    for s in range(0, len(train), CHUNK):
        f.write("\n".join(train.iloc[s:s + CHUNK].tolist()))
        f.write("\n")
from gensim.models import Word2Vec
model = Word2Vec(corpus_file=CORPUS, vector_size=DIM, window=WINDOW, min_count=MIN_COUNT,
                 sg=1, negative=NEG, epochs=EPOCHS, workers=WORKERS, seed=SEED)
kv = model.wv
kv.save(os.path.join(W2V_DIR, "word2vec.kv"))
out["vocabulary_size"] = len(kv.index_to_key)
out["corpus_words"] = int(model.corpus_total_words)
out["train_seconds"] = round(time.time() - t0)
print(f"vocabulary {len(kv.index_to_key):,} words from {model.corpus_total_words:,} tokens, "
      f"{out['train_seconds']}s")
del model
os.remove(CORPUS)

# Sanity check: nearest neighbours of a few words. If training worked, insults
# sit next to insults and the placeholders next to what they replaced.
probe = ["idiot", "stupid", "xxmasked", "excellent", "thanks"]
out["neighbours"] = {w: [[n, round(float(s), 3)] for n, s in kv.most_similar(w, topn=6)]
                     for w in probe if w in kv.key_to_index}
for w, nb in out["neighbours"].items():
    print(f"   {w:<10} -> {', '.join(n for n, _ in nb)}")

banner("2. DOCUMENT VECTORS (mean of word vectors)")
vocab = {w: i for i, w in enumerate(kv.index_to_key)}
E = kv.vectors.astype(np.float32)
counter = CountVectorizer(token_pattern=TOKEN, vocabulary=vocab, dtype=np.float32)


def embed(texts, dest):
    empty, known = 0, 0
    for s in range(0, len(texts), CHUNK):
        C = counter.transform(texts.iloc[s:s + CHUNK].tolist())
        n = np.asarray(C.sum(axis=1)).ravel()
        known += float(n.sum())
        empty += int((n == 0).sum())
        V = C @ E
        V[n > 0] /= n[n > 0, None]
        dest[s:s + len(V)] = V
    return empty, known


raw = {}
for split in SPLITS:
    texts = train if split == "train" else read_parquet(
        os.path.join(CLEAN, f"{split}.parquet"), columns=["text_bow"])["text_bow"]
    path = os.path.join(W2V_DIR, f"{split}.npy")
    M = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(texts), DIM))
    empty, known = embed(texts, M)
    total = float(texts.str.count(r"\S+").sum())
    raw[split] = path
    out.setdefault("coverage", {})[split] = {
        "rows": int(len(texts)), "comments_with_no_known_word": empty,
        "pct_tokens_in_vocabulary": 100 * known / total if total else 0.0}
    M.flush()
    del M
    print(f"{split:<11} {len(texts):>10,} comments, {empty:,} with no known word, "
          f"{out['coverage'][split]['pct_tokens_in_vocabulary']:.2f}% of tokens in vocabulary")
del train

banner("3. STANDARDISE WITH TRAINING STATISTICS")
Mtr = np.load(raw["train"], mmap_mode="r")
mu = np.zeros(DIM, dtype=np.float64)
sq = np.zeros(DIM, dtype=np.float64)
for s in range(0, Mtr.shape[0], CHUNK):
    blk = np.asarray(Mtr[s:s + CHUNK], dtype=np.float64)
    mu += blk.sum(axis=0)
    sq += (blk ** 2).sum(axis=0)
n = Mtr.shape[0]
mu /= n
sd = np.sqrt(sq / n - mu ** 2)
del Mtr
np.save(os.path.join(W2V_DIR, "scaler_mean.npy"), mu.astype(np.float32))
np.save(os.path.join(W2V_DIR, "scaler_std.npy"), sd.astype(np.float32))
for split in SPLITS:
    M = np.load(raw[split], mmap_mode="r+")
    for s in range(0, M.shape[0], CHUNK):
        M[s:s + CHUNK] = ((M[s:s + CHUNK] - mu) / sd).astype(np.float32)
    M.flush()
    out.setdefault("files", {})[f"{split}.npy"] = os.path.getsize(raw[split])
    del M
out["raw_dimension_std_range"] = [float(sd.min()), float(sd.max())]
chk = np.load(raw["train"], mmap_mode="r")[:200_000]
out["check_train_first_200k"] = {"mean_abs_mean": float(np.abs(chk.mean(axis=0)).mean()),
                                 "mean_std": float(chk.std(axis=0).mean())}
print("after scaling, first 200k training rows:", out["check_train_first_200k"])
out["seconds"] = round(time.time() - t0)
save_json(out, "04_word2vec.json")
