"""
Linear regression (Lecture 4) used as a classifier: fitted on the 0/1 label, positive
when the output is at least 0.5 (a linear probability model).

The normal equations W = (X'X)^-1 X'y are built in chunks, so all 1.78M rows fit in
memory. The publication one-hot columns sum to 1, like the intercept, so X'X is singular;
the pseudo-inverse gives the same minimum-norm solution as scikit-learn's LinearRegression.

Parameter choice: unweighted, or with balanced class weights; the better validation F1 wins.

Run:  python models/linear_regression.py [--features word2vec doc2vec]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (CHUNK, FEATURE_SETS, TARGETS, Timer, done, evaluate, features, label_column,
                         labels, save_result, score_all)

MODEL = "linear_regression"


def with_intercept(X):
    return np.hstack([X.astype(np.float64), np.ones((X.shape[0], 1))])


def gram_matrices(X, Y):
    """X'X and X'y over all rows, plus the same two sums over each target's positive rows."""
    d = X.shape[1] + 1
    k = Y.shape[1]
    A, b = np.zeros((d, d)), np.zeros((d, k))
    A1, b1 = np.zeros((k, d, d)), np.zeros((k, d))
    for s in range(0, X.shape[0], CHUNK):
        B = with_intercept(np.asarray(X[s:s + CHUNK]))
        y = np.asarray(Y[s:s + CHUNK]).astype(np.float64)
        A += B.T @ B
        b += B.T @ y
        for j in range(k):
            P = B[y[:, j] == 1]
            A1[j] += P.T @ P
            b1[j] += P.sum(axis=0)
    return A, b, A1, b1


def solve(A, b):
    return np.linalg.pinv(A, rcond=1e-12, hermitian=True) @ b


def main(texts):
    Y = labels("train")
    n = Y.shape[0]
    for text in texts:
        if all(done(MODEL, text, t) for t in TARGETS):
            continue
        with Timer() as gram_time:
            A, b, A1, b1 = gram_matrices(features("train", text), Y)
        Xv = with_intercept(np.asarray(features("validation", text)))
        for j, target in enumerate(TARGETS):
            yv = label_column("validation", target)
            with Timer() as fit_time:
                n1 = float(Y[:, j].sum())
                w1, w0 = n / (2 * n1), n / (2 * (n - n1))     # scikit-learn's "balanced" weights
                options = {
                    "none": solve(A, b[:, j]),
                    "balanced": solve(w0 * (A - A1[j]) + w1 * A1[j], w1 * b1[j]),
                }
                tried = {}
                for name, w in options.items():
                    out = Xv @ w
                    tried[name] = evaluate(yv, (out >= 0.5).astype(int), out)["f1"]
                best = max(tried, key=tried.get)
                w = options[best]

                def predict(block):
                    out = with_intercept(block) @ w
                    return (out >= 0.5).astype(int), out

                scores = score_all(predict, text, target)
            save_result(MODEL, target, text, {"class_weight": best, "tried_val_f1": tried}, scores,
                        gram_time.seconds / len(TARGETS) + fit_time.seconds, n_train=n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=FEATURE_SETS)
    main(ap.parse_args().features)
