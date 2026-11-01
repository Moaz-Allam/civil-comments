"""
Logistic regression (Lecture 5), one model per label, balanced class weights (Lab 2).

Parameter choice: C is chosen by validation F1 on a stratified 200,000-row sample, then the
model is refitted with that C on all training rows. A 5-fold search on all rows would need
25 full fits for each label and feature set.

Run:  python models/logistic_regression.py [--features word2vec doc2vec]
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (FEATURE_SETS, TARGETS, Timer, done, evaluate, features, label_column, save_result,
                         score_all)
from common.sample import stratified_indices

MODEL = "logistic_regression"
GRID = [0.01, 0.1, 1, 10, 100]
TUNE_ROWS = 200_000


def build(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=200, solver="lbfgs")


def main(texts):
    for text in texts:
        X = features("train", text)
        Xv = np.asarray(features("validation", text))
        for target in TARGETS:
            if done(MODEL, text, target):
                continue
            y = label_column("train", target)
            yv = label_column("validation", target)
            with Timer() as t:
                idx = stratified_indices(y, TUNE_ROWS, 0)
                Xs, ys = np.asarray(X[idx]), y[idx]
                tried = {}
                for C in GRID:
                    m = build(C).fit(Xs, ys)
                    tried[C] = evaluate(yv, m.predict(Xv))["f1"]
                best = max(tried, key=tried.get)
                model = build(best).fit(X, y)
                scores = score_all(lambda b: (model.predict(b), model.decision_function(b)), text, target)
            save_result(MODEL, target, text, {"C": best, "class_weight": "balanced", "tried_val_f1": tried,
                                              "iterations": int(model.n_iter_[0])}, scores, t.seconds, n_train=len(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=FEATURE_SETS)
    main(ap.parse_args().features)
