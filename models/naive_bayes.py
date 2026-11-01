"""
Gaussian Naive Bayes (Lecture 11), one model per label. The features are continuous, so
each feature gets a normal distribution per class. Fitted in chunks with partial_fit.

Parameter choice: the class priors, either learned from the data or uniform (0.5 / 0.5,
which plays the role of balanced class weights); the better validation F1 wins.

Run:  python models/naive_bayes.py [--features word2vec doc2vec]
"""
import argparse
import os
import sys

import numpy as np
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (CHUNK, FEATURE_SETS, TARGETS, Timer, done, evaluate, features, label_column,
                         save_result, score_all)

MODEL = "naive_bayes"


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
                model = GaussianNB()
                for s in range(0, len(y), CHUNK):
                    model.partial_fit(np.asarray(X[s:s + CHUNK]), y[s:s + CHUNK], classes=[0, 1])
                learned = model.class_prior_.copy()
                tried = {}
                for name, prior in {"learned": learned, "uniform": np.array([0.5, 0.5])}.items():
                    model.class_prior_ = prior
                    tried[name] = evaluate(yv, model.predict(Xv))["f1"]
                best = max(tried, key=tried.get)
                model.class_prior_ = learned if best == "learned" else np.array([0.5, 0.5])
                scores = score_all(lambda b: (model.predict(b), model.predict_joint_log_proba(b)[:, 1]
                                              - model.predict_joint_log_proba(b)[:, 0]), text, target)
            save_result(MODEL, target, text, {"priors": best, "tried_val_f1": tried}, scores, t.seconds, n_train=len(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=FEATURE_SETS)
    main(ap.parse_args().features)
