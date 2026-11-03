"""Perceptron trained in chunks with balanced class weights. Number of passes picked by validation F1."""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import Perceptron

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (CHUNK, FEATURE_SETS, TARGETS, Timer, done, evaluate, features, label_column,
                         save_result, score_all)

MODEL = "perceptron"
MAX_PASSES, PATIENCE = 10, 2


def main(texts):
    for text in texts:
        X = features("train", text)
        Xv = np.asarray(features("validation", text))
        for target in TARGETS:
            if done(MODEL, text, target):
                continue
            y = label_column("train", target)
            yv = label_column("validation", target)
            n, n1 = len(y), int(y.sum())
            weights = {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
            with Timer() as t:
                model = Perceptron(class_weight=weights, shuffle=True, random_state=0)
                best_f1, best_pass, best_w, wait, history = -1.0, 0, None, 0, []
                for p in range(1, MAX_PASSES + 1):
                    for s in range(0, n, CHUNK):
                        model.partial_fit(np.asarray(X[s:s + CHUNK]), y[s:s + CHUNK], classes=[0, 1])
                    f1 = evaluate(yv, model.predict(Xv))["f1"]
                    history.append(f1)
                    if f1 > best_f1:
                        best_f1, best_pass, wait = f1, p, 0
                        best_w = (model.coef_.copy(), model.intercept_.copy())
                    else:
                        wait += 1
                        if wait >= PATIENCE:
                            break
                model.coef_, model.intercept_ = best_w
                scores = score_all(lambda b: (model.predict(b), model.decision_function(b)), text, target)
            save_result(MODEL, target, text, {"passes": best_pass, "class_weight": "balanced",
                                              "val_f1_per_pass": history}, scores, t.seconds, n_train=n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=FEATURE_SETS)
    main(ap.parse_args().features)
