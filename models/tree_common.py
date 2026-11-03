"""
Shared code for the ID3 and CART trees. max_depth is picked on a 200k sample, then the tree is fitted
on all rows (about 16 GB of RAM) or on a sample (--final-rows).
"""
import argparse
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import FEATURE_SETS, TARGETS, Timer, done, evaluate, features, label_column, save_result, score_all
from common.sample import stratified_indices

DEPTHS = [5, 10, 15, 20]


def build(criterion, depth):
    return DecisionTreeClassifier(criterion=criterion, max_depth=depth, class_weight="balanced", random_state=0)


def fit_one(model_name, criterion, text, target, X, Xfull, Xv, tune_rows, final_rows):
    y = label_column("train", target)
    yv = label_column("validation", target)
    with Timer() as t:
        idx = stratified_indices(y, tune_rows, 0)
        Xs, ys = np.asarray(X[idx]), y[idx]
        tried = {}
        for depth in DEPTHS:
            m = build(criterion, depth).fit(Xs, ys)
            tried[depth] = evaluate(yv, m.predict(Xv))["f1"]
        best = max(tried, key=tried.get)
        del Xs, ys
        if Xfull is not None:
            model = build(criterion, best).fit(Xfull, y)
            used = len(y)
        else:
            idx = stratified_indices(y, final_rows, 0)
            model = build(criterion, best).fit(np.asarray(X[idx]), y[idx])
            used = len(idx)
        scores = score_all(lambda b: (model.predict(b), model.predict_proba(b)[:, 1]), text, target)
    save_result(model_name, target, text,
                {"criterion": criterion, "max_depth": best, "class_weight": "balanced", "tried_val_f1": tried,
                 "tune_rows": tune_rows, "nodes": int(model.tree_.node_count)},
                scores, t.seconds, n_train=used, sampled=used < len(y))


def run(model_name, criterion, texts, tune_rows, final_rows, targets, jobs):
    for text in texts:
        todo = [t for t in targets if not done(model_name, text, t)]
        if not todo:
            continue
        X = features("train", text)
        Xv = np.asarray(features("validation", text))
        Xfull = np.array(X) if final_rows == 0 else None
        Parallel(n_jobs=jobs, prefer="threads")(
            delayed(fit_one)(model_name, criterion, text, t, X, Xfull, Xv, tune_rows, final_rows) for t in todo)
        del Xfull


def cli(model_name, criterion):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=FEATURE_SETS)
    ap.add_argument("--tune-rows", type=int, default=200_000)
    ap.add_argument("--final-rows", type=int, default=0, help="0 = all training rows")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--jobs", type=int, default=6, help="labels fitted at the same time")
    a = ap.parse_args()
    run(model_name, criterion, a.features, a.tune_rows, a.final_rows, a.targets, a.jobs)
