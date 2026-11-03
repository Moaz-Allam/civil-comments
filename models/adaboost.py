"""AdaBoost with 100 decision stumps."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from model_common import Timer, done, label_column, load_train, parser, proba, save_result, score_all, \
    stratified_indices

MODEL = "adaboost"


# boost stumps for one label, save metrics
def run_one(text, target, X, rows):
    y = label_column("train", target)
    with Timer() as t:
        idx = stratified_indices(y, rows, 0) if rows and rows < len(y) else np.arange(len(y))  # 300k stratified rows
        Xs = X[idx] if len(idx) < len(y) else X
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100,  # 100 boosted stumps
                                   random_state=0).fit(Xs, y[idx])
        del Xs
        scores = score_all(proba(model), text, target)  # metrics on train/val/test
    save_result(MODEL, target, text,
                {"estimator": "DecisionTreeClassifier(max_depth=1)", "n_estimators": 100},
                scores, t.seconds, n_train=len(idx), sampled=len(idx) < len(y))


def main():
    ap = parser(__doc__)
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    for text in a.features:
        todo = [t for t in a.targets if not done(MODEL, text, t)]
        if todo:
            X, _, _ = load_train(text, todo[0], 0)
            Parallel(n_jobs=a.jobs, prefer="threads")(delayed(run_one)(text, t, X, a.rows) for t in todo)  # labels in parallel
            del X


if __name__ == "__main__":
    main()
