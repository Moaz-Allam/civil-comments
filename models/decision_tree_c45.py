"""Decision tree, C4.5 entry: entropy tree with validation-chosen pruning."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from model_common import Timer, done, evaluate, label_column, load_split, load_train, parser, proba, save_result, \
    score_all

MODEL = "decision_tree_c45"


# entropy tree with balanced class weights
def build(alpha, depth):
    return DecisionTreeClassifier(criterion="entropy", class_weight="balanced", max_depth=depth, ccp_alpha=alpha,
                                  random_state=0)


# pruning levels to try
def pick_alphas(path, k):
    a = np.unique(path[path > 0])[:-1]
    if len(a) <= k:
        return [0.0] + a.tolist()
    pos = np.searchsorted(a, np.geomspace(a[0], a[-1], k)).clip(0, len(a) - 1)
    return [0.0] + np.unique(a[pos]).tolist()


# choose alpha on a sample, fit the final tree, save metrics
def run_one(text, target, Xv, tune_rows, final_rows, depth, n_alphas, jobs):
    yv = label_column("validation", target)
    with Timer() as t:
        X, y, _ = load_train(text, target, tune_rows)  # 300k sample for tuning
        path = build(0.0, depth).cost_complexity_pruning_path(X, y).ccp_alphas  # all pruning levels

        def fit(alpha):
            pv = build(alpha, depth).fit(X, y).predict(Xv)
            return alpha, float((pv == yv).mean()), evaluate(yv, pv)["f1"]

        fits = Parallel(n_jobs=jobs, prefer="threads")(delayed(fit)(a) for a in pick_alphas(path, n_alphas))
        alpha = max(fits, key=lambda f: (f[1], f[0]))[0]  # best accuracy, ties -> more pruning
        tried_acc = {f"{f[0]:.3g}": round(f[1], 5) for f in fits}
        tried_f1 = {f"{f[0]:.3g}": round(f[2], 5) for f in fits}
        del X
        X, y, idx = load_train(text, target, final_rows)  # all training rows
        model = build(alpha, depth).fit(X, y)  # final pruned tree
        del X
        scores = score_all(proba(model), text, target)  # metrics on train/val/test
    save_result(MODEL, target, text,  # write outputs/*.json
                {"criterion": "entropy", "class_weight": "balanced", "max_depth": depth, "ccp_alpha": alpha,
                 "tune_rows": tune_rows, "path_alphas": int(len(path)), "tried_val_accuracy": tried_acc,
                 "tried_val_f1": tried_f1, "nodes": int(model.tree_.node_count),
                 "leaves": int(model.get_n_leaves()), "depth": int(model.get_depth())},
                scores, t.seconds, n_train=len(idx), sampled=len(idx) < len(label_column("train", target)))


def main():
    ap = parser(__doc__)
    ap.add_argument("--tune-rows", type=int, default=300_000)
    ap.add_argument("--final-rows", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--alphas", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    for text in a.features:
        Xv = load_split("validation", text)
        for target in a.targets:
            if not done(MODEL, text, target):
                run_one(text, target, Xv, a.tune_rows, a.final_rows, a.max_depth, a.alphas, a.jobs)


if __name__ == "__main__":
    main()
