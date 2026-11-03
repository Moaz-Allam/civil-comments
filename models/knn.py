"""KNN: k chosen by validation F1."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from model_common import Timer, done, evaluate, label_column, load_split, load_train, parser, save_result, \
    stratified_indices

MODEL = "knn"
KS = [1, 3, 5, 9, 15, 21]
TRAIN_EVAL_ROWS = 50_000


# share of positive labels among the k nearest neighbours
def votes(nn, ys, k):
    return ys[nn[:, :k]].mean(axis=1)


# find neighbours, choose k, save metrics
def run_one(text, target, Xv, Xt, rows):
    yv, yt = label_column("validation", target), label_column("test", target)
    with Timer() as t:
        X, ys, idx = load_train(text, target, rows)  # 150k stratified sample
        knn = KNeighborsClassifier(n_neighbors=max(KS), n_jobs=-1).fit(X, ys)
        nn_v = knn.kneighbors(Xv, return_distance=False)  # 21 neighbours per validation row
        tried = {k: evaluate(yv, (votes(nn_v, ys, k) > 0.5).astype(np.uint8))["f1"] for k in KS}  # validation F1 for each k
        best = max(tried, key=tried.get)  # best k

        def score(nn, y):
            p = votes(nn, ys, best)
            return evaluate(y, (p > 0.5).astype(np.uint8), p)  # majority vote at 0.5

        sub = stratified_indices(ys, TRAIN_EVAL_ROWS, 0)
        scores = {"train": score(knn.kneighbors(X[sub], return_distance=False), ys[sub]),  # metrics on train/val/test
                  "validation": score(nn_v, yv),
                  "test": score(knn.kneighbors(Xt, return_distance=False), yt)}
        del X
    save_result(MODEL, target, text, {"n_neighbors": best, "weights": "uniform", "tried_val_f1": tried},
                scores, t.seconds, n_train=len(idx), sampled=len(idx) < len(label_column("train", target)))


def main():
    ap = parser(__doc__)
    ap.add_argument("--rows", type=int, default=150_000)
    a = ap.parse_args()
    for text in a.features:
        Xv, Xt = load_split("validation", text), load_split("test", text)
        for target in a.targets:
            if not done(MODEL, text, target):
                run_one(text, target, Xv, Xt, a.rows)


if __name__ == "__main__":
    main()
