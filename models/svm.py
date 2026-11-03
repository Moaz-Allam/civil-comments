"""SVM: linear (SGD hinge) and RBF kernel."""
import os

from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from model_common import Timer, decision, done, label_column, load_train, parser, save_result, score_all


# linear SVM on all training rows
def linear(text, targets):
    todo = [t for t in targets if not done("svm_linear", text, t)]
    if not todo:
        return
    X, _, _ = load_train(text, todo[0], 0)  # all training rows
    for target in todo:
        y = label_column("train", target)
        with Timer() as t:
            model = SGDClassifier(loss="hinge", class_weight="balanced", random_state=0).fit(X, y)  # linear SVM by gradient descent
            scores = score_all(decision(model), text, target)  # AUC from the margin
        save_result("svm_linear", target, text,
                    {"loss": "hinge", "class_weight": "balanced", "alpha": model.alpha, "epochs": int(model.n_iter_)},
                    scores, t.seconds, n_train=len(y), sampled=False)


# RBF SVM on a stratified sample
def rbf_one(text, target, rows):
    with Timer() as t:
        X, y, idx = load_train(text, target, rows)  # 50k stratified sample
        model = SVC(kernel="rbf", gamma="scale", class_weight="balanced", cache_size=2000).fit(X, y)  # RBF kernel SVM
        del X
        scores = score_all(decision(model), text, target, train_idx=idx)
    save_result("svm_rbf", target, text,
                {"kernel": "rbf", "gamma": "scale", "C": 1.0, "class_weight": "balanced",
                 "support_vectors": int(model.n_support_.sum())},
                scores, t.seconds, n_train=len(idx), sampled=len(idx) < len(label_column("train", target)))


def main():
    ap = parser(__doc__)
    ap.add_argument("--part", nargs="+", default=["linear", "rbf"], choices=["linear", "rbf"])
    ap.add_argument("--rbf-rows", type=int, default=50_000)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if "linear" in a.part:
        for text in a.features:
            linear(text, a.targets)
    if "rbf" in a.part:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        work = [(text, t) for text in a.features for t in a.targets if not done("svm_rbf", text, t)]
        Parallel(n_jobs=a.jobs)(delayed(rbf_one)(text, t, a.rbf_rows) for text, t in work)  # labels fitted in parallel


if __name__ == "__main__":
    main()
