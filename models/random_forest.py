"""Random forest: n_estimators chosen by validation F1."""
import warnings

from sklearn.ensemble import RandomForestClassifier

from model_common import Timer, done, evaluate, label_column, load_split, load_train, parser, proba, save_result, \
    score_all

MODEL = "random_forest"
SIZES = [100, 200, 300]
warnings.filterwarnings("ignore", message="class_weight presets")


# grow the forest, keep the best size, save metrics
def run_one(text, target, Xv, rows):
    yv = label_column("validation", target)
    with Timer() as t:
        X, y, idx = load_train(text, target, rows)  # 300k stratified sample
        forest = RandomForestClassifier(n_estimators=SIZES[0], max_features="sqrt", class_weight="balanced_subsample",  # bagging + sqrt features per split
                                        n_jobs=-1, random_state=0, warm_start=True)
        tried = {}
        for n in SIZES:
            forest.set_params(n_estimators=n).fit(X, y)
            tried[n] = evaluate(yv, forest.predict(Xv))["f1"]  # validation F1
        best = max(tried, key=tried.get)  # best size
        forest.estimators_ = forest.estimators_[:best]
        forest.n_estimators = best
        del X
        scores = score_all(proba(forest), text, target)  # metrics on train/val/test
    save_result(MODEL, target, text,
                {"n_estimators": best, "max_features": "sqrt", "class_weight": "balanced_subsample",
                 "max_depth": None, "tried_val_f1": tried},
                scores, t.seconds, n_train=len(idx), sampled=len(idx) < len(label_column("train", target)))


def main():
    ap = parser(__doc__)
    ap.add_argument("--rows", type=int, default=300_000)
    a = ap.parse_args()
    for text in a.features:
        Xv = load_split("validation", text)
        for target in a.targets:
            if not done(MODEL, text, target):
                run_one(text, target, Xv, a.rows)


if __name__ == "__main__":
    main()
