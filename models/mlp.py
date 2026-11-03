"""Neural network (MLP): hidden layers chosen by validation F1."""
from sklearn.neural_network import MLPClassifier

from model_common import Timer, done, evaluate, label_column, load_split, load_train, parser, proba, save_result, \
    score_all

MODEL = "mlp"
ARCHS = [(64,), (128,), (128, 64)]  # hidden layers tried


# one MLP architecture
def build(arch):
    return MLPClassifier(hidden_layer_sizes=arch, learning_rate_init=0.001, early_stopping=True, random_state=0)


# compare architectures, refit the best on all rows, save metrics
def run_one(text, target, Xv, tune_rows):
    yv = label_column("validation", target)
    with Timer() as t:
        X, y, idx = load_train(text, target, tune_rows)  # 300k sample for tuning
        tried, fitted = {}, {}
        for arch in ARCHS:
            fitted[arch] = build(arch).fit(X, y)
            tried[str(arch)] = evaluate(yv, fitted[arch].predict(Xv))["f1"]  # validation F1
        best = ARCHS[[str(a) for a in ARCHS].index(max(tried, key=tried.get))]  # best architecture
        model = fitted[best]
        del fitted
        if len(idx) < len(label_column("train", target)):
            del X
            X, y, idx = load_train(text, target, 0)
            model = build(best).fit(X, y)  # refit the best
        del X
        scores = score_all(proba(model), text, target)  # metrics on train/val/test
    save_result(MODEL, target, text,
                {"hidden_layer_sizes": str(best), "learning_rate_init": 0.001, "early_stopping": True,
                 "tried_val_f1": tried, "tune_rows": tune_rows, "epochs": int(model.n_iter_)},
                scores, t.seconds, n_train=len(idx), sampled=False)


def main():
    ap = parser(__doc__)
    ap.add_argument("--tune-rows", type=int, default=300_000)
    a = ap.parse_args()
    for text in a.features:
        Xv = load_split("validation", text)
        for target in a.targets:
            if not done(MODEL, text, target):
                run_one(text, target, Xv, a.tune_rows)


if __name__ == "__main__":
    main()
