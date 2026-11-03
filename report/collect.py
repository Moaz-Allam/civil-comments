"""Collects outputs/*.json and outputs/top3/*.json into report/out/numbers.json for the report and figures."""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "out")
TOP3 = os.path.join(ROOT, "outputs", "top3")
TARGETS = ["toxicity", "insult", "identity_attack", "obscene", "threat", "sexual_explicit"]
MODELS = ["linear_regression", "logistic_regression", "perceptron", "naive_bayes", "decision_tree_id3",
          "decision_tree_cart", "decision_tree_c45", "random_forest", "svm_linear", "svm_rbf", "mlp", "adaboost", "knn"]
METRICS = ["accuracy", "error_rate", "precision", "recall", "specificity", "fpr", "f1", "roc_auc"]


def load_results():
    res = {}
    for path in glob.glob(os.path.join(ROOT, "outputs", "*.json")):
        with open(path, encoding="utf8") as f:
            r = json.load(f)
        res.setdefault(r["model"], {}).setdefault(r["features"], {})[r["label"]] = r
    return res


def mean_over_labels(runs, split, metric):
    return float(np.mean([runs[t][split][metric] for t in TARGETS]))


def per_model(res):
    out = {}
    for m in MODELS:
        if m not in res:
            continue
        sets = {f: runs for f, runs in res[m].items() if all(t in runs for t in TARGETS)}
        by_set = {f: {"mean_f1": mean_over_labels(r, "validation", "f1"),
                      "mean_auc": mean_over_labels(r, "validation", "roc_auc")} for f, r in sets.items()}
        best = max(by_set, key=lambda f: by_set[f]["mean_f1"])
        runs = sets[best]
        out[m] = {
            "best_features": best,
            "by_features": by_set,
            "n_train": {t: runs[t]["n_train"] for t in TARGETS},
            "seconds": {t: runs[t]["seconds"] for t in TARGETS},
            "params": {t: runs[t]["params"] for t in TARGETS},
            "labels": {t: {s: {k: runs[t][s][k] for k in METRICS + ["tp", "fp", "fn", "tn"]}
                           for s in ["train", "validation", "test"]}
                       for t in TARGETS},
            "mean": {s: {k: mean_over_labels(runs, s, k) for k in METRICS} for s in ["validation", "test"]},
        }
    return out


def feature_comparison(res):
    """Mean over the models that ran on all three text versions, and over the six labels."""
    sets = ["word2vec", "doc2vec_dbow", "doc2vec"]
    models = [m for m in MODELS if m in res and all(len(res[m].get(f, {})) == 6 for f in sets)]
    out = {"models": models}
    for f in sets:
        runs = [res[m][f][t] for m in models for t in TARGETS]
        out[f] = {"mean_auc": float(np.mean([r["validation"]["roc_auc"] for r in runs])),
                  "mean_f1": float(np.mean([r["validation"]["f1"] for r in runs])),
                  "best_toxicity_f1": float(max(res[m][f]["toxicity"]["validation"]["f1"] for m in models))}
    return out


def prep():
    """Phase 2 and Doc2Vec facts the report repeats."""
    p = os.path.join(ROOT, "prep", "out")
    load = lambda n: json.load(open(os.path.join(p, n), encoding="utf8"))
    fin = load("03_cleaning.json")["final"]
    out = {"rows": fin["rows"], "n_tabular": fin["n_features_tabular"],
           "positives_pct": {t: fin["positives"]["train"][t]["pct"] for t in TARGETS},
           "word2vec_max_r": abs(load("05_correlations.json")["max_abs_r"]["word2vec"]["r"])}
    for name, f in [("pv_dm", "07_doc2vec.json"), ("pv_dbow", "07_doc2vec_dbow.json")]:
        d = load(f)
        out[name] = {"settings": d["settings"], "cosine": d["stored_vs_inferred"]["mean_cosine"],
                     "max_r": d["leakage"]["max_abs_r"]}
    return out


def summarise_cv():
    cv = json.load(open(os.path.join(TOP3, "cv.json"), encoding="utf8"))
    out = {"pool_rows": cv["pool_rows"], "folds": cv["folds"], "train_rows_per_fold": cv["train_rows_per_fold"],
           "pool_positives": {t: int(np.load(os.path.join(ROOT, "data", "clean", "cache", f"cvpool_word2vec_{t}.npz"))["y"].sum())
                              for t in TARGETS},
           "models": {}}
    for m, labels in cv["results"].items():
        out["models"][m] = {}
        for t, folds in labels.items():
            out["models"][m][t] = {}
            for k in ["f1", "precision", "recall", "roc_auc"]:
                v = np.array([folds[f][k] for f in sorted(folds)], dtype=float)
                out["models"][m][t][k] = {"values": v.tolist(), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                          "min": float(v.min()), "max": float(v.max())}
    return out


def summarise_seeds():
    s = json.load(open(os.path.join(TOP3, "seeds.json"), encoding="utf8"))
    out = {}
    for m, labels in s["results"].items():
        for t, seeds in labels.items():
            for k in ["f1", "precision", "recall", "roc_auc"]:
                v = np.array([seeds[i][k] for i in sorted(seeds)], dtype=float)
                out.setdefault(m, {}).setdefault(t, {})[k] = {"values": v.tolist(), "mean": float(v.mean()),
                                                             "std": float(v.std(ddof=1)), "min": float(v.min()),
                                                             "max": float(v.max())}
    return out


def maybe(name, fn):
    return fn() if os.path.exists(os.path.join(TOP3, name)) else None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = load_results()
    numbers = {
        "models": per_model(res),
        "features": feature_comparison(res),
        "prep": prep(),
        "raw_f1": {m: {f: {t: r["validation"]["f1"] for t, r in runs.items()} for f, runs in res[m].items()}
                   for m in ["svm_rbf", "mlp", "logistic_regression"]},
        "cv": maybe("cv.json", summarise_cv),
        "seeds": maybe("seeds.json", summarise_seeds),
        "thresholds": maybe("thresholds.json", lambda: json.load(open(os.path.join(TOP3, "thresholds.json")))),
        "subgroups": maybe("subgroups.json", lambda: json.load(open(os.path.join(TOP3, "subgroups.json")))),
        "tests": maybe("tests.json", lambda: json.load(open(os.path.join(TOP3, "tests.json")))),
    }
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf8") as f:
        json.dump(numbers, f, indent=1)
    print("wrote report/out/numbers.json:", ", ".join(numbers["models"]))
