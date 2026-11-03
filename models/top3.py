"""
Variability and comparison runs for the top 3 models (SVM-RBF, MLP, logistic regression).

    python models/top3.py cv           5-fold stratified CV, all 6 labels, same folds for every model
    python models/top3.py seeds        MLP with 5 seeds, SVM-RBF on 5 samples (toxicity, threat)
    python models/top3.py thresholds   toxicity labels made at score >= 0.4 / 0.5 / 0.6
    python models/top3.py subgroups    test scores per identity subgroup + F1 against the decision cut
    python models/top3.py tests        bootstrap intervals and McNemar tests on the test scores

Results go to outputs/top3/.
"""
import argparse
import ast
import json
import os
import sys
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import CACHE, CLEAN, OUTPUTS, TARGETS, evaluate, features, label_column, result_path
from common.sample import stratified_indices

OUT = os.path.join(OUTPUTS, "top3")
os.makedirs(OUT, exist_ok=True)
FEATURES = {"svm_rbf": "doc2vec_dbow", "mlp": "word2vec", "logistic_regression": "word2vec"}
SVM_ROWS, POOL, FOLDS, SEEDS = 50_000, 62_500, 5, 5
IDENTITIES = ["male", "female", "transgender", "other_gender", "heterosexual", "homosexual_gay_or_lesbian",
              "bisexual", "other_sexual_orientation", "christian", "jewish", "muslim", "hindu", "buddhist",
              "atheist", "other_religion", "black", "white", "asian", "latino", "other_race_or_ethnicity",
              "physical_disability", "intellectual_or_learning_disability", "psychiatric_or_mental_illness",
              "other_disability"]


def chosen(model, target):
    return json.load(open(result_path(model, FEATURES[model], target), encoding="utf8"))["params"]


def build(model, target, seed=0):
    p = chosen(model, target)
    if model == "logistic_regression":
        return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=200, solver="lbfgs")
    if model == "mlp":
        return MLPClassifier(hidden_layer_sizes=ast.literal_eval(p["hidden_layer_sizes"]), learning_rate_init=0.001,
                             early_stopping=True, random_state=seed)
    return SVC(kernel="rbf", gamma="scale", class_weight="balanced", cache_size=500)


def scores(model, m, X):
    return m.decision_function(X) if model == "svm_rbf" else m.predict_proba(X)[:, 1]


def save(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf8") as f:
        json.dump(obj, f, indent=1, default=float)
    print("wrote outputs/top3/" + name, flush=True)


# ---------------------------------------------------------------- cross-validation
def pool_path(text, target):
    return os.path.join(CACHE, f"cvpool_{text}_{target}.npz")


def make_pools():
    for target in TARGETS:
        y = label_column("train", target)
        idx = stratified_indices(y, POOL, 0)
        for text in set(FEATURES.values()):
            p = pool_path(text, target)
            if not os.path.exists(p):
                np.savez(p, X=np.asarray(features("train", text)[idx], dtype=np.float32), y=y[idx])


def cv_task(task):
    model, target, fold = task
    d = np.load(pool_path(FEATURES[model], target))
    X, y = d["X"], d["y"]
    tr, te = list(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y))[fold]
    m = build(model, target).fit(X[tr], y[tr])
    r = evaluate(y[te], m.predict(X[te]), scores(model, m, X[te]))
    print(f"cv {model:<20} {target:<16} fold {fold}  F1 {r['f1']:.3f}  AUC {r['roc_auc']}", flush=True)
    return model, target, fold, r


def run_cv(procs):
    make_pools()
    cost = {"toxicity": 6, "insult": 5, "identity_attack": 2, "obscene": 2, "sexual_explicit": 1, "threat": 1}
    tasks = [(m, t, f) for m in FEATURES for t in TARGETS for f in range(FOLDS)]
    tasks.sort(key=lambda x: -(cost[x[1]] * (10 if x[0] == "svm_rbf" else 1)))
    res = {}
    with get_context("spawn").Pool(procs) as pool:
        for model, target, fold, r in pool.imap_unordered(cv_task, tasks):
            res.setdefault(model, {}).setdefault(target, {})[fold] = r
    save("cv.json", {"pool_rows": POOL, "folds": FOLDS, "train_rows_per_fold": POOL * (FOLDS - 1) // FOLDS,
                     "features": FEATURES, "results": res})


# ---------------------------------------------------------------- seeds
def seed_task(task):
    model, target, seed = task
    text = FEATURES[model]
    y = label_column("train", target)
    if model == "svm_rbf":
        idx = stratified_indices(y, SVM_ROWS, seed)
        X = np.asarray(features("train", text)[idx], dtype=np.float32)
        m = build(model, target, seed).fit(X, y[idx])
    else:
        m = build(model, target, seed).fit(np.array(features("train", text), dtype=np.float32), y)
    Xv = np.asarray(features("validation", text), dtype=np.float32)
    yv = label_column("validation", target)
    r = evaluate(yv, m.predict(Xv), scores(model, m, Xv))
    print(f"seed {model:<8} {target:<10} seed {seed}  F1 {r['f1']:.3f}  AUC {r['roc_auc']:.3f}", flush=True)
    return model, target, seed, r


def run_seeds(procs):
    tasks = [(m, t, s) for m in ["svm_rbf", "mlp"] for t in ["toxicity", "threat"] for s in range(SEEDS)]
    res = {}
    with get_context("spawn").Pool(procs) as pool:
        for model, target, seed, r in pool.imap_unordered(seed_task, tasks):
            res.setdefault(model, {}).setdefault(target, {})[seed] = r
    save("seeds.json", {"seeds": SEEDS, "svm_rows": SVM_ROWS, "features": FEATURES, "results": res,
                        "note": "SVM: a new stratified sample per seed; MLP: all rows, new weights and early-stopping split"})


# ---------------------------------------------------------------- label thresholds
def scores_column(split):
    return pq.read_table(os.path.join(CLEAN, f"{split}.parquet"), columns=["toxicity"]).column(0).to_numpy()


def threshold_task(task):
    model, cut = task
    text = FEATURES[model]
    y = (scores_column("train") >= cut).astype(np.uint8)
    if model == "svm_rbf":
        idx = stratified_indices(y, SVM_ROWS, 0)
        m = build(model, "toxicity").fit(np.asarray(features("train", text)[idx], dtype=np.float32), y[idx])
    else:
        m = build(model, "toxicity").fit(np.array(features("train", text), dtype=np.float32), y)
    out = {"train_positive_rate": float(y.mean())}
    for split in ["validation", "test"]:
        X = np.asarray(features(split, text), dtype=np.float32)
        ys = (scores_column(split) >= cut).astype(np.uint8)
        out[split] = evaluate(ys, m.predict(X), scores(model, m, X))
    print(f"threshold {model:<20} {cut}  val F1 {out['validation']['f1']:.3f}", flush=True)
    return model, cut, out


def run_thresholds(procs):
    tasks = [(m, c) for m in FEATURES for c in [0.4, 0.5, 0.6]]
    res = {}
    with get_context("spawn").Pool(procs) as pool:
        for model, cut, r in pool.imap_unordered(threshold_task, tasks):
            res.setdefault(model, {})[str(cut)] = r
    save("thresholds.json", {"label": "toxicity", "features": FEATURES, "results": res})


# ---------------------------------------------------------------- identity subgroups and the decision cut
def subgroup_metrics(y, s, mask):
    """Borkan et al. 2019: AUC inside the subgroup, BPSN and BNSP."""
    bg = ~mask

    def auc(sel):
        return float(roc_auc_score(y[sel], s[sel])) if 0 < y[sel].sum() < sel.sum() else None
    return {"n": int(mask.sum()), "positives": int(y[mask].sum()),
            "subgroup_auc": auc(mask),
            "bpsn_auc": auc((bg & (y == 1)) | (mask & (y == 0))),
            "bnsp_auc": auc((bg & (y == 0)) | (mask & (y == 1)))}


def subgroup_task(model):
    text = FEATURES[model]
    y = label_column("train", "toxicity")
    if model == "svm_rbf":
        idx = stratified_indices(y, SVM_ROWS, 0)
        m = build(model, "toxicity").fit(np.asarray(features("train", text)[idx], dtype=np.float32), y[idx])
    else:
        m = build(model, "toxicity").fit(np.array(features("train", text), dtype=np.float32), y)
    out = {}
    for split in ["validation", "test"]:
        s = scores(model, m, np.asarray(features(split, text), dtype=np.float32))
        np.save(os.path.join(OUT, f"scores_{model}_{split}.npy"), s.astype(np.float32))
        out[split] = s
    return model, out


def run_subgroups(procs):
    with get_context("spawn").Pool(procs) as pool:
        got = dict(pool.map(subgroup_task, list(FEATURES)))
    t = pq.read_table(os.path.join(CLEAN, "test.parquet"), columns=IDENTITIES + ["identity_annotated"])
    yt, yv = label_column("test", "toxicity"), label_column("validation", "toxicity")
    res = {"identities": {}, "overall_auc": {}, "cut_sweep": {}}
    for model, s in got.items():
        res["overall_auc"][model] = float(roc_auc_score(yt, s["test"]))
        for name in IDENTITIES:
            col = np.nan_to_num(t.column(name).to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0)
            mask = col >= 0.5
            if mask.sum() >= 100:
                res["identities"].setdefault(name, {})[model] = subgroup_metrics(yt, s["test"], mask)
        sv = s["validation"]
        cuts = np.quantile(sv, np.linspace(0.50, 0.995, 100))
        sweep = [{"cut": float(c), **{k: evaluate(yv, (sv >= c).astype(int))[k] for k in ["f1", "precision", "recall"]}}
                 for c in cuts]
        default = 0.0 if model == "svm_rbf" else 0.5
        best = max(sweep, key=lambda p: p["f1"])
        res["cut_sweep"][model] = {"default_cut": default, "points": sweep, "best_on_validation": best,
                                   "at_default": evaluate(yv, (sv >= default).astype(int)),
                                   "test_at_default": evaluate(yt, (s["test"] >= default).astype(int)),
                                   "test_at_best_cut": evaluate(yt, (s["test"] >= best["cut"]).astype(int))}
    save("subgroups.json", {"label": "toxicity", "features": FEATURES, "identity_mention": "score >= 0.5, test split",
                            "min_comments": 100, **res})


# ---------------------------------------------------------------- tests on the saved test scores
def run_tests(procs, boots=1000):
    """Bootstrap intervals for test F1 / ROC-AUC and McNemar tests between models (Dietterich 1998)."""
    y = label_column("test", "toxicity")
    s = {m: np.load(os.path.join(OUT, f"scores_{m}_test.npy")) for m in FEATURES}
    pred = {m: (s[m] >= (0.0 if m == "svm_rbf" else 0.5)).astype(int) for m in FEATURES}
    rng = np.random.default_rng(0)
    draws = {m: {"f1": [], "roc_auc": []} for m in FEATURES}
    for _ in range(boots):
        i = rng.integers(0, len(y), len(y))
        for m in FEATURES:
            draws[m]["f1"].append(evaluate(y[i], pred[m][i])["f1"])
            draws[m]["roc_auc"].append(float(roc_auc_score(y[i], s[m][i])))
    ci = {m: {k: {"mean": float(np.mean(v)), "low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
              for k, v in d.items()} for m, d in draws.items()}
    mcnemar = {}
    names = list(FEATURES)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ra, rb = pred[names[a]] == y, pred[names[b]] == y
            n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            mcnemar[f"{names[a]} vs {names[b]}"] = {"only_first_right": n01, "only_second_right": n10, "chi2": chi2}
    save("tests.json", {"label": "toxicity", "split": "test", "bootstrap_resamples": boots, "ci95": ci,
                        "mcnemar": mcnemar, "note": "chi2 > 3.84 means p < 0.05 (1 degree of freedom)"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cv", "seeds", "thresholds", "subgroups", "tests"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    {"cv": run_cv, "seeds": run_seeds, "thresholds": run_thresholds, "subgroups": run_subgroups,
     "tests": run_tests}[a.step](a.procs)
