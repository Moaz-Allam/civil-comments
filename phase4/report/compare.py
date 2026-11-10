"""
Test-set statistics for the final Phase 4 model, and its comparison with the Phase 3 MLP on the same test rows.

- 95% bootstrap intervals (1,000 resamples of test rows) for each label's PR-AUC, ROC-AUC and F1 at the final cut
- toxicity against the Phase 3 MLP: paired bootstrap of the PR-AUC / ROC-AUC difference, and McNemar's test on the
  predictions at each model's own cut (Phase 4: max-F1 cut on out-of-fold scores; Phase 3: max-F1 cut on validation)

Writes phase4/report/out/compare.json.
"""
import json
import os

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACK = os.path.join(ROOT, "data", "colab_pack")
RUN = os.path.join(ROOT, "outputs", "phase4", "run")
OUT = os.path.join(ROOT, "phase4", "report", "out")
B = 1000


def f1(y, pred):
    tp = int((pred & (y == 1)).sum())
    return 2 * tp / max(pred.sum() + (y == 1).sum(), 1)


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


if __name__ == "__main__":
    meta = json.load(open(os.path.join(PACK, "meta.json"), encoding="utf8"))
    targets = meta["targets"]
    y = np.load(os.path.join(PACK, "y_test.npy")).astype(int)
    p4 = np.load(os.path.join(RUN, "test_scores.npy"))
    fin = json.load(open(os.path.join(RUN, "final.json"), encoding="utf8"))
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]

    res = {"resamples": B, "labels": {}}
    for l, t in enumerate(targets):
        yl, pl = y[:, l], p4[:, l]
        pred = pl >= fin["cuts"][t]
        pr, roc, ff = [], [], []
        for i in idx:
            pr.append(average_precision_score(yl[i], pl[i]))
            roc.append(roc_auc_score(yl[i], pl[i]))
            ff.append(f1(yl[i], pred[i]))
        res["labels"][t] = {"pr_auc": float(average_precision_score(yl, pl)), "pr_auc_ci": ci(pr),
                            "roc_auc": float(roc_auc_score(yl, pl)), "roc_auc_ci": ci(roc),
                            "f1": f1(yl, pred), "f1_ci": ci(ff)}
        print(t, {k: round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] for k, v in res["labels"][t].items()})

    # toxicity vs the Phase 3 MLP (its saved test scores cover toxicity only)
    p3 = np.load(os.path.join(ROOT, "outputs", "top3", "scores_mlp_test.npy"))
    sweep = json.load(open(os.path.join(ROOT, "report", "out", "numbers.json"), encoding="utf8"))["subgroups"]["cut_sweep"]["mlp"]
    p3_cut = max(sweep["points"], key=lambda q: q["f1"])["cut"]
    yt, s4 = y[:, 0], p4[:, 0]
    d_pr = [average_precision_score(yt[i], s4[i]) - average_precision_score(yt[i], p3[i]) for i in idx]
    d_roc = [roc_auc_score(yt[i], s4[i]) - roc_auc_score(yt[i], p3[i]) for i in idx]
    a, b = s4 >= fin["cuts"]["toxicity"], p3 >= p3_cut
    ok4, ok3 = a == (yt == 1), b == (yt == 1)
    n01, n10 = int((ok4 & ~ok3).sum()), int((~ok4 & ok3).sum())          # only Phase 4 right / only Phase 3 right
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    res["vs_phase3"] = {
        "label": "toxicity", "phase3_cut": float(p3_cut), "phase4_cut": fin["cuts"]["toxicity"],
        "phase3": {"pr_auc": float(average_precision_score(yt, p3)), "roc_auc": float(roc_auc_score(yt, p3)), "f1": f1(yt, b),
                   "accuracy": float(ok3.mean())},
        "phase4": {"pr_auc": float(average_precision_score(yt, s4)), "roc_auc": float(roc_auc_score(yt, s4)), "f1": f1(yt, a),
                   "accuracy": float(ok4.mean())},
        "pr_auc_diff": float(np.mean(d_pr)), "pr_auc_diff_ci": ci(d_pr),
        "roc_auc_diff": float(np.mean(d_roc)), "roc_auc_diff_ci": ci(d_roc),
        "mcnemar": {"only_phase4_right": n01, "only_phase3_right": n10, "chi2": float(stat), "p": float(chi2.sf(stat, 1))},
    }
    print(json.dumps(res["vs_phase3"], indent=1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "compare.json"), "w", encoding="utf8") as f:
        json.dump(res, f, indent=1)
    print("wrote phase4/report/out/compare.json")
