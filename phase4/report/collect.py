"""Collects the Phase 4 search results into phase4/report/out/numbers.json for the report and figures."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "outputs", "phase4", "run")
PACK = os.path.join(ROOT, "data", "colab_pack")
OUT = os.path.join(ROOT, "phase4", "report", "out")
P3 = os.path.join(ROOT, "report", "out", "numbers.json")


def results():
    path = os.path.join(RUN, "results.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf8") if l.strip()] if os.path.exists(path) else []


def by_setting(rows, stage):
    """Per finished setting (5 folds): mean / std of mean PR-AUC, per-label PR-AUC and ROC-AUC, epochs, seconds."""
    groups = {}
    for r in rows:
        if r["stage"] == stage:
            groups.setdefault(r["tag"], []).append(r)
    out = []
    for rs in groups.values():
        if len(rs) < 5:
            continue
        labels = list(rs[0]["scores"])
        fold_mean = [np.mean([r["scores"][t]["pr_auc"] for t in labels]) for r in rs]
        out.append({
            "setting": rs[0]["setting"], "labels": labels,
            "mean_pr_auc": float(np.mean(fold_mean)), "std_pr_auc": float(np.std(fold_mean, ddof=1)),
            "fold_pr_auc": {r["fold"]: float(m) for r, m in zip(rs, fold_mean)},
            "pr_auc": {t: float(np.mean([r["scores"][t]["pr_auc"] for r in rs])) for t in labels},
            "roc_auc": {t: float(np.mean([r["scores"][t]["roc_auc"] for r in rs])) for t in labels},
            "mean_roc_auc": float(np.mean([np.mean([r["scores"][t]["roc_auc"] for t in labels]) for r in rs])),
            "epochs": float(np.mean([r["epochs"] for r in rs])), "seconds": float(sum(r["seconds"] for r in rs)),
        })
    return sorted(out, key=lambda t: -t["mean_pr_auc"])


def maybe(name):
    p = os.path.join(RUN, name)
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = results()
    numbers = {
        "meta": json.load(open(os.path.join(PACK, "meta.json"), encoding="utf8")),
        "stage1": by_setting(rows, "stage1"),
        "stage2": by_setting(rows, "stage2"),
        "separate": by_setting(rows, "separate"),
        "stage2_summary": maybe("stage2_summary.json"),
        "separate_summary": maybe("separate_summary.json"),
        "final": maybe("final.json"),
        "compare": json.load(open(os.path.join(OUT, "compare.json"), encoding="utf8")) if os.path.exists(os.path.join(OUT, "compare.json")) else None,
        "phase3": json.load(open(P3, encoding="utf8")) if os.path.exists(P3) else None,
    }
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf8") as f:
        json.dump(numbers, f, indent=1)
    print("wrote phase4/report/out/numbers.json:", {k: len(numbers[k]) for k in ["stage1", "stage2", "separate"]})
