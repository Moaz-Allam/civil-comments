"""
Progress of Kareem's six models: results done, rough time left, and at the end
the zip of all results for Moaz.

    python3 models/progress.py check     data ready? and the expected run time
    python3 models/progress.py status    results done and time left
    python3 models/progress.py finish    summary table + outputs/kareem_results.zip
"""
import glob
import json
import os
import statistics
import sys
import zipfile

from model_common import FEATURE_SETS, TARGETS, _data, missing_files

# result name -> (script, rough minutes for all 6 labels of one feature set on the M1 Pro)
MODELS = {
    "svm_linear": ("svm", 5),
    "svm_rbf": ("svm", 30),
    "random_forest": ("random_forest", 35),
    "decision_tree_c45": ("decision_tree_c45", 25),
    "knn": ("knn", 20),
    "adaboost": ("adaboost", 45),
    "mlp": ("mlp", 150),
}
PARALLEL = {"svm_rbf": 6, "adaboost": 6}       # labels fitted at the same time
OUT = _data.OUTPUTS
ZIP = os.path.join(OUT, "kareem_results.zip")


def features():
    return os.environ.get("FEATURES", " ".join(FEATURE_SETS)).split()


def results():
    found = {}
    for name in MODELS:
        for text in features():
            for target in TARGETS:
                path = os.path.join(OUT, f"{name}_{text}_{target}.json")
                if os.path.exists(path):
                    found[(name, text, target)] = path
    return found


def minutes_left(found):
    total = 0.0
    for name, (_, rough) in MODELS.items():
        secs = [json.load(open(p))["seconds"] for (n, _, _), p in found.items() if n == name]
        per_label = statistics.median(secs) / 60 / PARALLEL.get(name, 1) if secs else rough / 6
        todo = sum((name, f, t) not in found for f in features() for t in TARGETS)
        total += per_label * todo
    return total


def hm(minutes):
    h, m = divmod(int(round(minutes)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def status(short=False):
    found = results()
    total = len(MODELS) * len(features()) * len(TARGETS)
    left = minutes_left(found)
    if short:
        print(f"--- progress: {len(found)}/{total} results done, about {hm(left)} left")
        return
    print(f"{'model':<20}" + "".join(f"{f:>14}" for f in features()))
    for name in MODELS:
        cells = [sum((name, f, t) in found for t in TARGETS) for f in features()]
        print(f"{name:<20}" + "".join(f"{c:>12}/6" for c in cells))
    print(f"\n{len(found)}/{total} results done" +
          (f", about {hm(left)} left (rough estimate, gets better as results come in)" if len(found) < total else ""))


def check():
    miss = missing_files(features())
    if miss:
        print("Missing files in", _data.CLEAN)
        for m in miss:
            print("  -", m)
        return 1
    print("Data ready:", ", ".join(features()))
    status()
    return 0


def finish():
    found = results()
    total = len(MODELS) * len(features()) * len(TARGETS)
    rows = []
    for (name, text, target), p in sorted(found.items()):
        if target == "toxicity":
            v = json.load(open(p))["validation"]
            rows.append((v["f1"], name, text, v))
    print("\nValidation, toxicity label (best first):")
    print(f"{'model':<20}{'features':<14}{'F1':>7}{'prec':>7}{'recall':>7}{'AUC':>7}")
    for f1, name, text, v in sorted(rows, key=lambda r: -r[0]):
        auc = "" if v["roc_auc"] is None else f"{v['roc_auc']:.3f}"
        print(f"{name:<20}{text:<14}{f1:>7.3f}{v['precision']:>7.3f}{v['recall']:>7.3f}{auc:>7}")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(found.values()):
            z.write(p, os.path.join("outputs", os.path.basename(p)))
        summary = os.path.join(OUT, "summary.csv")
        if os.path.exists(summary):
            z.write(summary, "outputs/summary.csv")
    print(f"\n{len(found)}/{total} results in {os.path.relpath(ZIP, os.path.dirname(OUT))}")
    if len(found) < total:
        print("Missing:", ", ".join(sorted({f"{n} ({f})" for n in MODELS for f in features()
                                             for t in TARGETS if (n, f, t) not in found})))
        return 1
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "check":
        sys.exit(check())
    if cmd == "finish":
        sys.exit(finish())
    status(short="--short" in sys.argv)
