"""The one evaluation function every Phase 3 model uses (metrics of Lecture 2)."""
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate(y_true, y_pred, y_score=None):
    """y_score = any number that ranks comments by how toxic they look: a probability,
    a regression output or an SVM/perceptron decision value. ROC-AUC needs only the
    ranking, so it does not have to be a probability."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "accuracy": (tp + tn) / n,
        "error_rate": (fp + fn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "roc_auc": None,
    }
    if y_score is not None and tp + fn and tn + fp:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out
