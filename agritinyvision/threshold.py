import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def metrics(labels, scores, threshold):
    y, p = np.asarray(labels), np.asarray(scores)
    if len(y) == 0 or y.shape != p.shape or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary evaluation inputs")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    def div(a, b):
        return float(a / b) if b else None
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, len(y)),
        "precision": precision,
        "suspect_recall": recall,
        "recall": recall,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
        "pr_auc_definition": "average_precision (step integral)",
        "false_positive_rate": div(fp, fp + tn),
        "false_negative_rate": div(fn, fn + tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "samples": len(y),
        "decision_threshold": float(threshold),
    }


def tune(labels, scores, settings, split="val"):
    if split != "val":
        raise ValueError("Threshold tuning is permitted only on validation")
    results = [
        metrics(labels, scores, float(t))
        for t in np.arange(settings["min"], settings["max"] + 1e-8, settings["step"])
    ]
    feasible = [
        r
        for r in results
        if (r["suspect_recall"] or 0) >= settings["target_recall"]
        and r["false_positive_rate"] is not None
        and r["false_positive_rate"] <= settings["max_fpr"]
    ]
    if feasible:
        chosen = max(
            feasible, key=lambda r: (r["f1"] or 0, r["suspect_recall"] or 0, -r["false_positive_rate"])
        )
    else:
        chosen = max(
            results, key=lambda r: (r["suspect_recall"] or 0, -(r["false_positive_rate"] or 0), r["f1"] or 0)
        )
    return {
        "selected": chosen,
        "constraints_met": bool(feasible),
        "settings": settings,
        "split": "val",
        "search": results,
    }


def selection_key(tuned):
    r = tuned["selected"]
    if tuned["constraints_met"]:
        return (1, r["f1"] or 0, r["suspect_recall"] or 0, -(r["false_positive_rate"] or 0))
    return (0, r["suspect_recall"] or 0, -(r["false_positive_rate"] or 0), r["f1"] or 0)
