import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


def binary_metrics(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be nonempty and have matching shapes.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected binary labels and finite scores between zero and one.")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else None
    both = len(np.unique(labels)) == 2
    return {
        "n": int(labels.size), "threshold": threshold,
        "accuracy": float((tp + tn) / labels.size),
        "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "average_precision": float(average_precision_score(labels, scores)) if both else None,
        "brier_score": float(brier_score_loss(labels, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
