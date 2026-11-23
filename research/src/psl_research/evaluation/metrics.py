import numpy as np


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = cm.sum(1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def ece(confidences, correct, n_bins=10):
    """Expected calibration error."""
    confidences, correct = np.asarray(confidences), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confidences[m].mean())
    return float(total)


def summarize(y_true, probs, class_names):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    per_class = {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i, c in enumerate(class_names)}
    top5 = float(np.mean([t in np.argsort(p)[-5:] for t, p in zip(y_true, probs)]))
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "top5_accuracy": top5,
        "macro_f1": macro_f1(cm),
        "ece": ece(probs.max(1), y_pred == y_true),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
    }
