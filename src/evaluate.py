"""Metrics and plots. Recall is the headline number: a missed sick patient
(false negative) is worse than a false alarm (false positive)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=2),
        "roc_auc": roc_auc_score(y_true, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def best_threshold(y_true, proba, beta=2.0):
    """Threshold that maximises F-beta (beta=2 weights recall 4x as much as
    precision). Meant to be called on out-of-fold training predictions."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def plot_confusions(results, y_test, path):
    """results: {name: (proba, threshold)}; one confusion matrix per model."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    for ax, (name, (proba, thr)) in zip(np.atleast_1d(axes), results.items()):
        pred = (proba >= thr).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Healthy", "Disease"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{name}\n(threshold {thr:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(results, y_test, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (proba, thr) in results.items():
        p, r, _ = precision_recall_curve(y_test, proba)
        ax.plot(r, p, label=name)
        pred = (proba >= thr).astype(int)
        ax.scatter([recall_score(y_test, pred)], [precision_score(y_test, pred)], marker="o")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (dots = chosen threshold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
