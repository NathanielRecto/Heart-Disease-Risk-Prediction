"""Extra report figures built from the SAVED models (no retraining):  python -m src.report

Recreates the exact same train/test split as train.py (same seed), then draws:
  roc_curves.png          ROC curves for the three models on the test set
  threshold_tradeoff.png  recall / precision / F2 vs threshold (out-of-fold, training data)
  cv_comparison.png       5-fold CV scores of the three models (default settings)
  feature_importance.png  permutation importance of the final model on the test set
and checks that the recreated test metrics match the ones stored with each model."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.predict import LABELS
from src.preprocess import ROOT, get_xy, load_data
from src.train import SEED

REPORTS = ROOT / "reports"
COLORS = {"Logistic Regression": "#2563eb", "Random Forest": "#059669", "XGBoost": "#8a0f1d"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "font.size": 10})


def load_everything():
    arts = {a["model_name"]: a for a in (joblib.load(p) for p in sorted((ROOT / "models").glob("*.joblib")))}
    df = load_data()
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=df["target"], random_state=SEED)
    sparse = next(iter(arts.values()))["sparse"]
    X_train, y_train = get_xy(df.loc[idx_train], sparse)
    X_test, y_test = get_xy(df.loc[idx_test], sparse)
    return arts, X_train, y_train, X_test, y_test


def check_reproduction(arts, X_test, y_test):
    """The recreated split must give the metrics that were stored at training time."""
    for name, a in arts.items():
        pred = (a["model"].predict_proba(X_test)[:, 1] >= a["threshold"]).astype(int)
        assert abs(recall_score(y_test, pred) - a["test_recall"]) < 1e-9, f"{name}: split differs from training"
        assert abs(precision_score(y_test, pred) - a["test_precision"]) < 1e-9, f"{name}: split differs from training"
    print("OK: recreated test metrics match the stored ones for all models")


def roc_figure(arts, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, a in arts.items():
        p = a["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.2f})")
        pred = (p >= a["threshold"]).astype(int)
        healthy = y_test.to_numpy() == 0
        ax.scatter([(pred[healthy] == 1).mean()], [recall_score(y_test, pred)], color=COLORS[name], s=55, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="Chance")
    ax.set(xlabel="False positive rate (healthy patients flagged)", ylabel="True positive rate (recall)",
           title="ROC curves on the test set\n(dots = chosen threshold)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(REPORTS / "roc_curves.png", dpi=130); plt.close(fig)


def threshold_figure(arts, X_train, y_train):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grid = np.linspace(0.05, 0.95, 91)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (name, a) in zip(axes, arts.items()):
        oof = cross_val_predict(clone(a["model"]), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        rec = [recall_score(y_train, oof >= t) for t in grid]
        pre = [precision_score(y_train, oof >= t, zero_division=0) for t in grid]
        f2 = [fbeta_score(y_train, oof >= t, beta=2) for t in grid]
        ax.plot(grid, rec, label="Recall", color="#dc2626", lw=2)
        ax.plot(grid, pre, label="Precision", color="#2563eb", lw=2)
        ax.plot(grid, f2, label="F2 (recall-weighted)", color="#111827", lw=2, ls="--")
        ax.axvline(a["threshold"], color=COLORS[name], lw=2, alpha=.9)
        ax.axvline(0.5, color="gray", lw=1, ls=":")
        ax.set(title=f"{name}\nchosen threshold {a['threshold']:.2f} (default 0.50 dotted)", xlabel="Decision threshold")
    axes[0].set_ylabel("Score (out-of-fold, training data)")
    axes[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(REPORTS / "threshold_tradeoff.png", dpi=130); plt.close(fig)


def cv_figure():
    cv = pd.read_csv(REPORTS / "cv_comparison.csv")
    cv = cv[cv["sparse_cols"] == "drop"].set_index("model")
    metrics = ["recall", "precision", "f1", "roc_auc"]
    names = {"recall": "Recall", "precision": "Precision", "f1": "F1", "roc_auc": "ROC-AUC"}
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (model, row) in enumerate(cv.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=model, color=COLORS[model])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [names[m] for m in metrics]); ax.set_ylim(0.6, 1.0)
    ax.set(title="5-fold cross-validation on the training set (default settings)",
           ylabel="Mean score (axis starts at 0.6)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(REPORTS / "cv_comparison.png", dpi=130); plt.close(fig)


def importance_figure(arts, X_test, y_test):
    name = next(n for n, a in arts.items() if a["is_best"])
    r = permutation_importance(arts[name]["model"], X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED)
    out = pd.DataFrame({"feature": [LABELS[c] for c in X_test.columns], "importance": r.importances_mean,
                        "std": r.importances_std}).sort_values("importance")
    out.sort_values("importance", ascending=False).round(4).to_csv(REPORTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(out["feature"], out["importance"], xerr=out["std"], color=COLORS[name], alpha=.9, capsize=3)
    ax.set(xlabel="Drop in test ROC-AUC when the feature is shuffled",
           title=f"What the final model ({name}) relies on")
    fig.tight_layout(); fig.savefig(REPORTS / "feature_importance.png", dpi=130); plt.close(fig)
    print(out.sort_values("importance", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    arts, X_train, y_train, X_test, y_test = load_everything()
    check_reproduction(arts, X_test, y_test)
    roc_figure(arts, X_test, y_test)
    threshold_figure(arts, X_train, y_train)
    cv_figure()
    importance_figure(arts, X_test, y_test)
    print("Wrote roc_curves.png, threshold_tradeoff.png, cv_comparison.png, feature_importance.png to reports/")
