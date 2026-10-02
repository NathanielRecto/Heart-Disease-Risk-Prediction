"""Train and compare Logistic Regression, Random Forest and XGBoost.

Run from the project root:  python -m src.train

Flow
 1. Stratified 80/20 train/test split. The test set is not touched until the end.
 2. 5-fold CV on the training set, for each model x {drop, keep} sparse columns.
 3. Pick the sparse-column strategy, then tune each model (scored by F2).
 4. Calibrate each model (Platt scaling) and choose its decision threshold,
    both on out-of-fold training predictions.
 5. Evaluate once on the test set, plot, and save all three models.
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.calibration import calibrate, fit_calibrator
from src.evaluate import best_threshold, metrics_at, plot_confusions, plot_pr_curves
from src.preprocess import ROOT, build_preprocessor, feature_columns, get_xy, load_data

SEED = 42
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
f2_scorer = make_scorer(fbeta_score, beta=2)
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "f2": f2_scorer, "roc_auc": "roc_auc"}


def make_models():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED),
            {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10],
             "clf__max_features": ["sqrt", 0.5]}),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
            {"clf__n_estimators": [100, 200, 400], "clf__max_depth": [2, 3, 4, 6],
             "clf__learning_rate": [0.01, 0.05, 0.1], "clf__subsample": [0.7, 1.0],
             "clf__colsample_bytree": [0.7, 1.0]}),
    }


def pipe(sparse, clf):
    return Pipeline([("prep", build_preprocessor(sparse)), ("clf", clf)])


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=df["target"], random_state=SEED)
    train, test = df.loc[idx_train], df.loc[idx_test]
    print(f"train {len(train)}  test {len(test)}  disease rate {df['target'].mean():.2f}")

    # --- Step 2: baseline comparison, default hyperparameters ----------------
    rows = []
    for sparse in ["drop", "keep"]:
        X, y = get_xy(train, sparse)
        for name, (clf, _) in make_models().items():
            s = cross_validate(pipe(sparse, clf), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            rows.append({"sparse_cols": sparse, "model": name,
                         **{m: s[f"test_{m}"].mean() for m in SCORING}})
    cv_table = pd.DataFrame(rows)
    cv_table.to_csv(REPORTS_DIR / "cv_comparison.csv", index=False)
    print("\n== 5-fold CV on training data (default hyperparameters) ==")
    print(cv_table.round(3).to_string(index=False))

    # --- Step 3: choose sparse-column strategy -------------------------------
    by_strategy = cv_table.groupby("sparse_cols")["f2"].mean()
    print("\nmean CV F2 by strategy:", by_strategy.round(3).to_dict())
    # Prefer dropping unless keeping clearly helps (>0.02): the extra columns
    # are mostly hospital-specific missingness, and a clinician entering one
    # patient would not have them either.
    sparse = "keep" if by_strategy["keep"] - by_strategy["drop"] > 0.02 else "drop"
    print("using sparse columns:", sparse)

    X_train, y_train = get_xy(train, sparse)
    X_test, y_test = get_xy(test, sparse)

    # --- Step 3b/4: tune, threshold, evaluate --------------------------------
    results, table, fitted = {}, [], {}
    for name, (clf, grid) in make_models().items():
        search = RandomizedSearchCV(
            pipe(sparse, clf), grid, n_iter=15, scoring=f2_scorer, cv=cv,
            random_state=SEED, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        # Out-of-fold scores: each training patient scored by a model that never saw them.
        # They are used to fit the calibration and to choose the threshold, so the test
        # set plays no part in either.
        raw_oof = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
        calibrator = fit_calibrator(raw_oof, y_train)
        thr = best_threshold(y_train, calibrate(calibrator, raw_oof))
        proba = calibrate(calibrator, model.predict_proba(X_test)[:, 1])
        results[name] = (proba, thr)
        fitted[name] = (model, thr, search.best_score_, search.best_params_, calibrator)

        for label, t in [("threshold 0.50", 0.5), ("tuned threshold", thr)]:
            table.append({"model": name, "setting": label, "cv_f2": search.best_score_,
                          **metrics_at(y_test, proba, t)})
        print(f"\n{name}: best CV F2 {search.best_score_:.3f}  params {search.best_params_}")

    metrics = pd.DataFrame(table)
    metrics.to_csv(REPORTS_DIR / "test_metrics.csv", index=False)
    print("\n== Held-out test set ==")
    print(metrics.drop(columns=["f2"]).round(3).to_string(index=False))

    plot_confusions(results, y_test, REPORTS_DIR / "confusion_matrices.png")
    plot_pr_curves(results, y_test, REPORTS_DIR / "precision_recall.png")

    # Per-hospital view of the best model's test predictions (diagnostic only)
    best_name = max(fitted, key=lambda n: fitted[n][2])
    model, thr, cv_f2, params, _ = fitted[best_name]
    proba, _ = results[best_name]
    per_site = []
    for site, g in test.assign(p=proba).groupby("dataset"):
        pred = (g["p"] >= thr).astype(int)
        per_site.append({"hospital": site, "n": len(g), "disease_rate": g["target"].mean(),
                         # recall needs sick patients, precision needs flagged ones; a hospital
                         # where everyone is sick (Switzerland) still has a valid recall
                         "recall": recall_score(g["target"], pred) if g["target"].sum() else None,
                         "precision": precision_score(g["target"], pred) if pred.sum() else None,
                         "false_alarms": int(((pred == 1) & (g["target"] == 0)).sum())})
    per_site = pd.DataFrame(per_site)
    per_site.to_csv(REPORTS_DIR / "per_hospital.csv", index=False)
    print(f"\n== {best_name} by hospital (test set) ==\n{per_site.round(3).to_string(index=False)}")

    # --- Step 5: save the winner ---------------------------------------------
    # Every model is saved (with its own threshold and test metrics) so the
    # app can offer a dropdown; `is_best` marks the default selection.
    num, cat = feature_columns(sparse)
    for name, (m, t, m_cv_f2, _, cal) in fitted.items():
        tuned = metrics[(metrics.model == name) & (metrics.setting == "tuned threshold")].iloc[0]
        joblib.dump({"model": m, "calibrator": cal, "threshold": t, "sparse": sparse, "numeric": num,
                     "categorical": cat, "model_name": name, "is_best": name == best_name,
                     "cv_f2": m_cv_f2, "test_recall": tuned["recall"],
                     "test_precision": tuned["precision"], "test_roc_auc": tuned["roc_auc"],
                     "test_brier": tuned["brier"]},
                    MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
    summary = metrics[(metrics.model == best_name) & (metrics.setting == "tuned threshold")]
    (REPORTS_DIR / "summary.json").write_text(json.dumps({
        "best_model": best_name, "sparse_columns": sparse, "threshold": thr,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in params.items()},
        "test": summary.drop(columns=["model", "setting"]).iloc[0].astype(float).round(4).to_dict(),
    }, indent=2))
    print(f"\nSaved all {len(fitted)} models to models/ (best: {best_name}, threshold {thr:.2f})")


if __name__ == "__main__":
    main()
