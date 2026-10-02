"""Probability calibration (Platt scaling).

Raw model scores are not trustworthy probabilities: before calibration,
XGBoost only produced scores between 0.23 and 0.80, and patients it scored
around 30% were sick only 8% of the time. Platt scaling fits a 1-feature
logistic regression from the model's score (as log-odds) to the true label,
using out-of-fold predictions on the training set, so it never sees the test
set. It is monotonic: the ranking of patients, the ROC-AUC and the flag
decisions are unchanged; only the percentages become honest.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _log_odds(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(raw_scores, y):
    return LogisticRegression(penalty=None).fit(_log_odds(raw_scores), y)


def calibrate(calibrator, raw_scores):
    return calibrator.predict_proba(_log_odds(raw_scores))[:, 1]


def risk(art, X):
    """Calibrated probability of heart disease for the rows of X."""
    return calibrate(art["calibrator"], art["model"].predict_proba(X)[:, 1])
