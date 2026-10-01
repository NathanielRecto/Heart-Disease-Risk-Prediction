"""Loading, cleaning and the preprocessing pipeline.

Everything that learns from data (imputation values, scaling, one-hot
categories) lives inside a scikit-learn Pipeline, so it is fit on training
data only. This prevents data leakage and lets the Streamlit app apply the
exact same transformations at prediction time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart_disease_uci.csv"

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
# Columns that are mostly missing (ca 66%, thal 53%, slope 34%). Their
# missingness depends on which hospital collected the data, see README.
SPARSE = ["ca", "slope", "thal"]
BOOLEAN = ["fbs", "exang"]


def load_data(path=DATA_PATH):
    """Read the CSV, fix impossible values and build the binary target."""
    df = pd.read_csv(path)

    # num: 0 = no disease, 1-4 = disease of increasing severity -> binary.
    df["target"] = (df["num"] > 0).astype(int)

    # Cholesterol and resting blood pressure of 0 are impossible values that
    # actually mean "not recorded". Treat them as missing.
    for col in ["chol", "trestbps"]:
        df.loc[df[col] == 0, col] = np.nan

    # Booleans become "True"/"False" strings (NaN stays NaN) so they can be
    # one-hot encoded like the other categorical columns.
    for col in BOOLEAN:
        df[col] = df[col].map(lambda v: np.nan if pd.isna(v) else str(bool(v)))

    return df.drop(columns=["id", "num"])


def get_xy(df, sparse="drop"):
    """Select features and target. The hospital (`dataset`) column is never
    used as a feature: a doctor's patient doesn't come with that label and it
    would let the model learn hospital-specific disease rates."""
    num, cat = feature_columns(sparse)
    return df[num + cat], df["target"]


def feature_columns(sparse="drop"):
    keep = (lambda c: True) if sparse == "keep" else (lambda c: c not in SPARSE)
    return [c for c in NUMERIC if keep(c)], [c for c in CATEGORICAL if keep(c)]


def build_preprocessor(sparse="drop"):
    """`sparse="drop"` removes ca/thal/slope; `"keep"` keeps them and lets
    "was it missing?" become a feature."""
    num, cat = feature_columns(sparse)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num),
        ("cat", categorical_pipe, cat),
    ])
