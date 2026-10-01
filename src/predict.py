"""Everything the web app needs around the saved models: loading them,
validating user input, predicting, and explaining a prediction."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.preprocess import ROOT, load_data

LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalch": "Max heart rate", "exang": "Exercise-induced angina", "oldpeak": "ST depression",
}
UNITS = {"trestbps": " mm Hg", "chol": " mg/dL", "thalch": " bpm", "age": " yrs"}
CP_VALUES = {"typical angina", "atypical angina", "non-anginal", "asymptomatic"}
ECG_VALUES = {"normal", "st-t abnormality", "lv hypertrophy"}
BOOL_VALUES = {"True", "False"}


@lru_cache(maxsize=1)
def load_artifacts():
    """{model name: artifact dict}, best model first."""
    arts = [joblib.load(p) for p in sorted((ROOT / "models").glob("*.joblib"))]
    arts.sort(key=lambda a: not a["is_best"])
    return {a["model_name"]: a for a in arts}


def best_artifact():
    return next(iter(load_artifacts().values()))


@lru_cache(maxsize=1)
def reference_data():
    """A fixed sample of real patients, used to explain predictions."""
    return load_data().sample(150, random_state=0)


def _num(payload, key, lo, hi, errors, required=False):
    raw = payload.get(key)
    if raw in (None, ""):
        if required:
            errors[key] = "Required"
        return np.nan
    try:
        v = float(raw)
    except (TypeError, ValueError):
        errors[key] = "Enter a number"
        return np.nan
    if not lo <= v <= hi:
        errors[key] = f"Must be between {lo} and {hi}"
        return np.nan
    return v


def _choice(payload, key, allowed, errors, required=False):
    raw = payload.get(key)
    if raw in (None, ""):
        if required:
            errors[key] = "Required"
        return np.nan
    if raw not in allowed:
        errors[key] = "Invalid choice"
        return np.nan
    return raw


def parse_inputs(payload):
    """Validate the JSON from the form. Returns (model row dict, extras, errors).
    Optional fields left blank become NaN and are imputed by the pipeline."""
    errors = {}
    row = {
        "age": _num(payload, "age", 18, 100, errors, required=True),
        "sex": _choice(payload, "sex", {"Male", "Female"}, errors, required=True),
        "cp": _choice(payload, "cp", CP_VALUES, errors, required=True),
        "exang": _choice(payload, "exang", BOOL_VALUES, errors, required=True),
        "trestbps": _num(payload, "trestbps", 70, 250, errors),
        "chol": _num(payload, "chol", 100, 600, errors),
        "fbs": _choice(payload, "fbs", BOOL_VALUES, errors),
        "restecg": _choice(payload, "restecg", ECG_VALUES, errors),
        "thalch": _num(payload, "thalch", 60, 220, errors),
        "oldpeak": _num(payload, "oldpeak", -3, 7, errors),
    }
    height = _num(payload, "height_cm", 100, 250, errors)
    weight = _num(payload, "weight_kg", 30, 300, errors)
    bmi = None if np.isnan(height) or np.isnan(weight) else round(weight / (height / 100) ** 2, 1)
    return row, {"bmi": bmi}, errors


def _frame(art, rows):
    cols = art["numeric"] + art["categorical"]
    return pd.DataFrame(rows).reindex(columns=cols)


def predict_proba(art, row):
    return float(art["model"].predict_proba(_frame(art, [row]))[0, 1])


def _show(feature, value):
    if feature in ("fbs", "exang"):
        return "Yes" if value == "True" else "No"
    if isinstance(value, float):
        value = int(value) if value == int(value) else round(value, 1)
    return f"{value}{UNITS.get(feature, '')}"


def explain(art, row, prob, top_up=3, top_down=2, min_effect=0.02):
    """Which of the answers moved this estimate the most?

    For each provided answer: replace it with values from real patients,
    average the model's output, and take the difference. A positive number
    means "this answer pushes the estimate up compared with a typical
    patient". It describes the model, not a cause."""
    art_cols = art["numeric"] + art["categorical"]
    ref = reference_data()
    frames = []
    for f in art_cols:
        block = pd.DataFrame([row] * len(ref)).reindex(columns=art_cols)
        block[f] = ref[f].to_numpy()
        frames.append(block)
    probs = art["model"].predict_proba(pd.concat(frames, ignore_index=True))[:, 1]
    probs = probs.reshape(len(art_cols), len(ref)).mean(axis=1)

    effects = []
    for f, mean_p in zip(art_cols, probs):
        value = row.get(f)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue  # not provided: nothing meaningful to say
        effects.append((prob - mean_p, f, value))
    effects.sort(key=lambda e: e[0])

    def fmt(e):
        return {"feature": LABELS[e[1]], "value": _show(e[1], e[2]), "effect": round(e[0] * 100)}

    up = [fmt(e) for e in reversed(effects) if e[0] > min_effect][:top_up]
    down = [fmt(e) for e in effects if e[0] < -min_effect][:top_down]
    return {"raising": up, "lowering": down}


def guidance(high_risk, bmi):
    tips = []
    if high_risk:
        tips.append("Share these results with a doctor. Tests such as an ECG, stress test and blood work can "
                    "confirm or rule out a problem, which this tool cannot.")
    else:
        tips.append("A lower estimate does not rule out heart disease. Keep up regular check-ups, "
                    "especially if you have symptoms or a family history.")
    tips.append("General heart-health habits: about 150 minutes of moderate activity a week, "
                "not smoking, and a balanced diet.")
    if bmi and bmi >= 25:
        tips.append(f"Your BMI is {bmi}, in the overweight range or above. This model does not use BMI, "
                    "but a healthy weight is part of general heart-health advice.")
    return tips
