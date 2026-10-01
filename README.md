# PulseCheck: Heart Disease Risk Prediction

An end-to-end machine-learning project: clean a real multi-hospital medical dataset, compare three models,
tune for the metric that matters (**recall**: a missed sick patient is worse than a false alarm), and serve
the result as a Flask web app with a 4-step assessment, per-prediction explanations and private per-visitor history.

**Stack:** Python · scikit-learn · XGBoost · Pandas · Flask · SQLite/SQLAlchemy · HTML/CSS/JavaScript · Docker

<p align="center">
  <img src="docs/demo.gif" alt="Demo: filling in the 4-step assessment and getting a result" width="760">
</p>

## Results at a glance

Held-out test set of 184 patients, never used for training or tuning. The final model is XGBoost with a
decision threshold of 0.40, chosen on out-of-fold training predictions.

| Recall | Precision | F1 | ROC-AUC | Accuracy | Missed sick patients | False alarms |
|---|---|---|---|---|---|---|
| **96%** (98 of 102) | 74% | 0.84 | 0.92 | 79% | 4 | 34 |

Accuracy is deliberately not the headline: lowering the threshold from 0.50 to 0.40 cut missed patients from 12
to 4 at the cost of 10 more false alarms. The three models score within noise of each other on a test set this
small (see [Results](#results-in-detail)).

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/home.png" alt="Home page"></td>
    <td width="50%"><img src="docs/screenshots/assess-2-symptoms.png" alt="Assessment step 2: symptoms"></td>
  </tr>
  <tr>
    <td align="center"><sub>Home</sub></td>
    <td align="center"><sub>4-step assessment (symptoms step)</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/result-high-risk.png" alt="Result: elevated risk with explanation"></td>
    <td width="50%"><img src="docs/screenshots/result-low-risk.png" alt="Result: lower risk"></td>
  </tr>
  <tr>
    <td align="center"><sub>Result with "what influenced this estimate"</sub></td>
    <td align="center"><sub>A lower-risk result</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/screenshots/how-it-works.png" alt="How it works page"></td>
    <td width="50%"><img src="docs/screenshots/history.png" alt="Prediction history"></td>
  </tr>
  <tr>
    <td align="center"><sub>How it works: models, metrics, limitations, privacy</sub></td>
    <td align="center"><sub>Private per-visitor history</sub></td>
  </tr>
</table>

<p align="center">
  <img src="docs/screenshots/home-dark.png" alt="Dark mode" width="48%">
  <img src="docs/screenshots/mobile-assessment.png" alt="Phone layout" width="22%">
</p>
<p align="center"><sub>Dark mode and the phone layout</sub></p>

## Features

- **4-step assessment** (basics with live BMI, symptoms, health profile, heart tests), validated in the browser
  and again on the server, with a loading screen and an animated result.
- **Three models behind a dropdown** (Logistic Regression, Random Forest, XGBoost), each with its own tuned threshold.
- **Explanations:** for each prediction, which answers moved the estimate most, found by swapping each answer
  with values from real patients and measuring the change. It describes the model, not causes.
- **Optional fields:** anything left blank is imputed, and the page says the estimate is less reliable.
- **Private history:** each browser gets an anonymous ID in an HttpOnly cookie; history and "Clear history" only
  touch that visitor's rows, and rows are deleted automatically after 7 days.
- **How it works page** with live metrics, limitations and privacy details, plus dark mode and a responsive layout.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python app.py                     # open http://127.0.0.1:5000
```

The trained models are included in `models/`, so the app runs right away. To retrain everything:

```bash
python -m src.train               # compare 3 models, writes models/ and reports/
python -m src.load_db             # optional: load the cleaned dataset into SQLite
python tests/smoke_test_app.py    # tests every page, model, validation rule and privacy behaviour
```

`notebooks/01_eda.ipynb` contains the exploratory analysis.

### With Docker

Needs [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL 2 on Windows).

```bash
docker compose up --build         # then open http://localhost:5000
```

History lives in a Docker volume (`heart-data`) and survives restarts; `docker compose down -v` deletes it.
Without Compose: `docker build -t pulsecheck .` then `docker run -p 5000:5000 -v heart-data:/data pulsecheck`.

The image uses `python:3.10-slim` plus `libgomp1` (needed by XGBoost), installs only the runtime packages from
`requirements-app.txt`, runs as a non-root user, and serves with gunicorn. Those packages are **pinned** to the
versions that trained `models/*.joblib`, because pickled models only load reliably with the same scikit-learn and
XGBoost versions. If you retrain with different versions, update the pins. The port comes from `PORT` and the
database from `DATABASE_URL`, so the same image can run on other container hosts.

## Method

1. **Target.** `num` (0-4) is collapsed to binary: disease if `num > 0` (55% positive).
2. **Cleaning.** Cholesterol and blood pressure of 0 are impossible, so they are treated as missing.
   Numeric columns: median imputation + standard scaling. Categorical: a "missing" category + one-hot encoding.
   Everything sits inside a scikit-learn `Pipeline`, so it is fit on training data only (no leakage) and the
   app applies identical preprocessing.
3. **Models.** Logistic Regression (baseline), Random Forest, XGBoost. Stratified 80/20 split; 5-fold CV on the
   736 training rows; `RandomizedSearchCV` scored on F2 (recall weighted 4x).
4. **Threshold.** Each model's cutoff is chosen on out-of-fold training predictions to maximise F2, then
   evaluated once on the untouched 184-row test set.

## Results in detail

| Model | Threshold | Recall | Precision | F1 | ROC-AUC | Missed sick (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.892 | 0.812 | 0.850 | 0.897 | 11 | 21 |
| Logistic Regression | 0.32 (tuned) | 0.961 | 0.690 | 0.803 | 0.897 | 4 | 44 |
| Random Forest | 0.50 | 0.843 | 0.835 | 0.839 | 0.918 | 16 | 17 |
| Random Forest | 0.28 (tuned) | 0.980 | 0.730 | 0.837 | 0.918 | 2 | 37 |
| XGBoost | 0.50 | 0.882 | 0.789 | 0.833 | 0.917 | 12 | 24 |
| **XGBoost** | **0.40 (tuned)** | **0.961** | **0.742** | **0.838** | **0.917** | **4** | **34** |

XGBoost was selected because it had the best cross-validated F2 (0.865 vs 0.851 for Random Forest and 0.836 for
Logistic Regression). Full tables and plots are in `reports/`.

**Caveats, stated plainly**
- **The models are close.** With 184 test patients, one patient moves recall by about 1 point, so differences
  between models are within noise. Logistic Regression is competitive with XGBoost here.
- **Hospital effects.** `ca` is ~99% missing at three of the four hospitals, and Switzerland recorded no cholesterol.
  Disease rates differ a lot by hospital (Hungary 36%, Switzerland 93%), so missingness can act as a hidden
  hospital label. The hospital column is never used as a feature, and `ca`, `thal` and `slope` are dropped. Keeping
  them gave only a small CV gain (F2 +0.011), which is not worth the risk. Per-hospital test recall for the final
  model is 0.95-0.96 everywhere (`reports/per_hospital.csv`), but those subsets are tiny (28-64 patients).
- **"No chest pain" raises the estimate.** In this dataset, patients with no chest pain who were still examined had
  a 79% disease rate (they were referred for other reasons). The model learned that selection effect, so it can look
  backwards for a healthy person. The app warns about this next to the question.
- **Small, old, mostly male data** from four hospitals. This is an educational project, not a diagnostic tool.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask app: page routes and the `/api/predict` JSON endpoint |
| `src/preprocess.py` | Loading, cleaning, and the imputation/encoding/scaling pipeline |
| `src/train.py` | CV comparison, tuning, threshold selection, test evaluation, saves all three models |
| `src/evaluate.py` | Recall/precision/F1/AUC, confusion matrices, precision-recall curves |
| `src/predict.py` | Input validation, prediction, and per-prediction explanations |
| `src/db.py`, `src/load_db.py` | SQLAlchemy models (`patients`, `predictions`), per-visitor history, cleanup, migration |
| `templates/`, `static/` | Jinja2 pages, CSS, and the JavaScript for the 4-step wizard |
| `models/` | The three trained pipelines (`.joblib`), each with its own threshold and test metrics |
| `reports/` | Metrics tables and plots produced by training |
| `tests/` | End-to-end smoke test |
| `Dockerfile`, `docker-compose.yml`, `requirements-app.txt` | Container setup |
| `docs/` | Screenshots and demo GIF used in this README |

## Privacy

History is private per visitor: a random anonymous ID is kept in an HttpOnly cookie, and every history query and
"Clear history" is filtered by it, so visitors never see or delete each other's predictions. Only the model's
inputs are stored (never height or weight), and rows are deleted automatically after 7 days (`HISTORY_DAYS`).
Over HTTPS the cookie is sent with `Secure; SameSite=None` so it also works inside an iframe; locally it is
`SameSite=Lax`. Limits: clearing cookies or switching browser starts a fresh history, and there is no CSRF token
(the worst a forged request can do is add to or clear the victim's own demo history). The behaviour is covered by
the tests, but the app has not been run on a public host.

## Using PostgreSQL instead of SQLite

Set `DATABASE_URL` (for example `postgresql+psycopg2://user:pass@localhost:5432/heart`) and
`pip install psycopg2-binary`. No code changes are needed.

## Deploying (optional)

This repo is container-ready. To host it on Hugging Face Spaces, create a Docker Space and put this block at the
very top of `README.md` (Hugging Face reads it to know how to build the app):

```yaml
---
title: PulseCheck Heart Risk
emoji: ❤️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 5000
pinned: false
---
```

Free hosting tiers have temporary storage, so saved history resets when the app restarts.

## Data

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland, Hungary, Switzerland,
VA Long Beach), via the Kaggle mirror `heart_disease_uci.csv`. Donors: Janosi, Steinbrunn, Pfisterer, Detrano.

**Disclaimer:** PulseCheck is an educational machine-learning project. It is not a medical device and cannot
diagnose heart disease. If you have severe chest pain, trouble breathing or fainting, call your local emergency number.
