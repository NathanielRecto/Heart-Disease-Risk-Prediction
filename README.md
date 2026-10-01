# PulseCheck: Heart Disease Risk Prediction

An end-to-end machine-learning project: clean a real multi-hospital medical dataset, compare three models,
tune for the metric that matters (**recall**: a missed sick patient is worse than a false alarm), and serve
the result as a Flask web app with a 4-step assessment, per-prediction explanations and private per-visitor history.

**Stack:** Python · scikit-learn · XGBoost · Pandas · Flask · SQLite/SQLAlchemy · HTML/CSS/JavaScript · Docker

<p align="center">
  <img src="docs/demo.gif" alt="Demo: filling in the 4-step assessment and getting a result" width="760">
</p>

## Results at a glance

Held-out test set of 184 patients, never used for training, tuning or model selection. The final model is
XGBoost with a decision threshold of 0.40, chosen on out-of-fold training predictions.

| Recall | Precision | F1 | ROC-AUC | Accuracy | Missed sick patients | False alarms |
|---|---|---|---|---|---|---|
| **96%** (98 of 102) | 74% | 0.84 | 0.92 | 79% | 4 | 34 |

Accuracy is deliberately not the headline: lowering the threshold from 0.50 to 0.40 cut missed patients from 12
to 4 at the cost of 10 more false alarms. The three models score within noise of each other on a test set this
small, which is discussed under [Why XGBoost](#why-xgboost).

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

## The ML pipeline

```mermaid
flowchart TD
    A["Raw data<br/>920 patients, 4 hospitals<br/>heart_disease_uci.csv"] --> B["Clean<br/>binary target: num above 0<br/>impossible zeros become missing<br/>drop id and hospital"]
    B --> C{"Stratified split<br/>seed 42"}
    C -->|"80% = 736"| D["Training set"]
    C -->|"20% = 184"| T["Test set<br/>locked until the very end"]
    D --> E["Preprocessing pipeline<br/>impute, scale, one-hot<br/>refit inside every CV fold"]
    E --> F["5-fold CV comparison<br/>Logistic Regression, Random Forest, XGBoost<br/>with vs without the sparse columns"]
    F --> G["Tune each model<br/>RandomizedSearchCV, score = F2"]
    G --> H["Choose each threshold<br/>maximise F2 on out-of-fold predictions"]
    H --> I["Pick the final model<br/>best cross-validated F2"]
    I --> J["Evaluate once on the test set<br/>recall, precision, F1, ROC-AUC, confusion matrix"]
    T --> J
    J --> K["Save all 3 pipelines<br/>models/*.joblib"]
    K --> L["Flask app<br/>validate, predict, explain, log to SQLite"]
```

1. **Target.** `num` (0-4) is collapsed to binary: disease if `num > 0` (55% positive).
2. **Cleaning.** Cholesterol and blood pressure of 0 are impossible, so they are treated as missing. The hospital
   and the id are never used as features.
3. **Preprocessing.** Numeric columns: median imputation + standard scaling. Categorical: a "missing" category +
   one-hot encoding. Everything sits inside a scikit-learn `Pipeline`, so it is fit on training data only (no
   leakage) and the app applies identical preprocessing at prediction time.
4. **Model comparison.** Logistic Regression (baseline), Random Forest, XGBoost, each scored with stratified 5-fold
   cross-validation on the 736 training rows, with and without the mostly-missing columns `ca`, `thal`, `slope`.
5. **Tuning.** `RandomizedSearchCV` (up to 15 settings per model) scored on **F2**, which weights recall four times
   as much as precision.
6. **Threshold.** Each model's cutoff is chosen on out-of-fold training predictions, so the test set plays no part.
7. **Final evaluation.** The untouched 184-patient test set is used once.

## Why XGBoost

**The selection rule was fixed before looking at test results:** pick the model with the best *cross-validated F2*
after tuning. Choosing by test results would leak the test set into the decision.

| Model | CV F2, defaults (0.50 cutoff) | CV F2, tuned (0.50 cutoff) | Test recall* | Test precision* | Test ROC-AUC | Model file |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.836 | 0.961 | 0.690 | 0.897 | 6 KB |
| Random Forest | 0.838 | 0.851 | 0.980 | 0.730 | 0.918 | 3.6 MB |
| **XGBoost** | 0.831 | **0.865** | 0.961 | 0.742 | 0.917 | 253 KB |

\*At each model's own tuned threshold. Test columns are reported for information; they did not drive the choice.

Tuned settings: Logistic Regression `C=0.01` (strong regularisation); Random Forest `min_samples_leaf=3`,
`max_features="sqrt"`, unlimited depth; XGBoost 100 trees, `max_depth=6`, `learning_rate=0.01`, `subsample=0.7`,
`colsample_bytree=0.7`.

**Why XGBoost won.**
- It had the highest tuned cross-validated F2 (0.865 vs 0.851 and 0.836), which is the metric that matches the goal
  of catching sick patients without flagging everyone.
- It is the most precise of the three at the chosen threshold (0.742: 34 false alarms vs 37 and 44), with the same
  recall as Logistic Regression.
- Gradient-boosted trees can capture non-linear effects and interactions between answers (for example chest pain
  type together with exercise angina) that a plain Logistic Regression cannot, and the saved model is small (253 KB).

**Why not the others.**
- **Logistic Regression** is simpler and fully interpretable, and with default settings it had the *best* ROC-AUC in
  cross-validation (0.88, see the chart below). If interpretability mattered more than the last bit of F2, it would
  be a defensible choice.
- **Random Forest** had the highest test recall (98%, only 2 missed). But it did not have the best cross-validated
  score, and choosing it *because* of its test result would be selecting on the test set. It is also a 3.6 MB file.

**Why it looks last in the default-settings chart.** That chart is *before* tuning. XGBoost has the most settings and
its defaults overfit a small dataset, so it starts last and gains the most from tuning (see the right-hand panel
of the [cross-validation chart](#cross-validation-comparison)). It was chosen on the tuned scores, specifically
because it catches the most sick patients (tuned CV recall 0.885 vs 0.862 and 0.848). It is *not* the best on
ROC-AUC or precision, so it is "best at the metric we set", not "best at everything".

**The honest summary:** on 920 patients the three models are very hard to tell apart. A difference of one patient
moves test recall by about one point, and the default-setting CV scores put Logistic Regression ahead on ROC-AUC.
XGBoost was chosen by a pre-set rule, not because it clearly beats the others.

## Evaluation reports

All figures are in [`reports/`](reports/). The first two come from `python -m src.train`; the others come from
`python -m src.report`, which rebuilds them from the saved models without retraining and first checks that the
recreated test split reproduces the stored metrics exactly.

### Cross-validation comparison

<p align="center"><img src="reports/cv_comparison.png" alt="Cross-validation comparison of the three models, before and after tuning" width="900"></p>

Left: every model with its default settings. Right: the same models after tuning, scored on the same 5 folds
(`reports/cv_tuned.csv`). Both axes start at 0.6, which makes small gaps look bigger than they are.

- **With defaults, XGBoost is the weakest of the three** (ROC-AUC 0.84 vs 0.88 for Logistic Regression). Its default
  settings overfit a dataset of only 736 training rows.
- **After tuning, XGBoost has the best recall (0.885) and F2/F1**, a middle precision (0.80), and a ROC-AUC (0.868)
  that is essentially tied with the others (0.876 and 0.871). It gained the most from tuning: F2 +0.034, against
  +0.011 for Logistic Regression and +0.013 for Random Forest.
- **Caution:** because XGBoost gained the most from tuning, it is also the most exposed to optimistic bias. Its
  settings were the best of 15 random tries scored on these same folds, and there is no nested cross-validation to
  correct for that, so its tuned score is probably a little flattering. The held-out test set is the cleaner check.

**Dropping the sparse columns.** `ca` is ~99% missing at three of the four hospitals, `thal` and `slope` are also
heavily missing, and missingness depends on the hospital. Mean CV scores with and without them:

| Columns | Logistic Regression F2 | Random Forest F2 | XGBoost F2 | Mean F2 |
|---|---|---|---|---|
| Drop `ca`, `thal`, `slope` (used) | 0.825 | 0.838 | 0.831 | 0.831 |
| Keep them | 0.838 | 0.843 | 0.844 | 0.842 |

Keeping them gains only about 0.011 F2. That is not worth the risk that the model learns "which hospital" instead
of "which patient", and a person filling in a form would not have those values anyway, so they are dropped.

### Test-set confusion matrices

<p align="center"><img src="reports/confusion_matrices.png" alt="Confusion matrices on the test set" width="900"></p>

Each matrix is on the same 184 patients (102 sick, 82 healthy) at that model's tuned threshold. Missed sick patients
(bottom left): 4 for Logistic Regression, 2 for Random Forest, 4 for XGBoost. The cost is false alarms (top right):
44, 37 and 34. For XGBoost that is 34 of 82 healthy patients (41%) flagged, which is the price of 96% recall.

### ROC curves

<p align="center"><img src="reports/roc_curves.png" alt="ROC curves on the test set" width="520"></p>

The curves overlap almost everywhere (AUC 0.90, 0.92, 0.92). The dots mark each model's chosen threshold: far up
the curve (high recall) but well to the right (many healthy patients flagged), which is the deliberate trade.

### Precision-recall curves

<p align="center"><img src="reports/precision_recall.png" alt="Precision-recall curves on the test set" width="520"></p>

Precision is above 0.9 up to about 0.6 recall, slides to roughly 0.8 near 0.85 recall and to about 0.75 near 0.95.
The chosen thresholds (dots) sit at the steep end, where gaining the last few points of recall costs the most
precision. XGBoost and Random Forest are almost level; Logistic Regression is a little lower at high recall.

### Choosing the threshold

<p align="center"><img src="reports/threshold_tradeoff.png" alt="Recall, precision and F2 versus threshold" width="900"></p>

Out-of-fold results on the training data, which is what the thresholds were chosen from. Lowering the cutoff below
the default 0.50 raises recall and lowers precision. The F2 curve is nearly flat (about 0.86-0.89) over a wide range
of cutoffs, so the exact threshold matters less than the decision to move it down from 0.50.

**A sanity baseline.** More than half the patients are sick (55%), so a "model" that flags *everyone* already has
recall 1.00, precision 0.55 and F2 0.86 (the flat left edge of these curves). F2 is therefore a lenient yardstick
here, and the left end of each curve is exactly that trivial model. What the real models add is fewer false alarms at
almost the same recall:

| On the 184 test patients | Recall | Healthy patients flagged (of 82) | Precision | Test F2 |
|---|---|---|---|---|
| Flag everyone (trivial baseline) | 1.00 | 82 | 0.55 | 0.861 |
| Logistic Regression | 0.96 | 44 | 0.69 | 0.891 |
| Random Forest | 0.98 | 37 | 0.73 | 0.917 |
| **XGBoost** | 0.96 | 34 | 0.74 | 0.907 |

So the gain is real but modest: XGBoost gives up 4 points of recall to avoid 48 of the 82 possible false alarms.

### What the final model relies on

<p align="center"><img src="reports/feature_importance.png" alt="Permutation importance of the final model" width="560"></p>

Permutation importance on the test set: how much ROC-AUC drops when one answer is shuffled. Chest pain type matters
most by a wide margin, then ST depression, exercise-induced angina, cholesterol and sex. Age, resting ECG and
resting blood pressure add essentially nothing (their error bars cross zero). Treat this as indicative: with 184
test patients it is noisy, correlated features share credit, and chest pain's weight partly reflects the dataset's
selection effect (see below).

### Per-hospital check (final model, test set)

| Hospital | Patients | Disease rate | Recall | Precision |
|---|---|---|---|---|
| Cleveland | 55 | 47% | 0.96 | 0.66 |
| Hungary | 64 | 34% | 0.95 | 0.66 |
| Switzerland | 28 | 96% | 0.96 | 0.96 |
| VA Long Beach | 37 | 73% | 0.96 | 0.74 |

Recall is consistent across hospitals, but each subset is small. Raw tables: `reports/*.csv`.

## Caveats, stated plainly

- **F2 is lenient on this data.** Flagging every patient already scores F2 0.86, so F2 differences between models
  are small in absolute terms (see the baseline table above). Compare false alarms at a given recall as well.
- **The models are close.** With 184 test patients, one patient moves recall by about 1 point, so differences
  between models are within noise. There is no nested cross-validation, no repeated runs or confidence intervals,
  and no leave-one-hospital-out evaluation.
- **Hospital effects.** Disease rates differ a lot by hospital (Hungary 36%, Switzerland 93%), and Switzerland recorded
  no cholesterol. The hospital column is never used as a feature, but the split is random, so patients from the same
  hospital appear in both train and test.
- **"No chest pain" raises the estimate.** In this dataset, patients with no chest pain who were still examined had a
  79% disease rate (they were referred for other reasons). The model learned that selection effect, so it can look
  backwards for a healthy person. The app warns about this next to the question.
- **Small, old, mostly male data** from four hospitals. This is an educational project, not a diagnostic tool.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python app.py                     # open http://127.0.0.1:5000
```

The trained models are included in `models/`, so the app runs right away. To reproduce everything:

```bash
python -m src.train               # compare 3 models, writes models/ and reports/
python -m src.report              # extra figures from the saved models (no retraining)
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
database from `DATABASE_URL`.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask app: page routes and the `/api/predict` JSON endpoint |
| `src/preprocess.py` | Loading, cleaning, and the imputation/encoding/scaling pipeline |
| `src/train.py` | CV comparison, tuning, threshold selection, test evaluation, saves all three models |
| `src/evaluate.py` | Recall/precision/F1/AUC, confusion matrices, precision-recall curves |
| `src/report.py` | ROC, threshold, CV and feature-importance figures from the saved models |
| `src/predict.py` | Input validation, prediction, and per-prediction explanations |
| `src/db.py`, `src/load_db.py` | SQLAlchemy models (`patients`, `predictions`), per-visitor history, cleanup, migration |
| `templates/`, `static/` | Jinja2 pages, CSS, and the JavaScript for the 4-step wizard |
| `models/` | The three trained pipelines (`.joblib`), each with its own threshold and test metrics |
| `reports/` | Metrics tables and plots produced by training and `src.report` |
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

## Data

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland, Hungary, Switzerland,
VA Long Beach), via the Kaggle mirror `heart_disease_uci.csv`. Donors: Janosi, Steinbrunn, Pfisterer, Detrano.

**Disclaimer:** PulseCheck is an educational machine-learning project. It is not a medical device and cannot
diagnose heart disease. If you have severe chest pain, trouble breathing or fainting, call your local emergency number.
