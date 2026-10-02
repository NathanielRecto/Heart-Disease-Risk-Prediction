# PulseCheck: Heart Disease Risk Prediction

An end-to-end machine-learning project: clean a real multi-hospital medical dataset, compare three models,
tune for the metric that matters (**recall**: a missed sick patient is worse than a false alarm), calibrate the
outputs so the percentages mean something, and serve the result as a Flask web app with a 4-step assessment,
per-prediction explanations and private per-visitor history.

**Stack:** Python · scikit-learn · XGBoost · Pandas · Flask · SQLite/SQLAlchemy · HTML/CSS/JavaScript · Docker

<p align="center">
  <img src="docs/demo.gif" alt="Demo: filling in the 4-step assessment and getting a result" width="760">
</p>

## Results at a glance

Held-out test set of 184 patients (102 sick, 82 healthy), never used for training, tuning, calibration or model
selection. The final model is XGBoost, flagging anyone with a calibrated risk of 18% or more; the cutoff was chosen
on out-of-fold training predictions.

| Recall | Precision | F1 | ROC-AUC | Brier score | Accuracy | Missed sick patients | False alarms |
|---|---|---|---|---|---|---|---|
| **99%** (101 of 102) | 70% | 0.82 | 0.90 | 0.125 | 76% | 1 | 44 of 82 |

Accuracy is deliberately not the headline: lowering the cutoff from the default 0.50 to 0.18 cut missed patients
from 13 to 1 at the cost of 22 more false alarms. The three models are within noise of each other, and on this test
set Random Forest did as well on recall with fewer false alarms; see [Why XGBoost](#why-xgboost).

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
- **Calibrated probabilities:** "30%" means roughly 30 in 100 similar patients in the data had heart disease.
- **Three models behind a dropdown** (Logistic Regression, Random Forest, XGBoost), each with its own tuned cutoff.
- **Explanations:** for each prediction, which answers moved the estimate most, found by swapping each answer
  with values from real patients and measuring the change. It describes the model, not causes.
- **Optional fields:** anything left blank is filled with the typical training value, and a blank on its own never
  changes the estimate (this is tested).
- **Private history:** each browser gets an anonymous ID in an HttpOnly cookie; history and "Clear history" only
  touch that visitor's rows, and rows are deleted automatically after 7 days.
- **How it works page** with live metrics, limitations and privacy details, plus dark mode and a responsive layout.

## The ML pipeline

```mermaid
flowchart TD
    A["Raw data<br/>920 rows, 4 hospitals<br/>heart_disease_uci.csv"] --> B["Clean<br/>remove 2 exact duplicates: 918 patients<br/>binary target: num above 0<br/>impossible zeros become missing<br/>drop id and hospital"]
    B --> C{"Stratified split<br/>seed 42"}
    C -->|"80% = 734"| D["Training set"]
    C -->|"20% = 184"| T["Test set<br/>locked until the very end"]
    D --> E["Preprocessing pipeline<br/>fill blanks with typical values, no missing flags<br/>scale, one-hot<br/>refit inside every CV fold"]
    E --> F["5-fold CV comparison<br/>Logistic Regression, Random Forest, XGBoost<br/>with vs without the sparse columns"]
    F --> G["Tune each model<br/>RandomizedSearchCV, score = F2"]
    G --> H["Out-of-fold predictions<br/>fit calibration, then choose the cutoff<br/>that maximises F2"]
    H --> I["Pick the final model<br/>best cross-validated F2"]
    I --> J["Evaluate once on the test set<br/>recall, precision, F1, ROC-AUC, Brier, confusion matrix"]
    T --> J
    J --> K["Save all 3 pipelines + calibrators<br/>models/*.joblib"]
    K --> L["Flask app<br/>validate, predict, explain, log to SQLite"]
```

1. **Target.** `num` (0-4) is collapsed to binary: disease if `num > 0` (55% of patients).
2. **Cleaning.** Two patients appear twice (identical on every column), so one copy of each is dropped. Cholesterol
   and blood pressure of 0 are impossible, so they are treated as missing. The hospital and the id are never features.
3. **Preprocessing.** Numeric columns: median imputation + standard scaling. Categorical: most-frequent imputation +
   one-hot encoding. **Missingness is deliberately not a feature** (no "was missing" flags), because in this data it
   gives away the hospital; see [Review fixes](#review-fixes). Everything sits inside a scikit-learn `Pipeline`, so
   it is fit on training data only and the app applies identical preprocessing.
4. **Model comparison.** Logistic Regression (baseline), Random Forest, XGBoost, each scored with stratified 5-fold
   cross-validation on the 734 training patients, with and without the mostly-missing columns `ca`, `thal`, `slope`.
5. **Tuning.** `RandomizedSearchCV` (up to 15 settings per model) scored on **F2**, which weights recall four times
   as much as precision.
6. **Calibration and cutoff.** Out-of-fold predictions (each training patient scored by a model that never saw them)
   are used to fit Platt scaling and then to choose the cutoff, so the test set plays no part in either.
7. **Final evaluation.** The untouched 184-patient test set is used once.

## Why XGBoost

**The selection rule was fixed in advance:** pick the model with the best *cross-validated F2* after tuning.
Choosing by test results would leak the test set into the decision.

| Model | CV F2, defaults | CV F2, tuned | Test recall* | Test precision* | False alarms* | Test ROC-AUC | Model file |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.826 | 0.828 | 0.971 | 0.733 | 36 | 0.896 | 6 KB |
| Random Forest | 0.844 | 0.848 | 0.990 | 0.732 | 37 | 0.899 | 1.3 MB |
| **XGBoost** | 0.809 | **0.852** | 0.990 | 0.697 | 44 | 0.901 | 253 KB |

\*At each model's own tuned cutoff (0.28, 0.28 and 0.18). CV F2 is scored at the default 0.50 cutoff. The test
columns are reported for information; they did not drive the choice.

Tuned settings: Logistic Regression `C=10`; Random Forest `max_depth=5`, `max_features="sqrt"`; XGBoost 100 trees,
`max_depth=6`, `learning_rate=0.01`, `subsample=0.7`, `colsample_bytree=0.7`.

**Why it won.** XGBoost had the highest tuned cross-validated F2 (0.852 vs 0.848 and 0.828), driven by the best
cross-validated recall (0.874). That is the metric that matches the goal of catching sick patients.

**Why it looks last in the default-settings chart.** With default settings XGBoost is the weakest of the three; its
defaults overfit a dataset of only 734 training patients. Tuning (slow learning rate, row and column subsampling)
gives it the biggest improvement, from 0.809 to 0.852 F2. It is best at the metric we set, not best at everything:
its cross-validated precision (0.775) and ROC-AUC (0.856) are the lowest of the three.

**What the test set says.** Random Forest matched XGBoost's recall (both missed 1 of 102 sick patients) with 7
fewer false alarms (37 vs 44). The cross-validated gap between them (0.852 vs 0.848) is far smaller than the noise
on 734 patients, so the honest reading is that they are tied. I kept the pre-set rule rather than switching to
Random Forest after seeing the test results, because switching would be choosing on the test set. In hindsight, a
tie-breaker decided in advance (for example "within 0.01 F2, prefer the higher cross-validated precision") would
have been a better rule.

**Logistic Regression** is simpler, fully interpretable, and has the best cross-validated ROC-AUC. It missed 3 sick
patients on the test set instead of 1. If interpretability mattered more, it would be a defensible choice.

## Evaluation reports

All figures are in [`reports/`](reports/). The confusion matrices and precision-recall curves come from
`python -m src.train`; the others come from `python -m src.report`, which rebuilds them from the saved models without
retraining and first checks that the recreated test split reproduces the stored metrics exactly.

### Cross-validation comparison

<p align="center"><img src="reports/cv_comparison.png" alt="Cross-validation comparison of the three models, before and after tuning" width="900"></p>

Left: every model with default settings. Right: the same models after tuning, scored on the same 5 folds
(`reports/cv_tuned.csv`). Both axes start at 0.6, which makes small gaps look bigger than they are. XGBoost goes from
last to first on recall. Because it gained the most from tuning, it is also the most exposed to optimistic bias: its
settings were the best of 15 random tries scored on these same folds, with no nested cross-validation to correct
for that.

**Dropping the sparse columns.** `ca` is ~99% missing at three of the four hospitals, and `thal` and `slope` are also
heavily missing. Mean CV F2 with and without them (default settings):

| Columns | Logistic Regression | Random Forest | XGBoost | Mean |
|---|---|---|---|---|
| Drop `ca`, `thal`, `slope` (used) | 0.826 | 0.844 | 0.809 | 0.826 |
| Keep them | 0.834 | 0.841 | 0.818 | 0.831 |

Keeping them gains only about 0.005 F2, not worth the risk that the model learns "which hospital" instead of
"which patient". A person filling in the form would not have those values anyway.

### Calibration

<p align="center"><img src="reports/calibration.png" alt="Calibration curves before and after Platt scaling" width="900"></p>

Each point groups the test patients by predicted probability and shows how many of them actually had heart
disease; perfect calibration is the dotted diagonal. XGBoost's raw scores (grey) were badly compressed: they only
ranged from about 0.2 to 0.8, so low-risk patients were shown far too high a risk and high-risk patients too low.
Platt scaling, fitted on out-of-fold training predictions, puts it on the diagonal and improves its Brier score
from 0.151 to 0.125 (lower is better). Logistic Regression was already well calibrated (0.130 before and after).
Calibration is monotonic, so it does not change ROC-AUC, the ranking of patients, or which patients get flagged.

### Test-set confusion matrices

<p align="center"><img src="reports/confusion_matrices.png" alt="Confusion matrices on the test set" width="900"></p>

Each matrix is on the same 184 patients (102 sick, 82 healthy) at that model's tuned cutoff. Missed sick patients
(bottom left): 3 for Logistic Regression, 1 for Random Forest, 1 for XGBoost. The cost is false alarms (top right):
36, 37 and 44. For XGBoost that is 44 of 82 healthy patients (54%) flagged, which is the price of 99% recall.

### ROC curves

<p align="center"><img src="reports/roc_curves.png" alt="ROC curves on the test set" width="520"></p>

The curves overlap almost everywhere (AUC 0.896, 0.899 and 0.901). The dots mark each model's cutoff: high up the
curve (high recall) but well to the right (many healthy patients flagged), which is the deliberate trade.

### Precision-recall curves

<p align="center"><img src="reports/precision_recall.png" alt="Precision-recall curves on the test set" width="520"></p>

Precision stays around 0.9 or higher up to about 0.6 recall and slides to roughly 0.8 by 0.85 recall. The cutoffs
(dots) sit at the steep end near 0.97-0.99 recall, where the last few points of recall cost the most precision.

### Choosing the cutoff

<p align="center"><img src="reports/threshold_tradeoff.png" alt="Recall, precision and F2 versus cutoff" width="900"></p>

Out-of-fold results on the training data, on the calibrated scale the cutoffs were chosen on. Lowering the cutoff
raises recall and lowers precision. The F2 curve is nearly flat over a wide range of low cutoffs, so the exact
value (0.18 for XGBoost) matters less than the decision to move well below 0.50.

**A sanity baseline.** More than half the patients are sick (55%), so a "model" that flags *everyone* already has
recall 1.00, precision 0.55 and F2 0.86. F2 is therefore a lenient yardstick here. What the real models add is far
fewer false alarms at almost the same recall:

| On the 184 test patients | Recall | Healthy patients flagged (of 82) | Precision | Test F2 |
|---|---|---|---|---|
| Flag everyone (trivial baseline) | 1.00 | 82 | 0.55 | 0.861 |
| Logistic Regression | 0.97 | 36 | 0.73 | 0.912 |
| Random Forest | 0.99 | 37 | 0.73 | 0.925 |
| **XGBoost** | 0.99 | 44 | 0.70 | 0.913 |

So the gain is real but modest: XGBoost gives up one sick patient to avoid 38 of the 82 possible false alarms.

### What the final model relies on

<p align="center"><img src="reports/feature_importance.png" alt="Permutation importance of the final model" width="560"></p>

Permutation importance on the test set: how much ROC-AUC drops when one answer is shuffled. Chest pain type matters
most by a wide margin, then exercise-induced angina, ST depression, sex and cholesterol. Fasting blood sugar and
resting ECG add essentially nothing. Fasting blood sugar ranked 6th before the missing-value fix, most likely
because "blood sugar missing" was a stand-in for the Swiss and VA hospitals. Treat this as indicative: with 184 test
patients it is noisy, correlated features share credit, and chest pain's weight partly reflects the dataset's
selection effect (see Caveats).

### Per-hospital check (final model, test set)

| Hospital | Patients | Disease rate | Recall | Precision | False alarms |
|---|---|---|---|---|---|
| Cleveland | 55 | 47% | 1.00 | 0.62 | 16 |
| Hungary | 64 | 34% | 0.95 | 0.53 | 19 |
| Switzerland | 27 | 100% | 1.00 | 1.00 | 0 |
| VA Long Beach | 38 | 71% | 1.00 | 0.75 | 9 |

Recall holds across hospitals, but precision is lowest where disease is rarest (Hungary), as expected from a low
cutoff. Each subset is small. Raw tables: `reports/*.csv`.

## Review fixes

A code and methodology review of the first version found two problems, both fixed and covered by tests:

1. **Missing values leaked the hospital.** The first version added "was this value missing?" as a feature. In this
   dataset missingness depends on the hospital: all 123 Swiss patients lack cholesterol and 93% of them are sick, so
   patients with missing cholesterol had an 81% disease rate versus 48% otherwise. The model learned "blank means
   sick", and in the app, skipping all optional fields raised Logistic Regression's estimate for a typical patient from
   36% to 64% and flipped XGBoost from "lower" to "elevated". Now blanks are filled with typical values and carry no
   signal: a test checks that leaving any field blank gives *exactly* the same result as typing in the typical value.
2. **The percentages were not probabilities.** XGBoost's raw outputs only ranged from 0.23 to 0.80, and patients it
   scored around 30% were actually sick 8% of the time. All models are now calibrated (see above).

Smaller fixes: two duplicate patients removed, malformed API requests now return 400 instead of crashing, request size
is capped, `requirements.txt` is pinned so the saved models keep loading, and database start-up tolerates several
server workers starting at once.

The first version reported 96% recall, 74% precision and 0.92 ROC-AUC. The new numbers are 99%, 70% and 0.90. They are
not directly comparable (removing the duplicates changes the split), but a slightly lower ROC-AUC is the expected cost
of no longer letting the model read the hospital from missing values.

## Caveats, stated plainly

- **F2 is lenient on this data.** Flagging every patient already scores F2 0.86, so F2 differences between models are
  small in absolute terms. Compare false alarms at a given recall as well.
- **The models are close.** With 184 test patients, one patient moves recall by about 1 point. There is no nested
  cross-validation, no repeated runs or confidence intervals, and no leave-one-hospital-out evaluation.
- **Hospital effects.** Disease rates differ a lot by hospital (Hungary 36%, Switzerland 93%). The hospital column and
  missingness are never used, but the split is random, so patients from the same hospital appear in both train and test.
- **"No chest pain" raises the estimate.** In this dataset, patients with no chest pain who were still examined had a
  79% disease rate (they were referred for other reasons). The model learned that selection effect, so it can look
  backwards for a healthy person. The app warns about this next to the question.
- **Blank fields mean "average for these patients".** Typical values come from patients who were all referred for heart
  tests, so a blank nudges the estimate toward their average risk.
- **Small, old, mostly male data** (725 of 918 patients are men) from four hospitals. This is an educational project,
  not a diagnostic tool.

## Quick start

Tested with Python 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python app.py                     # open http://127.0.0.1:5000
```

The trained models are included in `models/`, so the app runs right away. To reproduce everything:

```bash
python -m src.train               # compare, tune, calibrate 3 models; writes models/ and reports/
python -m src.report              # extra figures from the saved models (no retraining)
python -m src.load_db             # optional: load the cleaned dataset into SQLite
python tests/smoke_test_app.py    # tests pages, models, validation, the missing-value fix and privacy
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
`requirements-app.txt`, runs as a non-root user, and serves with gunicorn. Both requirements files are **pinned** to
the versions that trained `models/*.joblib`, because pickled models only load reliably with the same scikit-learn and
XGBoost versions. If you retrain with different versions, update the pins. The port comes from `PORT` and the
database from `DATABASE_URL`.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask app: page routes and the `/api/predict` JSON endpoint |
| `src/preprocess.py` | Loading, cleaning, and the imputation/encoding/scaling pipeline |
| `src/train.py` | CV comparison, tuning, calibration, cutoff selection, test evaluation, saves all three models |
| `src/calibration.py` | Platt scaling fitted on out-of-fold predictions |
| `src/evaluate.py` | Recall/precision/F1/AUC/Brier, confusion matrices, precision-recall curves |
| `src/report.py` | ROC, cutoff, CV, calibration and feature-importance figures from the saved models |
| `src/predict.py` | Input validation, prediction, and per-prediction explanations |
| `src/db.py`, `src/load_db.py` | SQLAlchemy models (`patients`, `predictions`), per-visitor history, cleanup, migration |
| `templates/`, `static/` | Jinja2 pages, CSS, and the JavaScript for the 4-step wizard |
| `models/` | The three trained pipelines and calibrators (`.joblib`), each with its own cutoff and test metrics |
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
