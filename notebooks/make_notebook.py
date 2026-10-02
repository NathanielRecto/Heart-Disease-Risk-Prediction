"""Builds notebooks/01_eda.ipynb (run once; then open the notebook in VS Code)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# 01 - Exploratory data analysis\nUCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach): "
       "920 rows, 918 patients after removing 2 exact duplicates."),
    code("""import sys; sys.path.insert(0, '..')
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.preprocess import load_data
sns.set_theme(style='whitegrid')
df = load_data('../data/heart_disease_uci.csv')
df.head()"""),
    md("## 1. Target\nOriginal `num` is 0-4 (severity). We use `target = num > 0`."),
    code("""print(df['target'].value_counts(normalize=True).round(3))
df['target'].value_counts().plot.bar(title='Healthy (0) vs disease (1)');"""),
    md("## 2. Missing values\n`chol` and `trestbps` zeros were converted to NaN (impossible values)."),
    code("""miss = df.isna().mean().sort_values(ascending=False)
miss[miss > 0].plot.barh(title='Fraction missing');
miss[miss > 0].round(3)"""),
    md("## 3. Missingness depends on the hospital\n`ca` is almost only recorded at Cleveland; Switzerland has no cholesterol at all. "
       "Missingness therefore leaks *which hospital* a patient came from, and hospitals have very different disease rates, "
       "so the model must not learn from it. This is why the hospital column is never a feature, why `train.py` "
       "compares dropping vs keeping `ca`/`thal`/`slope`, and why blanks are filled with typical values without a "
       "\"was missing\" indicator."),
    code("""for c in ['chol', 'fbs', 'trestbps', 'thalch']:
    m = df[c].isna()
    print(f"{c:9s} missing n={m.sum():3d}  disease rate if missing {df.loc[m,'target'].mean():.2f}  vs present {df.loc[~m,'target'].mean():.2f}")"""),
    code("""by_site = df.groupby('dataset').agg(n=('target','size'), disease_rate=('target','mean'),
        ca_missing=('ca', lambda s: s.isna().mean()), thal_missing=('thal', lambda s: s.isna().mean()),
        chol_missing=('chol', lambda s: s.isna().mean())).round(2)
by_site"""),
    md("## 4. Numeric features vs target"),
    code("""num = ['age','trestbps','chol','thalch','oldpeak']
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
for ax, c in zip(axes, num):
    sns.kdeplot(data=df, x=c, hue='target', common_norm=False, fill=True, ax=ax)
fig.tight_layout()"""),
    md("## 5. Categorical features vs disease rate"),
    code("""cats = ['sex','cp','fbs','restecg','exang','slope','thal']
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, c in zip(axes.ravel(), cats):
    df.groupby(c)['target'].mean().sort_values().plot.barh(ax=ax, title=f'Disease rate by {c}')
axes.ravel()[-1].axis('off'); fig.tight_layout()"""),
    md("Chest pain type is striking: **asymptomatic** patients (no chest pain) have the highest disease rate. "
       "Men have a much higher rate than women, partly because women are under-represented here."),
]
nbf.write(nb, "01_eda.ipynb")
