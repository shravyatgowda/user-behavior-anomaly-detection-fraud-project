# User Behavior Anomaly Detection

A risk-analytics pipeline that flags account takeover, transaction velocity abuse, and
spending anomalies from multi-table user/device/login/transaction data — combining
**SQL-based rule signals** with an **unsupervised ML model (Isolation Forest)**, then
evaluating both against ground truth.

Built to mirror a real trust & safety analyst workflow: raw relational tables → SQL joins →
feature table → model → measured precision/recall, not just "here's a chart."

## Why this project

Most portfolio fraud-detection projects run a classifier on a pre-cleaned Kaggle CSV and stop.
This one instead:
- Starts from **4 related raw tables** (users, devices, login_events, transactions — ~92K
  transactions, ~60K logins) and does the joins in SQL, the way you'd pull signals from a
  production warehouse.
- Injects **three distinct, realistic fraud patterns** (account takeover, velocity abuse,
  amount anomaly) with ground truth kept separate, so detection is built blind and scored
  afterward — not fit to the answer.
- Compares a **rule-based approach vs. an ML approach vs. combining both**, and reports actual
  precision/recall/F1 for each, because "we built a model" means nothing without knowing how
  well it actually performs.

## Results

| Approach                  | Precision | Recall | F1    |
|----------------------------|-----------|--------|-------|
| Rule-based (SQL signals)   | 1.000     | 0.712  | 0.832 |
| Isolation Forest           | 0.833     | 1.000  | 0.909 |
| **Combined (rules OR ML)** | **0.833** | **1.000** | **0.909** |

Rule-based signals catch known patterns with zero false positives but miss what nobody wrote a
rule for. The ML model catches everything the rules do, plus more, at a modest precision cost.
Combining both gives the best balance — which is exactly why production risk systems layer
rules and ML rather than picking one.

Full write-up and all charts are in [`notebook/anomaly_detection.ipynb`](notebook/anomaly_detection.ipynb).

## Project structure

```
data/                    generated CSVs (users, devices, logins, transactions, ground truth)
sql/
  schema.sql             table definitions
  analysis_queries.sql   SQL views that build the risk signals (the core join logic)
scripts/
  generate_data.py       generates the synthetic multi-table dataset with injected fraud
  build_features.py      loads CSVs into SQLite, runs the SQL views, outputs features.csv
  build_notebook.py      builds the analysis notebook programmatically
notebook/
  anomaly_detection.ipynb   full EDA + SQL signals + ML model + evaluation, with outputs
```

## How to run it

```bash
pip install pandas numpy scikit-learn matplotlib jupyter

python scripts/generate_data.py      # generates data/*.csv
python scripts/build_features.py     # builds data/features.csv via SQL joins
jupyter notebook notebook/anomaly_detection.ipynb
```

## Data note

The dataset is **synthetically generated** (see `scripts/generate_data.py`) — no real user or
company data is used. Fraud patterns are injected deterministically (fixed random seed) so the
notebook's results are fully reproducible.

## Tech stack

Python (pandas, NumPy, scikit-learn), SQL (SQLite), matplotlib, Jupyter.
