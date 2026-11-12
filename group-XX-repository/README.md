# Group XX – ML Element 1: Build Machine-Learning-Ready Datasets

**Members:** IGIHOZO Bienvenue · MUGISHA Aime Patrick
**Course:** Competence 10 – Apply Machine Learning, Element 1 (Rwanda Polytechnic, Kigali College, 2026/2027)

| Question | Folder | Status |
|---|---|---|
| Q1 – Predicting maize yield for MCSU farmer cooperatives (`maize_yield_rwanda_rw.csv`, supplied by the lecturer) | `q1_maize_yield/` | complete |
| Q2 – Next-day rain early warning for MINEMA (NOAA station data, Kigali / Rusizi / Rubavu / Musanze, 2015–2025) | `q2_own_scenario/` | complete – scenario approval (Stage 0) pending signature |

Full report: `GroupXX_ML_Element1_Documentation.docx` (repository root).

## 1. How to reproduce (any Windows / Linux / macOS computer)
```bash
git clone <repository-url>
cd group-XX-repository
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main   # once per computer (Anaconda licence)
conda env create -f environment.yml      # creates the "ml-project" environment (Python 3.14, defaults channel)
conda activate ml-project
cd q1_maize_yield/notebooks
jupyter notebook                          # then run 00 -> 04 in order (Kernel > Restart & Run All)
```
Or from the command line (no browser):
```bash
cd q1_maize_yield
for n in 00_environment_check 01_acquire 02_eda 03_clean 04_features; do
  jupyter nbconvert --to notebook --execute --inplace notebooks/$n.ipynb
done
```
Running the notebooks in order on the raw file always gives the same final dataset (fixed seed 42). Q2 works the same way
in `q2_own_scenario/notebooks` (01 → 04).

## 2. Repository structure
```
group-XX-repository/
  environment.yml   README.md
  q1_maize_yield/
    data/raw/         original data – never edited
    data/interim/     clean data + un-encoded train/val/test
    data/processed/   model-ready train/val/test + preprocessor.joblib
    notebooks/        00_environment_check, 01_acquire, 02_eda, 03_clean, 04_features
    src/              maize_pipeline.py – reusable cleaning + feature functions
    tools/            build_notebooks.py, build_docs.py (generate the notebooks and Word documents)
    docs/             dictionary, source log, quality report, cleaning log, feature table, data card, memo, figures
  q2_own_scenario/  (same structure)
```

## 3. Why each tool and library was chosen
| Tool / library | Why we use it in this project |
|---|---|
| **Anaconda + conda** | One installer for Python and the data libraries; `conda` creates an isolated `ml-project` environment so library versions do not conflict with other projects, and `environment.yml` lets anyone rebuild it exactly. |
| **Jupyter Notebook** | Code, output, charts and Markdown explanations in one document – needed to justify every decision and to demo at the presentation. |
| **Python 3.14** | The task requires 3.10 or newer; 3.14 is the version shipped with Anaconda 2026.07 and supported by all libraries below. |
| **pandas** | Reads the CSV and does all table work: types, missing values, duplicates, text cleaning, grouping. |
| **numpy** | Fast maths on whole columns (NaN handling, log transform, conditions). |
| **matplotlib** | Base plotting library; saves the charts used in the report. |
| **seaborn** | Statistical charts (box plots, histograms, heatmaps) with little code. |
| **scikit-learn** | Train/validation/test split, `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`, `ColumnTransformer`/`Pipeline` (fitted on train only → no leakage), feature selection (`VarianceThreshold`, mutual information, Random Forest importance). |
| **joblib** | Saves the fitted preprocessing pipeline so new data is transformed exactly the same way. |
| **Git / GitHub** | Version history and proof of both members' contributions. |

## 4. Outputs for the modelling team (Q1)
* `data/processed/maize_train.csv`, `maize_val.csv`, `maize_test.csv` – encoded + scaled features, `record_id`,
  targets `yield_t_ha` / `yield_kg_ha` (regression) and `low_yield_risk` (classification, question E).
* `data/processed/preprocessor.joblib` – fitted pipeline; `src/maize_pipeline.py` – `clean_maize()` and `add_features()`.
* `docs/` – data dictionary, source log, data quality report, cleaning log, feature justification, data card, memo.
