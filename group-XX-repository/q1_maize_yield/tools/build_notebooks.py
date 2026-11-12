"""Builds the Question 1 notebooks (00-04) from the cell text below. Run: python tools/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

HEADER = """# Group XX - Machine Learning, Element 1: Apply dataset preparation mechanisms
| Item | Details |
|---|---|
| Group | **Group XX** |
| Members | **IGIHOZO Bienvenue** and **MUGISHA Aime Patrick** |
| Question | **Q1 - Predicting maize yield for farmer cooperatives in Rwanda (MCSU)** |
| Stage | **{stage}** |
| Driver / Navigator | {roles} |
"""


def nb(cells, stage, roles):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells.append(nbf.v4.new_markdown_cell(HEADER.format(stage=stage, roles=roles)))
    for kind, text in cells:
        text = text.strip("\n")
        book.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    return book


SETUP = r'''
import sys  # lets us tell Python where our own code (src/) is
from pathlib import Path  # file paths that work on every computer
import warnings  # hide harmless library notices
warnings.filterwarnings("ignore")  # keep the output readable
import numpy as np  # maths on whole columns
import pandas as pd  # the DataFrame table
ROOT = Path.cwd().parent  # notebooks/ is inside q1_maize_yield
sys.path.insert(0, str(ROOT / "src"))  # make src/maize_pipeline.py importable
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)  # show every column
RAW = ROOT / "data" / "raw" / "maize_yield_rwanda_rw.csv"  # the original file (never edited)
import maize_pipeline as mp  # our reusable functions
'''

# ============================================================== 00 environment check
nb00 = [
    ("md", r'''
## Stage 1 - Environment test
**Goal (PC 1.1):** prove that the `ml-project` conda environment works and that every library the project needs can be
imported. The versions printed below are the ones written in `environment.yml`.
'''),
    ("code", r'''
import sys  # knows which Python is running
import platform  # tells us the operating system
import numpy, pandas, matplotlib, seaborn, sklearn  # the five libraries required by the task
print("Python      :", sys.version.split()[0])  # Python version (3.10 or newer required)
print("Interpreter :", sys.executable)  # path shows which conda environment is used
print("OS          :", platform.platform())  # computer the test ran on
for lib in (numpy, pandas, matplotlib, seaborn, sklearn):  # each library
    print(f"{lib.__name__:<12}: {lib.__version__}")  # name and version
'''),
    ("md", "**Result:** all libraries import without error - the environment is ready."),
]

# ============================================================== 01 acquire
nb01 = [
    ("md", r'''
## Stage 2 - Define requirements and acquire data (PC 1.2)

### 2.1 Data requirements (from Mr. Habimana's message and the five business questions)
| Requirement | Decision |
|---|---|
| **Target** | maize yield per hectare - `yield_t_ha` in the file (we also give `yield_kg_ha`). For question E: `low_yield_risk` = bottom 10 % of training yields |
| **Unit of observation** | one farm in one season |
| **Features needed** | location (province, district, altitude), season and year, farm area, seed variety, farming practice, fertiliser **per hectare** (NPK, urea, manure), plant density, rainfall, temperature, soil type and pH, pest level, irrigation |
| **Must NOT be used** | anything known only after harvest (harvest weight, sale price...) - leakage (question E) |
| **Time period / area** | Seasons A, B, C over several years; all provinces |
| **Records needed** | at least ~1,000 usable farm-season records so that train/val/test (70/15/15) keep every season |
| **Quality** | one label per category, one unit per measure, realistic values, no duplicates |

### 2.2 Which data serves which business question
| Q | Needs | Columns in the file |
|---|---|---|
| A | reliable yield per farm | `yield_t_ha`, `farm_area_ha`, `plant_density_per_ha` |
| B | fair fertiliser comparison, correct seed | `npk_kg_ha`, `urea_kg_ha`, `manure_t_ha` (already per ha), `seed_variety`, `farming_practice` |
| C | weather, altitude, irrigation | `rainfall_mm`, `avg_temp_c`, `altitude_m`, `irrigation`, `soil_ph`, `soil_type` |
| D | consistent province/district/season | `province`, `district`, `season`, `year` |
| E | only pre-harvest information | all except the target (see timing note in Stage 3) |
'''),
    ("code", SETUP),
    ("md", r'''
### 2.3 Acquire the data
The file `maize_yield_rwanda_rw.csv` supplied by the lecturer is copied **unchanged** into `data/raw/`
(SHA-256 identical to the original). We only read it; every change happens later in code.
'''),
    ("code", r'''
raw = pd.read_csv(RAW)  # read the CSV into a DataFrame
print("File :", RAW.name)  # file name
print("Shape:", raw.shape, "-> rows, columns")  # size
raw.head()  # first 5 records
'''),
    ("code", r'''
raw.tail()  # last 5 records - the whole file was read
'''),
    ("code", r'''
raw.info()  # column names, non-empty counts and data types
'''),
    ("md", r'''
**First observations:** 1,426 rows x 20 columns. Numbers are already numeric and text columns are text; only
`rainfall_mm`, `soil_ph` and `pest_incidence_index` have empty cells.
'''),
    ("md", "### 2.4 Check: does the file meet the requirements?"),
    ("code", r'''
check = pd.DataFrame({  # requirement -> evidence from the file
    "requirement": ["at least 1,000 records", "target present", "seasons A, B, C", "all provinces",
                    "years covered", "fertiliser per hectare", "no post-harvest column"],
    "evidence": [len(raw), mp.TARGET in raw.columns, sorted(raw["season"].unique()), sorted(raw["province"].unique()),
                 f"{raw['year'].min()}-{raw['year'].max()}", [c for c in raw.columns if c.endswith("_ha")],
                 "none found (no harvest weight, sales or price)"],
})
check  # show
'''),
    ("md", r'''
**Conclusion:** the file covers all five questions: yield per hectare (A), fertiliser already **per hectare** plus seed and
practice (B), weather/altitude/irrigation (C), 5 provinces and 30 districts over 2015-2024 (D) and no post-harvest
column (E). Differences from the brief: yield is in **t/ha** (not kg/ha), fertiliser is **per ha** (not whole-farm), and there
is no farm/cooperative code or planting date - so questions about individual cooperatives or planting dates cannot be answered.
'''),
    ("md", "### 2.5 Data dictionary (Appendix B) - saved to `docs/data_dictionary.csv`"),
    ("code", r'''
dictionary = pd.DataFrame([  # column, meaning, type, unit, allowed values
    ("record_id", "Record code", "text (ID)", "-", "RW-MZ-00001 ... unique; not a feature"),
    ("year", "Year of the season", "integer", "year", "2015-2024"),
    ("season", "Agricultural season", "category", "-", "A (Sep-Jan), B (Feb-Jun), C (Jun-Sep)"),
    ("province", "Province", "category", "-", "Kigali City, Eastern, Northern, Southern, Western"),
    ("district", "District", "category", "-", "30 official districts"),
    ("altitude_m", "Altitude of the farm", "integer", "m", "900-2,600 in Rwanda"),
    ("rainfall_mm", "Rainfall during the season", "float", "mm", "> 0"),
    ("avg_temp_c", "Average season temperature", "float", "deg C", "maize best 18-27"),
    ("soil_ph", "Soil acidity", "float", "pH", "4-8 normal; maize best 5.5-7"),
    ("soil_type", "Soil type", "category", "-", "Clay, Clay loam, Loam, Sandy loam, Volcanic"),
    ("seed_variety", "Seed variety", "category", "-", "Local, ZM607, ZM621 (OPV), RHM104, WH505, PAN691 (hybrids)"),
    ("farming_practice", "Farming practice", "category", "-", "Improved / Traditional"),
    ("npk_kg_ha", "NPK fertiliser per hectare", "float", "kg/ha", ">= 0 (0 = none used)"),
    ("urea_kg_ha", "Urea fertiliser per hectare", "float", "kg/ha", ">= 0"),
    ("manure_t_ha", "Organic manure per hectare", "float", "t/ha", ">= 0"),
    ("plant_density_per_ha", "Maize plants per hectare", "integer", "plants/ha", "about 20,000-75,000"),
    ("farm_area_ha", "Area of the maize farm", "float", "ha", "> 0"),
    ("irrigation", "Plot irrigated", "category", "-", "Yes / No"),
    ("pest_incidence_index", "Pest pressure", "float", "0-1 index", "0 = none, 1 = severe"),
    ("yield_t_ha", "Maize yield - TARGET", "float", "t/ha", "> 0; typical 1-4 t/ha"),
], columns=["column", "meaning", "data_type", "unit", "allowed_values_notes"])
(ROOT / "docs").mkdir(exist_ok=True)  # make sure docs/ exists
dictionary.to_csv(ROOT / "docs" / "data_dictionary.csv", index=False)  # save
dictionary  # display
'''),
    ("md", "### 2.6 Source log - saved to `docs/source_log.csv`"),
    ("code", r'''
source_log = pd.DataFrame([{  # where the data came from
    "source": "File maize_yield_rwanda_rw.csv supplied by the lecturer for Question 1 (MCSU case study)",
    "link": "data/raw/maize_yield_rwanda_rw.csv (course material)",
    "date_accessed": "2026-11-11",
    "licence": "course material - for this course only",
    "collected_by": "MCSU field officers (case study); weather values per season",
    "limitations": "no farm/cooperative code or planting date; yield in t/ha; Season C and irrigated farms are few; rainfall is a season total",
    "ethical_issues": "no names, phones or IDs of farmers; possible bias: equal numbers per district may not reflect real maize area",
}])
source_log.to_csv(ROOT / "docs" / "source_log.csv", index=False)  # save
source_log.T  # display vertically
'''),
]

# ============================================================== 02 EDA
nb02 = [
    ("md", r'''
## Stage 3 - Explore and audit the data (EDA) (PC 1.3)
**Goal:** understand the data and find every quality problem **before changing anything**. Each check is shown, also when
it finds nothing, so the reader can see it was done.
'''),
    ("code", SETUP + r'''
import matplotlib.pyplot as plt  # charts
import seaborn as sns  # statistical charts
sns.set_theme(style="whitegrid")  # chart style
FIG = ROOT / "docs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)  # chart folder
raw = pd.read_csv(RAW)  # original file
num_cols = raw.select_dtypes("number").columns.drop("year").tolist()  # numeric measures
cat_cols = ["season", "province", "district", "soil_type", "seed_variety", "farming_practice", "irrigation"]  # categories
print(raw.shape)  # size
'''),
    ("md", "### 3.1 Describe the dataset"),
    ("code", r'''
raw.describe().T.round(2)  # count, mean, std, min, quartiles, max
'''),
    ("code", r'''
for c in cat_cols:  # every category column
    print(f"{c}: {raw[c].nunique()} labels ->", raw[c].value_counts().to_dict() if raw[c].nunique() < 10 else sorted(raw[c].unique()))  # labels
'''),
    ("md", r'''
**Reading the summary:** all minimums and maximums are realistic (pH 4.5-7.2, temperature 15.4-23 C, altitude 1,243-2,341 m,
yield 1.0-5.2 t/ha). Labels are consistent (one spelling each). `farm_area_ha` has a long right tail (max 4.2 ha vs median 0.67).
The province is written **"Kigali"** instead of the official **"Kigali City"**.
'''),
    ("md", "### 3.2 Missing values and duplicates"),
    ("code", r'''
missing = raw.isna().sum()  # empty cells per column
miss_tbl = pd.DataFrame({"missing": missing, "percent": (missing / len(raw) * 100).round(2)})[missing > 0]  # only columns with gaps
print("Exact duplicate rows                 :", raw.duplicated().sum())  # identical rows
print("Duplicate record_id                  :", raw["record_id"].duplicated().sum())  # repeated code
print("Same content under another record_id :", raw.drop(columns="record_id").duplicated().sum())  # copied forms
print("Rows by number of gaps               :", raw.isna().sum(axis=1).value_counts().sort_index().to_dict())  # gaps per row
miss_tbl  # table
'''),
    ("code", r'''
fig, ax = plt.subplots(figsize=(8, 3.5))  # chart
sns.heatmap(raw[["rainfall_mm", "soil_ph", "pest_incidence_index"]].isna().T, cbar=False, cmap="Reds", ax=ax, xticklabels=False)  # red = missing
ax.set_title("Figure 3.1 - Where the gaps are (red = missing), one column per farm"); ax.set_xlabel("farms (row order)")  # labels
plt.tight_layout(); plt.savefig(FIG / "fig3_1_missing.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** only 39 cells are empty (under 1 % per column), but they are **not random**: 11 farms miss all three
values and 3 farms miss two - the same farms had no soil test, no pest scouting and no rain record. We keep these farms,
impute the values (learned on the training set) and test an indicator `field_data_missing` in Stage 5. No duplicates were found.
'''),
    ("md", "### 3.3 Consistency checks"),
    ("code", r'''
official = raw["district"].map(mp.DISTRICT_PROVINCE)  # official province of each district
print("Districts:", raw["district"].nunique(), "| not in official list:", (~raw["district"].isin(mp.DISTRICT_PROVINCE)).sum())  # 30 expected
print("Province does not match district (ignoring 'Kigali' name):", (official != raw["province"].replace({"Kigali": "Kigali City"})).sum())  # mapping check
print("Rows with 'Kigali' instead of 'Kigali City':", (raw["province"] == "Kigali").sum())  # naming
for c, allowed in mp.ALLOWED.items():  # allowed labels
    print(f"{c:<17} labels outside allowed list: {(~raw[c].isin(allowed)).sum()}")  # 0 expected
print("\nRecords per season:", raw["season"].value_counts().to_dict())  # balance
pd.crosstab(raw["season"], raw["irrigation"], margins=True)  # irrigation by season
'''),
    ("md", r'''
**Finding:** labels and the district-province mapping are correct. Season C has far fewer records (226 vs 600) and only 128
farms are irrigated - mostly in Season C, as the agronomist expects, but also 66 in Seasons A and B (plausible for marshland
plots, kept). These small groups make results for Season C and irrigation less certain.
'''),
    ("md", "### 3.4 Outliers (box plots + IQR rule) and realistic ranges"),
    ("code", r'''
fig, axes = plt.subplots(2, 6, figsize=(17, 6)); axes = axes.ravel()  # 11 small charts
for ax, c in zip(axes, num_cols):  # one box per numeric column
    sns.boxplot(y=raw[c], ax=ax, color="#DD8452"); ax.set_title(c, fontsize=9); ax.set_ylabel("")  # box plot
axes[-1].axis("off")  # 12th panel unused
fig.suptitle("Figure 3.2 - Box plots of the numeric columns"); plt.tight_layout()  # title
plt.savefig(FIG / "fig3_2_boxplots.png", dpi=120); plt.show()  # save and display
'''),
    ("code", r'''
def iqr_report(s):  # IQR rule and agronomic range for one column
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1  # quartiles
    lo, hi = mp.RANGES[s.name]  # realistic range
    return pd.Series({"IQR_lower": q1 - 1.5 * i, "IQR_upper": q3 + 1.5 * i,
                      "IQR_outliers": int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum()),
                      "outside_realistic_range": int((~s.between(lo, hi) & s.notna()).sum()),
                      "skew": s.skew()})  # result
raw[num_cols].apply(iqr_report).T.round(2)  # every numeric column
'''),
    ("md", r'''
**Interpretation:** the IQR rule flags some values (75 farm areas above ~1.8 ha, 13 high pest indices, 10 pH values, 8 yields,
6 manure values) but **none is outside a realistic range** - they are real large farms, heavy pest years and good harvests,
not typing errors. Decision: **keep them** (removing them would hide real risk); reduce the farm-area skew (1.95) with a log.
'''),
    ("md", "### 3.5 The target"),
    ("code", r'''
y = raw[mp.TARGET]  # target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # two charts
sns.histplot(y, bins=40, ax=axes[0], color="#55A868"); axes[0].set_title(f"Figure 3.3a - yield_t_ha, skew = {y.skew():.2f}")  # distribution
axes[0].axvline(y.quantile(.10), color="red", ls="--", label=f"10th percentile = {y.quantile(.10):.2f} t/ha"); axes[0].legend()  # low-yield line
sns.boxplot(data=raw, x="season", y=mp.TARGET, order=["A", "B", "C"], ax=axes[1]); axes[1].set_title("Figure 3.3b - yield by season")  # by season
plt.tight_layout(); plt.savefig(FIG / "fig3_3_target.png", dpi=120); plt.show()  # save and display
print("farms below 1.0 t/ha (agronomist 'very low'):", (y < 1.0).sum())  # none
'''),
    ("md", r'''
**Interpretation:** yield is almost symmetric (skew 0.28) around 2.6 t/ha and stays within 1.0-5.2 t/ha. **No farm** is below
the agronomist's 1,000 kg/ha, so "very low harvest" (question E) is defined as the **bottom 10 % of the training farms**
(about 1.7 t/ha), computed after the split.
'''),
    ("md", "### 3.6 Relationships"),
    ("code", r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4))  # two charts
order = raw.groupby("seed_variety")[mp.TARGET].median().sort_values().index  # sort by median yield
sns.boxplot(data=raw, x="seed_variety", y=mp.TARGET, order=order, ax=axes[0], color="#8172B3")  # yield by seed
axes[0].set_title("Figure 3.4a - yield by seed variety")  # title
sns.boxplot(data=raw, x="farming_practice", y=mp.TARGET, ax=axes[1], color="#64B5CD")  # yield by practice
axes[1].set_title("Figure 3.4b - yield by farming practice")  # title
plt.tight_layout(); plt.savefig(FIG / "fig3_4_seed_practice.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** Local seed gives clearly lower yields than all improved varieties; the hybrids (PAN691, RHM104, WH505) are
highest. Improved practice also yields more - question B can be answered.
'''),
    ("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # three scatter plots
for ax, c in zip(axes, ["rainfall_mm", "npk_kg_ha", "plant_density_per_ha"]):  # three drivers
    ax.scatter(raw[c], raw[mp.TARGET], s=6, alpha=.4)  # one dot per farm
    ax.set_xlabel(c); ax.set_ylabel("yield t/ha"); ax.set_title(f"r = {raw[c].corr(raw[mp.TARGET]):.2f}")  # correlation in title
fig.suptitle("Figure 3.5 - Rainfall, NPK and plant density vs yield"); plt.tight_layout()  # title
plt.savefig(FIG / "fig3_5_drivers.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** rainfall (r about 0.36), plant density (0.30) and fertiliser (0.22) all raise yield, but with wide scatter:
yield depends on several factors together, which is why engineered combinations (nitrogen applied, input package) are useful.
'''),
    ("code", r'''
corr = raw[num_cols].corr()  # correlation matrix
fig, ax = plt.subplots(figsize=(9, 7))  # chart
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})  # heatmap
ax.set_title("Figure 3.6 - Correlation heatmap"); plt.tight_layout()  # title
plt.savefig(FIG / "fig3_6_corr.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** altitude and temperature are strongly linked (r = -0.88: highlands are cooler), so they carry overlapping
information. **No column is suspiciously close to the target** (no hidden post-harvest variable). **Timing note (question E):**
`rainfall_mm` and `avg_temp_c` are **season totals/averages**, fully known only at the end of the season, close to harvest;
`pest_incidence_index` is observed during the season. For an *early* warning the modelling team must use rainfall-to-date
instead. We keep the columns (needed for A-D) and document this risk.
'''),
    ("code", r'''
fig, ax = plt.subplots(figsize=(10, 4))  # chart
sns.boxplot(data=raw, x="province", y=mp.TARGET, hue="season", hue_order=["A", "B", "C"], ax=ax)  # province x season
ax.set_title("Figure 3.7 - Yield by province and season"); plt.tight_layout()  # title
plt.savefig(FIG / "fig3_7_province_season.png", dpi=120); plt.show()  # save and display
'''),
    ("md", "**Interpretation:** differences between provinces and seasons are visible - question D can be answered."),
    ("md", "### 3.7 Data Quality Report - saved to `docs/data_quality_report.csv`"),
    ("code", r'''
iqr = raw[num_cols].apply(iqr_report).T  # outlier summary from above
dqr = pd.DataFrame([  # problem, columns, records, severity, treatment
    (1, "Missing values that occur together (no soil test / scouting / rain record)", "rainfall_mm, soil_ph, pest_incidence_index",
     int(raw[["rainfall_mm", "soil_ph", "pest_incidence_index"]].isna().any(axis=1).sum()), "Medium",
     "keep rows; median imputation fitted on train; indicator tested in Stage 5"),
    (2, "IQR outliers (all within realistic ranges)", ", ".join(iqr.index[iqr.IQR_outliers > 0]), int(iqr.IQR_outliers.sum()), "Low",
     "keep (real farms); log transform for farm area"),
    (3, "Right-skewed farm area (skew 1.95)", "farm_area_ha", len(raw), "Low", "log(1 + area) feature"),
    (4, "True zeros in fertiliser/manure (no input used) - must not be read as missing", "npk_kg_ha, urea_kg_ha, manure_t_ha",
     int((raw[["npk_kg_ha", "urea_kg_ha", "manure_t_ha"]] == 0).any(axis=1).sum()), "Medium", "keep 0; add uses_mineral_fertiliser flag"),
    (5, "Strongly related columns (altitude vs temperature r = -0.88; NPK/urea vs nitrogen)", "altitude_m, avg_temp_c, urea_kg_ha",
     len(raw), "Low", "keep altitude + temperature deviation; drop urea after creating nitrogen feature"),
    (6, "Unbalanced groups: Season C (226 rows), irrigated farms (128)", "season, irrigation", 226, "Medium",
     "stratified split by season; report lower certainty"),
    (7, "Non-official province name 'Kigali'", "province", int((raw["province"] == "Kigali").sum()), "Low", "rename to 'Kigali City'"),
    (8, "Target unit t/ha while the brief and agronomist use kg/ha", "yield_t_ha", len(raw), "Low", "keep t/ha and add yield_kg_ha"),
    (9, "No farm below 1,000 kg/ha - agronomist threshold cannot define 'very low'", "yield_t_ha", 0, "Medium",
     "low_yield_risk = bottom 10 % of TRAINING yields"),
    (10, "Timing leakage risk: season rainfall/temperature known only at season end", "rainfall_mm, avg_temp_c, pest_incidence_index",
     len(raw), "High", "keep for A-D; warn modelling team to use data-to-date for early warning (E)"),
    (11, "Information missing vs brief: no farm/cooperative code, no planting date", "-", len(raw), "Medium",
     "cannot analyse per cooperative or planting date; recommend collecting"),
    (12, "Checks with nothing found: duplicates, wrong labels, wrong district-province, impossible values",
     "all", 0, "-", "documented checks; no change needed"),
], columns=["no", "problem", "columns", "records_affected", "severity", "proposed_treatment"])
dqr.to_csv(ROOT / "docs" / "data_quality_report.csv", index=False)  # save
dqr  # display
'''),
    ("md", r'''
**Checkpoint answer - two problems and why they matter for the model**
1. **Missing values together in 14 farms:** if we simply deleted them we would lose farms that may differ (no soil test =
   less supported farmers); if we filled them using all rows we would leak test information. So: impute with training
   medians and add an indicator.
2. **Timing of rainfall/temperature:** a model trained on full-season rainfall looks good in testing but cannot be used
   for an early warning before the season ends - a form of leakage for question E.
'''),
]

# ============================================================== 03 clean
nb03 = [
    ("md", r'''
## Stage 4 - Clean and pre-process the data (PC 1.3)
Every step calls a documented function in `src/maize_pipeline.py`. Each step shows a before/after view and writes to the
**Cleaning Log** - also the checks that found nothing.
'''),
    ("code", SETUP + r'''
raw = pd.read_csv(RAW)  # original file
log = mp.CleaningLog()  # empty log
def before_after(before, after, cols):  # helper: shape, unique values, missing counts
    print(f"shape: {before.shape} -> {after.shape}")  # size
    for c in cols:  # each column
        print(f"  {c:<22} unique {before[c].nunique() if c in before else '-'} -> {after[c].nunique() if c in after else '-'}"
              f" | missing {before[c].isna().sum() if c in before else '-'} -> {after[c].isna().sum() if c in after else '-'}")  # one line
'''),
    ("md", "### Steps 1-2: spaces and duplicates"),
    ("code", r'''
df = mp.trim_text(raw, log)  # remove spaces around labels
step = df.copy()  # before duplicates
df = mp.check_duplicates(df, log)  # same id / same content
before_after(step, df, ["record_id"])  # no change expected
'''),
    ("md", "### Step 3: categories, district-province, official names"),
    ("code", r'''
step = df.copy()  # before
df = mp.check_categories(df, log)  # allowed labels; province from district
before_after(step, df, ["province", "district", "season", "seed_variety"])  # province labels
print(step["province"].value_counts().to_dict(), "->", df["province"].value_counts().to_dict())  # 'Kigali' -> 'Kigali City'
'''),
    ("md", "### Step 4: realistic ranges"),
    ("code", r'''
step = df.copy()  # before
df = mp.check_ranges(df, log)  # out-of-range -> NaN
pd.concat({"before": step[list(mp.RANGES)].agg(["min", "max"]).T, "after": df[list(mp.RANGES)].agg(["min", "max"]).T}, axis=1)  # ranges
'''),
    ("md", "### Steps 5-6: missing-value indicator, target"),
    ("code", r'''
step = df.copy()  # before
df = mp.flag_missing(df, log)  # indicator field_data_missing
df = mp.check_target(df, log)  # valid target + yield_kg_ha
before_after(step, df, mp.SOIL_TEST_COLS + ["field_data_missing", "yield_kg_ha"])  # new columns
df[df["field_data_missing"] == 1].head()  # farms with gaps
'''),
    ("md", "### Reproducibility: the one-call pipeline gives exactly the same table"),
    ("code", r'''
again, _ = mp.clean_maize(pd.read_csv(RAW))  # all steps in one call
pd.testing.assert_frame_equal(df, again)  # error if anything differs
print("identical:", again.shape)  # confirmation
'''),
    ("md", "### Cleaning Log - saved to `docs/cleaning_log.csv`"),
    ("code", r'''
clean_log = log.to_frame()  # every decision
clean_log.to_csv(ROOT / "docs" / "cleaning_log.csv", index=False)  # save
(ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)  # folder
df.to_csv(ROOT / "data" / "interim" / "maize_clean.csv", index=False)  # clean dataset
clean_log[["step", "operation", "column", "problem", "action", "records_affected"]]  # display
'''),
    ("md", r'''
### Split: 70 / 15 / 15, seed 42, stratified by season - **before** imputing and scaling
Imputers and scalers learn numbers (medians, means, standard deviations); if they saw validation/test rows, information from
those rows would leak into training. Stratifying by season keeps the small Season C group in all three files.
'''),
    ("code", r'''
from sklearn.model_selection import train_test_split  # random split
train, temp = train_test_split(df, test_size=0.30, random_state=mp.SEED, stratify=df["season"])  # 70 % train
val, test = train_test_split(temp, test_size=0.50, random_state=mp.SEED, stratify=temp["season"])  # 15 % + 15 %
thr = mp.low_yield_threshold(train)  # bottom 10 % learned on TRAIN only
train, val, test = (mp.add_low_yield_flag(p, thr) for p in (train, val, test))  # same threshold everywhere
print(f"low-yield threshold (train 10th percentile): {thr:.3f} t/ha = {thr * 1000:.0f} kg/ha")  # value
for n, p in [("train", train), ("val", val), ("test", test)]:  # each split
    print(f"{n:<5} {p.shape}  season C {int((p.season == 'C').sum()):>3}  low-yield share {p.low_yield_risk.mean():.3f}")  # size + balance
    p.to_csv(ROOT / "data" / "interim" / f"{n}_clean.csv", index=False)  # save
'''),
    ("md", r'''
**Imbalance:** `low_yield_risk` is about 10 % positives by construction -> planned remedy for the modelling team:
`class_weight="balanced"` or resampling **on the training set only**.
'''),
    ("md", "### Imputation, encoding, scaling - one scikit-learn Pipeline fitted on train only"),
    ("code", r'''
from sklearn.compose import ColumnTransformer  # different steps for different columns
from sklearn.pipeline import Pipeline  # chain of steps
from sklearn.impute import SimpleImputer  # fills missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding and scaling
numeric_cols = ["altitude_m", "rainfall_mm", "avg_temp_c", "soil_ph", "npk_kg_ha", "urea_kg_ha", "manure_t_ha",
                "plant_density_per_ha", "farm_area_ha", "pest_incidence_index", "year"]  # numbers
nominal_cols = ["season", "district", "soil_type", "seed_variety", "farming_practice", "irrigation"]  # categories
base_prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),  # median: robust to the kept outliers
                      ("scale", StandardScaler())]), numeric_cols),  # mean 0, std 1
    ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols),  # one 0/1 column per label
])
base_prep.fit(train)  # LEARN medians, means, stds and labels from TRAIN only
Xtr, Xva, Xte = (base_prep.transform(p) for p in (train, val, test))  # APPLY to all three
print("train", Xtr.shape, "| val", Xva.shape, "| test", Xte.shape)  # same columns
pd.DataFrame(Xtr, columns=base_prep.get_feature_names_out()).iloc[:5, :12].round(2)  # first rows
'''),
    ("code", r'''
pd.DataFrame(Xtr[:, :len(numeric_cols)], columns=numeric_cols).agg(["mean", "std"]).round(2)  # ~0 and ~1 = scaler worked
'''),
    ("md", r'''
**Leakage question:** "Why fit the imputer and scaler on train only?" - validation and test stand for future farms; their
medians, means and standard deviations must not influence anything the model learns.
'''),
]

# ============================================================== 04 features
nb04 = [
    ("md", r'''
## Stage 5 - Engineer and select features (PC 1.4)
New features use only values of the same row, so they cannot leak between splits. Every statistic used for **selection** is
computed on the **training set only**.
'''),
    ("code", SETUP + r'''
import matplotlib.pyplot as plt  # charts
import seaborn as sns  # charts
INTERIM, PROCESSED, FIG = ROOT / "data" / "interim", ROOT / "data" / "processed", ROOT / "docs" / "figures"  # folders
PROCESSED.mkdir(parents=True, exist_ok=True)  # create if needed
splits = {n: mp.add_features(pd.read_csv(INTERIM / f"{n}_clean.csv")) for n in ["train", "val", "test"]}  # SAME steps on each split
train = splits["train"]  # shortcut
new_cols = ["n_applied_kg_ha", "uses_mineral_fertiliser", "seed_class", "improved_package", "log_farm_area_ha",
            "temp_dev_from_optimum", "ph_in_optimal_range", "is_irrigated", "altitude_zone", "plant_density_k_ha"]  # 10 new features
print({n: d.shape for n, d in splits.items()})  # sizes
train[["record_id", "npk_kg_ha", "urea_kg_ha", "seed_variety", "farming_practice", "farm_area_ha"] + new_cols].head()  # first rows
'''),
    ("md", "### 5.1 Skewness -> transformation"),
    ("code", r'''
sk = train[["farm_area_ha", "log_farm_area_ha", "pest_incidence_index", "manure_t_ha", mp.TARGET]].skew().round(2)  # skewness
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))  # two histograms
sns.histplot(train["farm_area_ha"], ax=axes[0]); axes[0].set_title(f"farm_area_ha, skew = {sk['farm_area_ha']}")  # before
sns.histplot(train["log_farm_area_ha"], ax=axes[1], color="#55A868"); axes[1].set_title(f"log(1+area), skew = {sk['log_farm_area_ha']}")  # after
fig.suptitle("Figure 5.1 - Log transform of farm area"); plt.tight_layout()  # title
plt.savefig(FIG / "fig5_1_log_area.png", dpi=120); plt.show()  # save and display
sk.to_frame("skewness")  # table
'''),
    ("md", "**Decision:** farm area is right-skewed; the log version is much closer to symmetric, so we keep the log. The target is not transformed."),
    ("md", "### 5.2 Relationship with the target (training set only)"),
    ("code", r'''
cands = ["altitude_m", "rainfall_mm", "avg_temp_c", "soil_ph", "npk_kg_ha", "urea_kg_ha", "manure_t_ha", "plant_density_k_ha",
         "farm_area_ha", "pest_incidence_index", "year", "field_data_missing", "n_applied_kg_ha", "uses_mineral_fertiliser",
         "improved_package", "log_farm_area_ha", "temp_dev_from_optimum", "ph_in_optimal_range", "is_irrigated"]  # numeric candidates
corr = train[cands + [mp.TARGET]].corr()  # correlation (train)
fig, ax = plt.subplots(figsize=(12, 9))  # chart
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)  # heatmap
ax.set_title("Figure 5.2 - Correlation of candidate features (training set)"); plt.tight_layout()  # title
plt.savefig(FIG / "fig5_2_feature_corr.png", dpi=120); plt.show()  # save and display
corr[mp.TARGET].drop(mp.TARGET).sort_values(key=abs, ascending=False).round(3).to_frame("corr_with_yield")  # ranked
'''),
    ("code", r'''
from sklearn.feature_selection import mutual_info_regression  # any kind of dependence
from sklearn.ensemble import RandomForestRegressor  # quick model for importance
cats = ["season", "province", "district", "soil_type", "seed_variety", "seed_class", "farming_practice", "altitude_zone"]  # categories
X = train[cands].fillna(train[cands].median())  # simple fill with TRAIN medians, only for measuring
X = X.join(train[cats].apply(lambda s: s.astype("category").cat.codes))  # categories as integer codes
mi = pd.Series(mutual_info_regression(X, train[mp.TARGET], random_state=mp.SEED), index=X.columns)  # mutual information
rf = RandomForestRegressor(n_estimators=300, random_state=mp.SEED, n_jobs=-1, min_samples_leaf=3).fit(X, train[mp.TARGET])  # forest
rank = pd.DataFrame({"mutual_info": mi, "rf_importance": rf.feature_importances_}, index=X.columns).sort_values("rf_importance", ascending=False)  # table
fig, ax = plt.subplots(figsize=(8, 7))  # chart
rank["rf_importance"].sort_values().plot.barh(ax=ax, color="#4C72B0")  # bars
ax.set_title("Figure 5.3 - Random Forest importance (training set)"); plt.tight_layout()  # title
plt.savefig(FIG / "fig5_3_rf_importance.png", dpi=120); plt.show()  # save and display
rank.round(3)  # table
'''),
    ("md", r'''
### 5.3 Feature selection
1. **Variance threshold** - remove constant / almost constant columns.
2. **Duplicated information** - `province` is fully determined by `district`; pairs with |r| > 0.9 keep the more useful one.
3. **Model-based evidence** - Random Forest importance and mutual information above.
'''),
    ("code", r'''
from sklearn.feature_selection import VarianceThreshold  # selection method 1
vt = VarianceThreshold(threshold=0.01).fit(X[cands])  # fit on train
print("near-constant:", [c for c, k in zip(cands, vt.get_support()) if not k] or "none")  # result
up = corr.loc[cands, cands].abs().where(np.triu(np.ones((len(cands),) * 2), k=1).astype(bool))  # upper triangle
print("pairs with |r| > 0.9:\n", up.stack()[lambda s: s > 0.9].round(3).to_string())  # strongly related pairs
print("provinces per district:", train.groupby("district")["province"].nunique().max(), "-> province is redundant")  # 1
'''),
    ("code", r'''
DROP = {  # removed -> reason (evidence above)
    "province": "fully determined by district (each district in one province)",
    "urea_kg_ha": f"r = {corr.loc['urea_kg_ha', 'n_applied_kg_ha']:.2f} with n_applied_kg_ha, which contains the urea nitrogen",
    "farm_area_ha": f"r = {corr.loc['farm_area_ha', 'log_farm_area_ha']:.2f} with log_farm_area_ha (less skewed)",
    "plant_density_per_ha": "same values as plant_density_k_ha (only divided by 1,000)",
    "irrigation": "replaced by numeric is_irrigated",
    "field_data_missing": f"near-constant (variance threshold): only {int(train['field_data_missing'].sum())} of {len(train)} training farms",
    "record_id": "identifier, not a feature",
    "yield_kg_ha": "same as the target in another unit (would leak the answer)",
}
FINAL_NUMERIC = ["altitude_m", "rainfall_mm", "temp_dev_from_optimum", "avg_temp_c", "soil_ph", "ph_in_optimal_range",
                 "npk_kg_ha", "n_applied_kg_ha", "manure_t_ha", "uses_mineral_fertiliser", "improved_package",
                 "plant_density_k_ha", "log_farm_area_ha", "pest_incidence_index", "is_irrigated", "year"]  # kept numbers
FINAL_NOMINAL = ["season", "district", "soil_type", "seed_variety", "seed_class", "farming_practice", "altitude_zone"]  # kept categories
print(len(FINAL_NUMERIC) + len(FINAL_NOMINAL), "selected input columns:", FINAL_NUMERIC + FINAL_NOMINAL)  # final list
'''),
    ("md", r'''
**Notes on choices:** `avg_temp_c` is kept although r = -0.88 with altitude (below the 0.9 rule) because temperature is what
the agronomist reasons with; `seed_class` and `seed_variety` are both kept - the class lets the model share information among
similar varieties, the variety keeps the differences (RF importance above). The modelling team can drop features with very
low importance after validation.
'''),
    ("md", "### 5.4 Final pipeline (fitted on train only) and model-ready files"),
    ("code", r'''
from sklearn.compose import ColumnTransformer  # column-wise recipe
from sklearn.pipeline import Pipeline  # ordered steps
from sklearn.impute import SimpleImputer  # missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding + scaling
import joblib  # saves the fitted pipeline
final_prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FINAL_NUMERIC),
    ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FINAL_NOMINAL),
], verbose_feature_names_out=False)  # readable column names
final_prep.fit(splits["train"])  # learn ONLY from training rows
cols = list(final_prep.get_feature_names_out())  # output names
for n, d in splits.items():  # train, val, test
    X_out = pd.DataFrame(final_prep.transform(d), columns=cols)  # identical transformation
    out = pd.concat([d[["record_id"]].reset_index(drop=True), X_out,
                     d[[mp.TARGET, "yield_kg_ha", "low_yield_risk"]].reset_index(drop=True)], axis=1)  # id + X + targets
    out.to_csv(PROCESSED / f"maize_{n}.csv", index=False)  # model-ready file
    d.to_csv(PROCESSED / f"maize_{n}_readable.csv", index=False)  # same rows un-scaled, for analysts
    print(f"{n:<5} -> {out.shape}, missing values: {int(out.isna().sum().sum())}")  # check
joblib.dump(final_prep, PROCESSED / "preprocessor.joblib")  # fitted pipeline for new data
print("same columns in all splits:", len({tuple(pd.read_csv(PROCESSED / f'maize_{n}.csv', nrows=1).columns) for n in splits}) == 1)  # consistency
'''),
    ("md", "### 5.5 Feature Justification table - saved to `docs/feature_justification.csv`"),
    ("code", r'''
r = lambda c: f"r with yield {corr.loc[c, mp.TARGET]:.2f}, RF importance {rank.loc[c, 'rf_importance']:.3f}"  # evidence text
fj = pd.DataFrame([  # feature, how created, question, evidence, decision
    ("n_applied_kg_ha", "0.17 x npk_kg_ha + 0.46 x urea_kg_ha (N content of NPK 17-17-17 and urea)", "B", r("n_applied_kg_ha"), "kept"),
    ("uses_mineral_fertiliser", "1 if npk + urea > 0", "B", "separates true zeros (no fertiliser) from small doses", "kept"),
    ("seed_class", "seed_variety -> Local / OPV / Hybrid", "B", "Local clearly lower (Fig. 3.4a)", "kept"),
    ("improved_package", "1 if OPV/Hybrid seed AND Improved practice", "B", r("improved_package"), "kept"),
    ("log_farm_area_ha", "log(1 + farm_area_ha)", "A", f"skew {sk['farm_area_ha']} -> {sk['log_farm_area_ha']}", "kept"),
    ("temp_dev_from_optimum", "degrees below 18 C or above 27 C", "C", r("temp_dev_from_optimum"), "kept"),
    ("ph_in_optimal_range", "1 if 5.5 <= soil_ph <= 7", "C", r("ph_in_optimal_range"), "kept"),
    ("is_irrigated", "irrigation Yes = 1 / No = 0", "C", "numeric form", "kept"),
    ("altitude_zone", "altitude < 1,600 / 1,600-1,900 / > 1,900 m", "C, D", "lowland / midland / highland advice groups", "kept"),
    ("plant_density_k_ha", "plant_density_per_ha / 1,000", "A", r("plant_density_k_ha"), "kept"),
    ("low_yield_risk", "1 if yield < 10th percentile of TRAIN yields", "E", "second target; ~10 % positives", "target"),
] + [(k, "-", "-", v, "removed") for k, v in DROP.items()],
    columns=["feature", "how_created_or_reason", "business_question", "evidence", "decision"])
fj.to_csv(ROOT / "docs" / "feature_justification.csv", index=False)  # save
fj  # display
'''),
    ("md", "### 5.6 Final check (Stage 6) - the processed files"),
    ("code", r'''
final_train = pd.read_csv(PROCESSED / "maize_train.csv")  # read back
final_train.info()  # all inputs numeric, no missing values
sorted(p.name for p in PROCESSED.iterdir())  # files delivered to the modelling team
'''),
]

for name, cells, stage, roles in [
    ("00_environment_check", nb00, "1 - Configure the environment", "Driver: IGIHOZO Bienvenue / Navigator: MUGISHA Aime Patrick"),
    ("01_acquire", nb01, "2 - Define requirements and acquire data", "Driver: MUGISHA Aime Patrick / Navigator: IGIHOZO Bienvenue"),
    ("02_eda", nb02, "3 - Explore and audit the data (EDA)", "Driver: IGIHOZO Bienvenue / Navigator: MUGISHA Aime Patrick"),
    ("03_clean", nb03, "4 - Clean and pre-process the data", "Driver: MUGISHA Aime Patrick / Navigator: IGIHOZO Bienvenue"),
    ("04_features", nb04, "5 - Engineer and select features (+ Stage 6 final files)", "Driver: IGIHOZO Bienvenue / Navigator: MUGISHA Aime Patrick"),
]:
    nbf.write(nb(cells, stage, roles), NB_DIR / f"{name}.ipynb")
    print("wrote", name)
