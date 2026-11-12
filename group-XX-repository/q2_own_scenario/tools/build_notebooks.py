"""Builds the Question 2 notebooks (01-04). Run: python tools/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
HEADER = """# Group XX - Machine Learning, Element 1: Apply dataset preparation mechanisms
| Item | Details |
|---|---|
| Group | **Group XX** |
| Members | **IGIHOZO Bienvenue** and **MUGISHA Aime Patrick** |
| Question | **Q2 - Next-day rain early warning for MINEMA district disaster offices (own scenario)** |
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
warnings.filterwarnings("ignore")  # keep output readable
import numpy as np  # maths on whole columns
import pandas as pd  # DataFrame tables
ROOT = Path.cwd().parent  # notebooks/ is inside q2_own_scenario
sys.path.insert(0, str(ROOT / "src"))  # make src/rain_pipeline.py importable
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)  # show all columns
RAW_DIR = ROOT / "data" / "raw" / "gsod"  # original NOAA files (never edited)
import rain_pipeline as rp  # our reusable functions
'''

# ============================================================== 01 acquire
nb01 = [
    ("md", r'''
## Stage 2 - Define requirements and acquire data (PC 1.2)

### Scenario (summary - full brief in `docs/Q2_Scenario_Brief_and_Proposal.docx`)
MINEMA (Ministry in charge of Emergency Management) works with district disaster management committees. Heavy and repeated
rain triggers floods and landslides - in May 2023 floods and landslides in the Western and Northern provinces (Rubavu, Nyabihu,
Ngororero, Musanze...) killed more than 130 people. Every evening the duty officer must decide whether to **send an SMS
warning and put response teams on standby for tomorrow**. MINEMA asks us to prepare a dataset so that a model can later
answer: **"Will it rain tomorrow at this station?"** (`rain_tomorrow` = 1 / 0, binary classification).

### 2.1 Data requirements
| Requirement | Decision |
|---|---|
| **Target** | `rain_tomorrow` - rain or drizzle observed at the station on the next calendar day (1/0) |
| **Unit of observation** | one weather station on one day |
| **Features needed** | today's temperature, dew point / humidity, min-max temperature, wind, visibility, rain amount, weather events (rain, thunder, fog), season, station/district |
| **Must NOT be used** | anything measured tomorrow (leakage); random split across time (would let the model see the future) |
| **Time period** | at least 5 years of daily data (seasons repeat every year) -> 2015-2025 |
| **Area** | stations near high-risk districts: Rubavu, Musanze, Rusizi + Kigali for comparison |
| **Records needed** | > 1,000 rows, >= 8 columns, numeric + categorical |
| **Licence** | open / free for education; no personal data |

### 2.2 Comparison of possible sources
| Source | Relevance | Size | Licence | Quality | Decision |
|---|---|---|---|---|---|
| **NOAA NCEI Global Summary of the Day (GSOD)** - real station observations | real measurements at Kigali, Kamembe, Gisenyi, Ruhengeri airports | ~10,600 station-days 2015-2025, 28 columns | free, no login (US Gov); WMO Res. 40 - non-commercial use | real problems: missing codes, gaps, empty columns, imperial units | **chosen** |
| Open-Meteo historical weather API (ERA5 reanalysis) | modelled values for any point | unlimited | CC BY 4.0 | complete and smooth (model output, not observation) - **too clean** for this task and not a real observation | rejected |
| Meteo Rwanda station data / NISR | best local source | - | request / registration needed | not downloadable in time | rejected (recommended for the future) |
'''),
    ("code", SETUP),
    ("md", r'''
### 2.3 Acquire the data
42 yearly CSV files were downloaded from `https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<year>/<station>.csv`
for 4 stations x 2015-2025 and saved **unchanged** in `data/raw/gsod/` (2023 is missing at NOAA for Gisenyi and Ruhengeri).
'''),
    ("code", r'''
files = sorted(RAW_DIR.glob("*.csv"))  # list the raw files
print(len(files), "files, e.g.", [f.name for f in files[:3]])  # how many and a few names
raw = rp.load_raw(RAW_DIR)  # read and stack them (all values as text)
print("Shape:", raw.shape)  # rows, columns
raw.head()  # first records
'''),
    ("code", r'''
raw.tail()  # last records
'''),
    ("code", r'''
raw.info()  # every column is text (str) because we read with dtype=str
'''),
    ("code", r'''
check = pd.DataFrame({  # requirement -> evidence
    "requirement": [">= 1,000 rows", ">= 8 columns", "numeric + categorical columns", "target can be built",
                    "4 stations", "period"],
    "evidence": [len(raw), raw.shape[1], "TEMP, PRCP ... numeric; NAME, FRSHTT categorical", "FRSHTT rain flag + DATE",
                 raw["NAME"].str.strip().unique().tolist(), f"{raw['DATE'].min()} to {raw['DATE'].max()}"],
})
check  # show
'''),
    ("md", r'''
**Conclusion:** the data meets the requirements (10,614 rows, 28 columns + file name, numeric and categorical, 2015-2025).
'''),
    ("md", r'''
### 2.4 Data dictionary (raw NOAA columns) - saved to `docs/data_dictionary.csv`
'''),
    ("code", r'''
dictionary = pd.DataFrame([  # column, meaning, type, unit, notes
    ("STATION", "NOAA station code", "text (ID)", "-", "64387099999 Kigali, 64380099999 Kamembe, 64381099999 Gisenyi, 64383099999 Ruhengeri"),
    ("DATE", "Day of the observation", "date", "YYYY-MM-DD", "2015-01-01 to 2025-08-24"),
    ("LATITUDE, LONGITUDE", "Station position", "float", "degrees", "constant per station"),
    ("ELEVATION", "Station height", "float", "m", "1,481-1,856"),
    ("NAME", "Station name", "category", "-", "e.g. 'KIGALI INTERNATIONAL, RW'"),
    ("TEMP", "Mean temperature", "float", "deg F", "9999.9 = missing"),
    ("DEWP", "Mean dew point", "float", "deg F", "9999.9 = missing"),
    ("SLP", "Sea-level pressure", "float", "mbar", "9999.9 = missing"),
    ("STP", "Station pressure", "float", "mbar", "999.9 = missing"),
    ("VISIB", "Mean visibility", "float", "miles", "999.9 = missing"),
    ("WDSP", "Mean wind speed", "float", "knots", "999.9 = missing"),
    ("MXSPD", "Max sustained wind", "float", "knots", "999.9 = missing"),
    ("GUST", "Max wind gust", "float", "knots", "999.9 = missing"),
    ("MAX, MIN", "Max / min temperature", "float", "deg F", "9999.9 = missing; '*' = derived from hourly data"),
    ("PRCP", "Total precipitation", "float", "inches", "99.99 = missing"),
    ("PRCP_ATTRIBUTES", "Precipitation flag", "category", "-", "A-H = report period, I = incomplete"),
    ("SNDP", "Snow depth", "float", "inches", "999.9 = missing (no snow in Rwanda)"),
    ("FRSHTT", "Weather events", "6-digit code", "-", "Fog Rain Snow Hail Thunder Tornado, 1 = occurred"),
    ("*_ATTRIBUTES", "Number of reports used for the daily value", "integer", "count", "quality indicator"),
    ("rain_tomorrow (created)", "TARGET: rain flag of the next day", "0/1", "-", "built in Stage 4"),
], columns=["column", "meaning", "data_type", "unit", "allowed_values_notes"])
(ROOT / "docs").mkdir(exist_ok=True)  # make sure docs/ exists
dictionary.to_csv(ROOT / "docs" / "data_dictionary.csv", index=False)  # save
dictionary  # display
'''),
    ("md", r'''
### 2.5 Source log - saved to `docs/source_log.csv`
'''),
    ("code", r'''
source_log = pd.DataFrame([{  # where the data came from
    "source": "NOAA National Centers for Environmental Information - Global Summary of the Day (GSOD)",
    "link": "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/ (station list: /pub/data/noaa/isd-history.csv)",
    "date_accessed": "2026-11-11",
    "licence": "free and open from NOAA; non-US stations under WMO Resolution 40 (free for education/research, no commercial resale)",
    "collected_by": "Rwanda airport weather stations (Meteo Rwanda / airport met offices), exchanged via WMO, summarised by NOAA",
    "limitations": "only 4 stations (not every district); days missing; precipitation totals often incomplete; 2023 missing for 2 stations",
    "ethical_issues": "no personal data; bias: stations at airports (valleys/plains), not on the steep slopes where landslides happen",
}])
source_log.to_csv(ROOT / "docs" / "source_log.csv", index=False)  # save
source_log.T  # display vertically
'''),
]

# ============================================================== 02 EDA
nb02 = [
    ("md", r'''
## Stage 3 - Explore and audit the data (EDA) (PC 1.3)
We only **look** here; nothing is overwritten. Numeric copies are made only for the checks and charts.
'''),
    ("code", SETUP + r'''
import matplotlib.pyplot as plt  # charts
import seaborn as sns  # statistical charts
sns.set_theme(style="whitegrid")  # chart style
FIG = ROOT / "docs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)  # chart folder
raw = rp.load_raw(RAW_DIR)  # all values as text, exactly as in the files
raw.columns = raw.columns.str.strip()  # column names
txt = raw.apply(lambda s: s.str.strip())  # text without padding, for the checks only
num = lambda c: pd.to_numeric(txt[c], errors="coerce")  # helper: a column as numbers
print(raw.shape)  # size
'''),
    ("md", r'''
### 3.1 Describe the dataset
'''),
    ("code", r'''
print("Padded values, e.g.:", raw["TEMP"].head(3).tolist())  # numbers stored as text with spaces
numeric = pd.DataFrame({c: num(c) for c in rp.MISSING_CODES})  # numeric copy of the measured columns
numeric.describe().T.round(2)  # summary statistics - look at the max values
'''),
    ("md", r'''
**Reading the summary:** several maximums are exactly **9999.9, 999.9 or 99.99** - NOAA's codes for "not measured".
`SLP` and `SNDP` are the code in **every** row, and `GUST`/`STP` in most rows. Temperatures are in **Fahrenheit**
(about 58-83), wind in **knots**, visibility in **miles**, rain in **inches**.
'''),
    ("code", r'''
for c in ["NAME", "PRCP_ATTRIBUTES", "FRSHTT"]:  # categorical / coded columns
    print(f"\n{c}:", txt[c].value_counts(dropna=False).head(10).to_dict())  # labels and counts
'''),
    ("md", r'''
### 3.2 Missing values (true and disguised), duplicates
'''),
    ("code", r'''
codes = pd.Series({c: int((numeric[c] == code).sum()) for c, code in rp.MISSING_CODES.items()})  # disguised missing
empty = txt.eq("").sum()  # empty strings
miss = pd.DataFrame({"missing_code": codes, "pct": (codes / len(raw) * 100).round(1)}).sort_values("missing_code", ascending=False)  # table
print("Empty cells per column (only flags):", empty[empty > 0].to_dict())  # e.g. PRCP_ATTRIBUTES
print("Duplicated station-days:", txt.duplicated(subset=["STATION", "DATE"]).sum())  # duplicates
miss  # disguised missing per column
'''),
    ("code", r'''
fig, ax = plt.subplots(figsize=(9, 4))  # empty chart
miss["pct"].plot.bar(ax=ax, color="#4C72B0")  # % missing per column
ax.set_ylabel("% of days"); ax.set_title("Figure Q2-3.1 - Share of 'not measured' codes per column")  # labels
plt.tight_layout(); plt.savefig(FIG / "q2_fig3_1_missing.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** `SLP` and `SNDP` are 100 % missing, `GUST` 96 % and `STP` 85 % - these columns cannot be used.
`MXSPD` (11 %) and `PRCP` (8 %) need imputation; the other columns are almost complete. No station-day is duplicated.
'''),
    ("md", r'''
### 3.3 Missing DAYS (gaps in the calendar)
'''),
    ("code", r'''
dates = pd.to_datetime(txt["DATE"])  # dates
cov = (pd.DataFrame({"station": txt["NAME"], "year": dates.dt.year}).value_counts().unstack(0).fillna(0).astype(int))  # days per station-year
fig, ax = plt.subplots(figsize=(10, 4))  # chart
sns.heatmap(cov.T, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=False)  # days recorded per year (max 365)
ax.set_title("Figure Q2-3.2 - Days recorded per station and year (365 = complete)")  # title
plt.tight_layout(); plt.savefig(FIG / "q2_fig3_2_coverage.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** Kigali is almost complete, but Gisenyi and Ruhengeri miss large parts of several years (and all of 2023),
and 2025 stops in August. Missing days mean "tomorrow" is sometimes unknown, and stations are **not equally represented**.
'''),
    ("md", r'''
### 3.4 Outliers and unrealistic values
'''),
    ("code", r'''
valid = numeric.mask(numeric.isin([9999.9, 999.9, 99.99]))  # hide the codes to see the real values
cols = ["TEMP", "DEWP", "MAX", "MIN", "WDSP", "MXSPD", "VISIB", "PRCP"]  # measured columns
fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))  # one box per column
for ax, c in zip(axes, cols):  # draw each
    sns.boxplot(y=valid[c], ax=ax, color="#DD8452"); ax.set_title(c)  # box plot
fig.suptitle("Figure Q2-3.3 - Box plots (codes removed, original units)"); plt.tight_layout()  # title
plt.savefig(FIG / "q2_fig3_3_boxplots.png", dpi=120); plt.show()  # save and display
'''),
    ("code", r'''
def iqr(s):  # IQR rule
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1  # quartiles and range
    return pd.Series({"lower": q1 - 1.5 * i, "upper": q3 + 1.5 * i, "n_outliers": int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())})  # result
out = valid[cols].apply(iqr).T.round(2)  # apply to each column
out["max"] = valid[cols].max().round(2)  # largest value for comparison
out  # table
'''),
    ("md", r'''
**Interpretation:** most IQR "outliers" are real weather (heavy-rain days, windy days) and are **kept**. Two values are not
credible: a daily **mean** wind of about 32 knots (59 km/h) and a max sustained wind of 60 knots (111 km/h) - both are
set to missing in Stage 4. Rain amounts are very right-skewed (most days 0, a few days > 1 inch) -> log transform later.
'''),
    ("md", r'''
### 3.5 Consistency checks and the precipitation problem
'''),
    ("code", r'''
dew_gt = (valid["DEWP"] > valid["TEMP"] + 1).sum()  # dew point above temperature (impossible)
max_lt = (valid["MAX"] < valid["MIN"]).sum()  # max below min (impossible)
rain_flag = txt["FRSHTT"].str.zfill(6).str[1].astype(int)  # rain observed (1/0)
p = valid["PRCP"]  # rain amount (inches)
incomplete = txt["PRCP_ATTRIBUTES"] == "I"  # incomplete totals
print("Dew point > temperature:", dew_gt, "| max < min:", max_lt)  # impossible combinations
print("PRCP flagged 'I' (incomplete):", incomplete.sum(), f"({incomplete.mean():.0%})")  # how many
print("Rain observed but PRCP = 0:", ((rain_flag == 1) & (p == 0)).sum())  # contradiction
print("Rain observed but PRCP = 0 AND flag I:", ((rain_flag == 1) & (p == 0) & incomplete).sum())  # explained by I
pd.crosstab(rain_flag.rename("rain flag"), pd.cut(p, [-0.1, 0, 0.1, 10], labels=["0", "0-0.1 in", ">0.1 in"]).rename("PRCP"))  # table
'''),
    ("md", r'''
**Finding:** the rain **flag** and the rain **amount** often disagree - 903 days say "rain observed" with 0.00 inch, and
961 days have more than 0.1 inch with no rain flag. They are recorded differently: the flag comes from the weather seen
during the reported hours, while the amount is a 24-hour total that can cover the previous night, and 36 % of totals are
also marked **incomplete ('I')**. We build the **target from the rain flag** because it is recorded every day (never
coded missing) and describes rain seen at the station - the event MINEMA warns about. The amount is kept only as a
feature with a `prcp_incomplete` indicator, and the disagreement is reported as a limitation.
'''),
    ("md", r'''
### 3.6 The target and relationships
'''),
    ("code", r'''
d = pd.DataFrame({"station": txt["NAME"].str.replace(", RW", "").str.title(), "date": dates, "rain": rain_flag}).sort_values(["station", "date"])  # small table
d["rain_tomorrow"] = d.groupby("station")["rain"].shift(-1).where(d.groupby("station")["date"].shift(-1) - d["date"] == pd.Timedelta(days=1))  # preview of target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # two charts
d["rain_tomorrow"].value_counts().sort_index().plot.bar(ax=axes[0], color=["#C44E52", "#4C72B0"])  # class balance
axes[0].set_xticklabels(["no rain (0)", "rain (1)"], rotation=0); axes[0].set_title("Figure Q2-3.4a - Target balance")  # labels
d.groupby(d["date"].dt.month)["rain"].mean().plot.bar(ax=axes[1], color="#55A868")  # rain frequency by month
axes[1].set_xlabel("month"); axes[1].set_ylabel("share of rainy days"); axes[1].set_title("Figure Q2-3.4b - Rainy days by month")  # labels
plt.tight_layout(); plt.savefig(FIG / "q2_fig3_4_target.png", dpi=120); plt.show()  # save and display
print("Share rain_tomorrow = 1:", round(d["rain_tomorrow"].mean(), 3))  # balance
'''),
    ("md", r'''
**Interpretation:** about 39 % of days are followed by a rainy day - **moderately imbalanced** (no resampling needed;
class weights are an option). Rain follows Rwanda's seasons clearly: the long rains (March-May) and short rains
(September-December) are wet, June-August is dry - so **season/month** is an important feature.
'''),
    ("code", r'''
t = (valid["TEMP"] - 32) * 5 / 9; dp = (valid["DEWP"] - 32) * 5 / 9  # quick Celsius copies for the chart
rh = 100 * np.exp(17.625 * dp / (243.04 + dp)) / np.exp(17.625 * t / (243.04 + t))  # relative humidity
plot = pd.DataFrame({"rel_humidity_today": rh, "station": d.sort_index()["station"], "rain_tomorrow": d.sort_index()["rain_tomorrow"]}).dropna()  # table
fig, ax = plt.subplots(figsize=(8, 4))  # chart
sns.boxplot(data=plot, x="station", y="rel_humidity_today", hue="rain_tomorrow", ax=ax)  # humidity today vs rain tomorrow
ax.set_title("Figure Q2-3.5 - Today's humidity vs rain tomorrow, per station")  # title
plt.tight_layout(); plt.savefig(FIG / "q2_fig3_5_humidity.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** at every station, days followed by rain have **higher humidity today** - a useful signal. Kigali is
drier overall, so the station must stay in the data.
'''),
    ("code", r'''
corr = pd.DataFrame({"TEMP": valid.TEMP, "DEWP": valid.DEWP, "MAX": valid.MAX, "MIN": valid.MIN, "WDSP": valid.WDSP,
                     "VISIB": valid.VISIB, "PRCP": valid.PRCP, "rain_today": rain_flag,
                     "rain_tomorrow": d.sort_index()["rain_tomorrow"]}).corr()  # correlation matrix
fig, ax = plt.subplots(figsize=(8, 6))  # chart
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)  # heatmap
ax.set_title("Figure Q2-3.6 - Correlation heatmap"); plt.tight_layout()  # title
plt.savefig(FIG / "q2_fig3_6_corr.png", dpi=120); plt.show()  # save and display
'''),
    ("md", r'''
**Interpretation:** rain **today** and the dew point are the clearest single signals for rain tomorrow; mean, max and min
temperature are strongly related to each other (redundant information). **Leakage risk:** any value from tomorrow
(e.g. tomorrow's PRCP or FRSHTT) must never be a feature, and the split must follow time.
'''),
    ("md", r'''
### 3.7 Data Quality Report - saved to `docs/data_quality_report.csv`
'''),
    ("code", r'''
gaps = int(d.groupby("station")["date"].diff().dt.days.gt(1).sum())  # number of breaks in the calendar
dqr = pd.DataFrame([  # problem, columns, records, severity, treatment
    (1, "Numbers and dates stored as padded text ('  70.0')", "all", int((raw["TEMP"].str.len() != raw["TEMP"].str.strip().str.len()).sum()), "Medium", "strip spaces, convert to float/datetime"),
    (2, "Disguised missing codes 9999.9 / 999.9 / 99.99", ", ".join(codes[codes > 0].index), int(codes.sum()), "High", "replace by NaN"),
    (3, "Columns (almost) completely empty", "SLP, SNDP, GUST, STP", int(codes[["SLP", "SNDP", "GUST", "STP"]].min()), "Medium", "drop columns > 50 % missing"),
    (4, "Imperial units (F, knots, miles, inches)", "TEMP, DEWP, MAX, MIN, WDSP, MXSPD, VISIB, PRCP", len(raw), "Medium", "convert to C, km/h, km, mm"),
    (5, "Several facts packed in one code (FRSHTT '010010')", "FRSHTT", len(raw), "Medium", "split into fog/rain/hail/thunder 0/1"),
    (6, "Incomplete precipitation totals (flag I)", "PRCP, PRCP_ATTRIBUTES", int(incomplete.sum()), "High", "do not use PRCP as target; keep prcp_incomplete flag"),
    (7, "Rain flag and rain amount disagree (flag 1 & 0 in, or flag 0 & > 0.1 in)", "PRCP vs FRSHTT",
     int(((rain_flag == 1) & (p == 0)).sum() + ((rain_flag == 0) & (p > 0.1)).sum()), "High", "target from rain flag; report as limitation"),
    (8, "Missing days / station-years (gaps)", "DATE", gaps, "High", "target only when next day exists; lags only over consecutive days"),
    (9, "Unequal station coverage", "NAME", int(cov.sum().max() - cov.sum().min()), "Low", "keep station as a feature; report bias"),
    (10, "Unrealistic wind values", "WDSP, MXSPD", int((valid.WDSP > 21.6).sum() + (valid.MXSPD > 48.6).sum()), "Low", "set to NaN"),
    (11, "Skewed rain amount", "PRCP", int((valid.PRCP > 0).sum()), "Low", "log(1 + mm) feature"),
    (12, "Leakage risk: target comes from tomorrow; random split would mix future into training", "rain_tomorrow, DATE", len(raw), "High", "only today/past features; chronological split"),
    (13, "Long station names with country suffix", "NAME", len(raw), "Low", "short station + district + province"),
], columns=["no", "problem", "columns", "records_affected", "severity", "proposed_treatment"])
dqr.to_csv(ROOT / "docs" / "data_quality_report.csv", index=False)  # save
dqr  # display (13 kinds of problems)
'''),
    ("md", r'''
**Checkpoint answer - two problems and why they matter**
1. **Incomplete precipitation (flag I):** a day with "0 inches" may really have been rainy; a target built from PRCP would
   teach the model wrong answers. We use the observed rain flag instead.
2. **Time leakage:** with a random split, days from 2025 would train a model tested on 2017 - it would look better than it
   can ever be in real use. We split by time.
'''),
]

# ============================================================== 03 clean
nb03 = [
    ("md", r'''
## Stage 4 - Clean and pre-process the data (PC 1.3)
Each step calls a documented function in `src/rain_pipeline.py`; every decision goes into the Cleaning Log.
'''),
    ("code", SETUP + r'''
raw = rp.load_raw(RAW_DIR)  # original data
log = rp.CleaningLog()  # empty log
def show(before, after):  # before/after helper
    print(f"shape {before.shape} -> {after.shape}")  # size change
    common = [c for c in after.columns if c in before.columns]  # columns in both
    print("missing (after):", after.isna().sum()[lambda s: s > 0].to_dict())  # remaining missing values
'''),
    ("md", "### Step 1: trim padded text, convert types"),
    ("code", r'''
df = rp.trim_and_type(raw, log)  # strip spaces, text -> numbers/dates
print(raw.dtypes.value_counts().to_dict(), "->", df.dtypes.astype(str).value_counts().to_dict())  # types before/after
'''),
    ("md", "### Step 2: missing codes -> NaN"),
    ("code", r'''
step = df.copy()  # before
df = rp.replace_missing_codes(df, log)  # 9999.9 etc. -> NaN
pd.DataFrame({"max_before": step[list(rp.MISSING_CODES)].max(), "max_after": df[list(rp.MISSING_CODES)].max(),
              "missing_after": df[list(rp.MISSING_CODES)].isna().sum()})  # codes disappear, NaN appear
'''),
    ("md", "### Step 3: drop empty columns"),
    ("code", r'''
step = df.copy()  # before
df = rp.drop_empty_columns(df, log)  # SLP, SNDP, GUST, STP + bookkeeping columns
print("removed:", sorted(set(step.columns) - set(df.columns)))  # which columns went
show(step, df)  # before/after
'''),
    ("md", "### Step 4-5: station names, metric units"),
    ("code", r'''
step = df.copy()  # before
df = rp.name_stations(df, log)  # short name, district, province
df = rp.convert_units(df, log)  # F -> C, knots -> km/h, miles -> km, inches -> mm
print(df[["station", "district", "province"]].drop_duplicates().to_string(index=False))  # station table
df[["station", "DATE", "temp_c", "dewpoint_c", "max_temp_c", "min_temp_c", "wind_kmh", "visibility_km", "prcp_mm"]].describe().round(1)  # metric ranges
'''),
    ("md", "### Step 6-7: decode flags, impossible values"),
    ("code", r'''
step = df.copy()  # before
df = rp.quality_flags(df, log)  # FRSHTT -> fog/rain/hail/thunder; PRCP flag; n_obs
df = rp.impossible_values(df, log)  # impossible combinations / unrealistic wind -> NaN
print(df[["fog", "rain", "hail", "thunder", "prcp_incomplete"]].mean().round(3).to_dict())  # share of days with each event
print("max wind before/after:", step["max_wind_kmh"].max(), "->", df["max_wind_kmh"].max())  # effect
'''),
    ("md", "### Step 8-9: duplicates, target `rain_tomorrow`"),
    ("code", r'''
step = df.copy()  # before
df = rp.remove_duplicates(df, log)  # one row per station-day
df = rp.add_target(df, log)  # rain flag of the NEXT calendar day; rows without a real tomorrow removed
show(step, df)  # before/after
print("rain_tomorrow share:", round(df[rp.TARGET].mean(), 3))  # balance
'''),
    ("md", "### Reproducibility: one call gives the same table"),
    ("code", r'''
again, _ = rp.clean_rain(rp.load_raw(RAW_DIR))  # all steps at once
pd.testing.assert_frame_equal(df, again)  # error if anything differs
print("identical:", again.shape)  # confirmation
'''),
    ("code", r'''
clean_log = log.to_frame()  # the log
clean_log.to_csv(ROOT / "docs" / "cleaning_log.csv", index=False)  # save
(ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)  # folder
df.to_csv(ROOT / "data" / "interim" / "rain_clean.csv", index=False)  # clean dataset
clean_log[["step", "operation", "column", "problem", "action", "records_affected"]]  # display
'''),
    ("md", r'''
### Chronological split (train 2015-2022, validation 2023, test 2024-2025)
Weather is a time series: a **random** split would let the model learn from days after the test days (leakage).
We train on the past and test on the most recent years - exactly how MINEMA would use the model.
'''),
    ("code", r'''
train, val, test = rp.chronological_split(df)  # split by date
for n, part in [("train", train), ("val", val), ("test", test)]:  # each part
    print(f"{n:<5} {part.shape} {part['DATE'].min().date()} -> {part['DATE'].max().date()}  rain share {part[rp.TARGET].mean():.3f}  stations {sorted(part.station.unique())}")
    part.to_csv(ROOT / "data" / "interim" / f"{n}_clean.csv", index=False)  # save un-encoded split
'''),
    ("md", r'''
**Notes:** the validation year 2023 has only Kigali and Kamembe (NOAA has no 2023 data for Gisenyi and Ruhengeri) - a known
limitation. The split is about 86 / 7 / 7 % instead of 70/15/15 because a time split must follow the calendar; it still
gives 640+ validation and 730+ test days. The target is moderately imbalanced (~39 % rain) -> planned remedy:
`class_weight="balanced"`.
'''),
    ("md", "### Imputation, encoding, scaling - fitted on the training set only"),
    ("code", r'''
from sklearn.compose import ColumnTransformer  # column-wise recipe
from sklearn.pipeline import Pipeline  # ordered steps
from sklearn.impute import SimpleImputer  # missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding and scaling
num_cols = ["temp_c", "dewpoint_c", "max_temp_c", "min_temp_c", "wind_kmh", "max_wind_kmh", "visibility_km", "prcp_mm", "n_obs"]  # numbers
flag_cols = ["fog", "rain", "hail", "thunder", "prcp_incomplete"]  # 0/1 flags (no scaling needed)
cat_cols = ["station"]  # category
prep = ColumnTransformer([  # recipe
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),  # median + z-score
    ("flag", SimpleImputer(strategy="most_frequent"), flag_cols),  # flags kept as 0/1
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),  # one column per station
])
prep.fit(train)  # LEARN from training rows only
Xtr, Xva, Xte = (prep.transform(p) for p in (train, val, test))  # apply to all
print("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)  # same columns
pd.DataFrame(Xtr, columns=prep.get_feature_names_out()).head().round(2)  # first encoded rows
'''),
]

# ============================================================== 04 features
nb04 = [
    ("md", r'''
## Stage 5 - Engineer and select features (PC 1.4)
All new features use **today and earlier days of the same station only** - never tomorrow - so they are safe.
Selection statistics are computed on the **training years only**.
'''),
    ("code", SETUP + r'''
import matplotlib.pyplot as plt  # charts
import seaborn as sns  # charts
INTERIM, PROCESSED, FIG = ROOT / "data" / "interim", ROOT / "data" / "processed", ROOT / "docs" / "figures"  # folders
PROCESSED.mkdir(parents=True, exist_ok=True)  # create if needed
clean = pd.read_csv(INTERIM / "rain_clean.csv", parse_dates=["DATE"])  # clean data
feat = rp.add_features(clean)  # features computed on the full timeline (lags need the previous day)
train, val, test = rp.chronological_split(feat)  # same time split as Stage 4
print({n: p.shape for n, p in [("train", train), ("val", val), ("test", test)]})  # sizes
'''),
    ("md", r'''
Lags (yesterday, last 3 days) are computed **before** splitting because the first day of 2023 legitimately knows the weather
of 31 Dec 2022 - that is the past, not the future, so there is no leakage.
'''),
    ("code", r'''
new_cols = ["month", "rw_season", "doy_sin", "doy_cos", "dewpoint_spread_c", "rel_humidity_pct", "diurnal_range_c",
            "rain_yesterday", "temp_change_c", "rain_days_last3", "log_prcp_mm"]  # 11 new features
train[["station", "DATE", "rain"] + new_cols + [rp.TARGET]].head(8)  # first rows with the new columns
'''),
    ("code", r'''
sk = train[["prcp_mm", "log_prcp_mm", "wind_kmh", "rel_humidity_pct"]].skew().round(2)  # skewness
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))  # two histograms
sns.histplot(train["prcp_mm"], bins=40, ax=axes[0]); axes[0].set_title(f"prcp_mm, skew = {sk['prcp_mm']}")  # before
sns.histplot(train["log_prcp_mm"], bins=40, ax=axes[1], color="#55A868"); axes[1].set_title(f"log(1+prcp_mm), skew = {sk['log_prcp_mm']}")  # after
fig.suptitle("Figure Q2-5.1 - Log transform of rain amount"); plt.tight_layout()  # title
plt.savefig(FIG / "q2_fig5_1_log_prcp.png", dpi=120); plt.show()  # save and display
'''),
    ("code", r'''
from sklearn.feature_selection import mutual_info_classif  # dependence measure for a 0/1 target
from sklearn.ensemble import RandomForestClassifier  # quick model for importance
cands = ["temp_c", "dewpoint_c", "max_temp_c", "min_temp_c", "wind_kmh", "max_wind_kmh", "visibility_km", "prcp_mm",
         "elevation_m", "fog", "rain", "hail", "thunder", "prcp_incomplete", "n_obs", "month", "doy_sin", "doy_cos",
         "dewpoint_spread_c", "rel_humidity_pct", "diurnal_range_c", "rain_yesterday", "temp_change_c",
         "rain_days_last3", "log_prcp_mm"]  # numeric candidates
X = train[cands].fillna(train[cands].median())  # simple fill (train medians) only for measuring
X = X.join(pd.get_dummies(train[["station", "rw_season"]], dtype=float))  # categories as 0/1
y = train[rp.TARGET]  # target
corr = train[cands + [rp.TARGET]].corr()  # correlations (train)
mi = pd.Series(mutual_info_classif(X, y, random_state=rp.SEED), index=X.columns)  # mutual information
rf = RandomForestClassifier(n_estimators=300, random_state=rp.SEED, n_jobs=-1, min_samples_leaf=5).fit(X, y)  # forest
rank = pd.DataFrame({"corr_with_target": corr[rp.TARGET].reindex(X.columns), "mutual_info": mi,
                     "rf_importance": rf.feature_importances_}, index=X.columns).sort_values("rf_importance", ascending=False)  # table
fig, ax = plt.subplots(figsize=(8, 8))  # chart
rank["rf_importance"].sort_values().plot.barh(ax=ax, color="#4C72B0")  # bars
ax.set_title("Figure Q2-5.2 - Random Forest importance (training years)"); plt.tight_layout()  # title
plt.savefig(FIG / "q2_fig5_2_rf_importance.png", dpi=120); plt.show()  # save and display
rank.round(3)  # table
'''),
    ("code", r'''
from sklearn.feature_selection import VarianceThreshold  # selection method 1
vt = VarianceThreshold(0.01).fit(X)  # near-constant columns
low_var = [c for c, k in zip(X.columns, vt.get_support()) if not k]  # removed by the threshold
print("near-constant:", low_var)  # e.g. hail (almost never)
up = corr.loc[cands, cands].abs().where(np.triu(np.ones((len(cands),) * 2), k=1).astype(bool))  # upper triangle
pairs = up.stack()[lambda s: s > 0.85].round(3)  # strongly related pairs
print("pairs with |r| > 0.85:\n", pairs.to_string())  # list
'''),
    ("code", r'''
DROP = {  # removed feature -> reason (evidence above)
    "hail": f"near-constant (variance threshold): hail on {train['hail'].mean():.1%} of days",
    "elevation_m": "one value per station - duplicates the station column",
    "max_temp_c": "strongly correlated with temp_c; its information kept in diurnal_range_c",
    "min_temp_c": "strongly correlated with temp_c/dewpoint_c; its information kept in diurnal_range_c",
    "prcp_mm": "very skewed and often incomplete; replaced by log_prcp_mm + prcp_incomplete",
    "month": "replaced by doy_sin/doy_cos (circular) and rw_season (agronomic seasons)",
    "dewpoint_c": f"r = {corr.loc['dewpoint_c', 'rel_humidity_pct']:.2f} with rel_humidity_pct, which is easier to explain",
    "dewpoint_spread_c": f"r = {corr.loc['dewpoint_spread_c', 'rel_humidity_pct']:.3f} with rel_humidity_pct (same information, |r| > 0.85)",
    "n_obs": f"low importance ({rank.loc['n_obs', 'rf_importance']:.3f}); quality bookkeeping, not weather",
}
FINAL_NUM = ["temp_c", "rel_humidity_pct", "diurnal_range_c", "wind_kmh", "max_wind_kmh",
             "visibility_km", "log_prcp_mm", "temp_change_c", "rain_days_last3", "doy_sin", "doy_cos"]  # scaled numbers
FINAL_FLAG = ["rain", "thunder", "fog", "rain_yesterday", "prcp_incomplete"]  # 0/1 flags
FINAL_CAT = ["station", "rw_season"]  # one-hot categories
print(len(FINAL_NUM + FINAL_FLAG + FINAL_CAT), "selected inputs:", FINAL_NUM + FINAL_FLAG + FINAL_CAT)  # final list
'''),
    ("code", r'''
from sklearn.compose import ColumnTransformer  # recipe
from sklearn.pipeline import Pipeline  # steps
from sklearn.impute import SimpleImputer  # missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding/scaling
import joblib  # save the fitted recipe
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FINAL_NUM),
    ("flag", SimpleImputer(strategy="most_frequent"), FINAL_FLAG),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FINAL_CAT),
], verbose_feature_names_out=False)  # readable names
prep.fit(train)  # learn from 2015-2022 only
names = list(prep.get_feature_names_out())  # output column names
for n, part in [("train", train), ("val", val), ("test", test)]:  # each split
    out = pd.concat([part[["station", "DATE"]].reset_index(drop=True).rename(columns={"station": "station_id"}),
                     pd.DataFrame(prep.transform(part), columns=names), part[[rp.TARGET]].reset_index(drop=True)], axis=1)  # ids + X + y
    out.to_csv(PROCESSED / f"rain_{n}.csv", index=False)  # model-ready file
    part.to_csv(PROCESSED / f"rain_{n}_readable.csv", index=False)  # un-scaled copy
    print(f"{n:<5} {out.shape}  missing: {int(out.drop(columns=['station_id', 'DATE']).isna().sum().sum())}")  # check
joblib.dump(prep, PROCESSED / "preprocessor.joblib")  # save the recipe
'''),
    ("code", r'''
fj = pd.DataFrame([  # feature table
    ("rw_season", "month -> Long rains / Short rains / Long dry / Short dry", "rain follows Rwanda's seasons (Fig. Q2-3.4b)", "kept"),
    ("doy_sin, doy_cos", "day of year on a circle", "31 Dec and 1 Jan are neighbours", "kept"),
    ("rel_humidity_pct", "Magnus formula from temp and dew point", f"r = {corr.loc['rel_humidity_pct', rp.TARGET]:.2f} with target", "kept"),
    ("dewpoint_spread_c", "temp_c - dewpoint_c", f"r = {corr.loc['dewpoint_spread_c', rp.TARGET]:.2f}; created, then removed (duplicate of humidity)", "removed"),
    ("diurnal_range_c", "max_temp_c - min_temp_c", f"r = {corr.loc['diurnal_range_c', rp.TARGET]:.2f}; cloudy days vary less", "kept"),
    ("rain_yesterday", "rain flag of the previous day (if consecutive)", f"r = {corr.loc['rain_yesterday', rp.TARGET]:.2f}; wet spells persist", "kept"),
    ("rain_days_last3", "rainy days among the last 3 consecutive days", f"r = {corr.loc['rain_days_last3', rp.TARGET]:.2f}", "kept"),
    ("temp_change_c", "temp today - temp yesterday", f"r = {corr.loc['temp_change_c', rp.TARGET]:.2f}; cooling before rain", "kept"),
    ("log_prcp_mm", "log(1 + prcp_mm)", f"skew {sk['prcp_mm']} -> {sk['log_prcp_mm']}", "kept"),
    ("rain, thunder, fog, prcp_incomplete", "decoded from FRSHTT / PRCP flag", f"rain today r = {corr.loc['rain', rp.TARGET]:.2f}", "kept"),
] + [(k, "-", v, "removed") for k, v in DROP.items()], columns=["feature", "how_created", "evidence", "decision"])
fj.to_csv(ROOT / "docs" / "feature_justification.csv", index=False)  # save
fj  # display
'''),
    ("code", r'''
final = pd.read_csv(PROCESSED / "rain_train.csv")  # read back
final.info()  # all inputs numeric, no missing values
sorted(p.name for p in PROCESSED.iterdir())  # delivered files
'''),
]

for name, cells, stage, roles in [
    ("01_acquire", nb01, "2 - Define requirements and acquire data", "Driver: IGIHOZO Bienvenue / Navigator: MUGISHA Aime Patrick"),
    ("02_eda", nb02, "3 - Explore and audit the data (EDA)", "Driver: MUGISHA Aime Patrick / Navigator: IGIHOZO Bienvenue"),
    ("03_clean", nb03, "4 - Clean and pre-process the data", "Driver: IGIHOZO Bienvenue / Navigator: MUGISHA Aime Patrick"),
    ("04_features", nb04, "5 - Engineer and select features (+ Stage 6 final files)", "Driver: MUGISHA Aime Patrick / Navigator: IGIHOZO Bienvenue"),
]:
    nbf.write(nb(cells, stage, roles), NB_DIR / f"{name}.ipynb")
    print("wrote", name)
