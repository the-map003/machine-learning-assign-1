"""
build_summaries.py - two short summary documents (Q1 and Q2), numbers read from the pipeline outputs.
Run from the repository root:  python setup/build_summaries.py
Creates Q1_Summary_Maize_Yield.docx and Q2_Summary_MINEMA_Rain_Warning.docx in the repository root.
"""
import importlib.util
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

REPO = Path(__file__).resolve().parents[1]
Q1, Q2 = REPO / "q1_maize_yield", REPO / "q2_own_scenario"
MEMBERS = "IGIHOZO Bienvenue and MUGISHA Aime Patrick"

spec = importlib.util.spec_from_file_location("q1docs", Q1 / "tools" / "build_docs.py")  # reuse Word helpers
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def head(d, title, subtitle):
    h.para(d, title, bold=True, size=17, color=h.BLUE)
    h.para(d, subtitle, italic=True, size=11)
    h.para(d, f"Group XX - {MEMBERS} - Machine Learning, Element 1 (dataset preparation) - 12 November 2026", size=9)


def section(d, title):
    p = d.add_heading(title, 2)
    p.paragraph_format.space_before = Pt(8)


def small_doc():
    d = h.base_doc()
    for s in d.sections:
        s.top_margin = s.bottom_margin = Cm(1.5)
        s.left_margin = s.right_margin = Cm(1.8)
    return d


# ====================================================================== Q1
def build_q1():
    raw = pd.read_csv(Q1 / "data" / "raw" / "maize_yield_rwanda_rw.csv")
    clean = pd.read_csv(Q1 / "data" / "interim" / "maize_clean.csv")
    proc = {n: pd.read_csv(Q1 / "data" / "processed" / f"maize_{n}.csv") for n in ["train", "val", "test"]}
    tr = pd.read_csv(Q1 / "data" / "processed" / "maize_train_readable.csv")
    fj = pd.read_csv(Q1 / "docs" / "feature_justification.csv")
    thr = tr.loc[tr.low_yield_risk == 0, "yield_t_ha"].min()
    seed = raw.groupby("seed_variety")["yield_t_ha"].mean()
    n_gap = int(raw[["rainfall_mm", "soil_ph", "pest_incidence_index"]].isna().any(axis=1).sum())
    n_inputs = proc["train"].shape[1] - 4

    d = small_doc()
    head(d, "Question 1 - Summary", "Predicting maize yield for farmer cooperatives in Rwanda (MCSU case study)")

    section(d, "1. Task")
    h.para(d, "MCSU wants a yield forecast before each harvest. Our job was to prepare the supplied data so the modelling "
              "team can use it directly - without building a model - and to answer five business questions: expected yield "
              "(A), effect of fertiliser and seed (B), effect of weather, altitude and irrigation (C), differences by place "
              "and season (D) and which farms risk a very low harvest (E).")

    section(d, "2. Data received")
    h.table(d, pd.DataFrame([
        ("File", "maize_yield_rwanda_rw.csv (supplied by the lecturer; kept unchanged in data/raw/)"),
        ("Size", f"{raw.shape[0]:,} farm-season records x {raw.shape[1]} columns"),
        ("Coverage", f"{raw['year'].min()}-{raw['year'].max()}, Seasons A/B/C, 5 provinces, {raw['district'].nunique()} districts"),
        ("Target", "yield_t_ha (mean 2.58 t/ha, range 1.0-5.2)"),
        ("Usable after cleaning", f"{len(clean):,} records (none removed)"),
    ], columns=["Item", "Details"]), widths=[4, 13.4], font=9)

    section(d, "3. What the audit found")
    h.bullets(d, [
        f"Missing values: 39 empty cells, all in {n_gap} farms with no soil test, pest score and (mostly) rainfall - a pattern, not random.",
        "No duplicates, no misspelled labels, no wrong district-province pairs and no impossible values (all checks documented).",
        "IQR outliers (mostly large farms, skew 1.95) are realistic - kept, farm area log-transformed.",
        "True zeros in fertiliser/manure (farms using none) - kept as real values, not missing.",
        "Altitude and temperature strongly related (r = -0.88); Season C (226) and irrigated farms (128) are few.",
        "No farm below 1,000 kg/ha, so 'very low harvest' had to be redefined.",
        "Timing leakage risk: season rainfall and temperature are only fully known at the end of the season.",
        "File differs from the brief: yield in t/ha, fertiliser per ha, no cooperative code or planting date.",
    ])

    section(d, "4. Cleaning and preparation")
    h.bullets(d, [
        "All steps in code (src/maize_pipeline.py), every decision written in the Cleaning Log; raw file never edited.",
        "'Kigali' renamed to the official 'Kigali City'; yield also given in kg/ha.",
        "Split 70 / 15 / 15 with seed 42, stratified by season: "
        f"{len(proc['train'])} / {len(proc['val'])} / {len(proc['test'])} rows.",
        "Missing values filled with medians, categories one-hot encoded, numbers standardised - all learned on the training set only.",
        f"Question E target low_yield_risk = bottom 10 % of training yields (below {thr:.2f} t/ha), learned on train only.",
    ])

    section(d, "5. New features (10)")
    kept = fj[fj.decision == "kept"][["feature", "how_created_or_reason", "business_question"]]
    h.table(d, kept.rename(columns={"how_created_or_reason": "how it was created", "business_question": "Q"}),
            widths=[4.4, 11.5, 1.5], font=8.5)
    h.para(d, f"Most informative (training set): seed variety (Local {seed['Local']:.2f} t/ha vs PAN691 {seed['PAN691']:.2f} t/ha), "
              "rainfall, season, pest pressure and plant density; the improved seed + practice package has r = 0.46 with yield. "
              "Removed: province (same as district), urea (inside nitrogen applied), raw farm area (log kept), a near-constant flag.",
           size=9.5)

    section(d, "6. Delivered")
    h.bullets(d, [
        f"data/processed/maize_train.csv, maize_val.csv, maize_test.csv - {n_inputs} encoded inputs, no missing values, "
        "targets yield_t_ha, yield_kg_ha and low_yield_risk.",
        "preprocessor.joblib (fitted pipeline) and reusable cleaning/feature functions for new data.",
        "Data dictionary, source log, data quality report, cleaning log, feature table, data card and memo.",
    ])

    section(d, "7. Limits and recommendation")
    h.para(d, "The data is ready for modelling and supports questions A-D now, and E if rainfall measured up to the forecast "
              "date replaces season totals. It cannot give results per cooperative or by planting date. Next season MCSU "
              "should record a cooperative code, planting date and mid-season rainfall, and test the soil of every farm.")
    out = REPO / "Q1_Summary_Maize_Yield.docx"
    d.save(out)
    return out


# ====================================================================== Q2
def build_q2():
    raw_n = sum(len(pd.read_csv(f)) for f in (Q2 / "data" / "raw" / "gsod").glob("*.csv"))
    n_files = len(list((Q2 / "data" / "raw" / "gsod").glob("*.csv")))
    clean = pd.read_csv(Q2 / "data" / "interim" / "rain_clean.csv")
    proc = {n: pd.read_csv(Q2 / "data" / "processed" / f"rain_{n}.csv") for n in ["train", "val", "test"]}
    fj = pd.read_csv(Q2 / "docs" / "feature_justification.csv")
    dqr = pd.read_csv(Q2 / "docs" / "data_quality_report.csv")
    share = clean["rain_tomorrow"].mean()
    n_inputs = proc["train"].shape[1] - 3

    d = small_doc()
    head(d, "Question 2 - Summary", "Next-day rain early warning for MINEMA district disaster offices (own scenario)")

    section(d, "1. Scenario")
    h.para(d, "Heavy and repeated rain causes floods and landslides in Rwanda (May 2023: more than 130 deaths in the Western "
              "and Northern provinces). Every evening MINEMA's duty officer decides whether to send SMS warnings and put "
              "district teams on standby for the next day. We prepared the data for a future model that answers: "
              "\"Will it rain tomorrow at this station?\" - target rain_tomorrow (1/0), binary classification.")
    h.table(d, pd.DataFrame([
        ("Stakeholder", "MINEMA Early Warning unit + District Disaster Management Committees"),
        ("Area", "Kigali (Kicukiro), Rusizi (Kamembe), Rubavu (Gisenyi), Musanze (Ruhengeri)"),
        ("Who benefits", "households in high-risk zones, district teams, MINEMA, farmers"),
    ], columns=["Item", "Details"]), widths=[4, 13.4], font=9)

    section(d, "2. Data")
    h.table(d, pd.DataFrame([
        ("Source", "NOAA NCEI Global Summary of the Day - real airport station observations (free, WMO Res. 40)"),
        ("Compared with", "Open-Meteo historical API - rejected (modelled values, too clean); Meteo Rwanda - needs a request"),
        ("Size", f"{raw_n:,} station-days in {n_files} yearly files, 28 columns, 2015-2025"),
        ("Usable after cleaning", f"{len(clean):,} station-days; rain tomorrow on {share:.0%} of days"),
    ], columns=["Item", "Details"]), widths=[4, 13.4], font=9)

    section(d, f"3. What the audit found ({len(dqr)} kinds of problems)")
    h.bullets(d, [
        "'Not measured' codes 9999.9 / 999.9 / 99.99 in most columns; numbers stored as padded text.",
        "Columns almost or fully empty: sea-level pressure and snow depth (100 %), gusts (96 %), station pressure (85 %).",
        "Imperial units (F, knots, miles, inches); six weather events packed in one code (FRSHTT).",
        "36 % of rain totals marked incomplete; rain flag and rain amount disagree on about 1,860 days.",
        "Missing days and a missing year (2023) for Gisenyi and Ruhengeri - unequal coverage of stations.",
        "Two unrealistic wind values; very skewed rain amounts.",
        "Leakage risk: the target comes from tomorrow, and a random split would mix future days into training.",
    ])

    section(d, "4. Cleaning and preparation")
    h.bullets(d, [
        "All steps in code (src/rain_pipeline.py), every decision in the Cleaning Log; raw files never edited.",
        "Codes -> missing; empty columns dropped; units converted to C, km/h, km, mm; FRSHTT split into fog/rain/hail/thunder.",
        "Target rain_tomorrow built from the next day's rain flag, only when the next calendar day exists.",
        f"Chronological split: train 2015-2022 ({len(proc['train'])}), validation 2023 ({len(proc['val'])}), "
        f"test 2024-2025 ({len(proc['test'])}) - the model never learns from the future.",
        "Median imputation, one-hot encoding and scaling learned on the training years only.",
    ])

    section(d, "5. New features")
    kept = fj[fj.decision == "kept"][["feature", "how_created"]]
    h.table(d, kept.rename(columns={"how_created": "how it was created"}), widths=[5, 12.4], font=8.5)
    h.para(d, "Most informative (training years): time of year, humidity, recent wet days and rain today. Removed: dew-point "
              "spread (r = 0.99 with humidity), hail (almost never), max/min temperature, raw rain amount, month, elevation.",
           size=9.5)

    section(d, "6. Delivered")
    h.bullets(d, [
        f"data/processed/rain_train.csv, rain_val.csv, rain_test.csv - {n_inputs} encoded inputs, no missing values, target rain_tomorrow.",
        "preprocessor.joblib and reusable cleaning/feature functions; scenario brief + proposal form for approval.",
        "Data dictionary, source log, data quality report, cleaning log, feature table and data card.",
    ])

    section(d, "7. Limits")
    h.para(d, "Only four airport stations, in valleys rather than on the slopes where landslides start; the rain flag says "
              "whether it rained, not how much; validation (2023) contains only Kigali and Kamembe. Meteo Rwanda station "
              "and rain-gauge data would make the warning more local.")
    out = REPO / "Q2_Summary_MINEMA_Rain_Warning.docx"
    d.save(out)
    return out


if __name__ == "__main__":
    for f in (build_q1(), build_q2()):
        print("wrote", f)
