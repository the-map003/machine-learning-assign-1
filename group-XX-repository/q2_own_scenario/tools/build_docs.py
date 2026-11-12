"""
build_docs.py (Question 2) - run AFTER the Q2 notebooks.   python tools/build_docs.py
Creates docs/Q2_Scenario_Brief_and_Proposal.docx, docs/Q2_Code_Explained_Line_by_Line.docx, docs/data_card.md.
add_q2_section(doc, h) is called by the Q1 builder to put Question 2 into the single group report.
"""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

Q2 = Path(__file__).resolve().parents[1]
DOCS, FIG, NB, PROC, INTERIM = Q2 / "docs", Q2 / "docs" / "figures", Q2 / "notebooks", Q2 / "data" / "processed", Q2 / "data" / "interim"


def load_h():
    """Re-use the Word helpers (tables, figures, screenshot boxes) of the Q1 builder."""
    spec = importlib.util.spec_from_file_location("q1docs", Q2.parent / "q1_maize_yield" / "tools" / "build_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def numbers():
    raw_rows = sum(len(pd.read_csv(f)) for f in (Q2 / "data" / "raw" / "gsod").glob("*.csv"))
    clean = pd.read_csv(INTERIM / "rain_clean.csv")
    splits = {n: pd.read_csv(PROC / f"rain_{n}.csv") for n in ["train", "val", "test"]}
    return dict(raw_rows=raw_rows, n_files=len(list((Q2 / "data" / "raw" / "gsod").glob("*.csv"))), clean=clean,
                splits=splits, n_feat=splits["train"].shape[1] - 3, rain_share=clean["rain_tomorrow"].mean(),
                dqr=pd.read_csv(DOCS / "data_quality_report.csv"), clog=pd.read_csv(DOCS / "cleaning_log.csv"),
                fj=pd.read_csv(DOCS / "feature_justification.csv"), ddict=pd.read_csv(DOCS / "data_dictionary.csv"),
                slog=pd.read_csv(DOCS / "source_log.csv"))


BRIEF = [
    ("Who is asking",
     "The Early Warning and Disaster Risk Information unit of MINEMA (Ministry in charge of Emergency Management), which "
     "supports the District Disaster Management Committees of Rubavu, Musanze, Rusizi and the City of Kigali."),
    ("The problem",
     "Heavy and repeated rain causes floods and landslides in Rwanda. In early May 2023, floods and landslides in the "
     "Western and Northern provinces killed more than 130 people and destroyed thousands of houses. Every evening the "
     "MINEMA duty officer must decide whether to send SMS warnings to people living in high-risk zones, ask district teams "
     "to clear drainage channels and keep response teams on standby for the next day. Today this decision is made from "
     "general forecasts and experience. Warnings sent too often are ignored; warnings missed cost lives."),
    ("The decision a prediction supports",
     "Send / do not send a next-day rain alert for each station area. Target to predict later: rain_tomorrow "
     "(1 = rain observed at the station tomorrow, 0 = no rain) - binary classification. Predicting it matters because "
     "rain on already wet days (wet spells) is what saturates slopes and triggers landslides."),
    ("What data exists",
     "Daily observations of the four airport weather stations - Kigali (Kanombe, Kicukiro), Kamembe (Rusizi), Gisenyi "
     "(Rubavu) and Ruhengeri (Musanze) - published by NOAA's Global Summary of the Day for 2015-2025: temperature, dew "
     "point, min/max temperature, wind, visibility, precipitation and weather events (rain, thunder, fog). The data has "
     "real quality problems: 'not measured' codes, empty columns, imperial units, incomplete rain totals and missing days."),
    ("What the manager asks us to deliver",
     "Do not build the model yet. Set up a reproducible environment; collect the data from a legitimate source; find and "
     "rate every quality problem; clean the data without letting information from the future leak into training; create "
     "features from today's and recent weather; deliver train/validation/test files split by time, a reusable pipeline, "
     "and documentation that the Disaster Management Committees can trust."),
]


def build_brief(h, N):
    d = h.base_doc()
    h.para(d, "Question 2 - Scenario Brief", bold=True, size=16, color=h.BLUE)
    h.para(d, "Next-day rain early warning for MINEMA district disaster offices", italic=True, size=12)
    h.para(d, "Group XX - IGIHOZO Bienvenue, MUGISHA Aime Patrick", size=10)
    for title, text in BRIEF:
        p = d.add_paragraph()
        p.add_run(title + ". ").bold = True
        p.add_run(text)
    d.add_page_break()
    h.para(d, "Appendix A - Question 2 Scenario Proposal Form", bold=True, size=14, color=h.BLUE)
    h.table(d, pd.DataFrame([
        ("Group number and members", "Group XX - IGIHOZO Bienvenue, MUGISHA Aime Patrick"),
        ("Scenario title", "Next-day rain early warning for MINEMA district disaster offices"),
        ("Location in Rwanda (district / sector)", "Rubavu (Gisenyi), Musanze (Ruhengeri), Rusizi (Kamembe), Kicukiro - City of Kigali (Kanombe)"),
        ("Stakeholder (who asks for the work)", "MINEMA - Early Warning unit, with District Disaster Management Committees"),
        ("Scenario story and decision to support",
         "Floods and landslides after heavy or repeated rain kill people every rainy season (May 2023: more than 130 deaths "
         "in Western and Northern provinces). Each evening MINEMA must decide whether to send SMS alerts and put district "
         "teams on standby for the next day. A model that predicts rain tomorrow from today's station observations would "
         "make this decision faster and more consistent. Our task is to prepare the data the model needs. The model must "
         "only use information available today, and must be tested on later years than it was trained on."),
        ("Who will benefit from a solution?",
         "Households in high-risk zones (earlier warnings), district disaster committees (better planning of teams), "
         "MINEMA (fewer false alarms, more trust in alerts), farmers (planning field work)."),
        ("Target variable", "rain_tomorrow - rain or drizzle observed at the station on the next calendar day (1/0)"),
        ("Problem type", "Binary classification"),
        ("Expected data source and link",
         "NOAA NCEI Global Summary of the Day - https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/ "
         "(stations 64387, 64380, 64381, 64383); alternative compared: Open-Meteo historical API (rejected: modelled, too clean)"),
        ("Expected number of rows and columns", f"about {N['raw_rows']:,} station-days, 28 columns ({N['n_files']} yearly files)"),
        ("Known data risks",
         "No personal data. Licence: free, WMO Resolution 40 (education/research). Missing values coded as 9999.9; empty "
         "columns; incomplete rain totals; missing days and a missing year (2023) for two stations; stations at airports "
         "not on landslide slopes (bias)."),
        ("Approval", "Approved / Needs changes     Signature: ______________     Date: __________"),
    ], columns=["Field", "Group entry"]), widths=[5, 11.5], font=9.5)
    out = DOCS / "Q2_Scenario_Brief_and_Proposal.docx"
    d.save(out)
    return out


def data_card(N):
    s = N["splits"]
    return f"""# Data Card - Next-day rain early warning (Question 2)
## Purpose
Model-ready data to predict whether it will rain tomorrow at four Rwandan weather stations, supporting MINEMA's next-day flood/landslide alerts.
## Source
NOAA NCEI Global Summary of the Day (stations Kigali 64387, Kamembe 64380, Gisenyi 64381, Ruhengeri 64383), 2015-2025, downloaded 11 Nov 2026. Free for education/research (WMO Resolution 40).
## Size
Raw: {N['raw_rows']:,} station-days in {N['n_files']} files. Clean: {len(N['clean']):,} rows. Train (2015-2022) / validation (2023) / test (2024-2025): {len(s['train'])} / {len(s['val'])} / {len(s['test'])} rows, {N['n_feat']} encoded inputs.
## Target
rain_tomorrow - rain observed at the station the next calendar day (1/0); {N['rain_share']:.1%} positive.
## Cleaning and feature steps
- Padding removed, text converted to numbers/dates; NOAA missing codes (9999.9, 999.9, 99.99) set to NaN.
- Columns more than 50 % empty dropped (SLP, SNDP, GUST, STP); imperial units converted to metric.
- FRSHTT decoded to fog/rain/hail/thunder; incomplete precipitation flagged; unrealistic winds removed.
- Target built only where the next day exists; features use only today and past days (lags only over consecutive days).
- 11 engineered features (season, circular day of year, humidity, dew-point spread, diurnal range, rain yesterday, rain days in last 3, temperature change, log rain).
- Chronological split; median imputation, scaling and one-hot encoding fitted on 2015-2022 only.
## Known limitations and bias risks
- Only 4 stations at airports (valleys) - not on steep slopes where landslides start; not every district covered.
- Gisenyi and Ruhengeri have many missing days and no 2023 data, so validation contains only Kigali and Kamembe.
- Precipitation totals are often incomplete; the rain flag does not tell how heavy the rain was.
## How to reproduce
conda env create -f environment.yml; conda activate ml-project; run q2_own_scenario/notebooks 01 -> 04 in order.
"""


def add_q2_section(d, h):
    """Writes Question 2 (Stages 0-6) into the group report document d."""
    N = numbers()
    d.add_heading("Question 2 - Next-day rain early warning for MINEMA district disaster offices", 1)
    d.add_heading("Stage 0 - Scenario and proposal", 2)
    for title, text in BRIEF:
        p = d.add_paragraph()
        p.add_run(title + ". ").bold = True
        p.add_run(text)
    h.para(d, "The signed Scenario Proposal Form is in Q2_Scenario_Brief_and_Proposal.docx (Appendix A of the task).", italic=True)

    d.add_heading("Stage 1 - Environment (reused from Question 1)", 2)
    h.para(d, "The same ml-project environment and repository are used; Q2 has its own folder q2_own_scenario with the "
              "same structure. One extra step: the data was downloaded with PowerShell (Invoke-WebRequest).")

    d.add_heading("Stage 2 - Requirements and data acquisition (PC 1.2)", 2)
    h.table(d, pd.DataFrame([
        ("Target", "rain_tomorrow (1/0) - rain flag of the next calendar day"),
        ("Features", "today's temperature, dew point/humidity, min/max, wind, visibility, rain amount, weather events, season, station"),
        ("Excluded", "anything measured tomorrow; no random split across time"),
        ("Period / area", "2015-2025; Kigali, Rusizi, Rubavu, Musanze stations"),
        ("Size", f"{N['raw_rows']:,} raw rows, 28 columns - meets >= 1,000 rows and >= 8 columns"),
    ], columns=["Requirement", "Decision"]), widths=[3.5, 12.5])
    h.para(d, "Two sources were compared: NOAA GSOD (real station observations, chosen) and the Open-Meteo historical API "
              "(modelled reanalysis values, complete and smooth - rejected because it is not an observation and is too "
              "clean for this task). Meteo Rwanda's own records would be best but need a formal request.")
    h.screenshot(d, "Q2 data requirements and source comparison", "Notebook q2 01_acquire sections 2.1-2.2.",
                 "Notebook q2 01_acquire in Jupyter: scenario summary, requirements, comparison of two sources, the 42 loaded files, the requirement check, data dictionary and source log. Decision: NOAA GSOD chosen (real observations).")
    h.screenshot(d, "Q2 raw files and loaded data", "File Explorer of q2_own_scenario/data/raw/gsod + notebook output "
                 "of raw.head() and raw.info().", f"{N['n_files']} yearly files, {N['raw_rows']:,} rows; every column is text.")

    d.add_heading("Stage 3 - Explore and audit (PC 1.3)", 2)
    h.figure(d, "q2_fig3_1_missing.png", "Share of 'not measured' codes per column",
             "SLP and SNDP are always missing, GUST and STP almost always; PRCP and MXSPD partly.", folder=FIG)
    h.figure(d, "q2_fig3_2_coverage.png", "Days recorded per station and year",
             "Kigali is almost complete; Gisenyi and Ruhengeri have large gaps and no 2023 data.", folder=FIG)
    h.figure(d, "q2_fig3_3_boxplots.png", "Box plots of the measured columns",
             "Most extreme values are real weather; two wind values are not credible. Rain amount is very skewed.", folder=FIG)
    h.figure(d, "q2_fig3_4_target.png", "Target balance and rainy days by month",
             f"{N['rain_share']:.0%} of days are followed by rain; rain follows the long rains (Mar-May) and short rains (Sep-Dec).",
             folder=FIG)
    h.figure(d, "q2_fig3_5_humidity.png", "Today's humidity vs rain tomorrow",
             "Days followed by rain have higher humidity today at every station.", folder=FIG)
    h.para(d, "Limitation found: the rain flag and the rain amount disagree on many days (different recording windows and "
              "incomplete totals). The target uses the rain flag, which is recorded every day.", italic=True)
    h.figure(d, "q2_fig3_6_corr.png", "Correlation heatmap",
             "Rain today and dew point relate most to rain tomorrow; temperature columns are redundant with each other.", folder=FIG)
    h.screenshot(d, "Q2 missing values, duplicates and summary statistics", "Notebook q2 02_eda sections 3.1-3.2.",
                 "Notebook q2 02_eda in Jupyter: summary statistics, missing codes per column, missing days per station-year, box plots and IQR, the rain flag vs amount check, target balance and the Data Quality Report (13 problems).")
    d.add_heading("Data Quality Report (Question 2)", 3)
    h.table(d, N["dqr"].rename(columns={"records_affected": "records"}), widths=[0.8, 5, 3.2, 1.4, 1.5, 5], font=8)

    d.add_heading("Stage 4 - Clean and pre-process (PC 1.3)", 2)
    h.para(d, f"All steps are functions of src/rain_pipeline.py (clean_rain runs them in order); {len(N['clog'])} decisions "
              f"are logged (Appendix Q2-C). Clean data: {len(N['clean']):,} rows.")
    h.table(d, pd.DataFrame([
        ("Missing codes", "9999.9 / 999.9 / 99.99 -> NaN; then median (numbers) or most frequent (flags) imputation, learned on train"),
        ("Empty columns", "SLP, SNDP, GUST, STP dropped (> 50 % missing)"),
        ("Units", "F -> C, knots -> km/h, miles -> km, inches -> mm"),
        ("Coded columns", "FRSHTT -> fog, rain, hail, thunder; PRCP flag I -> prcp_incomplete"),
        ("Outliers", "real heavy-rain/windy days kept; 2 impossible wind values set to NaN"),
        ("Target", "rain_tomorrow from the next day's rain flag, only when the next calendar day exists"),
        ("Encoding / scaling", "station and season one-hot; numeric columns standardised; 0/1 flags unchanged"),
        ("Split", "chronological: train 2015-2022, validation 2023, test 2024-2025 (no future data in training)"),
        ("Imbalance", f"{N['rain_share']:.0%} rain days - moderate; planned remedy class_weight='balanced'"),
    ], columns=["Problem", "Treatment"]), widths=[3.5, 12.5])
    h.table(d, pd.DataFrame([(n, len(s), s.shape[1]) for n, s in N["splits"].items()], columns=["split", "rows", "columns"]),
            widths=[4, 4, 4])
    h.screenshot(d, "Q2 cleaning before/after and split shapes", "Notebook q2 03_clean: steps 2, 3, 8-9 and the split cell.",
                 "Notebook q2 03_clean in Jupyter: missing codes replaced, empty columns dropped, metric units, decoded flags, the rain_tomorrow target, the reproducibility check, the cleaning log and the chronological split.")

    d.add_heading("Stage 5 - Features (PC 1.4)", 2)
    h.figure(d, "q2_fig5_1_log_prcp.png", "Log transform of the rain amount", "log(1 + mm) reduces the strong skew.", folder=FIG)
    h.figure(d, "q2_fig5_2_rf_importance.png", "Random Forest importance on the training years",
             "The time of year (doy_cos/doy_sin) and humidity carry the most information; recent wet days and rain today "
             "follow. dewpoint_spread_c was removed because it duplicates humidity (r = 0.99).", folder=FIG)
    h.para(d, f"Final files contain {N['n_feat']} encoded inputs plus station_id, DATE and the target. The Feature "
              "Justification table is in Appendix Q2-D.")
    h.screenshot(d, "Q2 new features and final list", "Notebook q2 04_features: first rows with new columns + 'selected inputs'.",
                 "Notebook q2 04_features in Jupyter: the new weather features, log transform, importance ranking, selection with reasons and the saved model-ready files.")

    d.add_heading("Stage 6 - Final product", 2)
    h.bullets(d, ["data/processed/rain_train.csv, rain_val.csv, rain_test.csv + *_readable.csv + preprocessor.joblib.",
                  "Restart & Run All of notebooks 01-04 gives the same files. Data card: Appendix Q2-E."])
    h.screenshot(d, "Q2 processed files and final info", "File Explorer of q2_own_scenario/data/processed + final.info().",
                 "All inputs numeric with no missing values.")

    d.add_heading("Appendix Q2-A - Data dictionary", 2)
    h.table(d, N["ddict"], widths=[3.2, 4.3, 2, 2, 5], font=8)
    d.add_heading("Appendix Q2-B - Source log", 2)
    h.table(d, N["slog"].T.reset_index().rename(columns={"index": "field", 0: "entry"}), widths=[3.5, 12.5], font=8.5)
    d.add_heading("Appendix Q2-C - Cleaning log", 2)
    h.table(d, N["clog"][["step", "column", "problem", "action", "records_affected", "reason"]].rename(
        columns={"records_affected": "records"}), widths=[0.9, 2.8, 3.8, 3.0, 1.3, 4.5], font=7.5)
    d.add_heading("Appendix Q2-D - Feature justification", 2)
    h.table(d, N["fj"], widths=[3.6, 4.6, 6, 1.6], font=8)
    d.add_heading("Appendix Q2-E - Data card", 2)
    for line in data_card(N).splitlines():
        if line.startswith("# ") or not line.strip():
            continue
        if line.startswith("## "):
            d.add_heading(line[3:], 3)
        elif line.startswith("- "):
            d.add_paragraph(line[2:], style="List Bullet")
        else:
            d.add_paragraph(line)


def build_explained(h):
    d = h.base_doc(landscape=True)
    h.para(d, "Question 2 - Code explained line by line", bold=True, size=18, color=h.BLUE)
    h.para(d, "Group XX - IGIHOZO Bienvenue, MUGISHA Aime Patrick")
    d.add_heading("Key ideas for the oral questions", 2)
    h.bullets(d, [
        "Disguised missing values: NOAA writes 9999.9 / 999.9 / 99.99 when nothing was measured.",
        "Why the target comes from the rain flag: precipitation totals are often incomplete (flag I).",
        "Chronological split: in a time series a random split leaks the future into training.",
        "Lag features (rain yesterday, last 3 days) only use the past and only consecutive days.",
        "Fit imputers/scalers on 2015-2022 only, then transform 2023 and 2024-2025 unchanged.",
        "Relative humidity (Magnus formula): how close the air is to saturation - moist air before rain.",
        "Circular day of year (sin/cos): 31 December and 1 January are next to each other.",
    ])
    for nb_name in ["01_acquire", "02_eda", "03_clean", "04_features"]:
        book = json.loads((NB / f"{nb_name}.ipynb").read_text(encoding="utf-8"))
        d.add_page_break()
        d.add_heading(f"Notebook {nb_name}.ipynb", 1)
        k = 0
        for cell in book["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                first = next((l for l in src.splitlines() if l.startswith("#")), None)
                if first and not first.startswith("# Group"):
                    d.add_heading(first.lstrip("# ").strip(), 3)
                continue
            k += 1
            h.para(d, f"Code cell {k}", bold=True, size=9, color=h.BLUE)
            h.code_table(d, h.explain_lines(src))
    d.add_page_break()
    d.add_heading("Module src/rain_pipeline.py", 1)
    src = (Q2 / "src" / "rain_pipeline.py").read_text(encoding="utf-8")
    src = re.sub(r'"""(.*?)"""', lambda m: "#DOC# " + " ".join(m.group(1).split()), src, flags=re.S)
    h.code_table(d, h.explain_lines(src))
    out = DOCS / "Q2_Code_Explained_Line_by_Line.docx"
    d.save(out)
    return out


if __name__ == "__main__":
    h = load_h()
    N = numbers()
    (DOCS / "data_card.md").write_text(data_card(N), encoding="utf-8")
    for f in (build_brief(h, N), build_explained(h)):
        print("wrote", f)
