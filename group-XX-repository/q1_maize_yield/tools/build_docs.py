"""
build_docs.py - builds the Word documents for Question 1 from the outputs of the notebooks.
Run AFTER notebooks 00-04 have been executed:   python tools/build_docs.py

Creates in docs/:
  GroupXX_ML_Element1_Documentation.docx  - report organised by stage, with figures, tables and screenshot placeholders
  Q1_Code_Explained_Line_by_Line.docx     - every line of code with its explanation
  Q1_Memo_to_Mr_Habimana.docx             - one-page memo in simple language
  data_card.md                            - one-page data card
"""
import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS, FIG, NB, PROC, INTERIM = ROOT / "docs", ROOT / "docs" / "figures", ROOT / "notebooks", ROOT / "data" / "processed", ROOT / "data" / "interim"
BLUE = RGBColor(0x11, 0x6F, 0xBF)

# ------------------------------------------------------------------ numbers from the pipeline outputs
raw = pd.read_csv(ROOT / "data" / "raw" / "maize_yield_rwanda_rw.csv")
clean = pd.read_csv(INTERIM / "maize_clean.csv")
dqr = pd.read_csv(DOCS / "data_quality_report.csv")
clog = pd.read_csv(DOCS / "cleaning_log.csv")
fj = pd.read_csv(DOCS / "feature_justification.csv")
ddict = pd.read_csv(DOCS / "data_dictionary.csv")
slog = pd.read_csv(DOCS / "source_log.csv")
splits = {n: pd.read_csv(PROC / f"maize_{n}.csv") for n in ["train", "val", "test"]}
readable = {n: pd.read_csv(PROC / f"maize_{n}_readable.csv") for n in ["train", "val", "test"]}
n_feat = splits["train"].shape[1] - 4  # minus record_id and the three target columns
low_share = pd.concat(splits.values())["low_yield_risk"].mean()
low_thr = readable["train"].loc[readable["train"].low_yield_risk == 0, "yield_t_ha"].min()  # first yield not flagged
n_gap_farms = int(raw[["rainfall_mm", "soil_ph", "pest_incidence_index"]].isna().any(axis=1).sum())
seed_means = raw.groupby("seed_variety")["yield_t_ha"].mean().round(2)
rec = lambda op, col=None: int(clog[(clog.operation == op) & ((clog.column == col) if col else True)].records_affected.sum())


# ------------------------------------------------------------------ helpers
def base_doc(landscape=False):
    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2)
        s.top_margin = s.bottom_margin = Cm(1.8)
        if landscape:
            s.orientation = WD_ORIENT.LANDSCAPE
            s.page_width, s.page_height = s.page_height, s.page_width
    for lvl in (1, 2, 3):
        h = d.styles[f"Heading {lvl}"]
        h.font.color.rgb = BLUE
        h.font.name = "Calibri"
    return d


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"), shd.set(qn("w:color"), "auto"), shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(d, df, widths=None, font=8.5, header_fill="116FBF"):
    t = d.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        r.bold, r.font.size, r.font.color.rgb = True, Pt(font), RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, header_fill)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if pd.isna(v) else str(v))
            r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    d.add_paragraph()
    return t


def para(d, text, bold=False, italic=False, size=None, align=None, color=None):
    p = d.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    return p


def bullets(d, items):
    for it in items:
        d.add_paragraph(it, style="List Bullet")


FIG_NO = [0]


def figure(d, png, caption, explanation, folder=None):
    FIG_NO[0] += 1
    folder = folder or FIG
    if (folder / png).exists():
        d.add_picture(str(folder / png), width=Cm(16))
        d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(d, f"Figure {FIG_NO[0]}: {caption}", bold=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(d, explanation, size=9.5)


SHOT_DIR = ROOT.parent / "screenshots"
SHOT_MAP = json.loads((SHOT_DIR / "shot_map.json").read_text(encoding="utf-8")) if (SHOT_DIR / "shot_map.json").exists() else {}


def screenshot(d, caption, what_to_capture, explanation):
    """Real screenshot(s) taken with setup/take_screenshots.ps1 (mapped in screenshots/shot_map.json);
    a yellow placeholder box only if no screenshot was mapped for this caption."""
    patterns = SHOT_MAP.get(caption, [])
    if patterns == ["skip"]:  # covered by the notebook pages shown in another figure of the same stage
        return
    files = sorted({f for p in patterns for f in SHOT_DIR.glob(p)})
    if files:
        from PIL import Image  # Pillow comes with matplotlib
        jpg_dir = SHOT_DIR / "_report_jpg"
        jpg_dir.mkdir(exist_ok=True)
        FIG_NO[0] += 1
        para(d, f"Figure {FIG_NO[0]}: {caption} (screenshots, {len(files)} page(s))", bold=True, size=9)
        para(d, explanation, size=9.5)
        for k, f in enumerate(files, 1):
            jpg = jpg_dir / (f.stem + ".jpg")  # compressed copy for the report; the PNG stays unchanged
            if not jpg.exists():
                im = Image.open(f).convert("RGB")
                im.thumbnail((1500, 1500))
                im.save(jpg, quality=72)
            d.add_picture(str(jpg), width=Cm(16.5))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if len(files) > 1:
                para(d, f"Figure {FIG_NO[0]}.{k} - {f.stem.replace('_', ' ')}", italic=True, size=8,
                     align=WD_ALIGN_PARAGRAPH.CENTER)
        return
    FIG_NO[0] += 1
    t = d.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.rows[0].cells[0]
    shade(c, "FFF2CC")
    c.text = ""
    r = c.paragraphs[0].add_run(f"PASTE SCREENSHOT HERE  (Win + Shift + S)\n{what_to_capture}")
    r.italic, r.font.size = True, Pt(9)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(d, f"Figure {FIG_NO[0]}: {caption}", bold=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(d, explanation, size=9.5)


# ================================================================== 1. DOCUMENTATION REPORT
def build_report():
    d = base_doc()
    # ---- cover
    for _ in range(4):
        d.add_paragraph()
    para(d, "Rwanda Polytechnic - Kigali College", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(d, "Department of Information and Communication Technology", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    d.add_paragraph()
    para(d, "PROJECT-BASED PRACTICAL TASK", bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLUE)
    para(d, "Two Rwandan Scenarios: Build Machine-Learning-Ready Datasets", italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(d, "Competence 10: Apply Machine Learning - Element 1: Apply dataset preparation mechanisms", size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    d.add_paragraph()
    table(d, pd.DataFrame([
        ("Group", "Group XX"),
        ("Members", "IGIHOZO Bienvenue - MUGISHA Aime Patrick"),
        ("Question 1", "Predicting Maize Yield for Farmer Cooperatives in Rwanda (MCSU)"),
        ("Question 2", "Next-day rain early warning for MINEMA district disaster offices (NOAA station data 2015-2025)"),
        ("Repository", "[https://github.com/<account>/group-XX-repository]"),
        ("Academic year", "2026/2027"),
        ("Presentation", "Thursday 12 November 2026 (morning)"),
    ], columns=["Item", "Details"]), widths=[4, 12], font=10)
    d.add_page_break()

    # ---- contents
    d.add_heading("Contents", 1)
    for line in ["Question 1 - Predicting maize yield", "   Stage 1 - Configure the environment",
                 "   Stage 2 - Define requirements and acquire data", "   Stage 3 - Explore and audit the data (EDA)",
                 "   Stage 4 - Clean and pre-process the data", "   Stage 5 - Engineer and select features",
                 "   Stage 6 - Finalise, package and present", "Question 2 - Own scenario (Stages 0-6)",
                 "Limitations and lessons learned",
                 "Appendices: A Data dictionary, B Source log, C Cleaning log, D Feature table, E Data card, F Contribution log"]:
        d.add_paragraph(line)
    d.add_page_break()

    # ---- Q1 intro
    d.add_heading("Question 1 - Predicting Maize Yield for Farmer Cooperatives in Rwanda", 1)
    para(d, "Problem. The Maize Cooperative Support Unit (MCSU) wants a reliable maize yield forecast before every harvest. "
            "The data science team will build the model; our task was to make the data ready for modelling and prove that "
            "it is ready - without building a model.")
    para(d, f"Data. File maize_yield_rwanda_rw.csv supplied by the lecturer: {raw.shape[0]:,} farm-season records, "
            f"{raw.shape[1]} columns, {raw['year'].min()}-{raw['year'].max()}, 5 provinces and 30 districts. Its columns "
            "differ from the description in the brief (yield in t/ha, fertiliser per hectare, no farm/cooperative code, no "
            "planting date, no harvest weight); every step below was designed for the file actually received.", italic=True)

    # ---- Stage 1
    d.add_heading("Stage 1 - Configure the environment (PC 1.1)", 2)
    para(d, "What we did:")
    bullets(d, [
        "Installed Anaconda 2026.07 (Python distribution with conda, Navigator and Jupyter).",
        "Created a dedicated environment 'ml-project' from environment.yml (conda env create -f environment.yml) with "
        "Python 3.14, pandas, numpy, matplotlib, seaborn, scikit-learn and Jupyter; activated it with conda activate ml-project.",
        "environment.yml pins the tested versions so anyone can rebuild the same environment.",
        "Created the Git repository with the required folder structure (data/raw, data/interim, data/processed, "
        "notebooks, docs) and a README that explains why each tool was chosen.",
        "Ran notebook 00_environment_check, which imports every library and prints its version.",
        "Issue met and solved: Windows Smart App Control blocked a matplotlib DLL from the community conda-forge channel. "
        "We rebuilt the environment from Anaconda's own 'defaults' channel (signed files) after accepting its Terms of Service.",
    ])
    screenshot(d, "Anaconda Navigator with Jupyter Notebook", "Anaconda Navigator, 'ml-project' selected, Jupyter tile visible.",
               "Anaconda Navigator is installed and Jupyter is available in the ml-project environment.")
    screenshot(d, "Anaconda Prompt - active environment and list of environments",
               "Anaconda Prompt after: conda activate ml-project  and  conda env list",
               "The asterisk (*) marks ml-project as the active environment.")
    screenshot(d, "Library versions printed by 00_environment_check", "Notebook 00_environment_check output.",
               "All five required libraries import without error; versions match environment.yml.")

    # ---- Stage 2
    d.add_heading("Stage 2 - Define requirements and acquire data (PC 1.2)", 2)
    para(d, "Data requirements, derived from Mr. Habimana's message and the five business questions:")
    table(d, pd.DataFrame([
        ("Target", "yield_t_ha (regression, also given as yield_kg_ha); low_yield_risk = bottom 10 % of training yields (question E)"),
        ("Unit of observation", "one farm in one season"),
        ("Features", "province, district, altitude, season, year, farm area, seed variety, farming practice, NPK, urea, manure "
                     "(per ha), plant density, rainfall, temperature, soil type and pH, pest index, irrigation"),
        ("Must not be used", "anything known only after harvest; statistics from validation/test data"),
        ("Size", ">= ~1,000 usable records"),
    ], columns=["Requirement", "Decision"]), widths=[4, 12])
    para(d, f"Check: {raw.shape[0]:,} rows, all three seasons, 5 provinces, 30 districts, {raw['year'].min()}-{raw['year'].max()}, "
            "fertiliser already per hectare, no post-harvest column. The file meets the requirements for questions A-E. "
            "It cannot support analysis per cooperative or by planting date (those columns do not exist).")
    screenshot(d, "Data requirements Markdown cell (01_acquire)", "Notebook 01_acquire sections 2.1-2.2.",
               "Notebook 01_acquire in Jupyter: the data requirements, the loaded file (first/last rows, shape, info), the check against the requirements, the data dictionary and the source log. Finding: 1,426 x 20, types already correct, three columns with gaps. Decision: the file meets the requirements for questions A-E.")
    screenshot(d, "First rows, shape and info of the loaded data", "Notebook 01_acquire: raw.head(), shape and raw.info().",
               f"{raw.shape[0]:,} x {raw.shape[1]} loaded; types are already correct; three columns have empty cells.")
    para(d, "The completed data dictionary is in Appendix A and the source log in Appendix B.")

    # ---- Stage 3
    d.add_heading("Stage 3 - Explore and audit the data (PC 1.3)", 2)
    para(d, f"We described the data, counted missing values and duplicates, checked every label against an allowed list and every "
            f"district against its official province, compared every number with a realistic range, detected outliers with "
            f"box plots and the IQR rule, studied the target and looked for leakage. The file is much cleaner than the brief "
            f"suggests: no duplicates, no wrong labels and no impossible values. The Data Quality Report lists the "
            f"{len(dqr) - 1} real issues found and the checks that found nothing.")
    figure(d, "fig3_1_missing.png", "Where the missing values are",
           f"Only 39 cells are empty, but they sit together in {n_gap_farms} farms (no soil test, no pest scouting, no rain "
           "record). The farms are kept and the values imputed from the training set.")
    figure(d, "fig3_2_boxplots.png", "Box plots of the numeric columns",
           "The IQR rule flags 112 values (mostly large farm areas) but none is outside a realistic range - they are real farms "
           "and are kept. Farm area is right-skewed.")
    figure(d, "fig3_3_target.png", "Distribution of the target and yield by season",
           f"Yield is almost symmetric around 2.6 t/ha (1.0-5.2). No farm is below 1,000 kg/ha, so 'very low harvest' is "
           f"defined as the bottom 10 % of training farms ({low_thr:.2f} t/ha).")
    figure(d, "fig3_4_seed_practice.png", "Yield by seed variety and farming practice",
           f"Local seed averages {seed_means['Local']} t/ha against {seed_means['PAN691']} t/ha for PAN691; improved practice "
           "yields more - question B can be answered.")
    figure(d, "fig3_5_drivers.png", "Rainfall, NPK and plant density against yield",
           "All three raise yield but with wide scatter - yield depends on several factors together.")
    figure(d, "fig3_6_corr.png", "Correlation heatmap",
           "Altitude and temperature are strongly linked (r = -0.88). No column is suspiciously close to the target, so there "
           "is no hidden post-harvest column. Season rainfall and temperature are only fully known at season end - a timing "
           "risk for early warning (question E).")
    figure(d, "fig3_7_province_season.png", "Yield by province and season",
           "Differences between provinces and seasons are visible - question D can be answered.")
    screenshot(d, "Missing values, duplicate count and summary statistics", "Notebook 02_eda sections 3.1-3.3.",
               "Notebook 02_eda in Jupyter: summary statistics, missing values and duplicates, consistency checks, box plots and IQR table, target and relationship charts, and the Data Quality Report. Finding: 39 empty cells in 14 farms, no duplicates, no invalid labels or values; outliers are real farms. Decision: keep all rows, impute on train only.")
    d.add_heading("Data Quality Report", 3)
    table(d, dqr.rename(columns={"records_affected": "records"}), widths=[0.8, 5, 3.2, 1.4, 1.5, 5], font=8)

    # ---- Stage 4
    d.add_heading("Stage 4 - Clean and pre-process the data (PC 1.3)", 2)
    para(d, f"All steps are reusable functions in src/maize_pipeline.py (clean_maize runs them in order); the raw file is "
            f"never edited. The cleaning log has {len(clog)} entries, including checks that changed nothing. The clean dataset "
            f"keeps all {clean.shape[0]:,} rows.")
    table(d, pd.DataFrame([
        ("Duplicates", "checked by record_id and by content - none found"),
        ("Categories", "all labels valid; district-province pairs valid; 'Kigali' renamed to the official 'Kigali City' (132 rows)"),
        ("Ranges", "every number checked against a realistic range - none outside"),
        ("Missing values", "rows kept; numeric median imputation learned on the training set only (inside the Pipeline)"),
        ("Outliers", "IQR outliers kept (real farms); farm area log-transformed in Stage 5"),
        ("Target", "all present and realistic; yield_kg_ha added (= t/ha x 1,000)"),
        ("Encoding", "season, district, soil type, seed variety/class, practice, altitude zone one-hot encoded"),
        ("Scaling", "StandardScaler on numeric columns (mean 0, std 1) - fitted on train only"),
        ("Split", "70 / 15 / 15 train / validation / test, random_state = 42, stratified by season (keeps Season C in all files)"),
        ("Imbalance", f"low_yield_risk about 10 % positives - planned remedy class_weight='balanced'"),
    ], columns=["Problem", "Treatment"]), widths=[3.5, 12.5])
    table(d, pd.DataFrame([(n, s.shape[0], s.shape[1]) for n, s in splits.items()], columns=["split", "rows", "columns"]),
          widths=[4, 4, 4])
    screenshot(d, "Cleaning steps with before/after views", "Notebook 03_clean steps 3-6.",
               "Notebook 03_clean in Jupyter: every cleaning step with before/after views, the reproducibility check, the cleaning log, the 70/15/15 split stratified by season, and the imputation/encoding/scaling pipeline fitted on train only.")
    screenshot(d, "Encoding and scaling output, split shapes", "Notebook 03_clean last cells.",
               "The pipeline is fitted on the training set only and applied unchanged to validation and test.")
    para(d, "The full cleaning log is in Appendix C.")

    # ---- Stage 5
    d.add_heading("Stage 5 - Engineer and select features (PC 1.4)", 2)
    para(d, "Ten new features were created with add_features() using only values of the same row (no leakage between splits). "
            "Relationships were measured on the training set only (correlation, mutual information, Random Forest importance). "
            "Selection used a variance threshold, removal of duplicated information (|r| > 0.9, province vs district) and the "
            "model-based evidence.")
    figure(d, "fig5_1_log_area.png", "Log transformation of farm area", "log(1 + area) halves the skewness.")
    figure(d, "fig5_2_feature_corr.png", "Correlation of candidate features (training set)",
           "The improved seed + practice package (r = 0.46) and rainfall (0.39) relate most to yield; urea and nitrogen "
           "applied carry the same information (r = 0.95).")
    figure(d, "fig5_3_rf_importance.png", "Random Forest importance (training set)",
           "Seed variety, rainfall, season, pest pressure and plant density carry most of the information.")
    para(d, f"Final model-ready files contain {n_feat} encoded input columns plus record_id and the targets yield_t_ha, "
            "yield_kg_ha and low_yield_risk. The Feature Justification table is in Appendix D.")
    screenshot(d, "New features, importance output and final feature list", "Notebook 04_features sections 5.0, 5.2, 5.3.",
               "Notebook 04_features in Jupyter: the ten new features, the log transform, correlation, mutual information and Random Forest importance on the training set, the selection with reasons, the saved model-ready files and the final dataset info (no missing values).")

    # ---- Stage 6
    d.add_heading("Stage 6 - Finalise, package and present", 2)
    bullets(d, [
        "data/processed/maize_train.csv, maize_val.csv, maize_test.csv - model-ready (encoded, scaled, no missing values).",
        "data/processed/*_readable.csv - the same rows un-scaled; preprocessor.joblib - the fitted pipeline.",
        "Reproducibility: Kernel > Restart & Run All on notebooks 00-04 finishes without errors and gives the same files.",
        "Memo to Mr. Habimana: Q1_Memo_to_Mr_Habimana.docx. Data card: Appendix E.",
    ])
    screenshot(d, "Restart & Run All finished without errors", "A notebook after Kernel > Restart & Run All.",
               "All cells ran in order without errors.")
    screenshot(d, "Saved final files in the project folder", "File Explorer: q1_maize_yield/data/processed.",
               "Train, validation, test files and the fitted preprocessor are saved.")
    screenshot(d, "Final dataset info", "Notebook 04_features section 5.6 (final_train.info()).",
               "All inputs numeric with no missing values - ready for Element 2.")
    screenshot(d, "Repository page", "GitHub page showing folders and commits by both members.",
               "Commits from both members are visible.")
    d.add_page_break()

    # ---- Q2 (built by q2_own_scenario/tools/build_docs.py)
    q2_builder = ROOT.parent / "q2_own_scenario" / "tools" / "build_docs.py"
    if (ROOT.parent / "q2_own_scenario" / "data" / "processed" / "rain_train.csv").exists():
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("q2docs", q2_builder)
        q2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q2)
        q2.add_q2_section(d, sys.modules[__name__])
    else:
        d.add_heading("Question 2 - Own Rwandan scenario", 1)
        para(d, "[Run the Q2 notebooks first.]", italic=True)
    d.add_page_break()

    # ---- limitations
    d.add_heading("Limitations and lessons learned", 1)
    bullets(d, [
        "Q1: the supplied file differs from the brief (t/ha, fertiliser per ha, no cooperative code or planting date).",
        "Q1: season rainfall/temperature are known only at season end - early warning needs data-to-date.",
        "Q1: missing values were imputed with simple statistics; Season C and irrigated farms are few.",
        "Lesson: a clean-looking file still needs documented checks (duplicates, labels, ranges, district-province) - "
        "showing that a check found nothing is part of the proof.",
        "Lesson: split before fitting imputers/scalers, and think about WHEN each value is known, to avoid leakage.",
        "Q2: real data is messy in different ways than the case study - missing codes, empty columns, missing days, and "
        "two fields (rain flag and rain amount) that disagree. Reading the data provider's documentation was essential.",
        "Q2: for a time series the split must follow time; lag features must only use consecutive past days.",
        "Q2 limitation: only 4 airport stations, unequal coverage, and no 2023 data for Gisenyi and Ruhengeri.",
    ])
    d.add_page_break()

    # ---- appendices
    d.add_heading("Appendix A - Data dictionary", 1)
    table(d, ddict, widths=[3.2, 4.3, 2, 2, 5], font=8)
    d.add_heading("Appendix B - Source log", 1)
    table(d, slog.T.reset_index().rename(columns={"index": "field", 0: "entry"}), widths=[3.5, 12.5], font=8.5)
    d.add_heading("Appendix C - Cleaning log", 1)
    table(d, clog[["step", "column", "problem", "action", "records_affected", "reason"]].rename(
        columns={"records_affected": "records"}), widths=[0.9, 2.6, 3.8, 3.2, 1.3, 4.5], font=7.5)
    d.add_heading("Appendix D - Feature justification", 1)
    table(d, fj.rename(columns={"how_created_or_reason": "how created / reason", "business_question": "Q"}),
          widths=[3.6, 4.4, 0.9, 5.3, 1.6], font=8)
    d.add_heading("Appendix E - Data card", 1)
    for line in data_card_text().splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            d.add_heading(line[3:], 3)
        elif line.startswith("- "):
            d.add_paragraph(line[2:], style="List Bullet")
        elif line.strip():
            d.add_paragraph(line)
    d.add_heading("Appendix F - Individual contribution log", 1)
    table(d, pd.DataFrame([(s, "", "", "", "") for s in range(1, 7)],
                          columns=["Stage", "Driver", "Navigator", "IGIHOZO Bienvenue contribution", "MUGISHA Aime Patrick contribution"]),
          widths=[1.3, 2.7, 2.7, 4.6, 4.6], font=9)
    para(d, "IGIHOZO Bienvenue signature: ____________________        MUGISHA Aime Patrick signature: ____________________")
    out = ROOT.parent / "GroupXX_ML_Element1_Documentation.docx"  # one report per group, at the repository root
    d.save(out)
    old = DOCS / "GroupXX_ML_Element1_Documentation.docx"  # earlier Q1-only version
    if old.exists():
        old.unlink()
    return out


# ================================================================== 2. DATA CARD
def data_card_text():
    return f"""# Data Card - MCSU maize yield dataset (Question 1)
## Purpose
Model-ready data for forecasting maize yield per farm and flagging farms at risk of a very low harvest (MCSU cooperatives).
## Source
File maize_yield_rwanda_rw.csv supplied by the lecturer (course material), copied unchanged into data/raw/.
## Size
Raw: {raw.shape[0]:,} rows x {raw.shape[1]} columns ({raw['year'].min()}-{raw['year'].max()}, 5 provinces, 30 districts). Clean: {clean.shape[0]:,} rows. Train / validation / test: {splits['train'].shape[0]} / {splits['val'].shape[0]} / {splits['test'].shape[0]} rows, {n_feat} encoded input columns.
## Target
yield_t_ha (regression; also yield_kg_ha). low_yield_risk = 1 if yield < {low_thr:.2f} t/ha (bottom 10 % of training farms; {low_share:.1%} positives).
## Cleaning and feature steps
- Checked duplicates, labels, district-province pairs and realistic ranges (nothing to fix); 'Kigali' renamed 'Kigali City'.
- 39 missing values in 14 farms kept and imputed with training medians.
- 10 engineered features (nitrogen applied, fertiliser-use flag, seed class, improved package, log farm area, temperature deviation, pH flag, irrigation flag, altitude zone, plant density in thousands).
- One-hot encoding and standard scaling fitted on train only; 70/15/15 split stratified by season, seed 42.
## Known limitations and bias risks
- Season rainfall and temperature are only fully known at season end - an early-warning model must use data-to-date.
- Few Season C and irrigated farms; equal records per district may not reflect real maize area.
- No farm/cooperative code or planting date - results cannot be given per cooperative.
## How to reproduce
conda env create -f environment.yml; conda activate ml-project; run notebooks 00 -> 04 in order. Fixed seed 42.
"""


# ================================================================== 3. MEMO
def build_memo():
    d = base_doc()
    for s in d.sections:
        s.top_margin = s.bottom_margin = Cm(1.4)
    para(d, "MEMO", bold=True, size=16, color=BLUE)
    for k, v in [("To", "Mr. Habimana, Head of Planning and Data, MCSU"),
                 ("From", "IGIHOZO Bienvenue and MUGISHA Aime Patrick, Junior Data Analysts (Group XX)"),
                 ("Date", "12 November 2026"), ("Subject", "Maize yield data: ready for modelling?")]:
        p = d.add_paragraph()
        p.add_run(f"{k}: ").bold = True
        p.add_run(v)
        p.paragraph_format.space_after = Pt(0)
    d.add_paragraph()
    sections = [
        ("1. What we received",
         f"The file had {raw.shape[0]:,} farm records and {raw.shape[1]} columns from {raw['year'].min()} to "
         f"{raw['year'].max()}, covering all 5 provinces, 30 districts and the three seasons. All {clean.shape[0]:,} "
         "records can be used - none had to be removed."),
        ("2. Main problems found",
         f"The data is in good condition: no repeated farms, no spelling differences and no impossible values. The three "
         f"issues that matter are: (1) {n_gap_farms} farms have no soil test, pest count and (mostly) rainfall - we kept "
         "them and filled the gaps with typical values, so they are not lost; (2) season rainfall and temperature are only "
         "known at the end of the season, so a forecast made early in the season must use rain measured so far; (3) Season C "
         "(226 farms) and irrigated farms (128) are few, so results for them are less certain."),
        ("3. Data removed or changed",
         "No records were removed. We used the official name 'Kigali City' instead of 'Kigali', added the yield in kg per "
         "hectare next to tonnes per hectare, and filled 39 empty values with typical values learned only from the farms used "
         "for training. No column had to be removed for leakage - the file has no information collected after harvest. "
         "Every check and change is listed in the Cleaning Log."),
        ("4. New features",
         "Nitrogen applied per hectare and a fertiliser-use flag (question B), seed type - local, open-pollinated or hybrid - "
         "and an 'improved seed + improved practice' indicator (B), temperature outside the ideal 18-27 C, an ideal-pH flag, "
         "an irrigation flag and altitude zone (C, D), farm size on a log scale and plants per hectare (A), and a 'very low "
         f"harvest' flag for the lowest 10 % of farms, below {low_thr:.1f} t/ha (E)."),
        ("5. Limits and risks",
         "There is no farm or cooperative code and no planting date, so results cannot be given per cooperative or by planting "
         "time. Each district has about the same number of farms, which may not match where maize is really grown."),
        ("6. Recommendation",
         "The data is ready for the modelling team and supports questions A, B, C and D now; question E (early warning) is "
         "possible if rainfall measured up to the forecast date is used. Next season, please record a cooperative code, the "
         "planting date and rainfall at mid-season, and do a soil test on every farm."),
    ]
    for h, txt in sections:
        p = d.add_paragraph()
        p.add_run(h).bold = True
        p.paragraph_format.space_after = Pt(0)
        q = d.add_paragraph(txt)
        q.paragraph_format.space_after = Pt(4)
    out = DOCS / "Q1_Memo_to_Mr_Habimana.docx"
    d.save(out)
    return out


# ================================================================== 4. LINE-BY-LINE CODE EXPLANATION
def auto_explain(code):
    """Explanation for a line that has no comment of its own (mostly lines inside a multi-line statement)."""
    s = code.strip()
    first_str = re.search(r"""["']([^"']+)["']""", s)
    if s.startswith("def "):
        return f"defines the function {s[4:].split('(')[0]}() so the steps below can be reused by calling its name"
    if s.startswith("class "):
        return f"defines the class {s[6:].split('(')[0].split(':')[0]} (a container for data + functions)"
    if s.startswith("return "):
        return f"gives the result ({s[7:]}) back to the code that called this function"
    if s.startswith("for ") and s.endswith(":"):
        return "repeats the indented lines below once for each item"
    if s.startswith("if ") and s.endswith(":"):
        return "runs the indented lines below only when this condition is true"
    if re.fullmatch(r"[\)\]\}]+[,;]?(\s*,?\s*columns=.*)?", s) or s.startswith("], columns=") or s.startswith("})"):
        return "closes the brackets opened above - end of this multi-line statement" + \
            (" and names the table columns" if "columns=" in s else "")
    if s.startswith('("') and ("Pipeline" in s or "Imputer" in s or "Encoder" in s or "Scaler" in s):
        return f"pipeline part '{first_str.group(1)}': the transformations applied to the listed columns"
    if s.startswith("(") and first_str:
        return f"one row of the table being built: '{first_str.group(1)}' and its values"
    if re.match(r"""^["'][^"']+["']\s*:""", s):
        return f"entry '{first_str.group(1)}' of the dictionary (name -> value) being built"
    if s.startswith("[") or s.startswith('"') or s.startswith("'"):
        return "more items of the list started on the line above"
    return "continues the statement started on the line above"


def explain_lines(source):
    """Split code into (code, explanation) rows using the comments written on each line."""
    rows, pending = [], []
    for line in source.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#DOC#"):  # a function's docstring -> attach to its def line
            doc = s[5:].strip()
            if rows and rows[-1][0].lstrip().startswith(("def ", "class ")):
                rows[-1] = (rows[-1][0], rows[-1][1] + ". Purpose: " + doc)
            else:
                pending.append(doc)
            continue
        if s.startswith("#"):
            txt = re.sub(r"-{5,}", "", s.lstrip("# ")).strip()  # drop decorative ----- lines
            if txt:
                pending.append(txt)
            continue
        m = re.match(r"^(.*?)(?<![\"'\w])\s+#\s(.*)$", line)
        code, expl = (m.group(1).rstrip(), m.group(2).strip()) if m else (line.rstrip(), "")
        if pending:
            expl = (" ".join(pending) + (" - " + expl if expl else "")).strip()
            pending = []
        if not expl:
            expl = auto_explain(code)
        rows.append((code, expl))
    return rows


def code_table(d, rows):
    t = d.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    for i, h in enumerate(["#", "Code", "What this line does"]):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(8.5), RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "116FBF")
    for n, (code, expl) in enumerate(rows, 1):
        cells = t.add_row().cells
        cells[0].text = str(n)
        cells[0].paragraphs[0].runs[0].font.size = Pt(7.5)
        cells[1].text = ""
        r = cells[1].paragraphs[0].add_run(code)
        r.font.name, r.font.size = "Consolas", Pt(7.5)
        cells[2].text = ""
        cells[2].paragraphs[0].add_run(expl).font.size = Pt(8)
    for row in t.rows:
        for i, w in enumerate([0.9, 13.5, 10.5]):
            row.cells[i].width = Cm(w)
    d.add_paragraph()


def build_explained():
    d = base_doc(landscape=True)
    para(d, "Question 1 - Code explained line by line", bold=True, size=18, color=BLUE)
    para(d, "Group XX - IGIHOZO Bienvenue, MUGISHA Aime Patrick")
    para(d, "How to use this document: every code cell of notebooks 00-04 and every line of src/maize_pipeline.py is "
            "listed in order with the explanation written next to it in the code. Read it with the notebook open; for the "
            "oral questions, be ready to explain WHY each step was done (the Markdown text in the notebooks gives the reason).")
    d.add_heading("Key ideas to master before the presentation", 2)
    bullets(d, [
        "Data leakage: using information the model will not have at prediction time (e.g. full-season rainfall for an "
        "early forecast; statistics computed on the test set).",
        "Fit on train only: imputers, encoders and scalers learn from the training rows, then transform val/test unchanged.",
        "Disguised missing values: numbers or words (-999, 0, 'unknown') that mean 'not recorded' - convert to NaN first.",
        "Median vs mean imputation: the median is not pulled by extreme values.",
        "One-hot encoding: one 0/1 column per category (district, seed variety) because the categories have no order.",
        "Low-yield threshold learned on train only (10th percentile) - computing it on all rows would leak test information.",
        "Kept outliers: IQR outliers that are realistic (large farms, heavy pest years) are real information, not errors.",
        "Standardisation: (x - mean) / std, so all numeric features have mean 0 and std 1.",
        "Stratified split by season: keeps the small Season C group in train, validation and test.",
        "IQR rule: outlier if below Q1 - 1.5 IQR or above Q3 + 1.5 IQR.",
    ])
    for nb_name in ["00_environment_check", "01_acquire", "02_eda", "03_clean", "04_features"]:
        book = json.loads((NB / f"{nb_name}.ipynb").read_text(encoding="utf-8"))
        d.add_page_break()
        d.add_heading(f"Notebook {nb_name}.ipynb", 1)
        cell_no = 0
        for cell in book["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                first = next((l for l in src.splitlines() if l.startswith("#")), None)
                if first and not first.startswith("# Group"):
                    d.add_heading(first.lstrip("# ").strip(), 3)
                continue
            cell_no += 1
            para(d, f"Code cell {cell_no}", bold=True, size=9, color=BLUE)
            code_table(d, explain_lines(src))
    d.add_page_break()
    d.add_heading("Module src/maize_pipeline.py", 1)
    src = (ROOT / "src" / "maize_pipeline.py").read_text(encoding="utf-8")
    src = re.sub(r'"""(.*?)"""', lambda m: "#DOC# " + " ".join(m.group(1).split()), src, flags=re.S)  # docstrings
    code_table(d, explain_lines(src))
    out = DOCS / "Q1_Code_Explained_Line_by_Line.docx"
    d.save(out)
    return out


if __name__ == "__main__":
    (DOCS / "data_card.md").write_text(data_card_text(), encoding="utf-8")
    for f in (build_report(), build_memo(), build_explained()):
        print("wrote", f)
