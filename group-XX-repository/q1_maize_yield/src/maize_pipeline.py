"""
maize_pipeline.py  -  reusable cleaning and feature functions for Question 1
(maize yield forecasting for MCSU farmer cooperatives, file maize_yield_rwanda_rw.csv).

Every notebook imports this file, so the SAME code cleans the training data now and new
data later. Each cleaning function takes a DataFrame and a CleaningLog, changes a copy and
writes one line per decision (what, why, how many records) - including checks that found
nothing, so the reader can see they were done.
"""
from pathlib import Path  # file paths that work on every computer

import numpy as np  # maths on whole columns
import pandas as pd  # the DataFrame table

SEED = 42  # one fixed random seed -> same split every run
TARGET = "yield_t_ha"  # maize yield in tonnes per hectare (1 t/ha = 1,000 kg/ha)
ID = "record_id"  # record code (not a feature)

# Official list: each of Rwanda's 30 districts and its province.
DISTRICT_PROVINCE = {
    **{d: "Kigali City" for d in ["Gasabo", "Kicukiro", "Nyarugenge"]},
    **{d: "Eastern" for d in ["Bugesera", "Gatsibo", "Kayonza", "Kirehe", "Ngoma", "Nyagatare", "Rwamagana"]},
    **{d: "Northern" for d in ["Burera", "Gakenke", "Gicumbi", "Musanze", "Rulindo"]},
    **{d: "Southern" for d in ["Gisagara", "Huye", "Kamonyi", "Muhanga", "Nyamagabe", "Nyanza", "Nyaruguru", "Ruhango"]},
    **{d: "Western" for d in ["Karongi", "Ngororero", "Nyabihu", "Nyamasheke", "Rubavu", "Rusizi", "Rutsiro"]},
}
# Allowed labels for every category column.
ALLOWED = {
    "season": {"A", "B", "C"},
    "soil_type": {"Clay", "Clay loam", "Loam", "Sandy loam", "Volcanic"},
    "seed_variety": {"Local", "ZM607", "ZM621", "RHM104", "WH505", "PAN691"},
    "farming_practice": {"Improved", "Traditional"},
    "irrigation": {"Yes", "No"},
}
# Seed varieties grouped by type (breeder descriptions): open-pollinated (OPV) or hybrid.
SEED_CLASS = {"Local": "Local", "ZM607": "OPV", "ZM621": "OPV", "RHM104": "Hybrid", "WH505": "Hybrid", "PAN691": "Hybrid"}
# Realistic range of each number for Rwandan maize farms (agronomist notes + national context).
RANGES = {
    "year": (2010, 2026), "altitude_m": (900, 2600), "rainfall_mm": (0, 1500), "avg_temp_c": (10, 30),
    "soil_ph": (4, 8), "npk_kg_ha": (0, 400), "urea_kg_ha": (0, 300), "manure_t_ha": (0, 30),
    "plant_density_per_ha": (10_000, 100_000), "farm_area_ha": (0.02, 20), "pest_incidence_index": (0, 1),
    TARGET: (0.2, 12),
}
SOIL_TEST_COLS = ["soil_ph", "pest_incidence_index", "rainfall_mm"]  # columns that go missing together


class CleaningLog:
    """Collects one row per decision; saved as docs/cleaning_log.csv."""

    def __init__(self):  # runs when we write CleaningLog()
        self.rows = []  # list of decisions

    def add(self, operation, column, problem, action, n_rows, reason):  # record one decision
        self.rows.append({"step": len(self.rows) + 1, "operation": operation, "column": column, "problem": problem,
                          "action": action, "records_affected": int(n_rows), "reason": reason})  # one row

    def to_frame(self):  # the log as a table
        return pd.DataFrame(self.rows)


def trim_text(df, log):
    """Step 1: remove spaces around text values (a check: ' Loam' and 'Loam' would be two soils)."""
    df = df.copy()  # never change the caller's table
    n = 0  # cells changed
    for c in df.select_dtypes(include=["object", "string"]).columns:  # text columns only
        new = df[c].str.strip()  # without spaces
        n += int((new != df[c]).sum())  # count changes
        df[c] = new  # save
    log.add("trim_text", "all text columns", "leading/trailing spaces", "stripped (check)", n,
            "the same label with a space would be counted as another category")
    return df


def check_duplicates(df, log):
    """Step 2: remove repeated records - same ID, or same content under a different ID."""
    df = df.copy()  # work on a copy
    n0 = len(df)  # rows before
    df = df.drop_duplicates(subset=[ID])  # same record code twice
    n1 = len(df)  # after
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID])  # same farm data typed under two codes
    log.add("duplicates", ID, "same record_id repeated", "removed (keep first)", n0 - n1, "a farm must count once")
    log.add("duplicates", "all except record_id", "same content under a different record_id", "removed (keep first)",
            n1 - len(df), "a copied form would double the weight of one farm")
    return df.reset_index(drop=True)


def check_categories(df, log):
    """Step 3: every label must be an allowed value; province must match the district; official names."""
    df = df.copy()  # work on a copy
    for c, allowed in ALLOWED.items():  # each category column
        bad = ~df[c].isin(allowed) & df[c].notna()  # unexpected labels
        if bad.any():  # stop loudly instead of guessing
            raise ValueError(f"{c}: unexpected labels {sorted(df.loc[bad, c].unique())}")
        log.add("categories", c, "labels outside the allowed list", "check only", bad.sum(), "one label per category")
    unknown = ~df["district"].isin(DISTRICT_PROVINCE)  # district not in the official list
    if unknown.any():
        raise ValueError(f"unknown districts {sorted(df.loc[unknown, 'district'].unique())}")
    renamed = (df["province"] == "Kigali").sum()  # short form of the official name
    expected = df["district"].map(DISTRICT_PROVINCE)  # official province of each district
    mismatch = (expected != df["province"].replace({"Kigali": "Kigali City"})).sum()  # wrong province
    df["province"] = expected  # official names everywhere
    log.add("categories", "province", "'Kigali' instead of the official 'Kigali City'", "renamed", renamed,
            "use official administrative names (NISR) so the data joins with other national data")
    log.add("categories", "district, province", "district written under the wrong province", "province taken from district",
            mismatch, "each of the 30 districts belongs to exactly one province")
    return df


def check_ranges(df, log):
    """Step 4: values outside realistic ranges become NaN (none expected - this is a documented check)."""
    df = df.copy()  # work on a copy
    for c, (lo, hi) in RANGES.items():  # each numeric column
        bad = ~df[c].between(lo, hi) & df[c].notna()  # outside the range
        df.loc[bad, c] = np.nan  # an impossible value is treated as unknown
        log.add("ranges", c, f"value outside {lo}-{hi}", "set to NaN (check)", bad.sum(), "agronomist / national ranges")
    return df


def flag_missing(df, log):
    """Step 5: keep rows with gaps, but record WHICH rows had no soil test / pest scouting / rain record."""
    df = df.copy()  # work on a copy
    for c in SOIL_TEST_COLS:  # each column with gaps
        log.add("missing", c, "empty cells", "kept; imputed later (median learned on train only)", df[c].isna().sum(),
                "deleting rows would lose the other 16 good values of these farms")
    df["field_data_missing"] = df[SOIL_TEST_COLS].isna().any(axis=1).astype(int)  # 1 = at least one gap
    log.add("missing", ", ".join(SOIL_TEST_COLS), "gaps occur together in the same farms (pattern, not random)",
            "indicator column field_data_missing", df["field_data_missing"].sum(),
            "the model can learn whether farms without a soil test/scouting visit differ")
    return df


def check_target(df, log):
    """Step 6: the target must exist and be realistic; also give it in kg/ha (the brief's unit)."""
    df = df.copy()  # work on a copy
    bad = df[TARGET].isna()  # no target (incl. values set to NaN by the range check)
    df = df[~bad].copy()  # cannot train without the answer
    log.add("target", TARGET, "missing or impossible yield", "row removed", bad.sum(), "a row without a target is useless")
    df["yield_kg_ha"] = (df[TARGET] * 1000).round(0)  # same value in kg/ha for the cooperatives
    log.add("target", TARGET, "unit t/ha, but the brief and agronomist use kg/ha", "kept t/ha + added yield_kg_ha", len(df),
            "both units available; 1 t/ha = 1,000 kg/ha")
    return df.reset_index(drop=True)


def clean_maize(raw, log=None):
    """Run all cleaning steps in order. Same function for the training data and new data."""
    log = log or CleaningLog()  # new log if none given
    df = trim_text(raw, log)  # 1
    df = check_duplicates(df, log)  # 2
    df = check_categories(df, log)  # 3
    df = check_ranges(df, log)  # 4
    df = flag_missing(df, log)  # 5
    df = check_target(df, log)  # 6
    return df, log


def add_features(df):
    """New features from values of the SAME row only -> safe to compute before or after the split."""
    df = df.copy()  # work on a copy
    df["n_applied_kg_ha"] = 0.17 * df["npk_kg_ha"] + 0.46 * df["urea_kg_ha"]  # F1 nitrogen: NPK 17-17-17 has 17 % N, urea 46 % N (B)
    df["uses_mineral_fertiliser"] = ((df["npk_kg_ha"] + df["urea_kg_ha"]) > 0).astype(int)  # F2 any NPK/urea at all (B)
    df["seed_class"] = df["seed_variety"].map(SEED_CLASS)  # F3 Local / OPV / Hybrid (B)
    df["improved_package"] = (df["seed_class"].isin(["OPV", "Hybrid"])
                              & (df["farming_practice"] == "Improved")).astype(int)  # F4 improved seed + improved practice (B)
    df["log_farm_area_ha"] = np.log1p(df["farm_area_ha"])  # F5 farm area is right-skewed -> log (A)
    df["temp_dev_from_optimum"] = ((18 - df["avg_temp_c"]).clip(lower=0)
                                   + (df["avg_temp_c"] - 27).clip(lower=0))  # F6 degrees outside the 18-27 C optimum (C)
    df["ph_in_optimal_range"] = df["soil_ph"].between(5.5, 7.0).astype(float)  # F7 maize prefers pH 5.5-7 (C)
    df.loc[df["soil_ph"].isna(), "ph_in_optimal_range"] = np.nan  # unknown pH stays unknown
    df["is_irrigated"] = (df["irrigation"] == "Yes").astype(int)  # F8 Yes/No -> 1/0 (C)
    df["altitude_zone"] = pd.cut(df["altitude_m"], [0, 1600, 1900, 3000],
                                 labels=["Lowland", "Midland", "Highland"]).astype(str)  # F9 altitude bands (C, D)
    df["plant_density_k_ha"] = df["plant_density_per_ha"] / 1000  # F10 plants in thousands per ha (easier to read) (A)
    return df


def low_yield_threshold(train):
    """Question E: 'very low harvest' = bottom 10 % of TRAINING yields (learned on train only -> no leakage)."""
    return float(train[TARGET].quantile(0.10))


def add_low_yield_flag(df, threshold):
    """1 = yield below the threshold learned on the training set."""
    df = df.copy()
    df["low_yield_risk"] = (df[TARGET] < threshold).astype(int)
    return df


def project_root():
    """Folder q1_maize_yield (works from notebooks/ or the project root)."""
    here = Path.cwd()
    return here.parent if here.name == "notebooks" else here
