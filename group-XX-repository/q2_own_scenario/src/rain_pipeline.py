"""
rain_pipeline.py  -  reusable cleaning and feature functions for Question 2
(next-day rain early warning for MINEMA district disaster offices, NOAA GSOD data).

Every notebook imports this file, so the SAME code cleans the training data now and any
new daily data later. Each cleaning function takes a DataFrame and a CleaningLog, changes a
copy and writes one line per decision into the log (what, why, how many records).
"""
from pathlib import Path  # Path builds file paths that work on every computer

import numpy as np  # maths on whole columns
import pandas as pd  # the DataFrame table

SEED = 42  # fixed seed for anything random -> same result every run
TARGET = "rain_tomorrow"  # 1 = rain observed at the station the next day, 0 = no rain

# Station code -> short name, district and province (where MINEMA would send the warning).
STATIONS = {
    "64387099999": ("Kigali", "Kicukiro", "Kigali City"),
    "64380099999": ("Kamembe", "Rusizi", "Western"),
    "64381099999": ("Gisenyi", "Rubavu", "Western"),
    "64383099999": ("Ruhengeri", "Musanze", "Northern"),
}
# NOAA writes these numbers when a value was NOT measured (disguised missing values).
MISSING_CODES = {"TEMP": 9999.9, "DEWP": 9999.9, "SLP": 9999.9, "STP": 999.9, "VISIB": 999.9, "WDSP": 999.9,
                 "MXSPD": 999.9, "GUST": 999.9, "MAX": 9999.9, "MIN": 9999.9, "PRCP": 99.99, "SNDP": 999.9}
NUMERIC = list(MISSING_CODES) + ["LATITUDE", "LONGITUDE", "ELEVATION"]  # columns that must be numbers
# Rwanda's four climate seasons (used for a feature and for the charts).
RW_SEASON = {1: "Short dry", 2: "Short dry", 3: "Long rains", 4: "Long rains", 5: "Long rains", 6: "Long dry",
             7: "Long dry", 8: "Long dry", 9: "Short rains", 10: "Short rains", 11: "Short rains", 12: "Short rains"}


class CleaningLog:
    """Collects one row per cleaning decision; saved as docs/cleaning_log.csv."""

    def __init__(self):  # runs when we write CleaningLog()
        self.rows = []  # list of decisions

    def add(self, operation, column, problem, action, n_rows, reason):  # record one decision
        self.rows.append({"step": len(self.rows) + 1, "operation": operation, "column": column, "problem": problem,
                          "action": action, "records_affected": int(n_rows), "reason": reason})  # one row

    def to_frame(self):  # the log as a table
        return pd.DataFrame(self.rows)


def load_raw(folder):
    """Read every yearly station file in data/raw/gsod and stack them into one table (all as text)."""
    files = sorted(Path(folder).glob("*.csv"))  # e.g. 2015_64387099999.csv
    parts = [pd.read_csv(f, dtype=str).assign(source_file=f.name) for f in files]  # read as text, remember the file
    return pd.concat(parts, ignore_index=True)  # one table, rows renumbered


def trim_and_type(df, log):
    """Step 1: remove padding spaces, convert text to numbers and dates."""
    df = df.copy()  # never change the caller's table
    df.columns = df.columns.str.strip()  # column names without spaces
    padded = 0  # counter of padded cells
    for c in df.columns:  # every column
        s = df[c].astype("string")  # as text
        padded += int((s.str.len() != s.str.strip().str.len()).sum())  # cells that had spaces
        df[c] = s.str.strip()  # remove the spaces
    log.add("trim_and_type", "all", "values padded with spaces ('  70.0')", "spaces removed", padded,
            "padded text cannot be read as numbers")
    for c in NUMERIC:  # numeric columns
        df[c] = pd.to_numeric(df[c], errors="coerce")  # text -> float
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")  # text -> date
    log.add("trim_and_type", ", ".join(NUMERIC + ["DATE"]), "numbers and dates stored as text",
            "converted to float / datetime", len(df), "calculations need real numbers and dates")
    return df


def replace_missing_codes(df, log):
    """Step 2: NOAA codes 9999.9 / 999.9 / 99.99 mean 'not measured' -> NaN."""
    df = df.copy()  # work on a copy
    for c, code in MISSING_CODES.items():  # each column has its own code
        mask = df[c] == code  # rows with the code
        df.loc[mask, c] = np.nan  # real missing value
        log.add("missing_codes", c, f"missing code {code}", "set to NaN", mask.sum(),
                "a code like 9999.9 would be read as a real (impossible) value")
    return df


def drop_empty_columns(df, log, limit=0.5):
    """Step 3: drop columns that are more than 50 % missing, and columns only used for bookkeeping."""
    df = df.copy()  # work on a copy
    share = df[list(MISSING_CODES)].isna().mean()  # share missing per measured column
    empty = share[share > limit]  # columns above the limit
    for c, s in empty.items():  # log each one
        log.add("drop_empty_columns", c, f"{s:.0%} of values missing", "column removed", len(df),
                "too few values to learn from; imputing > 50 % would invent data")
    attrs = [c for c in df.columns if c.endswith("_ATTRIBUTES") and c not in ("TEMP_ATTRIBUTES", "PRCP_ATTRIBUTES")]
    df = df.drop(columns=list(empty.index) + attrs + ["LATITUDE", "LONGITUDE"])  # remove them
    log.add("drop_empty_columns", ", ".join(attrs), "observation-count columns of other variables",
            "removed (TEMP count and PRCP flag kept)", len(df), "bookkeeping only; one count is enough as a quality signal")
    log.add("drop_empty_columns", "LATITUDE, LONGITUDE", "constant for each station", "removed", len(df),
            "the same information is in the station column")
    return df


def name_stations(df, log):
    """Step 4: station code -> short name, district and province; drop the long NOAA name."""
    df = df.copy()  # work on a copy
    info = df["STATION"].map(STATIONS)  # tuple (name, district, province)
    unknown = info.isna()  # a station we did not expect
    if unknown.any():  # stop loudly
        raise ValueError(f"unknown station codes: {df.loc[unknown, 'STATION'].unique()}")
    df["station"] = info.str[0]  # 'Kigali'
    df["district"] = info.str[1]  # 'Kicukiro'
    df["province"] = info.str[2]  # 'Kigali City'
    log.add("name_stations", "NAME", "long names ('KIGALI INTERNATIONAL, RW')", "short station, district, province",
            len(df), "MINEMA works by district; short labels are easier to read")
    return df.drop(columns=["NAME", "source_file"])  # NAME replaced; file name no longer needed


def convert_units(df, log):
    """Step 5: imperial -> metric (F -> C, knots -> km/h, miles -> km, inches -> mm)."""
    df = df.copy()  # work on a copy
    for c, new in [("TEMP", "temp_c"), ("DEWP", "dewpoint_c"), ("MAX", "max_temp_c"), ("MIN", "min_temp_c")]:
        df[new] = ((df[c] - 32) * 5 / 9).round(2)  # Fahrenheit -> Celsius
    df["wind_kmh"] = (df["WDSP"] * 1.852).round(2)  # mean wind, knots -> km/h
    df["max_wind_kmh"] = (df["MXSPD"] * 1.852).round(2)  # strongest sustained wind
    df["visibility_km"] = (df["VISIB"] * 1.609).round(2)  # statute miles -> km
    df["prcp_mm"] = (df["PRCP"] * 25.4).round(2)  # inches -> mm
    df["elevation_m"] = df["ELEVATION"]  # already metres
    df = df.drop(columns=["TEMP", "DEWP", "MAX", "MIN", "WDSP", "MXSPD", "VISIB", "PRCP", "ELEVATION"])  # old columns
    log.add("convert_units", "TEMP, DEWP, MAX, MIN, WDSP, MXSPD, VISIB, PRCP", "imperial units (F, knots, miles, inches)",
            "converted to C, km/h, km, mm", len(df), "Rwanda (Meteo Rwanda, MINEMA) uses metric units")
    return df


def quality_flags(df, log):
    """Step 6: decode the coded columns - FRSHTT weather flags, PRCP flag, observation count."""
    df = df.copy()  # work on a copy
    code = df["FRSHTT"].str.zfill(6)  # 6 digits: Fog Rain Snow Hail Thunder Tornado
    df["fog"] = code.str[0].astype(int)  # 1 = fog observed
    df["rain"] = code.str[1].astype(int)  # 1 = rain or drizzle observed
    df["hail"] = code.str[3].astype(int)  # 1 = hail
    df["thunder"] = code.str[4].astype(int)  # 1 = thunderstorm
    log.add("quality_flags", "FRSHTT", "6 weather events packed in one code ('010010')", "split into fog, rain, hail, thunder",
            len(df), "each event becomes a 0/1 column a model can use")
    df["prcp_incomplete"] = (df["PRCP_ATTRIBUTES"] == "I").astype(int)  # 'I' = not all reports received
    log.add("quality_flags", "PRCP_ATTRIBUTES", "flag 'I' = incomplete precipitation total", "kept as 0/1 prcp_incomplete",
            df["prcp_incomplete"].sum(), "an incomplete total of 0 mm does not prove a dry day")
    df["n_obs"] = pd.to_numeric(df["TEMP_ATTRIBUTES"], errors="coerce")  # how many reports built the daily mean
    few = df["n_obs"] < 4  # fewer than 4 reports in 24 h
    log.add("quality_flags", "TEMP_ATTRIBUTES", "daily mean built from fewer than 4 reports", "kept, count stored as n_obs",
            few.sum(), "values are less reliable; the model can learn how much to trust them")
    return df.drop(columns=["FRSHTT", "PRCP_ATTRIBUTES", "TEMP_ATTRIBUTES"])  # decoded columns replace them


def impossible_values(df, log):
    """Step 7: physically impossible combinations and unrealistic values -> NaN."""
    df = df.copy()  # work on a copy
    dew = df["dewpoint_c"] > df["temp_c"] + 0.5  # dew point cannot be above air temperature
    df.loc[dew, "dewpoint_c"] = np.nan  # not trustworthy
    log.add("impossible", "dewpoint_c", "dew point above air temperature", "set to NaN", dew.sum(), "physically impossible")
    mm = df["max_temp_c"] < df["min_temp_c"]  # maximum below minimum
    df.loc[mm, ["max_temp_c", "min_temp_c"]] = np.nan  # cannot know which one is wrong
    log.add("impossible", "max_temp_c, min_temp_c", "maximum below minimum", "both set to NaN", mm.sum(), "impossible")
    out = ~df["temp_c"].between(df["min_temp_c"] - 1, df["max_temp_c"] + 1) & df[["temp_c", "min_temp_c", "max_temp_c"]].notna().all(axis=1)
    df.loc[out, ["max_temp_c", "min_temp_c"]] = np.nan  # mean outside its own min-max range
    log.add("impossible", "max_temp_c, min_temp_c", "daily mean outside its own min-max range", "min/max set to NaN",
            out.sum(), "min/max are often derived from incomplete hourly reports")
    wind = df["wind_kmh"] > 40  # daily MEAN wind above 40 km/h is not realistic at these inland stations
    df.loc[wind, "wind_kmh"] = np.nan  # treat as error
    log.add("impossible", "wind_kmh", "daily mean wind above 40 km/h", "set to NaN", wind.sum(),
            "station climatology: normal daily mean 3-15 km/h; such values come with very few reports")
    gust = df["max_wind_kmh"] > 90  # sustained wind above 90 km/h never recorded in Rwanda
    df.loc[gust, "max_wind_kmh"] = np.nan  # treat as error
    log.add("impossible", "max_wind_kmh", "max sustained wind above 90 km/h", "set to NaN", gust.sum(), "not credible")
    return df


def remove_duplicates(df, log):
    """Step 8: one row per station per day."""
    df = df.copy()  # work on a copy
    n0 = len(df)  # before
    df = df.drop_duplicates(subset=["STATION", "DATE"], keep="first")  # same station and date twice
    log.add("remove_duplicates", "STATION, DATE", "same station-day twice", "removed (kept first)", n0 - len(df),
            "one day must count once")
    return df.sort_values(["station", "DATE"]).reset_index(drop=True)  # time order inside each station


def add_target(df, log):
    """Step 9: rain_tomorrow = rain flag of the NEXT calendar day at the same station (only if that day exists)."""
    df = df.copy()  # work on a copy
    nxt = df.groupby("station")["DATE"].shift(-1)  # date of the next row of the same station
    rain_next = df.groupby("station")["rain"].shift(-1)  # rain flag of that next row
    gap = (nxt - df["DATE"]).dt.days  # 1 = consecutive day
    df[TARGET] = np.where(gap == 1, rain_next, np.nan)  # only a true 'tomorrow' counts
    df["days_since_prev_record"] = df.groupby("station")["DATE"].diff().dt.days  # gaps before the day (for lags)
    no_target = df[TARGET].isna()  # tomorrow not recorded
    log.add("add_target", TARGET, "next day missing in the record (gaps, end of data)", "row removed", no_target.sum(),
            "without tomorrow's observation there is no answer to learn")
    return df[~no_target].assign(**{TARGET: lambda d: d[TARGET].astype(int)}).reset_index(drop=True)


def clean_rain(raw, log=None):
    """Run every cleaning step in order (same function for training data and new data)."""
    log = log or CleaningLog()  # new log if none given
    df = trim_and_type(raw, log)  # 1
    df = replace_missing_codes(df, log)  # 2
    df = drop_empty_columns(df, log)  # 3
    df = name_stations(df, log)  # 4
    df = convert_units(df, log)  # 5
    df = quality_flags(df, log)  # 6
    df = impossible_values(df, log)  # 7
    df = remove_duplicates(df, log)  # 8
    df = add_target(df, log)  # 9
    return df, log


def add_features(df):
    """Create new features using ONLY today and earlier days of the same station (never tomorrow)."""
    df = df.sort_values(["station", "DATE"]).copy()  # time order inside each station
    g = df.groupby("station")  # per-station groups for 'yesterday' values
    df["month"] = df["DATE"].dt.month  # F1 month number
    df["rw_season"] = df["month"].map(RW_SEASON)  # F2 Rwanda climate season
    df["doy_sin"] = np.sin(2 * np.pi * df["DATE"].dt.dayofyear / 365.25)  # F3 day of year as a circle (Dec 31 ~ Jan 1)
    df["doy_cos"] = np.cos(2 * np.pi * df["DATE"].dt.dayofyear / 365.25)  # F3 second part of the circle
    df["dewpoint_spread_c"] = df["temp_c"] - df["dewpoint_c"]  # F4 small spread = moist air
    a, b = 17.625, 243.04  # Magnus formula constants
    df["rel_humidity_pct"] = (100 * np.exp(a * df["dewpoint_c"] / (b + df["dewpoint_c"]))
                              / np.exp(a * df["temp_c"] / (b + df["temp_c"]))).clip(upper=100)  # F5 relative humidity
    df["diurnal_range_c"] = df["max_temp_c"] - df["min_temp_c"]  # F6 cloudy/rainy days have a small range
    consecutive = df["days_since_prev_record"] == 1  # yesterday exists
    df["rain_yesterday"] = g["rain"].shift(1).where(consecutive)  # F7 persistence of rain
    df["temp_change_c"] = (df["temp_c"] - g["temp_c"].shift(1)).where(consecutive)  # F8 cooling/warming since yesterday
    df["rain_days_last3"] = g["rain"].transform(lambda s: s.rolling(3, min_periods=3).sum())  # F9 wet spell
    window_ok = g["DATE"].transform(lambda s: (s - s.shift(2)).dt.days) == 2  # the 3 rows are 3 consecutive days
    df["rain_days_last3"] = df["rain_days_last3"].where(window_ok)  # otherwise unknown
    df["log_prcp_mm"] = np.log1p(df["prcp_mm"])  # F10 rain amount is very skewed -> log
    return df.reset_index(drop=True)


def chronological_split(df, train_end="2022-12-31", val_end="2023-12-31"):
    """Time split: learn from the past, validate on 2023, test on 2024-2025 (no future data in training)."""
    train = df[df["DATE"] <= train_end]  # 2015-2022
    val = df[(df["DATE"] > train_end) & (df["DATE"] <= val_end)]  # 2023
    test = df[df["DATE"] > val_end]  # 2024-2025
    return train.copy(), val.copy(), test.copy()


def project_root():
    """Folder q2_own_scenario (works from notebooks/ or the project root)."""
    here = Path.cwd()  # where Jupyter was started
    return here.parent if here.name == "notebooks" else here
