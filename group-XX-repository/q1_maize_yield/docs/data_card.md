# Data Card - MCSU maize yield dataset (Question 1)
## Purpose
Model-ready data for forecasting maize yield per farm and flagging farms at risk of a very low harvest (MCSU cooperatives).
## Source
File maize_yield_rwanda_rw.csv supplied by the lecturer (course material), copied unchanged into data/raw/.
## Size
Raw: 1,426 rows x 20 columns (2015-2024, 5 provinces, 30 districts). Clean: 1,426 rows. Train / validation / test: 998 / 214 / 214 rows, 68 encoded input columns.
## Target
yield_t_ha (regression; also yield_kg_ha). low_yield_risk = 1 if yield < 1.67 t/ha (bottom 10 % of training farms; 9.9% positives).
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
