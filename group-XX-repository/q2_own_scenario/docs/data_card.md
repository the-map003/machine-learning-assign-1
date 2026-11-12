# Data Card - Next-day rain early warning (Question 2)
## Purpose
Model-ready data to predict whether it will rain tomorrow at four Rwandan weather stations, supporting MINEMA's next-day flood/landslide alerts.
## Source
NOAA NCEI Global Summary of the Day (stations Kigali 64387, Kamembe 64380, Gisenyi 64381, Ruhengeri 64383), 2015-2025, downloaded 11 Nov 2026. Free for education/research (WMO Resolution 40).
## Size
Raw: 10,614 station-days in 42 files. Clean: 9,818 rows. Train (2015-2022) / validation (2023) / test (2024-2025): 8438 / 641 / 739 rows, 24 encoded inputs.
## Target
rain_tomorrow - rain observed at the station the next calendar day (1/0); 39.2% positive.
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
