# 02 Data Collection & Integration

## Data Sources

### Primary Source — football-data.co.uk
- Historical EPL match CSVs downloaded from `https://www.football-data.co.uk/englandm.php`.
- Each CSV (E0.csv) represents one EPL season and contains match results, shots, corners, fouls, cards, and bookmaker odds.
- **19 season files** are currently in `data/raw/`, covering from the 2010/11 season through the 2024/25 season.

### Secondary Source — Understat.com (xG Data)
- Expected Goals (xG) data fetched programmatically via the `soccerdata` Python library.
- Source: `https://understat.com` — a respected industry standard for xG statistics.
- **6 seasons** fetched (2020/21 → 2025/26) = **2,280 match records** with `home_xg` and `away_xg` per match.
- Stored in `data/raw/xg_data.csv` (generated at runtime by `src/data/fetch_xg_data.py`).

## Extraction and Gathering

### football-data.co.uk CSVs
- All `E0*.csv` files reside in `data/raw/` and are treated as **immutable raw data**.
- `src/data/make_dataset.py` merges all season CSVs into `data/interim/merged_data.csv`.

### Expected Goals (xG) — `src/data/fetch_xg_data.py`
- Uses `soccerdata.Understat` to fetch match schedules with xG for the last 6 seasons.
- Applies a team name mapping to align Understat names (e.g. `"Wolverhampton Wanderers"`) with football-data.co.uk names (e.g. `"Wolves"`).
- Saves output to `data/raw/xg_data.csv`.
- `make_dataset.py` left-joins this file onto the main interim dataset by `Date + HomeTeam + AwayTeam`.

## Data Governance
- Raw data files are excluded from Git via `.gitignore` (`data/raw/`).
- `xg_data.csv` is regenerated automatically at the start of every CT retrain cycle (Step 0 of `src/retrain_pipeline.py`).
- No PII is involved — all data is publicly available aggregate match statistics.

## Final Interim Dataset
- **7,189 rows** (match records) × **30 columns**
- Key new columns added: `Home_xG`, `Away_xG`
