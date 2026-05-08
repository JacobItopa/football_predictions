# 04 Data Preprocessing & Feature Engineering

## Data Cleaning
- Dropped all matches missing the `FTR` (Full Time Result) target.
- Merged the 21 season CSV files into a unified interim dataset (`merged_data.csv`).
- Handled inconsistent date formatting and sorted the dataset chronologically.

## Feature Engineering
We wrote a script (`src/features/build_features.py`) to create historical, rolling-average features that provide the model with context for every match.
- **Points Rolling (5 matches):** Calculates the average points (3 for Win, 1 for Draw, 0 for Loss) a team accumulated in their last 5 matches. This indicates current "form".
- **Goals Scored/Conceded Rolling (5 matches):** Indicates attacking and defensive strength over the short term.
- **Shots on Target Rolling (5 matches):** A strong underlying indicator of performance.

## Feature Selection
- Dropped single-match raw statistics (like `HS`, `AS`, `FTHG`) from the final modeling dataset because these are post-match statistics (data leakage). We only kept the rolling averages, team names, dates, and the target (`FTR`).
- Target mapped to numerical labels: `H`=2, `D`=1, `A`=0.
