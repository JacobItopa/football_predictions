# 03 Exploratory Data Analysis (EDA)

## Descriptive Statistics
- The raw dataset is composed of 21 CSV files containing EPL matches over multiple seasons.
- The `FTR` (Full Time Result) column shows the distribution of match outcomes (Home wins are typically the most common, followed by Away wins, then Draws).

## Identified Issues
- **Missing Values:** Early season data may have empty betting columns or missing referee information. We filtered out these columns as they are not needed for core predictive modeling. We only kept matches with non-null `FTR` and standard stats.
- **Date Formats:** The `football-data.co.uk` CSVs have inconsistent date formats across the years (`dd/mm/yy` vs `dd/mm/yyyy`). We used pandas `pd.to_datetime(..., format='mixed', dayfirst=True)` to handle this reliably.
- **Rolling Windows:** The first few matches of each team don't have enough history for a 5-match rolling average, so those rows will be dropped in the preprocessing stage.
