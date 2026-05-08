# 02 Data Collection & Integration

## Data Sources
- The dataset comprises multiple CSV files downloaded from `https://www.football-data.co.uk/englandm.php`.
- Each CSV file represents an EPL season (E0).

## Extraction and Gathering
- All `E0*.csv` files have been moved from the root directory into the `data/raw/` folder to adhere to the Cookiecutter Data Science structure.
- The raw data is immutable and will not be edited directly. All future transformations will be performed programmatically and saved to `data/interim/` and `data/processed/`.
