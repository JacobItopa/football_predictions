# 09 Continuous Training (CT) Pipeline

## Objective
To keep the model fresh and accurate as the EPL season progresses, we implemented
a fully automated **Continuous Training** pipeline that can be triggered at any time.

## Components

### `src/data/update_results.py`
Fetches all FINISHED EPL matches from the `football-data.org` API and appends
any new matches (identified by Date + HomeTeam + AwayTeam) to `data/interim/merged_data.csv`.
- **Deduplication**: Matches already in the dataset are never added twice.
- **Schema mapping**: API team names (e.g. "Manchester City FC") are translated to
  the CSV training names (e.g. "Man City") via `TEAM_NAME_MAP`.
- **Free-tier graceful handling**: Shots/Corners/Fouls are not provided by the free
  API tier, so those columns are left as `None`. The rolling averages fall back to
  the team's existing historical data for those stats.

### `src/retrain_pipeline.py`
Orchestrates the full CT cycle in one call:
1. `update_results.py` → append new results
2. `build_features.py` → rebuild rolling averages + Elo ratings
3. `train_model.py` logic → retrain Logistic Regression, Random Forest, XGBoost
4. Saves the best model to `models/best_model.joblib`
5. Writes an entry to `reports/retrain_log.json` for audit purposes

### FastAPI `/retrain` endpoint
Triggers the pipeline as a non-blocking **background task**, so the server
stays responsive to prediction requests while retraining is in progress.
- `POST /retrain` — start the pipeline
- `GET /retrain/status` — poll for progress/completion

### Live Hot-Swap
After retraining completes, the app calls `reload_model_and_stats()` to swap
in the new model and updated team statistics **without restarting the server**.
Predictions immediately reflect the latest data.

## Usage

### Manual (CLI)
```powershell
# Set API key
$env:FOOTBALL_DATA_API_KEY = "your_key_here"

# Run the full CT pipeline
python src/retrain_pipeline.py
```

### Via the UI
Click **⚡ Update & Retrain** on the Upcoming Fixtures page.
The button shows live progress and reports the new accuracy when done.

## Recommended Schedule
Run the pipeline once per **matchweek** (i.e. every 1–2 weeks during the season)
to keep rolling averages and Elo ratings current.
