# 05 Model Training

## Setup
Three different machine learning models were trained to predict the Full Time Result (FTR) of English Premier League matches:

1. **Logistic Regression** — A linear baseline. Interpretable and fast.
2. **Random Forest** — An ensemble method, less prone to overfitting on structured tabular data.
3. **XGBoost** — A powerful gradient-boosting algorithm, known for top performance on tabular tasks.

## Training Process
- Models were trained on **chronological data** to prevent "future" data leaking into past predictions.
- **80/20 chronological train-test split** (no shuffling): 5,616 training matches / 1,404 test matches.
- Features include rolling averages for points, goals scored/conceded, shots on target, **Expected Goals (xG)**, Elo ratings, rest days, head-to-head history, and bookmaker odds.

### Target Variable
| Label | Meaning | Numeric Code |
|---|---|---|
| `H` | Home Win | `2` |
| `D` | Draw | `1` |
| `A` | Away Win | `0` |

## Feature Set (20 Features)
```python
[
    'HomePointsRolling', 'AwayPointsRolling',
    'HomeGoalsScoredRolling', 'AwayGoalsScoredRolling',
    'HomeGoalsConcededRolling', 'AwayGoalsConcededRolling',
    'HomeShotsOnTargetRolling', 'AwayShotsOnTargetRolling',
    'Home_xGRolling', 'Away_xGRolling',
    'Home_xGConcededRolling', 'Away_xGConcededRolling',
    'HomeElo', 'AwayElo',
    'HomeRestDays', 'AwayRestDays',
    'H2H_HomePoints',
    'B365H', 'B365D', 'B365A'
]
```

## Scripts
- **`src/models/train_model.py`** — Standalone training script for manual runs.
- **`src/retrain_pipeline.py`** — Full CT pipeline (Step 3 handles model training).
