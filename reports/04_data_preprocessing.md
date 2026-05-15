# 04 Data Preprocessing & Feature Engineering

## Data Cleaning (`src/data/make_dataset.py`)
- Dropped all matches missing the `FTR` (Full Time Result) target.
- Merged 19 season CSV files into a unified interim dataset (`data/interim/merged_data.csv`).
- Handled inconsistent date formatting and sorted the dataset chronologically.
- Left-joined xG data from `data/raw/xg_data.csv` onto the main dataset using `Date + HomeTeam + AwayTeam` keys.
- Excluded `xg_data.csv` from the raw CSV file glob to prevent double-counting.

**Interim dataset: 7,189 rows × 30 columns**

---

## Feature Engineering (`src/features/build_features.py`)

All rolling features use a **5-match window** and are **shifted by 1** to avoid data leakage (the current match is never included in its own rolling average).

### Rolling Average Features (per team, from their own perspective)

| Feature | Description |
|---|---|
| `HomePointsRolling` / `AwayPointsRolling` | Average league points (W=3, D=1, L=0) over last 5 matches — measures current "form" |
| `HomeGoalsScoredRolling` / `AwayGoalsScoredRolling` | Average goals scored — attacking strength |
| `HomeGoalsConcededRolling` / `AwayGoalsConcededRolling` | Average goals conceded — defensive solidity |
| `HomeShotsOnTargetRolling` / `AwayShotsOnTargetRolling` | Average shots on target — underlying attacking quality |
| **`Home_xGRolling` / `Away_xGRolling`** | ⭐ **NEW** — Average xG generated over last 5 matches — true shot quality signal |
| **`Home_xGConcededRolling` / `Away_xGConcededRolling`** | ⭐ **NEW** — Average xG conceded over last 5 matches — true defensive exposure |

> **xG Fallback:** For historical matches predating the xG era, actual goals (`FTHG`/`FTAG`) are substituted so no rows are lost.

### Other Engineered Features

| Feature | Description |
|---|---|
| `HomeElo` / `AwayElo` | Dynamic Elo rating calculated sequentially from all historical matches |
| `HomeRestDays` / `AwayRestDays` | Days since each team's last match — captures fatigue and fixture congestion |
| `H2H_HomePoints` | Average points the home team earned in their last 5 head-to-head meetings |

### Bookmaker Odds (Market Features)
| Feature | Description |
|---|---|
| `B365H` / `B365D` / `B365A` | Bet365 odds for Home Win, Draw, Away Win — encodes market consensus |

---

## Feature Selection
- Post-match raw statistics (`HS`, `AS`, `FTHG`, `FTAG`, `Home_xG`, `Away_xG`) are **excluded** from the final modelling dataset — they are only used during the rolling average calculation phase (cannot be known before the match).
- Target mapped to numerical labels: `H`=2, `D`=1, `A`=0.

**Processed dataset: 7,020 rows × 25 columns (20 model features + metadata)**

### Final Feature List (20 features)
```python
[
    'HomePointsRolling', 'AwayPointsRolling',
    'HomeGoalsScoredRolling', 'AwayGoalsScoredRolling',
    'HomeGoalsConcededRolling', 'AwayGoalsConcededRolling',
    'HomeShotsOnTargetRolling', 'AwayShotsOnTargetRolling',
    'Home_xGRolling', 'Away_xGRolling',            # NEW
    'Home_xGConcededRolling', 'Away_xGConcededRolling',  # NEW
    'HomeElo', 'AwayElo',
    'HomeRestDays', 'AwayRestDays',
    'H2H_HomePoints',
    'B365H', 'B365D', 'B365A'
]
```
