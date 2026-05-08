# 05 Model Training

## Setup
We trained three different machine learning models to predict the Full Time Result (FTR) of English Premier League matches:
1. **Logistic Regression:** A linear baseline.
2. **Random Forest:** An ensemble method less prone to overfitting on tabular data.
3. **XGBoost:** A powerful gradient-boosting algorithm known for top performance on structured tabular tasks.

## Training Process
- The models were trained on chronological data to prevent "future" data from leaking into past predictions.
- We used an 80/20 train-test split without shuffling.
- The features included the rolling averages for points, goals scored/conceded, and shots on target for both the Home and Away teams.
- The target variable `FTR` was encoded as: 
  - `0` = Away Win
  - `1` = Draw
  - `2` = Home Win
