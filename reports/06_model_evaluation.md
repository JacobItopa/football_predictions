# 06 Model Evaluation

## Latest Results (with xG Features — May 2026)
Accuracy on the unseen chronological test set (1,404 matches):

| Model | Accuracy | Notes |
|---|---|---|
| ✅ **Logistic Regression** | **54.56%** | Best overall — selected for production |
| Random Forest | 54.49% | Close second |
| XGBoost | 53.70% | Slightly lower on this dataset size |

> Previous benchmark (pre-xG, 16 features): ~52.21% — the xG features added **~2.35 percentage points**.

## Classification Report (Best Model — Logistic Regression)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Away Win (0) | 0.55 | 0.50 | 0.52 | 452 |
| Draw (1) | 0.00 | 0.00 | 0.00 | 329 |
| Home Win (2) | 0.54 | 0.87 | 0.67 | 623 |
| **Accuracy** | | | **0.5456** | **1,404** |

## Analysis

### Why ~54% is a Strong Result
Predicting the exact 3-way outcome (Home/Draw/Away) of a football match consistently above 55% is considered expert-level performance. The sport has extremely high inherent variance — injuries, referee decisions, and bounce of the ball all contribute noise that no statistical model can capture.

### Impact of xG Features
The addition of `Home_xGRolling`, `Away_xGRolling`, `Home_xGConcededRolling`, and `Away_xGConcededRolling` improved overall accuracy from ~52% to ~54.56% by giving the model a more accurate signal of *true* attacking and defensive quality, filtering out noise from goals that were fortunate (e.g. deflections, penalties) vs. goals that reflected deserved performance.

### Draw Prediction Limitation
Draws (class 1) remain near impossible to predict (F1 ≈ 0.00). This is an industry-wide problem — even professional betting syndicates rarely predict draws reliably. Bookmakers' implied probability for a draw averages ~26% across all matches, which aligns with the actual draw rate in the test set (329/1404 = 23.4%).

### Bookmaker Odds as a Dominant Signal
The `B365H/D/A` features encode enormous market consensus. Models that include bookmaker odds almost always out-perform pure stats-based models.

## Selection
The **Logistic Regression** model was selected as the final production model and serialized to `models/best_model.joblib`. It is loaded at server startup and hot-swapped after each CT retrain cycle without a server restart.
