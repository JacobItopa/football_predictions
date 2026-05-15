# 01 Problem Definition & Scoping

## Business Objective
The objective is to build a highly realistic and robust English Premier League (EPL) Match Predictor. While sports are inherently unpredictable with high variance, the aim is to build a model that achieves strong predictive accuracy in predicting the Full-Time Result (FTR: Home Win, Draw, Away Win), using both classical match statistics and advanced analytics such as Expected Goals (xG).

## Success Metrics
- Achieve >54% accuracy on an unseen chronological test set (professional models consistently sit in the 52–58% band for 3-way football prediction).
- Deploy an easy-to-use modern UI where users can query the model for upcoming Premier League fixtures.
- Automate continuous retraining so the model always reflects the most recent team form.

## ML Task
- **Supervised Learning**: Multi-class classification (3 classes: H=Home Win, D=Draw, A=Away Win).
- **Evaluation Strategy**: Strict chronological train/test split (no shuffling) to prevent future data leaking into past predictions.

## Feasibility
- Historical EPL data from `football-data.co.uk` provides structured match stats and betting odds going back over 10 seasons.
- **Expected Goals (xG)** data from Understat.com provides 6 seasons of advanced shot quality metrics — a key signal used by professional analysts.
- Together, these sources give the model a rich blend of market-consensus signals (bookmaker odds) and performance-quality signals (xG).
