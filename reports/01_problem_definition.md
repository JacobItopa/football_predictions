# 01 Problem Definition & Scoping

## Business Objective
The objective is to build a highly realistic and robust English Premier League (EPL) Match Predictor. While sports are inherently unpredictable with high variance, the aim is to build a model that can achieve an accuracy of 60-65% in predicting the Full-Time Result (FTR: Home Win, Draw, Away Win), matching or exceeding the baseline accuracy of traditional bookmakers.

## Success Metrics
- Achieve 60-65% accuracy on an unseen test set.
- Deploy an easy-to-use modern UI where users can query the model.

## ML Task
- **Supervised Learning**: Multi-class classification (3 classes: H, D, A).

## Feasibility
- Historical EPL data from `football-data.co.uk` contains an extensive, structured set of historical matches including betting odds, which provides more than enough signal to train a model.
