# 06 Model Evaluation

## Results
The accuracy of the models on the unseen test set (latest chronological matches):
- **Logistic Regression:** 52.21%
- **Random Forest:** 51.44%
- **XGBoost:** 50.86%

## Analysis
- **Logistic Regression** remains the best performing model at **52.21%**.
- Adding **Bookmaker Odds (B365)**, **Rest Days**, and **Head-to-Head (H2H)** introduced a massive amount of market-consensus data. While the raw accuracy appears to hover around ~52%, this is actually a highly realistic number for a professional predictive model. Predicting the exact 3-way outcome (Home/Draw/Away) of a football match with >55% accuracy consistently over long periods is notoriously difficult due to the sport's high variance.
- **Draws (1)** remain incredibly hard to predict (F1-score near 0.00). The model defaults to predicting a decisive winner because betting odds heavily penalize predicting draws over a large sample size.

## Selection
The **Logistic Regression** model was selected as the final production model and has been serialized to `models/best_model.joblib`.
