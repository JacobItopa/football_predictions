# 06 Model Evaluation

## Results
The accuracy of the models on the 20% unseen test set (latest chronological matches):
- **Logistic Regression:** 53.99%
- **Random Forest:** 53.06%
- **XGBoost:** 53.92%

## Analysis
- **Logistic Regression** performed the best with an accuracy of **53.99%**, very closely followed by XGBoost at 53.92%.
- Adding the custom **Elo Rating** features significantly improved the model accuracy (jumping from 51.4% to ~54%). This proves that historical team strength is a massive indicator of match outcome beyond just short-term 5-match form.
- **Draws (1)** are the hardest to predict because they are the least common outcome and often happen due to high variance or late equalizers. The models overwhelmingly prefer predicting a Home Win (2) or Away Win (0), which is typical in football modeling.

## Selection
The **Logistic Regression** model was selected as the final production model and has been serialized to `models/best_model.joblib`.
