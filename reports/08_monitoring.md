# 08 Monitoring & Maintenance

## Production Realities
In sports betting and prediction, the underlying dynamics change frequently (Concept Drift). Teams change managers, sign new players, or change tactics.
- **Data Drift:** The `football-data.co.uk` CSVs update weekly. An automated script should be scheduled to download the latest CSV, append it to `data/raw/`, and re-run `src/data/make_dataset.py` and `src/features/build_features.py` to keep the rolling averages current.
- **Model Retraining (CT - Continuous Training):** Because recent form is crucial, the XGBoost model should ideally be retrained every month on the latest data to capture new tactical trends.

## Future Enhancements
- **Logging:** Implement structured JSON logging in FastAPI to track prediction requests and latency.
- **More Features:** Integrate xG (Expected Goals), betting odds, or Elo ratings to push accuracy beyond 51%.
