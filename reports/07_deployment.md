# 07 Deployment & UI

## Architecture
We designed a lightweight, scalable deployment architecture:
- **Backend:** `FastAPI` to serve the serialized `XGBoost` model via a REST endpoint (`/predict`). It loads the latest computed rolling averages for each team from `processed_features.csv` to make predictions seamlessly without requiring the user to manually input stats.
- **Frontend:** A beautiful, responsive Web App using Jinja2 templates, HTML, vanilla Javascript, and a modern "Glassmorphism" CSS UI. It fetches available teams, populates dropdowns, and asynchronously fetches predictions from the FastAPI backend.

## Run Instructions
To run the server locally:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000` to interact with the UI.
