from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
import os
import logging
import requests
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("epl_predictor")

templates = Jinja2Templates(directory="src/templates")

# -----------------------------------------------------------------------
# Team name mapping: football-data.org short names → CSV training names
# The API uses different abbreviations from the historical CSVs.
# Add/update entries here if the API returns a name your model doesn't know.
# -----------------------------------------------------------------------
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Arsenal": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brentford": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton": "Brighton",
    "Burnley FC": "Burnley",
    "Burnley": "Burnley",
    "Chelsea FC": "Chelsea",
    "Chelsea": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton",
    "Everton": "Everton",
    "Fulham FC": "Fulham",
    "Fulham": "Fulham",
    "Leeds United FC": "Leeds",
    "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Leicester City FC": "Leicester",
    "Leicester City": "Leicester",
    "Leicester": "Leicester",
    "Liverpool FC": "Liverpool",
    "Liverpool": "Liverpool",
    "Luton Town FC": "Luton",
    "Luton Town": "Luton",
    "Luton": "Luton",
    "Manchester City FC": "Man City",
    "Manchester City": "Man City",
    "Man City": "Man City",
    "Manchester United FC": "Man United",
    "Manchester United": "Man United",
    "Man United": "Man United",
    "Newcastle United FC": "Newcastle",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield United FC": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Southampton FC": "Southampton",
    "Southampton": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "West Ham United FC": "West Ham",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolves": "Wolves",
}

# -----------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------
MODEL_PATH = "models/best_model.joblib"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# -----------------------------------------------------------------------
# Load latest team stats from processed features
# -----------------------------------------------------------------------
DATA_PATH = "data/processed/processed_features.csv"
latest_stats: dict = {}
teams: list = []

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    all_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    teams = sorted(list(all_teams))

    for team in teams:
        team_df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        if not team_df.empty:
            last_match = team_df.iloc[-1]
            if last_match["HomeTeam"] == team:
                latest_stats[team] = {
                    "PointsRolling": last_match["HomePointsRolling"],
                    "GoalsScoredRolling": last_match["HomeGoalsScoredRolling"],
                    "GoalsConcededRolling": last_match["HomeGoalsConcededRolling"],
                    "ShotsOnTargetRolling": last_match["HomeShotsOnTargetRolling"],
                    "Elo": last_match["HomeElo"],
                }
            else:
                latest_stats[team] = {
                    "PointsRolling": last_match["AwayPointsRolling"],
                    "GoalsScoredRolling": last_match["AwayGoalsScoredRolling"],
                    "GoalsConcededRolling": last_match["AwayGoalsConcededRolling"],
                    "ShotsOnTargetRolling": last_match["AwayShotsOnTargetRolling"],
                    "Elo": last_match["AwayElo"],
                }

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
FEATURE_COLS = [
    "HomePointsRolling", "AwayPointsRolling",
    "HomeGoalsScoredRolling", "AwayGoalsScoredRolling",
    "HomeGoalsConcededRolling", "AwayGoalsConcededRolling",
    "HomeShotsOnTargetRolling", "AwayShotsOnTargetRolling",
    "HomeElo", "AwayElo",
]
OUTCOME_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def build_features(home_team: str, away_team: str) -> pd.DataFrame | None:
    if home_team not in latest_stats or away_team not in latest_stats:
        return None
    h = latest_stats[home_team]
    a = latest_stats[away_team]
    return pd.DataFrame([{
        "HomePointsRolling": h["PointsRolling"],
        "AwayPointsRolling": a["PointsRolling"],
        "HomeGoalsScoredRolling": h["GoalsScoredRolling"],
        "AwayGoalsScoredRolling": a["GoalsScoredRolling"],
        "HomeGoalsConcededRolling": h["GoalsConcededRolling"],
        "AwayGoalsConcededRolling": a["GoalsConcededRolling"],
        "HomeShotsOnTargetRolling": h["ShotsOnTargetRolling"],
        "AwayShotsOnTargetRolling": a["ShotsOnTargetRolling"],
        "HomeElo": h["Elo"],
        "AwayElo": a["Elo"],
    }])


def run_prediction(home_team: str, away_team: str) -> dict | None:
    X = build_features(home_team, away_team)
    if X is None:
        return None
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "prediction": OUTCOME_MAP[pred],
        "probabilities": {
            "Home Win": round(float(proba[2]), 4),
            "Draw": round(float(proba[1]), 4),
            "Away Win": round(float(proba[0]), 4),
        },
    }

# -----------------------------------------------------------------------
# Retrain status (shared between scheduler and /retrain/status endpoint)
# -----------------------------------------------------------------------
_retrain_status = {"running": False, "last_result": None}


# -----------------------------------------------------------------------
# Scheduled Retraining  (runs automatically - not exposed in the UI)
# -----------------------------------------------------------------------
def scheduled_retrain():
    """
    Called automatically by APScheduler on the configured schedule.
    Runs the full CT pipeline silently in the background:
      1. Fetch latest finished matches from football-data.org
      2. Rebuild rolling features + Elo ratings
      3. Retrain 3 models, save the best
      4. Hot-reload the model into the running server
    """
    if _retrain_status["running"]:
        log.info("[Scheduler] Retrain already running — skipping this trigger.")
        return

    log.info("[Scheduler] Starting scheduled retrain...")
    _retrain_status["running"] = True
    try:
        import importlib.util
        _root = os.path.abspath(".")
        _spec = importlib.util.spec_from_file_location(
            "retrain_pipeline",
            os.path.join(_root, "src", "retrain_pipeline.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        result = _mod.run_pipeline()
        _retrain_status["last_result"] = result
        log.info(
            f"[Scheduler] Retrain complete — "
            f"best model: {result.get('best_model')} "
            f"({result.get('best_accuracy', 0):.2%}) | "
            f"+{result.get('new_matches_added', 0)} new matches"
        )
    except Exception as e:
        log.error(f"[Scheduler] Retrain failed: {e}")
        _retrain_status["last_result"] = {"error": str(e)}
    finally:
        _retrain_status["running"] = False
        reload_model_and_stats()


# -----------------------------------------------------------------------
# FastAPI app with lifespan (starts/stops scheduler cleanly)
# -----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- Startup ----------
    scheduler = BackgroundScheduler()
    # Run every Tuesday at 06:00 AM  (catches weekend results)
    scheduler.add_job(
        scheduled_retrain,
        CronTrigger(day_of_week="tue", hour=6, minute=0),
        id="retrain_tuesday",
        replace_existing=True,
    )
    # Run every Friday at 06:00 AM  (catches mid-week results)
    scheduler.add_job(
        scheduled_retrain,
        CronTrigger(day_of_week="fri", hour=6, minute=0),
        id="retrain_friday",
        replace_existing=True,
    )
    scheduler.start()
    log.info("[Scheduler] Automated retraining scheduled: Tue & Fri at 06:00 AM")

    yield  # Server runs here

    # ---------- Shutdown ----------
    scheduler.shutdown(wait=False)
    log.info("[Scheduler] Shut down.")


# Create app AFTER lifespan is defined, and mount static files
app = FastAPI(title="EPL Match Predictor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/static"), name="static")


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "teams": teams})


@app.get("/upcoming", response_class=HTMLResponse)
async def read_upcoming(request: Request):
    return templates.TemplateResponse("upcoming.html", {"request": request})


@app.post("/predict")
async def predict_match(req: PredictionRequest):
    if not model:
        return JSONResponse({"error": "Model not loaded."}, status_code=500)
    if req.home_team not in latest_stats or req.away_team not in latest_stats:
        return JSONResponse({"error": f"Team data not found. Known teams: {teams}"}, status_code=400)
    result = run_prediction(req.home_team, req.away_team)
    return result


@app.get("/fixtures")
async def get_upcoming_fixtures():
    """
    Fetches upcoming EPL fixtures from football-data.org and runs
    model predictions on each one automatically.
    Requires FOOTBALL_DATA_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        # Return a helpful demo response if no key is configured
        return JSONResponse(
            {
                "error": "API key not set",
                "message": (
                    "Set the FOOTBALL_DATA_API_KEY environment variable. "
                    "Get a free key at https://www.football-data.org/client/register"
                ),
                "demo_fixtures": _demo_fixtures(),
            }
        )

    headers = {"X-Auth-Token": api_key}
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    try:
        resp = requests.get(url, headers=headers, params={"status": "SCHEDULED"}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return JSONResponse({"error": f"Failed to fetch fixtures: {e}"}, status_code=502)

    raw_matches = resp.json().get("matches", [])

    if not raw_matches:
        return {"fixtures": [], "total": 0, "matchweek": None}

    # --- Only show the CURRENT (nearest) matchweek ---
    # Find the earliest matchday number among all scheduled matches
    current_matchweek = min(m["matchday"] for m in raw_matches)
    # Filter to only that matchweek
    matchweek_matches = [m for m in raw_matches if m["matchday"] == current_matchweek]

    fixtures_with_predictions = []
    for match in matchweek_matches:
        api_home = match["homeTeam"]["shortName"]
        api_away = match["awayTeam"]["shortName"]
        # Translate API names -> training dataset names
        csv_home = TEAM_NAME_MAP.get(api_home) or TEAM_NAME_MAP.get(match["homeTeam"]["name"])
        csv_away = TEAM_NAME_MAP.get(api_away) or TEAM_NAME_MAP.get(match["awayTeam"]["name"])

        entry = {
            "matchday": match["matchday"],
            "date": match["utcDate"][:10],
            "home_team": api_home,
            "away_team": api_away,
            "prediction": None,
            "probabilities": None,
            "warning": None,
        }

        if csv_home and csv_away:
            pred = run_prediction(csv_home, csv_away)
            if pred:
                entry["prediction"] = pred["prediction"]
                entry["probabilities"] = pred["probabilities"]
            else:
                entry["warning"] = f"Stats not found for '{csv_home}' or '{csv_away}'"
        else:
            missing = []
            if not csv_home: missing.append(api_home)
            if not csv_away: missing.append(api_away)
            entry["warning"] = f"Name mapping missing for: {', '.join(missing)}"

        fixtures_with_predictions.append(entry)

    return {
        "fixtures": fixtures_with_predictions,
        "total": len(fixtures_with_predictions),
        "matchweek": current_matchweek,
    }


def _demo_fixtures() -> list:
    """Return hardcoded demo fixtures so the UI works without an API key."""
    demo = [
        {"home_team": "Arsenal", "away_team": "Man City"},
        {"home_team": "Liverpool", "away_team": "Chelsea"},
        {"home_team": "Man United", "away_team": "Tottenham"},
        {"home_team": "Newcastle", "away_team": "Aston Villa"},
        {"home_team": "Brighton", "away_team": "Fulham"},
    ]
    results = []
    for i, m in enumerate(demo, 1):
        pred = run_prediction(m["home_team"], m["away_team"])
        results.append({
            "matchday": "Demo",
            "date": "2026-05-10",
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "prediction": pred["prediction"] if pred else "N/A",
            "probabilities": pred["probabilities"] if pred else None,
            "warning": "Demo data – set API key for real fixtures",
        })
    return results


# -----------------------------------------------------------------------
# Reload model & latest_stats in-memory after retraining
# -----------------------------------------------------------------------
def reload_model_and_stats():
    """Hot-reload the model and team stats from disk without restarting."""
    global model, latest_stats, teams

    # Reload model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

    # Rebuild latest_stats from updated processed features
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date")

        all_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
        teams = sorted(list(all_teams))

        new_stats: dict = {}
        for team in teams:
            team_df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
            if not team_df.empty:
                last_match = team_df.iloc[-1]
                if last_match["HomeTeam"] == team:
                    new_stats[team] = {
                        "PointsRolling": last_match["HomePointsRolling"],
                        "GoalsScoredRolling": last_match["HomeGoalsScoredRolling"],
                        "GoalsConcededRolling": last_match["HomeGoalsConcededRolling"],
                        "ShotsOnTargetRolling": last_match["HomeShotsOnTargetRolling"],
                        "Elo": last_match["HomeElo"],
                    }
                else:
                    new_stats[team] = {
                        "PointsRolling": last_match["AwayPointsRolling"],
                        "GoalsScoredRolling": last_match["AwayGoalsScoredRolling"],
                        "GoalsConcededRolling": last_match["AwayGoalsConcededRolling"],
                        "ShotsOnTargetRolling": last_match["AwayShotsOnTargetRolling"],
                        "Elo": last_match["AwayElo"],
                    }
        latest_stats = new_stats


# -----------------------------------------------------------------------
# /retrain endpoint (admin use via CLI only - not shown in UI)
# -----------------------------------------------------------------------

def _run_retrain_background():
    """Background task: update data → rebuild features → retrain models."""
    _retrain_status["running"] = True
    try:
        import importlib.util
        _root = os.path.abspath(".")
        _spec = importlib.util.spec_from_file_location(
            "retrain_pipeline",
            os.path.join(_root, "src", "retrain_pipeline.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        result = _mod.run_pipeline()
        _retrain_status["last_result"] = result
    except Exception as e:
        _retrain_status["last_result"] = {"error": str(e)}
    finally:
        _retrain_status["running"] = False
        # Hot-swap model & stats so new predictions use the updated model immediately
        reload_model_and_stats()


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Starts the full CT pipeline in the background:
    1. Fetch latest finished matches from football-data.org
    2. Append new results to training data
    3. Rebuild rolling features + Elo ratings
    4. Retrain 3 models, save the best one
    5. Hot-reload the model into the running server
    """
    if _retrain_status["running"]:
        return JSONResponse({"status": "already_running", "message": "Retraining is already in progress."}, status_code=409)

    background_tasks.add_task(_run_retrain_background)
    return {"status": "started", "message": "Retraining pipeline started in the background. Poll /retrain/status for progress."}


@app.get("/retrain/status")
async def get_retrain_status():
    """Poll this endpoint to check if retraining is still running."""
    return {
        "running": _retrain_status["running"],
        "last_result": _retrain_status["last_result"],
    }
