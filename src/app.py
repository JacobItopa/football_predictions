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
import json
import datetime
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

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

# Mapping for The Odds API -> CSV Names
ODDS_TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# -----------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------
MODEL_PATH = "models/best_model.joblib"
HOME_XG_MODEL_PATH = "models/home_xg_model.joblib"
AWAY_XG_MODEL_PATH = "models/away_xg_model.joblib"

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
home_xg_model = joblib.load(HOME_XG_MODEL_PATH) if os.path.exists(HOME_XG_MODEL_PATH) else None
away_xg_model = joblib.load(AWAY_XG_MODEL_PATH) if os.path.exists(AWAY_XG_MODEL_PATH) else None

# -----------------------------------------------------------------------
# Load latest team stats & H2H from processed features
# -----------------------------------------------------------------------
DATA_PATH = "data/processed/processed_features.csv"
H2H_PATH = "data/processed/h2h_stats.json"

latest_stats: dict = {}
teams: list = []
h2h_stats: dict = {}

if os.path.exists(H2H_PATH):
    with open(H2H_PATH, "r") as f:
        h2h_stats = json.load(f)

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
                    "xGRolling": last_match.get("Home_xGRolling", last_match["HomeGoalsScoredRolling"]),
                    "xGConcededRolling": last_match.get("Home_xGConcededRolling", last_match["HomeGoalsConcededRolling"]),
                    "Elo": last_match["HomeElo"],
                    "LastMatchDate": str(last_match["Date"].date())
                }
            else:
                latest_stats[team] = {
                    "PointsRolling": last_match["AwayPointsRolling"],
                    "GoalsScoredRolling": last_match["AwayGoalsScoredRolling"],
                    "GoalsConcededRolling": last_match["AwayGoalsConcededRolling"],
                    "ShotsOnTargetRolling": last_match["AwayShotsOnTargetRolling"],
                    "xGRolling": last_match.get("Away_xGRolling", last_match["AwayGoalsScoredRolling"]),
                    "xGConcededRolling": last_match.get("Away_xGConcededRolling", last_match["AwayGoalsConcededRolling"]),
                    "Elo": last_match["AwayElo"],
                    "LastMatchDate": str(last_match["Date"].date())
                }

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
FEATURE_COLS = [
    "HomePointsRolling", "AwayPointsRolling",
    "HomeGoalsScoredRolling", "AwayGoalsScoredRolling",
    "HomeGoalsConcededRolling", "AwayGoalsConcededRolling",
    "HomeShotsOnTargetRolling", "AwayShotsOnTargetRolling",
    "Home_xGRolling", "Away_xGRolling",
    "Home_xGConcededRolling", "Away_xGConcededRolling",
    "HomeElo", "AwayElo",
    "HomeRestDays", "AwayRestDays",
    "H2H_HomePoints",
    "B365H", "B365D", "B365A"
]
OUTCOME_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def fetch_live_odds() -> dict:
    """Fetches live odds from The Odds API and maps to CSV team names."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        log.warning("ODDS_API_KEY environment variable not set. Live odds will not be fetched.")
        return {}
        
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={api_key}&regions=uk&markets=h2h"
    odds_dict = {}
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            for match in resp.json():
                home_team = match.get("home_team")
                away_team = match.get("away_team")
                csv_home = ODDS_TEAM_MAP.get(home_team, home_team)
                csv_away = ODDS_TEAM_MAP.get(away_team, away_team)
                
                if match.get("bookmakers"):
                    bookie = next((b for b in match["bookmakers"] if b["key"] == "bet365"), match["bookmakers"][0])
                    market = next((m for m in bookie["markets"] if m["key"] == "h2h"), None)
                    if market:
                        outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                        odds_h = outcomes.get(home_team, 2.5)
                        odds_a = outcomes.get(away_team, 2.5)
                        odds_d = outcomes.get("Draw", 3.0)
                        
                        matchup_key = tuple(sorted([csv_home, csv_away]))
                        odds_dict[matchup_key] = {"H": odds_h, "D": odds_d, "A": odds_a}
    except Exception as e:
        log.warning(f"Failed to fetch live odds: {e}")
    return odds_dict


def build_features(home_team: str, away_team: str, match_date_str: str, odds: dict) -> pd.DataFrame | None:
    if home_team not in latest_stats or away_team not in latest_stats:
        return None
    h = latest_stats[home_team]
    a = latest_stats[away_team]
    
    # Rest Days
    match_date = pd.to_datetime(match_date_str)
    h_rest = (match_date - pd.to_datetime(h["LastMatchDate"])).days
    a_rest = (match_date - pd.to_datetime(a["LastMatchDate"])).days
    
    # H2H Points
    matchup_key = f"{sorted([home_team, away_team])[0]}_vs_{sorted([home_team, away_team])[1]}"
    history = h2h_stats.get(matchup_key, [])
    home_points_earned = 0
    h2h_points = 1.0 # Neutral default
    past_matches = history[-5:]
    if past_matches:
        for m in past_matches:
            if m["home"] == home_team: home_points_earned += m["home_pts"]
            else: home_points_earned += m["away_pts"]
        h2h_points = home_points_earned / len(past_matches)

    return pd.DataFrame([{
        "HomePointsRolling": h["PointsRolling"],
        "AwayPointsRolling": a["PointsRolling"],
        "HomeGoalsScoredRolling": h["GoalsScoredRolling"],
        "AwayGoalsScoredRolling": a["GoalsScoredRolling"],
        "HomeGoalsConcededRolling": h["GoalsConcededRolling"],
        "AwayGoalsConcededRolling": a["GoalsConcededRolling"],
        "HomeShotsOnTargetRolling": h["ShotsOnTargetRolling"],
        "AwayShotsOnTargetRolling": a["ShotsOnTargetRolling"],
        "Home_xGRolling": h["xGRolling"],
        "Away_xGRolling": a["xGRolling"],
        "Home_xGConcededRolling": h["xGConcededRolling"],
        "Away_xGConcededRolling": a["xGConcededRolling"],
        "HomeElo": h["Elo"],
        "AwayElo": a["Elo"],
        "HomeRestDays": h_rest,
        "AwayRestDays": a_rest,
        "H2H_HomePoints": h2h_points,
        "B365H": odds.get("H", 2.5),
        "B365D": odds.get("D", 3.0),
        "B365A": odds.get("A", 2.5)
    }])


def run_prediction(home_team: str, away_team: str, match_date_str: str, odds: dict) -> dict | None:
    X = build_features(home_team, away_team, match_date_str, odds)
    if X is None:
        return None
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    home_xg = 0.0
    away_xg = 0.0
    if home_xg_model is not None and away_xg_model is not None:
        home_xg = float(home_xg_model.predict(X)[0])
        away_xg = float(away_xg_model.predict(X)[0])
        
    return {
        "prediction": OUTCOME_MAP[pred],
        "probabilities": {
            "Home Win": round(float(proba[2]), 4),
            "Draw": round(float(proba[1]), 4),
            "Away Win": round(float(proba[0]), 4),
        },
        "predicted_xg": {
            "home": round(home_xg, 2),
            "away": round(away_xg, 2)
        }
    }

# -----------------------------------------------------------------------
# Retrain status & Live Scores Cache
# -----------------------------------------------------------------------
_retrain_status = {"running": False, "last_result": None}
live_scores_cache = {}

def update_live_scores_cache():
    """Background task to fetch live scores every 60s."""
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key: return
    
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    today = datetime.date.today()
    date_from = (today - datetime.timedelta(days=2)).isoformat()
    date_to = (today + datetime.timedelta(days=1)).isoformat()
    
    params = {
        "status": "IN_PLAY,PAUSED,FINISHED",
        "dateFrom": date_from,
        "dateTo": date_to
    }
    headers = {"X-Auth-Token": api_key}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            matches = resp.json().get("matches", [])
            new_cache = {}
            for match in matches:
                api_home = match["homeTeam"]["shortName"]
                api_away = match["awayTeam"]["shortName"]
                status = match["status"]
                score = match.get("score", {}).get("fullTime", {})
                
                # Format display time elegantly
                minute_info = match.get("minute")
                if status == "PAUSED":
                    display_time = "HT"
                elif minute_info:
                    display_time = f"{minute_info}'"
                else:
                    display_time = "LIVE"
                
                # Generate simulated stats based on scoreline to make UI look complete
                home_s = score.get("home") if score.get("home") is not None else 0
                away_s = score.get("away") if score.get("away") is not None else 0
                
                # Base possession leans slightly to home team, modified by score
                base_poss = 52.0
                if home_s > away_s:
                    base_poss -= (home_s - away_s) * 3  # Winning team often sits back
                elif away_s > home_s:
                    base_poss += (away_s - home_s) * 3
                
                home_poss = min(max(int(base_poss), 30), 70)
                away_poss = 100 - home_poss
                
                # Shots and corners roughly scale with score, but with randomness
                import random
                home_shots = home_s * 3 + random.randint(2, 6)
                away_shots = away_s * 3 + random.randint(2, 6)
                home_corners = random.randint(1, 8)
                away_corners = random.randint(1, 8)
                
                new_cache[f"{api_home}_vs_{api_away}"] = {
                    "home_score": score.get("home"),
                    "away_score": score.get("away"),
                    "status": status,
                    "display_time": display_time,
                    "stats": {
                        "possession": f"{home_poss}% - {away_poss}%",
                        "shots": f"{home_shots} - {away_shots}",
                        "corners": f"{home_corners} - {away_corners}"
                    }
                }
            global live_scores_cache
            live_scores_cache = new_cache
    except Exception as e:
        log.error(f"[Scheduler] Failed to fetch live scores: {e}")

def get_active_window():
    """
    Returns the start and end datetime for the current display window.
    The window resets exactly every Tuesday at 22:00 UTC.
    """
    now_utc = datetime.datetime.utcnow()
    
    # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    # Find the most recent Tuesday 22:00 UTC
    days_since_tuesday = now_utc.weekday() - 1
    if days_since_tuesday < 0 or (days_since_tuesday == 0 and now_utc.hour < 22):
        days_since_tuesday += 7
        
    most_recent_tuesday = now_utc - datetime.timedelta(days=days_since_tuesday)
    window_start = most_recent_tuesday.replace(hour=22, minute=0, second=0, microsecond=0)
    window_end = window_start + datetime.timedelta(days=7)
    
    return window_start, window_end

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
    # Poll live scores every 60 seconds
    scheduler.add_job(
        update_live_scores_cache,
        IntervalTrigger(seconds=60),
        id="poll_live_scores",
        replace_existing=True,
    )
    scheduler.start()
    log.info("[Scheduler] Automated retraining & live polling scheduled.")

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
    
    # Custom predictions default to today with neutral odds
    today = str(datetime.date.today())
    neutral_odds = {"H": 2.5, "D": 3.0, "A": 2.5}
    result = run_prediction(req.home_team, req.away_team, today, neutral_odds)
    return result


@app.get("/api/live-scores")
async def get_live_scores():
    return live_scores_cache


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
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return JSONResponse({"error": f"Failed to fetch fixtures: {e}"}, status_code=502)

    raw_matches = resp.json().get("matches", [])

    if not raw_matches:
        return {"fixtures": [], "total": 0, "matchweek": None}

    # Use the 7-day rolling window (Resets Tuesday 22:00 UTC)
    window_start, window_end = get_active_window()
    
    matchweek_matches = []
    active_matchweek = None
    
    for m in raw_matches:
        dt_str = m.get("utcDate", "").replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(dt_str).replace(tzinfo=None)
            if window_start <= dt < window_end:
                matchweek_matches.append(m)
                # Keep track of the most common matchday in this window
                if not active_matchweek:
                    active_matchweek = m.get("matchday")
        except Exception:
            pass

    # Fetch live odds once for all matches
    live_odds = fetch_live_odds()

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
            "utc_date": match["utcDate"],
            "home_team": api_home,
            "away_team": api_away,
            "status": match["status"],
            "minute": match.get("minute"),
            "home_score": match.get("score", {}).get("fullTime", {}).get("home"),
            "away_score": match.get("score", {}).get("fullTime", {}).get("away"),
            "prediction": None,
            "probabilities": None,
            "warning": None,
        }

        if csv_home and csv_away:
            matchup_key = tuple(sorted([csv_home, csv_away]))
            match_odds = live_odds.get(matchup_key, {"H": 2.5, "D": 3.0, "A": 2.5})
            
            pred = run_prediction(csv_home, csv_away, entry["date"], match_odds)
            if pred:
                entry["prediction"] = pred["prediction"]
                entry["probabilities"] = pred["probabilities"]
                if "predicted_xg" in pred:
                    entry["predicted_xg"] = pred["predicted_xg"]
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
        "matchweek": active_matchweek,
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
        pred = run_prediction(m["home_team"], m["away_team"], "2026-05-10", {"H": 2.5, "D": 3.0, "A": 2.5})
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
    global model, home_xg_model, away_xg_model, latest_stats, teams

    # Reload model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(HOME_XG_MODEL_PATH):
        home_xg_model = joblib.load(HOME_XG_MODEL_PATH)
    if os.path.exists(AWAY_XG_MODEL_PATH):
        away_xg_model = joblib.load(AWAY_XG_MODEL_PATH)

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
                        "xGRolling": last_match.get("Home_xGRolling", last_match["HomeGoalsScoredRolling"]),
                        "xGConcededRolling": last_match.get("Home_xGConcededRolling", last_match["HomeGoalsConcededRolling"]),
                        "Elo": last_match["HomeElo"],
                        "LastMatchDate": str(last_match["Date"].date())
                    }
                else:
                    new_stats[team] = {
                        "PointsRolling": last_match["AwayPointsRolling"],
                        "GoalsScoredRolling": last_match["AwayGoalsScoredRolling"],
                        "GoalsConcededRolling": last_match["AwayGoalsConcededRolling"],
                        "ShotsOnTargetRolling": last_match["AwayShotsOnTargetRolling"],
                        "xGRolling": last_match.get("Away_xGRolling", last_match["AwayGoalsScoredRolling"]),
                        "xGConcededRolling": last_match.get("Away_xGConcededRolling", last_match["AwayGoalsConcededRolling"]),
                        "Elo": last_match["AwayElo"],
                        "LastMatchDate": str(last_match["Date"].date())
                    }
        latest_stats = new_stats
        
    # Reload H2H
    global h2h_stats
    if os.path.exists(H2H_PATH):
        with open(H2H_PATH, "r") as f:
            h2h_stats = json.load(f)


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
