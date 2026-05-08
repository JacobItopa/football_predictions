"""
src/data/update_results.py
==========================
Fetches recently FINISHED EPL matches from football-data.org API
and appends any new results to data/interim/merged_data.csv.

Usage:
    python src/data/update_results.py

Set FOOTBALL_DATA_API_KEY environment variable before running.
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
COMPETITION = "PL"

# Reverse lookup: API name → CSV training name
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
    "Brighton & Hove Albion": "Brighton",
    "Brighton Hove Albion": "Brighton",
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

INTERIM_DATA_FILE = os.path.join("data", "interim", "merged_data.csv")


def fetch_finished_matches(season: int | None = None) -> list[dict]:
    """
    Fetch FINISHED EPL matches from the API.
    If season is None, fetches the current season.
    season format: 2024 means the 2024-25 season.
    """
    url = f"{BASE_URL}/competitions/{COMPETITION}/matches"
    params = {"status": "FINISHED"}
    if season:
        params["season"] = season

    print("  >> Fetching finished matches from football-data.org...")
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("matches", [])


def api_match_to_row(match: dict) -> dict | None:
    """Convert an API match object to a row matching our CSV schema."""
    score = match.get("score", {})
    ft = score.get("fullTime", {})
    ht = score.get("halfTime", {})

    fthg = ft.get("home")
    ftag = ft.get("away")

    if fthg is None or ftag is None:
        return None  # incomplete score

    if fthg > ftag:
        ftr = "H"
    elif ftag > fthg:
        ftr = "A"
    else:
        ftr = "D"

    hthg = ht.get("home", 0) or 0
    htag = ht.get("away", 0) or 0
    if hthg > htag:
        htr = "H"
    elif htag > hthg:
        htr = "A"
    else:
        htr = "D"

    home_api = match["homeTeam"]["name"]
    away_api = match["awayTeam"]["name"]

    home_csv = TEAM_NAME_MAP.get(home_api) or TEAM_NAME_MAP.get(match["homeTeam"]["shortName"], "")
    away_csv = TEAM_NAME_MAP.get(away_api) or TEAM_NAME_MAP.get(match["awayTeam"]["shortName"], "")

    if not home_csv or not away_csv:
        print(f"  [SKIP] Unmapped team '{home_api}' or '{away_api}'")
        return None

    utc_date = match["utcDate"][:10]  # YYYY-MM-DD

    return {
        "Div": "E0",
        "Date": utc_date,
        "HomeTeam": home_csv,
        "AwayTeam": away_csv,
        "FTHG": int(fthg),
        "FTAG": int(ftag),
        "FTR": ftr,
        "HTHG": int(hthg),
        "HTAG": int(htag),
        "HTR": htr,
        # Shots/Corners/Fouls not available on free API tier — left blank
        # The rolling averages will fall back to existing history.
        "HS": None, "AS": None,
        "HST": None, "AST": None,
        "HF": None, "AF": None,
        "HC": None, "AC": None,
        "HY": None, "AY": None,
        "HR": None, "AR": None,
    }


def update_interim_data(season: int | None = None) -> int:
    """
    Main function. Fetches finished matches and appends new ones to
    the interim CSV. Returns the number of new rows added.
    """
    if not API_KEY:
        print("ERROR: FOOTBALL_DATA_API_KEY environment variable not set.")
        sys.exit(1)

    # ── 1. Load existing data ────────────────────────────────────────────
    print(f"Loading existing data from {INTERIM_DATA_FILE}...")
    existing_df = pd.read_csv(INTERIM_DATA_FILE)
    existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.strftime("%Y-%m-%d")

    # Create a set of (Date, HomeTeam, AwayTeam) to detect duplicates
    existing_keys = set(
        zip(existing_df["Date"], existing_df["HomeTeam"], existing_df["AwayTeam"])
    )
    print(f"  Existing rows: {len(existing_df):,}")

    # ── 2. Fetch finished matches from API ───────────────────────────────
    raw_matches = fetch_finished_matches(season)
    print(f"  API returned {len(raw_matches)} finished matches.")

    # ── 3. Convert & deduplicate ─────────────────────────────────────────
    new_rows = []
    skipped_dup = 0
    skipped_map = 0

    for m in raw_matches:
        row = api_match_to_row(m)
        if row is None:
            skipped_map += 1
            continue
        key = (row["Date"], row["HomeTeam"], row["AwayTeam"])
        if key in existing_keys:
            skipped_dup += 1
            continue
        new_rows.append(row)
        existing_keys.add(key)  # prevent duplicates within the new batch

    print(f"  >> New matches to add: {len(new_rows)}")
    print(f"  >> Duplicates skipped: {skipped_dup}")
    print(f"  >> Unmapped skipped:   {skipped_map}")

    if not new_rows:
        print("[OK] No new data to add. Dataset is already up to date.")
        return 0

    # ── 4. Append and save ───────────────────────────────────────────────
    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Sort by Date and save
    updated_df["Date"] = pd.to_datetime(updated_df["Date"])
    updated_df = updated_df.sort_values("Date").reset_index(drop=True)

    updated_df.to_csv(INTERIM_DATA_FILE, index=False)
    print(f"[OK] Appended {len(new_rows)} new rows -> total: {len(updated_df):,} rows")
    print(f"   Saved to: {INTERIM_DATA_FILE}")
    return len(new_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update interim data with finished matches.")
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (e.g. 2024 for 2024-25). Defaults to current season.",
    )
    args = parser.parse_args()
    update_interim_data(args.season)
