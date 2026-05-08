import os
import requests
import pandas as pd
import json

# -----------------------------------------------------------------
# Get a FREE API key at https://www.football-data.org/client/register
# Then either set the environment variable FOOTBALL_DATA_API_KEY
# or paste it directly below.
# -----------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.football-data.org/v4"

HEADERS = {"X-Auth-Token": API_KEY}

# Premier League competition code is PL
COMPETITION_CODE = "PL"


def fetch_upcoming_fixtures(num_matches=10) -> list[dict]:
    """Fetch the next upcoming EPL fixtures from football-data.org."""
    url = f"{BASE_URL}/competitions/{COMPETITION_CODE}/matches"
    params = {"status": "SCHEDULED"}

    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()

    data = resp.json()
    matches = data.get("matches", [])

    upcoming = []
    for match in matches[:num_matches]:
        upcoming.append({
            "matchday": match["matchday"],
            "date": match["utcDate"][:10],  # YYYY-MM-DD
            "home_team": match["homeTeam"]["shortName"],
            "away_team": match["awayTeam"]["shortName"],
            "home_team_full": match["homeTeam"]["name"],
            "away_team_full": match["awayTeam"]["name"],
        })

    return upcoming


if __name__ == "__main__":
    fixtures = fetch_upcoming_fixtures(20)
    print(f"\nUpcoming {len(fixtures)} EPL fixtures:")
    for f in fixtures:
        print(f"  [{f['date']}] MD{f['matchday']:>2}: {f['home_team']} vs {f['away_team']}")
