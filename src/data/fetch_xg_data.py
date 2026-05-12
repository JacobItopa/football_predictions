import os
import soccerdata as sd
import pandas as pd
from datetime import datetime

TEAM_MAPPING = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': "Nott'm Forest",
    'Wolverhampton Wanderers': 'Wolves',
    'Sheffield United': 'Sheffield United',
    'Leeds United': 'Leeds',
    'Leicester City': 'Leicester',
    'Norwich City': 'Norwich',
    'Cardiff City': 'Cardiff',
    'Swansea City': 'Swansea',
    'Stoke City': 'Stoke',
    'Hull City': 'Hull',
    'Birmingham City': 'Birmingham',
    'Queens Park Rangers': 'QPR',
    'West Bromwich Albion': 'West Brom',
    'Huddersfield Town': 'Huddersfield',
    'Charlton Athletic': 'Charlton',
    'Bolton Wanderers': 'Bolton',
    'Blackburn Rovers': 'Blackburn',
    'Wigan Athletic': 'Wigan',
    'Luton': 'Luton',
    'Ipswich Town': 'Ipswich'
}

def fetch_xg_history():
    print("Initializing soccerdata Understat scraper...")
    # Fetch data for the last 6 seasons to have enough history for rolling averages
    # Format for seasons is e.g. "1920" for 2019/2020
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # If we are before August, the current season started last year
    start_year = current_year - 1 if current_month < 8 else current_year
    
    seasons = []
    for i in range(6):
        y1 = str(start_year - i)[-2:]
        y2 = str(start_year - i + 1)[-2:]
        seasons.append(f"{y1}{y2}")
    
    print(f"Fetching seasons: {seasons}")
    
    understat = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
    
    print("Reading match schedule with xG...")
    df = understat.read_schedule()
    
    # Clean up the dataframe
    df = df.reset_index()
    
    # Map team names to match football-data.co.uk format
    df['home_team'] = df['home_team'].replace(TEAM_MAPPING)
    df['away_team'] = df['away_team'].replace(TEAM_MAPPING)
    
    # Parse dates to standard format
    df['Date'] = pd.to_datetime(df['date']).dt.date
    
    # Rename columns to match what we need
    df = df.rename(columns={
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_xg': 'Home_xG',
        'away_xg': 'Away_xG'
    })
    
    # Keep only what we need
    columns_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'Home_xG', 'Away_xG']
    final_df = df[columns_to_keep]
    
    # Ensure raw data directory exists
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    
    output_path = os.path.join("data", "raw", "xg_data.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"Successfully saved {len(final_df)} xG records to {output_path}")

if __name__ == "__main__":
    fetch_xg_history()
