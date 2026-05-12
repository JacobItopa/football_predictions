import os
import pandas as pd
import numpy as np
import numpy as np
import json

def calculate_elo(df, base_elo=1500, k=20):
    elos = {}
    home_elos = []
    away_elos = []
    
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        if home not in elos: elos[home] = base_elo
        if away not in elos: elos[away] = base_elo
            
        elo_h = elos[home]
        elo_a = elos[away]
        
        home_elos.append(elo_h)
        away_elos.append(elo_a)
        
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        exp_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
        
        if row['FTR'] == 'H':
            act_h, act_a = 1, 0
        elif row['FTR'] == 'A':
            act_h, act_a = 0, 1
        else:
            act_h, act_a = 0.5, 0.5
            
        elos[home] = elo_h + k * (act_h - exp_h)
        elos[away] = elo_a + k * (act_a - exp_a)
        
    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    return df

def calculate_h2h(df, window=5):
    """
    Calculates the historical average points the Home team earned against the Away team 
    in their last `window` meetings.
    """
    h2h_points = []
    h2h_history = {}
    
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        matchup = tuple(sorted([home, away]))
        
        if matchup not in h2h_history:
            h2h_history[matchup] = []
            
        past_matches = h2h_history[matchup][-window:]
        
        # Calculate points home team earned in past matches
        home_points_earned = 0
        if past_matches:
            for match in past_matches:
                if match['home'] == home:
                    home_points_earned += match['home_pts']
                else:
                    home_points_earned += match['away_pts']
            h2h_points.append(home_points_earned / len(past_matches))
        else:
            h2h_points.append(1.0) # Assume neutral (1 pt average)
        
        # Add current match to history
        if row['FTR'] == 'H':
            hp, ap = 3, 0
        elif row['FTR'] == 'A':
            hp, ap = 0, 3
        else:
            hp, ap = 1, 1
            
        h2h_history[matchup].append({
            'date': str(row['Date']),
            'home': home,
            'away': away,
            'home_pts': hp,
            'away_pts': ap
        })
        
    df['H2H_HomePoints'] = h2h_points
    
    # Save h2h history for live inference
    serializable_h2h = {f"{k[0]}_vs_{k[1]}": v for k, v in h2h_history.items()}
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    with open(os.path.join("data", "processed", "h2h_stats.json"), "w") as f:
        json.dump(serializable_h2h, f)
        
    return df


def create_rolling_features(df, stats_cols, window=5):
    """
    Creates rolling average features for teams.
    """
    # Create an empty DataFrame to store features
    features_list = []
    
    # Process each team separately
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Sort dataframe by date to ensure rolling works correctly
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    team_stats_dict = {}
    for team in teams:
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        
        # Determine stats from the perspective of the team
        for col in stats_cols:
            home_col = f'H{col}' if f'H{col}' in df.columns else (f'FH{col[-1]}' if col.startswith('T') else None)
            # Custom mappings for standard football-data columns
            if col == 'GoalsScored':
                team_matches[col] = np.where(team_matches['HomeTeam'] == team, team_matches['FTHG'], team_matches['FTAG'])
            elif col == 'GoalsConceded':
                team_matches[col] = np.where(team_matches['HomeTeam'] == team, team_matches['FTAG'], team_matches['FTHG'])
            elif col == 'Shots':
                team_matches[col] = np.where(team_matches['HomeTeam'] == team, team_matches['HS'], team_matches['AS'])
            elif col == 'ShotsOnTarget':
                team_matches[col] = np.where(team_matches['HomeTeam'] == team, team_matches['HST'], team_matches['AST'])
            elif col == 'xGScored':
                # Fallback to actual goals if xG is missing
                h_xg = team_matches['Home_xG'].fillna(team_matches['FTHG'])
                a_xg = team_matches['Away_xG'].fillna(team_matches['FTAG'])
                team_matches[col] = np.where(team_matches['HomeTeam'] == team, h_xg, a_xg)
            elif col == 'xGConceded':
                h_xg_c = team_matches['Away_xG'].fillna(team_matches['FTAG'])
                a_xg_c = team_matches['Home_xG'].fillna(team_matches['FTHG'])
                team_matches[col] = np.where(team_matches['HomeTeam'] == team, h_xg_c, a_xg_c)
            elif col == 'Points':
                # Win = 3, Draw = 1, Loss = 0
                team_matches[col] = 0
                team_matches.loc[(team_matches['HomeTeam'] == team) & (team_matches['FTR'] == 'H'), col] = 3
                team_matches.loc[(team_matches['AwayTeam'] == team) & (team_matches['FTR'] == 'A'), col] = 3
                team_matches.loc[team_matches['FTR'] == 'D', col] = 1
        
        # Calculate rolling averages (shift by 1 to not include current match)
        for col in stats_cols:
            team_matches[f'{col}Rolling{window}'] = team_matches[col].rolling(window=window).mean().shift(1)
            
        # Calculate Rest Days
        team_matches['RestDays'] = team_matches['Date'].diff().dt.days
        team_matches['RestDays'] = team_matches['RestDays'].fillna(14.0) # Default for first match
            
        team_stats_dict[team] = team_matches

    # Now merge these rolling features back to the main dataframe
    df['HomePointsRolling'] = np.nan
    df['AwayPointsRolling'] = np.nan
    df['HomeGoalsScoredRolling'] = np.nan
    df['AwayGoalsScoredRolling'] = np.nan
    df['HomeGoalsConcededRolling'] = np.nan
    df['AwayGoalsConcededRolling'] = np.nan
    df['HomeShotsOnTargetRolling'] = np.nan
    df['AwayShotsOnTargetRolling'] = np.nan
    df['Home_xGRolling'] = np.nan
    df['Away_xGRolling'] = np.nan
    df['Home_xGConcededRolling'] = np.nan
    df['Away_xGConcededRolling'] = np.nan
    df['HomeRestDays'] = np.nan
    df['AwayRestDays'] = np.nan

    print("Merging rolling features back to main dataframe...")
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        h_stats = team_stats_dict[home_team].loc[index]
        a_stats = team_stats_dict[away_team].loc[index]
        
        df.at[index, 'HomePointsRolling'] = h_stats['PointsRolling5']
        df.at[index, 'AwayPointsRolling'] = a_stats['PointsRolling5']
        
        df.at[index, 'HomeGoalsScoredRolling'] = h_stats['GoalsScoredRolling5']
        df.at[index, 'AwayGoalsScoredRolling'] = a_stats['GoalsScoredRolling5']
        
        df.at[index, 'HomeGoalsConcededRolling'] = h_stats['GoalsConcededRolling5']
        df.at[index, 'AwayGoalsConcededRolling'] = a_stats['GoalsConcededRolling5']
        
        df.at[index, 'HomeShotsOnTargetRolling'] = h_stats['ShotsOnTargetRolling5']
        df.at[index, 'AwayShotsOnTargetRolling'] = a_stats['ShotsOnTargetRolling5']
        
        df.at[index, 'Home_xGRolling'] = h_stats['xGScoredRolling5']
        df.at[index, 'Away_xGRolling'] = a_stats['xGScoredRolling5']
        
        df.at[index, 'Home_xGConcededRolling'] = h_stats['xGConcededRolling5']
        df.at[index, 'Away_xGConcededRolling'] = a_stats['xGConcededRolling5']
        
        df.at[index, 'HomeRestDays'] = h_stats['RestDays']
        df.at[index, 'AwayRestDays'] = a_stats['RestDays']
        
    # Drop rows with NaN in the calculated rolling features (the first few matches of each team)
    cols_to_check = [
        'HomePointsRolling', 'AwayPointsRolling',
        'HomeGoalsScoredRolling', 'AwayGoalsScoredRolling',
        'Home_xGRolling', 'Away_xGRolling'
    ]
    df = df.dropna(subset=cols_to_check).reset_index(drop=True)
    # Map Target Variable FTR to numbers: H=2, D=1, A=0
    target_mapping = {'H': 2, 'D': 1, 'A': 0}
    df['Target'] = df['FTR'].map(target_mapping)
    
    return df

def main(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Process Bookmaker Odds
    for col in ['B365H', 'B365D', 'B365A']:
        if col not in df.columns:
            df[col] = np.nan
    if 'AvgH' in df.columns:
        df['B365H'] = df['B365H'].fillna(df['AvgH'])
        df['B365D'] = df['B365D'].fillna(df['AvgD'])
        df['B365A'] = df['B365A'].fillna(df['AvgA'])
    # Fill remaining with median
    df['B365H'] = df['B365H'].fillna(df['B365H'].median())
    df['B365D'] = df['B365D'].fillna(df['B365D'].median())
    df['B365A'] = df['B365A'].fillna(df['B365A'].median())
    
    # Ensure xG columns exist if they weren't merged (defensive)
    if 'Home_xG' not in df.columns:
        df['Home_xG'] = np.nan
        df['Away_xG'] = np.nan
        
    stats_to_roll = ['Points', 'GoalsScored', 'GoalsConceded', 'Shots', 'ShotsOnTarget', 'xGScored', 'xGConceded']
    
    print("Creating features...")
    processed_df = create_rolling_features(df, stats_to_roll, window=5)
    
    print("Calculating Elo ratings...")
    processed_df = calculate_elo(processed_df)
    
    print("Calculating Head-to-Head...")
    processed_df = calculate_h2h(processed_df, window=5)
    
    # Select final columns for modeling
    final_columns = [
        'Date', 'HomeTeam', 'AwayTeam', 
        'HomePointsRolling', 'AwayPointsRolling',
        'HomeGoalsScoredRolling', 'AwayGoalsScoredRolling',
        'HomeGoalsConcededRolling', 'AwayGoalsConcededRolling',
        'HomeShotsOnTargetRolling', 'AwayShotsOnTargetRolling',
        'Home_xGRolling', 'Away_xGRolling',
        'Home_xGConcededRolling', 'Away_xGConcededRolling',
        'HomeElo', 'AwayElo',
        'HomeRestDays', 'AwayRestDays',
        'H2H_HomePoints',
        'B365H', 'B365D', 'B365A',
        'Target', 'FTR'
    ]
    
    processed_df = processed_df[final_columns]
    
    print(f"Processed dataset shape: {processed_df.shape}")
    processed_df.to_csv(output_file, index=False)
    print(f"Saved processed features to {output_file}")

if __name__ == "__main__":
    INTERIM_DATA_FILE = os.path.join("data", "interim", "merged_data.csv")
    PROCESSED_DATA_FILE = os.path.join("data", "processed", "processed_features.csv")
    main(INTERIM_DATA_FILE, PROCESSED_DATA_FILE)
