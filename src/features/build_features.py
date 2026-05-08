import os
import pandas as pd
import numpy as np
import numpy as np

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
            elif col == 'Points':
                # Win = 3, Draw = 1, Loss = 0
                team_matches[col] = 0
                team_matches.loc[(team_matches['HomeTeam'] == team) & (team_matches['FTR'] == 'H'), col] = 3
                team_matches.loc[(team_matches['AwayTeam'] == team) & (team_matches['FTR'] == 'A'), col] = 3
                team_matches.loc[team_matches['FTR'] == 'D', col] = 1
        
        # Calculate rolling averages (shift by 1 to not include current match)
        for col in stats_cols:
            team_matches[f'{col}Rolling{window}'] = team_matches[col].rolling(window=window).mean().shift(1)
            
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
        
    # Drop rows with NaN (the first few matches of each team)
    df = df.dropna().reset_index(drop=True)
    
    # Map Target Variable FTR to numbers: H=2, D=1, A=0
    target_mapping = {'H': 2, 'D': 1, 'A': 0}
    df['Target'] = df['FTR'].map(target_mapping)
    
    return df

def main(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    stats_to_roll = ['Points', 'GoalsScored', 'GoalsConceded', 'Shots', 'ShotsOnTarget']
    
    print("Creating features...")
    processed_df = create_rolling_features(df, stats_to_roll, window=5)
    
    print("Calculating Elo ratings...")
    processed_df = calculate_elo(processed_df)
    
    # Select final columns for modeling
    final_columns = [
        'Date', 'HomeTeam', 'AwayTeam', 
        'HomePointsRolling', 'AwayPointsRolling',
        'HomeGoalsScoredRolling', 'AwayGoalsScoredRolling',
        'HomeGoalsConcededRolling', 'AwayGoalsConcededRolling',
        'HomeShotsOnTargetRolling', 'AwayShotsOnTargetRolling',
        'HomeElo', 'AwayElo',
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
