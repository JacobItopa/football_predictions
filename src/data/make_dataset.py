import os
import glob
import pandas as pd

def parse_date(date_series):
    # football-data.co.uk dates can be either dd/mm/yy or dd/mm/yyyy
    return pd.to_datetime(date_series, format='mixed', dayfirst=True)

def merge_raw_data(input_dir, output_file):
    print("Finding raw CSV files...")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    all_files = [f for f in all_files if not f.endswith("xg_data.csv")]
    print(f"Found {len(all_files)} matching raw files.")
    
    df_list = []
    
    # We only care about essential columns for our prediction model
    # Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR
    # HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR
    columns_to_keep = [
        'Div', 'Date', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'FTR', 
        'HTHG', 'HTAG', 'HTR',
        'HS', 'AS', 'HST', 'AST', 
        'HF', 'AF', 'HC', 'AC', 
        'HY', 'AY', 'HR', 'AR',
        'B365H', 'B365D', 'B365A',
        'AvgH', 'AvgD', 'AvgA'
    ]
    
    for file in all_files:
        try:
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
            
        # Ensure the columns exist before keeping them
        cols_present = [c for c in columns_to_keep if c in df.columns]
        df = df[cols_present].copy()
        
        # Drop rows where FTR is missing (match didn't happen or missing data)
        if 'FTR' in df.columns:
            df = df.dropna(subset=['FTR'])
            
        df_list.append(df)
        
    print("Concatenating files...")
    merged_df = pd.concat(df_list, ignore_index=True)
    
    print("Parsing dates...")
    merged_df['Date'] = parse_date(merged_df['Date'])
    
    # Sort by date
    merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)
    
    # Drop rows without HomeTeam or AwayTeam
    merged_df = merged_df.dropna(subset=['HomeTeam', 'AwayTeam', 'Date'])
    
    # Merge Expected Goals (xG) data if available
    xg_path = os.path.join(input_dir, "xg_data.csv")
    if os.path.exists(xg_path):
        print("Merging xG data...")
        xg_df = pd.read_csv(xg_path)
        xg_df['Date'] = pd.to_datetime(xg_df['Date'])
        
        merged_df = pd.merge(
            merged_df, 
            xg_df, 
            on=['Date', 'HomeTeam', 'AwayTeam'], 
            how='left'
        )
        print("xG data merged successfully.")
    else:
        print("Warning: xg_data.csv not found in raw data. xG features will be missing.")
    
    print(f"Merged dataset shape: {merged_df.shape}")
    merged_df.to_csv(output_file, index=False)
    print(f"Saved interim data to {output_file}")

if __name__ == "__main__":
    RAW_DATA_DIR = os.path.join("data", "raw")
    INTERIM_DATA_FILE = os.path.join("data", "interim", "merged_data.csv")
    merge_raw_data(RAW_DATA_DIR, INTERIM_DATA_FILE)
