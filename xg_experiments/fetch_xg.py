import soccerdata as sd
import pandas as pd

def fetch_test_xg():
    print("Initializing soccerdata Understat scraper for 23/24 season...")
    
    understat = sd.Understat(leagues="ENG-Premier League", seasons=["2324"])
    
    print("Available methods on Understat:")
    print([m for m in dir(understat) if not m.startswith('_')])
    
    print("\nReading match results (schedule)...")
    try:
        df = understat.read_schedule()
        print("\nData Shape:", df.shape)
        print("\nSample Columns:")
        print(df.columns.tolist())
        print("\nSample Data:")
        print(df.head())
        df.head(20).to_csv("understat_sample.csv")
        print("\nSaved sample to understat_sample.csv")
    except Exception as e:
        print("Failed to read results:", e)
    


if __name__ == "__main__":
    fetch_test_xg()
