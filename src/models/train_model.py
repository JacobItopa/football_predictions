import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_evaluate_models(data_path, models_dir):
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    
    # We will sort by date to maintain chronological order, though it should already be sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    
    # Define features and target
    feature_cols = [
        'HomePointsRolling', 'AwayPointsRolling',
        'HomeGoalsScoredRolling', 'AwayGoalsScoredRolling',
        'HomeGoalsConcededRolling', 'AwayGoalsConcededRolling',
        'HomeShotsOnTargetRolling', 'AwayShotsOnTargetRolling',
        'Home_xGRolling', 'Away_xGRolling',
        'Home_xGConcededRolling', 'Away_xGConcededRolling',
        'HomeElo', 'AwayElo',
        'HomeRestDays', 'AwayRestDays',
        'H2H_HomePoints',
        'B365H', 'B365D', 'B365A'
    ]
    
    X = df[feature_cols]
    y = df['Target']
    
    # Chronological train-test split (80% train, 20% test)
    # We don't shuffle because future matches shouldn't predict past matches
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training set: {X_train.shape[0]} matches")
    print(f"Testing set: {X_test.shape[0]} matches")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    best_model = None
    best_acc = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"[{name}] Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=['Away (0)', 'Draw (1)', 'Home (2)']))
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name

    print(f"\n======================================")
    print(f"Best Model: {best_model_name} with Accuracy: {best_acc:.4f}")
    print(f"======================================")
    
    # Save the best model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # ---------------------------------------------------------
    # Train xG Regressors
    # ---------------------------------------------------------
    if 'Home_xG' in df.columns and 'Away_xG' in df.columns:
        print("\n======================================")
        print("Training xG Regressors...")
        print("======================================")
        
        xg_df = df.dropna(subset=['Home_xG', 'Away_xG'])
        X_xg = xg_df[feature_cols]
        y_home_xg = xg_df['Home_xG']
        y_away_xg = xg_df['Away_xG']
        
        # We can use the same chronological split
        _, _, y_train_hxG, y_test_hxG = train_test_split(X_xg, y_home_xg, test_size=0.2, shuffle=False)
        _, _, y_train_axG, y_test_axG = train_test_split(X_xg, y_away_xg, test_size=0.2, shuffle=False)
        X_train_xg, X_test_xg, _, _ = train_test_split(X_xg, y_home_xg, test_size=0.2, shuffle=False)
        
        # Train Home xG Regressor
        home_xg_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        home_xg_model.fit(X_train_xg, y_train_hxG)
        hxG_preds = home_xg_model.predict(X_test_xg)
        print(f"[Home xG] MAE: {mean_absolute_error(y_test_hxG, hxG_preds):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test_hxG, hxG_preds)):.4f}")
        
        # Train Away xG Regressor
        away_xg_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        away_xg_model.fit(X_train_xg, y_train_axG)
        axG_preds = away_xg_model.predict(X_test_xg)
        print(f"[Away xG] MAE: {mean_absolute_error(y_test_axG, axG_preds):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test_axG, axG_preds)):.4f}")
        
        # Save regressors
        home_xg_path = os.path.join(models_dir, "home_xg_model.joblib")
        away_xg_path = os.path.join(models_dir, "away_xg_model.joblib")
        joblib.dump(home_xg_model, home_xg_path)
        joblib.dump(away_xg_model, away_xg_path)
        print(f"Saved Home xG model to {home_xg_path}")
        print(f"Saved Away xG model to {away_xg_path}")

if __name__ == "__main__":
    PROCESSED_DATA_FILE = os.path.join("data", "processed", "processed_features.csv")
    MODELS_DIR = "models"
    train_and_evaluate_models(PROCESSED_DATA_FILE, MODELS_DIR)
