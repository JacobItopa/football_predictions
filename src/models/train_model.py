import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

if __name__ == "__main__":
    PROCESSED_DATA_FILE = os.path.join("data", "processed", "processed_features.csv")
    MODELS_DIR = "models"
    train_and_evaluate_models(PROCESSED_DATA_FILE, MODELS_DIR)
