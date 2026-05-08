"""
src/retrain_pipeline.py
========================
Full continuous-training pipeline:
  1. Fetch latest finished matches -> update interim data
  2. Rebuild features (rolling averages + Elo)
  3. Retrain 3 models -> save best model

Usage:
    python src/retrain_pipeline.py

Can also be triggered via the FastAPI /retrain endpoint.
"""

import os
import sys
import json
import importlib
from datetime import datetime

# -- Paths --------------------------------------------------------------
INTERIM_DATA_FILE = os.path.join("data", "interim", "merged_data.csv")
PROCESSED_DATA_FILE = os.path.join("data", "processed", "processed_features.csv")
MODELS_DIR = "models"
LOG_FILE = os.path.join("reports", "retrain_log.json")


def run_pipeline(season: int | None = None) -> dict:
    """
    Execute the full update -> feature build -> retrain cycle.
    Returns a dict summarising results (for the API endpoint).
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "new_matches_added": 0,
        "dataset_size": 0,
        "model_results": {},
        "best_model": "",
        "best_accuracy": 0.0,
        "errors": [],
    }

    # -- Step 1: Fetch & append latest results ---------------------------
    print("\n" + "=" * 60)
    print("STEP 1 - Fetching latest finished match results")
    print("=" * 60)
    try:
        # Ensure project root is on path for sibling module imports
        import importlib.util, sys as _sys
        _root = os.path.abspath(".")
        if _root not in _sys.path:
            _sys.path.insert(0, _root)

        _spec = importlib.util.spec_from_file_location(
            "update_results",
            os.path.join(_root, "src", "data", "update_results.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        new_rows = _mod.update_interim_data(season)
        result["new_matches_added"] = new_rows
    except Exception as e:
        msg = f"Data update failed: {e}"
        print(f"ERROR: {msg}")
        result["errors"].append(msg)
        # Don't abort - rebuild features and retrain on existing data anyway

    # -- Step 2: Rebuild features -----------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 - Rebuilding rolling features & Elo ratings")
    print("=" * 60)
    try:
        import pandas as pd
        import importlib.util, sys as _sys
        _root = os.path.abspath(".")
        if _root not in _sys.path:
            _sys.path.insert(0, _root)

        _spec = importlib.util.spec_from_file_location(
            "build_features",
            os.path.join(_root, "src", "features", "build_features.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.main(INTERIM_DATA_FILE, PROCESSED_DATA_FILE)

        df = pd.read_csv(PROCESSED_DATA_FILE)
        result["dataset_size"] = len(df)
        print(f"Feature dataset: {len(df):,} rows, {df.shape[1]} columns")
    except Exception as e:
        msg = f"Feature build failed: {e}"
        print(f"ERROR: {msg}")
        result["errors"].append(msg)
        return result  # Cannot retrain without features

    # -- Step 3: Retrain models --------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 - Retraining models")
    print("=" * 60)
    try:
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        df = pd.read_csv(PROCESSED_DATA_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        feature_cols = [
            "HomePointsRolling", "AwayPointsRolling",
            "HomeGoalsScoredRolling", "AwayGoalsScoredRolling",
            "HomeGoalsConcededRolling", "AwayGoalsConcededRolling",
            "HomeShotsOnTargetRolling", "AwayShotsOnTargetRolling",
            "HomeElo", "AwayElo",
        ]

        X = df[feature_cols]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric="mlogloss", verbosity=0
            ),
        }

        best_model = None
        best_acc = 0.0
        best_name = ""

        for name, mdl in models.items():
            print(f"\n  Training {name}...")
            mdl.fit(X_train, y_train)
            acc = accuracy_score(y_test, mdl.predict(X_test))
            print(f"  [{name}] Accuracy: {acc:.4f}")
            result["model_results"][name] = round(acc, 4)
            if acc > best_acc:
                best_acc = acc
                best_model = mdl
                best_name = name

        result["best_model"] = best_name
        result["best_accuracy"] = round(best_acc, 4)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"\n[OK] Best model: {best_name} ({best_acc:.4f}) -> saved to {model_path}")

    except Exception as e:
        msg = f"Model training failed: {e}"
        print(f"ERROR: {msg}")
        result["errors"].append(msg)

    # -- Log results -------------------------------------------------------
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                logs = json.load(f)
        logs.append(result)
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"\nRetrain log saved -> {LOG_FILE}")
    except Exception:
        pass

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    result = run_pipeline(args.season)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  New matches added : {result['new_matches_added']}")
    print(f"  Dataset size      : {result['dataset_size']:,}")
    print(f"  Best model        : {result['best_model']} ({result['best_accuracy']:.2%})")
    if result["errors"]:
        print(f"  Errors            : {result['errors']}")
