

import argparse
import os
try:
    import joblib
except ImportError:
    try:
        # Older scikit-learn versions vendored joblib under sklearn.externals
        from sklearn.externals import joblib  # type: ignore
    except Exception:
        raise ImportError("Missing required dependency 'joblib'. Install it with: pip install joblib")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DEFAULT_FILES = [
    "heart.csv",
    "/mnt/data/heart.csv",
    "/mnt/data/heart (1).csv",
    "./heart.csv"
]

def find_data_path(provided_path=None):
    if provided_path and os.path.exists(provided_path):
        return provided_path
    for p in DEFAULT_FILES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find data file. Provide --data path.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default="model_pipeline.joblib", help="Output filename for saved pipeline")
    parser.add_argument("--target", type=str, default="target", help="Name of the target column in the CSV (default: target)")
    args = parser.parse_args()

    data_path = find_data_path(args.data)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data columns: {df.columns.tolist()}")

    # Basic housekeeping
    df = df.copy()
    target_col = args.target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If categorical object columns exist, try to convert to numeric where reasonable
    for c in X.columns:
        if X[c].dtype == object:
            try:
                X[c] = pd.to_numeric(X[c])
            except:
                # fallback: simple factorize (numeric codes)
                X[c] = pd.factorize(X[c])[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: imputer -> scaler -> classifier
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save pipeline and feature names (so app knows input order)
    to_save = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "target_name": target_col
    }
    joblib.dump(to_save, args.out)
    print(f"Saved trained pipeline and meta to: {args.out}")

if __name__ == "__main__":
    main()
