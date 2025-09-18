"""
Train RandomForest model for predictive maintenance.
Input: features/features_df.csv
Outputs:
 - models/predictive_model.joblib
 - reports/metrics.json
 - reports/metrics.txt (classification report)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json

SEED = 42
np.random.seed(SEED)

FEAT_PATH = "02-predictive-maintenance/features/features_df.csv"
MODEL_DIR = "02-predictive-maintenance/models"
REPORT_DIR = "02-predictive-maintenance/reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def load_features():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df

def prepare_xy(df):
    # drop non-feature cols
    y = df["label"]
    drop_cols = ["machine_id", "timestamp", "label"]
    X = df.drop(columns=drop_cols)
    X = X.fillna(0)
    return X, y

def train_and_evaluate(X, y):
    # time-based split: last 20% as test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba) if len(set(y_test))>1 else None
    cm = confusion_matrix(y_test, y_pred).tolist()

    return model, report, auc, cm, X_test, y_test, y_pred

def main():
    df = load_features()
    X, y = prepare_xy(df)
    print("Data shape:", X.shape)
    model, report, auc, cm, X_test, y_test, y_pred = train_and_evaluate(X, y)

    # save model
    model_path = os.path.join(MODEL_DIR, "predictive_model.joblib")
    joblib.dump(model, model_path)
    print("Saved model ->", model_path)

    # save metrics
    metrics = {"classification_report": report, "roc_auc": auc, "confusion_matrix": cm}
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORT_DIR, "metrics.txt"), "w") as f:
        f.write("ROC AUC: " + str(auc) + "\n\n")
        f.write("Classification Report:\n")
        f.write(pd.DataFrame(report).transpose().to_string())
    print("Saved metrics ->", REPORT_DIR)

if __name__ == "__main__":
    main()

