"""
Train ML models for Customer Churn Prediction
Author: Keshav Jha
Description:
 - Loads preprocessed features (from 02_preprocessing.py)
 - Trains multiple classifiers: Logistic Regression, RandomForest, GradientBoosting
 - Evaluates them on the test set
 - Saves metrics and trained models
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths
FEATURES_DIR = Path("01-customer-churn/features")
MODELS_DIR = Path("01-customer-churn/models")
REPORTS_DIR = Path("01-customer-churn/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    X_train = pd.read_csv(FEATURES_DIR / "train_features.csv")
    y_train = pd.read_csv(FEATURES_DIR / "train_labels.csv").values.ravel()
    X_test = pd.read_csv(FEATURES_DIR / "test_features.csv")
    y_test = pd.read_csv(FEATURES_DIR / "test_labels.csv").values.ravel()
    return X_train, y_train, X_test, y_test

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    return metrics

def main():
    X_train, y_train, X_test, y_test = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

    # Save metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    print("\nModel training complete. Metrics saved to reports/model_metrics.csv")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
