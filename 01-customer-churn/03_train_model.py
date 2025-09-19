"""
Author: Keshav Jha
Train baseline models for customer churn prediction.
Handles class imbalance by using class_weight="balanced".
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

# Input data
X_train = pd.read_csv(FEATURES_DIR / "train_features.csv")
y_train = pd.read_csv(FEATURES_DIR / "train_labels.csv").values.ravel()
X_test = pd.read_csv(FEATURES_DIR / "test_features.csv")
y_test = pd.read_csv(FEATURES_DIR / "test_labels.csv").values.ravel()

# Models to train (with class_weight balancing)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)  # no class_weight param, will leave default
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)

    # Save model
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")

# Save metrics
df_results = pd.DataFrame(results)
df_results.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
print("\nModel training complete. Metrics saved to reports/model_metrics.csv")
print(df_results.to_string(index=False))

