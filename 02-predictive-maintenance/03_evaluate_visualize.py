"""
Produce evaluation visualizations: ROC, confusion matrix, feature importance.
Requires model and last test split saved by training step.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "02-predictive-maintenance/models/predictive_model.joblib"
FEAT_PATH = "02-predictive-maintenance/features/features_df.csv"
REPORT_DIR = "02-predictive-maintenance/reports"
VIS_DIR = os.path.join(REPORT_DIR, "visuals")
os.makedirs(VIS_DIR, exist_ok=True)

def load_test_data():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    y_test = test_df["label"]
    X_test = test_df.drop(columns=["machine_id","timestamp","label"]).fillna(0)
    return X_test, y_test

def plot_roc(y_test, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)

def plot_confusion(y_test, y_pred, out_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)

def plot_feature_importances(model, feature_names, out_path, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]
    plt.figure(figsize=(8,6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    # ROC
    plot_roc(y_test, y_proba, os.path.join(VIS_DIR, "roc_curve.png"))
    # Confusion
    plot_confusion(y_test, y_pred, os.path.join(VIS_DIR, "confusion_matrix.png"))
    # Feature importances
    plot_feature_importances(model, X_test.columns.tolist(), os.path.join(VIS_DIR, "feature_importances.png"))

if __name__ == "__main__":
    main()

