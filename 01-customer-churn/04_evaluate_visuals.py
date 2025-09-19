"""
Customer Churn - Evaluation Visuals
Generates ROC curves, confusion matrices, and feature importance plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Paths
BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "visuals"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_test_data():
    """Load test features and labels."""
    X_test = pd.read_csv(FEATURES_DIR / "test_features.csv")
    y_test = pd.read_csv(FEATURES_DIR / "test_labels.csv").values.ravel()
    feature_names = open(FEATURES_DIR / "feature_names.txt").read().splitlines()
    return X_test, y_test, feature_names

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curves.png")
    plt.close()
    print("Saved: reports/visuals/roc_curves.png")

def plot_confusion_matrix(model, X_test, y_test, model_name="LogisticRegression"):
    """Plot confusion matrix for a selected model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{model_name}.png")
    plt.close()
    print(f"Saved: reports/visuals/confusion_matrix_{model_name}.png")

def plot_feature_importance(model, feature_names):
    """Plot feature importance for RandomForest (if available)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-15:]  # Top 15 features
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[idx], importances[idx])
        plt.title("RandomForest Feature Importance (Top 15)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png")
        plt.close()
        print("Saved: reports/visuals/feature_importance.png")

def main():
    X_test, y_test, feature_names = load_test_data()

    # Load models
    models = {}
    for name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    if not models:
        raise FileNotFoundError("No trained models found in models/")

    # ROC Curves
    plot_roc_curves(models, X_test, y_test)

    # Confusion Matrix (LogisticRegression if available)
    if "LogisticRegression" in models:
        plot_confusion_matrix(models["LogisticRegression"], X_test, y_test, "LogisticRegression")

    # Feature Importance (RandomForest)
    if "RandomForest" in models:
        plot_feature_importance(models["RandomForest"], feature_names)

if __name__ == "__main__":
    main()
