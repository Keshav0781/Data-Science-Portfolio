"""
Train TF-IDF + Logistic Regression and RandomForest (optional).
Saves:
 - models/vectorizer.joblib
 - models/model_logreg.joblib
 - models/model_rf.joblib
 - reports/metrics.csv
 - reports/predictions_test.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

ROOT = Path(".")
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    X_train = pd.read_csv(FEATURES_DIR / 
"train_texts.csv").squeeze("columns")
    y_train = pd.read_csv(FEATURES_DIR / 
"train_labels.csv").squeeze("columns")
    X_test = pd.read_csv(FEATURES_DIR / 
"test_texts.csv").squeeze("columns")
    y_test = pd.read_csv(FEATURES_DIR / 
"test_labels.csv").squeeze("columns")
    return X_train, y_train, X_test, y_test

def evaluate_and_save(name, y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc}

def main():
    X_train, y_train, X_test, y_test = load_features()
    print("Loaded features. Training sizes:", len(X_train), len(X_test))

    # Vectorize (fit on train)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_t, y_train)
    pred_lr = logreg.predict(X_test_t)
    prob_lr = logreg.predict_proba(X_test_t)[:, 1]

    # Random Forest (optional)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_t, y_train)
    pred_rf = rf.predict(X_test_t)
    prob_rf = rf.predict_proba(X_test_t)[:, 1]

    # Evaluate
    results = []
    results.append(evaluate_and_save("LogisticRegression", y_test, pred_lr, y_score=prob_lr))
    results.append(evaluate_and_save("RandomForest", y_test, pred_rf, y_score=prob_rf))

    metrics_df = pd.DataFrame(results).round(4)
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    print("Saved metrics to:", REPORTS_DIR / "metrics.csv")
    print(metrics_df.to_string(index=False))

    # Save predictions (probabilities and predicted labels)
    preds_df = pd.DataFrame({
        "text": X_test,
        "label": y_test,
        "pred_logreg": pred_lr,
        "prob_logreg": prob_lr,
        "pred_rf": pred_rf,
        "prob_rf": prob_rf
    })
    preds_df.to_csv(REPORTS_DIR / "predictions_test.csv", index=False)
    print("Saved predictions to:", REPORTS_DIR / "predictions_test.csv")

    # Save models + vectorizer
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    joblib.dump(logreg, MODELS_DIR / "model_logreg.joblib")
    joblib.dump(rf, MODELS_DIR / "model_rf.joblib")
    print("Saved models to:", MODELS_DIR)

if __name__ == "__main__":
    main()
