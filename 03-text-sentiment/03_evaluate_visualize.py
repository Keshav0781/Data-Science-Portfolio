"""
Create evaluation visuals: ROC curves, confusion matrix, top logistic features.
Saves PNGs under reports/visuals/
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import joblib

ROOT = Path(".")
REPORTS = ROOT / "reports"
VIS_DIR = REPORTS / "visuals"
VIS_DIR.mkdir(parents=True, exist_ok=True)

def plot_roc(y_true, prob_dict, outpath):
    plt.figure(figsize=(8,6))
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)

def plot_confusion(y_true, y_pred, outpath, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)

def top_logreg_features(model_path, vectorizer_path, outpath, top_n=20):
    vec = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    feature_names = np.array(vec.get_feature_names_out())
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        idx_pos = np.argsort(coefs)[-top_n:][::-1]
        idx_neg = np.argsort(coefs)[:top_n]
        top_pos = feature_names[idx_pos]
        top_neg = feature_names[idx_neg]
        # bar chart combined
        vals = np.concatenate([coefs[idx_pos], coefs[idx_neg]])
        labels = np.concatenate([top_pos, top_neg])
        plt.figure(figsize=(8,6))
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, np.concatenate([coefs[idx_neg][::-1], coefs[idx_pos]]))
        plt.yticks(y_pos, np.concatenate([top_neg[::-1], top_pos]))
        plt.title("Top Logistic Regression Features (neg -> pos)")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        print("Saved", outpath)

def main():
    preds = pd.read_csv(REPORTS / "predictions_test.csv")
    y_true = preds["label"].values
    prob_dict = {}
    if "prob_logreg" in preds.columns:
        prob_dict["LogisticRegression"] = preds["prob_logreg"].values
    if "prob_rf" in preds.columns:
        prob_dict["RandomForest"] = preds["prob_rf"].values

    # ROC
    plot_roc(y_true, prob_dict, VIS_DIR / "roc_curve.png")

    # Confusion matrices (LogReg preferred since interpretable)
    if "pred_logreg" in preds.columns:
        plot_confusion(y_true, preds["pred_logreg"].values, VIS_DIR / "confusion_matrix_logreg.png",
                       title="Confusion Matrix (LogisticRegression)")

    # Feature importance from logistic
    top_logreg_features(ROOT / "models" / "model_logreg.joblib", ROOT / "models" / "vectorizer.joblib", 
                        VIS_DIR / "top_logreg_features.png", top_n=20)

if __name__ == "__main__":
    main()
