"""
Preprocess reviews: simple cleaning and train/test split.
Outputs:
 - features/train_texts.csv, features/train_labels.csv
 - features/test_texts.csv, features/test_labels.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

ROOT = Path(".")
DATA_IN = ROOT / "data/processed/reviews.csv"
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[@#]\S+", " ", s)
    s = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß\s]", " ", s)  # keep basic German characters
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(test_size=0.2, random_state=42):
    if not DATA_IN.exists():
        raise FileNotFoundError(f"Input CSV not found: {DATA_IN}. Run 00_get_data.py first.")
    df = pd.read_csv(DATA_IN)
    print(f"Loaded dataset: {DATA_IN} (rows: {len(df):,})")

    # Basic cleaning
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    # train/test split (stratify on label)
    X = df["text_clean"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save
    
    X_train.to_csv(FEATURES_DIR / "train_texts.csv", index=False, header=True)
    y_train.to_csv(FEATURES_DIR / "train_labels.csv", index=False, header=True)
    X_test.to_csv(FEATURES_DIR / "test_texts.csv", index=False, header=True)
    y_test.to_csv(FEATURES_DIR / "test_labels.csv", index=False, header=True)

    print(f"Train/test split: {len(X_train):,} train | {len(X_test):,} test")
    print("Saved preprocessed texts and labels under:", FEATURES_DIR)

if __name__ == "__main__":
    main()
