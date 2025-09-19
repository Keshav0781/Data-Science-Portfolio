#!/usr/bin/env python3
"""
02_preprocessing.py
Load customer churn CSV, create features, build a ColumnTransformer pipeline
(OneHotEncoder for categoricals, StandardScaler for numerics), split train/test,
save transformed CSVs and the fitted preprocessor for reuse.

Outputs:
 - 01-customer-churn/features/train_features.csv
 - 01-customer-churn/features/train_labels.csv
 - 01-customer-churn/features/test_features.csv
 - 01-customer-churn/features/test_labels.csv
 - 01-customer-churn/models/preprocessor.joblib
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT = Path.cwd()
PROJECT = ROOT / "01-customer-churn"
IN_CSV = PROJECT / "data" / "processed" / "customer_churn.csv"
FEATURE_DIR = PROJECT / "features"
MODEL_DIR = PROJECT / "models"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% test

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataset: {path} (rows: {len(df):,}, cols: {len(df.columns)})")
    return df

def normalize_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    # Map common text labels to 0/1 if present
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0}).fillna(df[target_col])
    # Ensure numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if df[target_col].isna().any():
        raise ValueError("Target column contains non-numeric values after mapping.")
    return df

def prepare_columns(df: pd.DataFrame):
    # Drop obvious identifier columns if present
    for id_col in ["customerID", "customer_id", "customerId", "customer"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
            break

    # Convert TotalCharges (or similar) to numeric if present
    for col in ["TotalCharges", "total_charges", "Total_Charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    # Identify numeric and categorical columns (exclude target)
    target = "Churn"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]
    return df, numeric_cols, categorical_cols

def build_and_save_pipeline(X_train: pd.DataFrame, numeric_cols, categorical_cols, save_path: Path):
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols))

    if not transformers:
        raise ValueError("No features found for transformation.")

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline([("preprocessor", ct)])
    pipeline.fit(X_train)
    joblib.dump(pipeline, save_path)
    # Get feature names
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        # fallback: make simple names
        feature_names = [f"f{i}" for i in range(len(pipeline.transform(X_train)[0]))]
    return pipeline, list(feature_names)

def transform_and_save(pipeline, feature_names, X, y, prefix: str):
    arr = pipeline.transform(X)
    df_trans = pd.DataFrame(arr, columns=feature_names, index=X.index)
    feat_path = FEATURE_DIR / f"{prefix}_features.csv"
    lbl_path = FEATURE_DIR / f"{prefix}_labels.csv"
    df_trans.to_csv(feat_path, index=False)
    pd.Series(y, name="Churn").to_csv(lbl_path, index=False)
    print(f"Saved {feat_path} (shape: {df_trans.shape}) and {lbl_path} (rows: {len(y)})")

def main():
    df = load_data(IN_CSV)
    df = normalize_target(df, target_col="Churn")
    df, numeric_cols, categorical_cols = prepare_columns(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    # Train-test split (stratify by target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train/test split: {len(X_train):,} train | {len(X_test):,} test")
    pipeline_path = MODEL_DIR / "preprocessor.joblib"

    pipeline, feature_names = build_and_save_pipeline(X_train, numeric_cols, categorical_cols, pipeline_path)
    print(f"Saved preprocessing pipeline to: {pipeline_path}")

    # Transform and persist
    transform_and_save(pipeline, feature_names, X_train, y_train, prefix="train")
    transform_and_save(pipeline, feature_names, X_test, y_test, prefix="test")

    # Also save a small schema file for reference
    schema_path = FEATURE_DIR / "feature_names.txt"
    schema_path.write_text("\n".join(feature_names))
    print(f"Saved feature schema to: {schema_path}")

if __name__ == "__main__":
    main()
