#!/usr/bin/env python3
"""
00_get_data.py
Create a customer churn dataset for Project 01 (Customer Churn).
Behavior:
 - If a user-supplied download URL is provided via --url, script will try to download and save it.
 - Otherwise (or if download fails), the script will generate a realistic synthetic dataset
   and save it to data/processed/customer_churn.csv.
"""
from pathlib import Path
import argparse
import csv
import urllib.request
import io
import sys

import numpy as np
import pandas as pd

RNG_SEED = 42


def try_download_csv(url: str, out_path: Path) -> bool:
    """Attempt to download CSV from a URL. Returns True on success, False on failure."""
    try:
        print(f"Attempting to download dataset from: {url}")
        resp = urllib.request.urlopen(url, timeout=20)
        raw = resp.read()
        # Try to parse with pandas to validate
        df = pd.read_csv(io.BytesIO(raw))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Downloaded and saved dataset to: {out_path} (rows: {len(df):,})")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def generate_synthetic_churn(n_customers: int = 7000) -> pd.DataFrame:
    """Generate a synthetic Telco-like customer churn dataset."""
    rng = np.random.default_rng(RNG_SEED)

    customer_ids = [f"CUST{100000 + i}" for i in range(1, n_customers + 1)]
    gender = rng.choice(["Male", "Female"], size=n_customers)
    senior = rng.choice([0, 1], size=n_customers, p=[0.84, 0.16]).astype(int)
    partner = rng.choice(["Yes", "No"], size=n_customers, p=[0.45, 0.55])
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(rng.choice(["Yes", "No"], p=[0.35, 0.65]))
        else:
            dependents.append(rng.choice(["Yes", "No"], p=[0.12, 0.88]))

    # tenure (months) skewed: many short tenures, fewer long-tenure customers
    tenure = rng.integers(0, 73, size=n_customers)
    phone_service = rng.choice(["Yes", "No"], size=n_customers, p=[0.9, 0.1])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.3, 0.7]))

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_customers, p=[0.35, 0.45, 0.20])

    # For internet-based services, set feature to Yes/No or "No internet service"
    def internet_feature_array():
        arr = []
        for s in internet_service:
            if s == "No":
                arr.append("No internet service")
            else:
                arr.append(rng.choice(["Yes", "No"], p=[0.25, 0.75]))
        return arr

    online_security = internet_feature_array()
    online_backup = internet_feature_array()
    device_protection = internet_feature_array()
    tech_support = internet_feature_array()
    streaming_tv = internet_feature_array()
    streaming_movies = internet_feature_array()

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_customers, p=[0.55, 0.25, 0.20])
    paperless = rng.choice(["Yes", "No"], size=n_customers, p=[0.6, 0.4])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_customers,
        p=[0.35, 0.20, 0.25, 0.20]
    )

    # Monthly charges base on subscription choices
    base_charge = np.where(internet_service == "No", 20.0, 45.0)
    phone_add = np.where(phone_service == "Yes", 10.0, 0.0)
    addon = rng.normal(loc=0.0, scale=8.0, size=n_customers)
    monthly_charges = np.round(np.clip(base_charge + phone_add + addon, 10.0, 200.0), 2)

    # Total charges roughly monthly_charges * tenure with small noise; tenure 0 -> total 0
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 20.0, size=n_customers), 2)
    total_charges = np.where(tenure == 0, 0.0, total_charges)
    total_charges = np.round(np.where(total_charges < 0, monthly_charges * tenure, total_charges), 2)

    # Churn probability: higher for month-to-month, lower for long contracts; shorter tenure increases churn
    churn_prob = np.full(n_customers, 0.05)
    churn_prob += np.where(contract == "Month-to-month", 0.20, 0.0)
    churn_prob += np.where(payment_method == "Electronic check", 0.05, 0.0)
    churn_prob += np.where(tenure < 3, 0.05, 0.0)
    churn_prob += np.where(monthly_charges > 100, 0.05, 0.0)
    churn_prob = np.clip(churn_prob, 0.01, 0.95)

    churn = rng.random(n_customers) < churn_prob
    churn = np.where(churn, "Yes", "No")

    # Assemble dataframe
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })

    # Shuffle rows for realism
    df = df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Download or generate customer churn CSV.")
    parser.add_argument("--url", type=str, default=None, help="Optional URL to a CSV to download.")
    parser.add_argument("--out", type=str, default="data/processed/customer_churn.csv", help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=7000, help="Number of synthetic rows (if generated).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If the file already exists, do nothing
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            print(f"Found existing dataset at {out_path} (rows: {len(existing):,}). No action taken.")
            return
        except Exception:
            # If existing file is unreadable, overwrite
            print(f"Existing file at {out_path} is unreadable. It will be overwritten.")

    # Try download if URL provided
    if args.url:
        success = try_download_csv(args.url, out_path)
        if success:
            return
        else:
            print("Download failed — proceeding to generate synthetic dataset.")

    # Generate synthetic dataset as fallback
    print("Generating synthetic customer churn dataset...")
    df = generate_synthetic_churn(n_customers=args.rows)
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to: {out_path} (rows: {len(df):,})")
    print("If you have a real dataset CSV, place it at this path to use real data: ", out_path.resolve())


if __name__ == "__main__":
    main()
