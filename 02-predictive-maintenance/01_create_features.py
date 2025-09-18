"""
Create rolling features and labels for predictive maintenance.
Reads:
 - data/processed/sensor_readings.csv
 - data/processed/failures.csv
Writes:
 - features/features_df.csv
"""

import os
import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

RAW_DIR = "02-predictive-maintenance/data/processed"
OUT_DIR = "02-predictive-maintenance/features"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON_DAYS = 7  # label: failure within next 7 days
ROLL_WINDOWS = [3, 7, 14]  # days for rolling stats

def load_data():
    sensors = pd.read_csv(os.path.join(RAW_DIR, "sensor_readings.csv"), parse_dates=["timestamp"])
    sensors = sensors.sort_values(["machine_id", "timestamp"])
    failures_path = os.path.join(RAW_DIR, "failures.csv")
    if os.path.exists(failures_path):
        failures = pd.read_csv(failures_path, parse_dates=["failure_date"])
    else:
        failures = pd.DataFrame(columns=["machine_id", "failure_date"])
    return sensors, failures

def create_labels(sensors, failures):
    # merge failure dates per machine
    failures_map = failures.groupby("machine_id")["failure_date"].min().to_dict()
    sensors["failure_date"] = sensors["machine_id"].map(failures_map)
    # label: does failure occur within next HORIZON_DAYS?
    sensors["timestamp_date"] = pd.to_datetime(sensors["timestamp"])
    sensors["label"] = 0
    mask = ~sensors["failure_date"].isna()
    sensors.loc[mask, "failure_date"] = pd.to_datetime(sensors.loc[mask, "failure_date"])
    sensors.loc[mask, "days_to_failure"] = (sensors.loc[mask, "failure_date"] - sensors.loc[mask, "timestamp_date"]).dt.days
    sensors.loc[mask & (sensors["days_to_failure"] >= 0) & (sensors["days_to_failure"] <= HORIZON_DAYS), "label"] = 1
    sensors["label"] = sensors["label"].fillna(0).astype(int)
    sensors = sensors.drop(columns=["failure_date", "days_to_failure"])
    return sensors

def engineer_features(sensors):
    sensors = sensors.sort_values(["machine_id", "timestamp"])
    feats = []
    numeric_cols = ["sensor_1","sensor_2","sensor_3","sensor_4"]
    # groupby machine and compute rolling windows
    for machine_id, group in sensors.groupby("machine_id"):
        g = group.copy().set_index("timestamp")
        for w in ROLL_WINDOWS:
            roll = g[numeric_cols].rolling(window=w, min_periods=1)
            g[[f"{c}_ma_{w}" for c in numeric_cols]] = roll.mean()
            g[[f"{c}_std_{w}" for c in numeric_cols]] = roll.std().fillna(0)
        # add current values (already present)
        g = g.reset_index()
        feats.append(g)
    feats_df = pd.concat(feats, ignore_index=True)
    # keep relevant columns
    keep_cols = ["machine_id","timestamp"] + numeric_cols + \
                [c for c in feats_df.columns if any(s in c for s in ["_ma_","_std_"])] + ["label"]
    feats_df = feats_df[keep_cols]
    # drop rows with all NaNs in rolling (shouldn't happen)
    feats_df = feats_df.dropna(subset=["sensor_1"])
    return feats_df

def main():
    sensors, failures = load_data()
    sensors_with_labels = create_labels(sensors, failures)
    features_df = engineer_features(sensors_with_labels)
    out_path = os.path.join(OUT_DIR, "features_df.csv")
    features_df.to_csv(out_path, index=False)
    print(f"Saved features -> {out_path}")
    print("Sample feature preview:")
    print(features_df.head().T)

if __name__ == "__main__":
    main()

