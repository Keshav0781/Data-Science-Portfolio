"""
Generate synthetic sensor data for predictive maintenance demo.
Outputs:
 - data/processed/machines.csv
 - data/processed/sensor_readings.csv
 - data/processed/failures.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

OUT_DIR = "02-predictive-maintenance/data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_machines(n_machines=50):
    machines = []
    for i in range(n_machines):
        machines.append({
            "machine_id": f"M_{i+1:03d}",
            "install_date": (datetime(2020,1,1) + pd.to_timedelta(np.random.randint(0, 365*2), unit='D')).strftime("%Y-%m-%d"),
            "machine_type": np.random.choice(["A", "B", "C"])
        })
    return pd.DataFrame(machines)

def generate_sensor_timeseries(machines_df, days=365, freq='D'):
    # Create timestamps
    start = datetime(2022, 1, 1)
    periods = days
    timestamps = pd.date_range(start, periods=periods, freq=freq)

    records = []
    failures = []

    for _, row in machines_df.iterrows():
        machine_id = row['machine_id']
        # base sensor levels (different per machine)
        base = np.random.uniform(20, 50, size=4)
        trend = np.random.uniform(-0.01, 0.02)  # small drift
        # choose whether this machine will have a failure event in the period
        will_fail = np.random.rand() < 0.25  # 25% machines fail in timeframe
        fail_day = None
        if will_fail:
            fail_day = np.random.choice(range(int(periods * 0.3), int(periods * 0.95)))
            fail_date = timestamps[fail_day]
            failures.append({"machine_id": machine_id, "failure_date": fail_date.strftime("%Y-%m-%d")})

        for t_idx, ts in enumerate(timestamps):
            # seasonal weekly cycle and random noise
            seasonal = 2.0 * np.sin(2 * np.pi * (t_idx % 30) / 30.0)  # monthly-ish seasonality
            sensors = base + trend * t_idx + seasonal + np.random.normal(0, 1.0, size=4)

            # if near failure, inject anomaly pattern (rising variance/drift)
            if will_fail and fail_day is not None:
                days_to_fail = fail_day - t_idx
                # when days_to_fail between 0 and 7, we simulate pre-failure anomaly
                if 0 <= days_to_fail <= 7:
                    sensors += np.random.normal(5 + (7 - days_to_fail), 2.0, size=4)

            records.append({
                "machine_id": machine_id,
                "timestamp": ts.strftime("%Y-%m-%d"),
                "sensor_1": round(sensors[0], 3),
                "sensor_2": round(sensors[1], 3),
                "sensor_3": round(sensors[2], 3),
                "sensor_4": round(sensors[3], 3)
            })

    sensor_df = pd.DataFrame(records)
    failures_df = pd.DataFrame(failures)
    return sensor_df, failures_df

def main():
    machines_df = generate_machines(n_machines=50)
    machines_df.to_csv(os.path.join(OUT_DIR, "machines.csv"), index=False)
    print(f"Saved machines -> {os.path.join(OUT_DIR, 'machines.csv')}")

    sensor_df, failures_df = generate_sensor_timeseries(machines_df, days=730, freq='D')
    sensor_path = os.path.join(OUT_DIR, "sensor_readings.csv")
    sensor_df.to_csv(sensor_path, index=False)
    print(f"Saved sensor readings -> {sensor_path}")

    if not failures_df.empty:
        failures_path = os.path.join(OUT_DIR, "failures.csv")
        failures_df.to_csv(failures_path, index=False)
        print(f"Saved failures -> {failures_path}")
    else:
        print("No failures generated in this run (rare).")

if __name__ == "__main__":
    main()

