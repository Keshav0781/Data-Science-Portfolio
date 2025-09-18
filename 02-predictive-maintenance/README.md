# Predictive Maintenance — Project 02

## Aim
Build a reproducible predictive maintenance pipeline that simulates industrial sensor data, engineers time-window features, trains a classification model to predict equipment failure within a short horizon, and produces evaluation artifacts and visualizations suitable for stakeholder review.

## Dataset
- Synthetic dataset is generated programmatically (no private data).
- Outputs saved locally to `02-predictive-maintenance/data/processed/`:
  - `machines.csv` (metadata)
  - `sensor_readings.csv` (time series: timestamp, machine_id, sensor_1..sensor_4)
  - `failures.csv` (failure events per machine)
- The generator simulates seasonality, trend, noise, and realistic pre-failure anomalies.

## Pipeline & Files
- `00_generate_sensor_data.py` — generate sensor timeseries + failure events.
- `01_create_features.py` — create rolling-window features and label (failure within horizon).
- `02_train_model.py` — train RandomForest classifier, evaluate and save model & metrics.
- `03_evaluate_visualize.py` — plot ROC, confusion matrix, feature importances and save PNGs.
- `requirements.txt` — dependency list.

## How to run (after pulling repo to local)
```bash
# from repo root
pip install -r 02-predictive-maintenance/requirements.txt

# generate synthetic data
python 02-predictive-maintenance/00_generate_sensor_data.py

# create features and labels
python 02-predictive-maintenance/01_create_features.py

# train model and save metrics/model
python 02-predictive-maintenance/02_train_model.py

# generate evaluation visualizations
python 02-predictive-maintenance/03_evaluate_visualize.py

