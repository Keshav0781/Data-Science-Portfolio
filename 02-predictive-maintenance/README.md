# Predictive Maintenance — Project 02

## 📌 Aim
The goal of this project is to build a **reproducible predictive maintenance pipeline** that simulates industrial sensor data, engineers time-window features, and trains a classification model to predict equipment failures within a short horizon.  
The pipeline also generates evaluation artifacts and stakeholder-ready visualizations, helping industries reduce downtime and optimize maintenance schedules.

## 📊 Dataset
This project uses a **synthetic dataset generated programmatically** (no private or proprietary data).  
The generator simulates **seasonality, trends, noise, and realistic pre-failure anomalies** to mimic industrial IoT sensor readings.

Outputs are stored locally in `02-predictive-maintenance/data/processed/`:

- `machines.csv` → machine metadata (IDs, type, age, etc.)  
- `sensor_readings.csv` → time series of sensor values (`timestamp`, `machine_id`, `sensor_1..sensor_4`)  
- `failures.csv` → recorded failure events per machine  

## 🔧 Pipeline & Files
The predictive maintenance workflow is structured into modular scripts:

- `00_generate_sensor_data.py` → generates synthetic sensor time series and failure events.  
- `01_create_features.py` → engineers rolling-window statistical features and creates failure labels.  
- `02_train_model.py` → trains a **RandomForest classifier**, evaluates performance, and saves the model + metrics.  
- `03_evaluate_visualize.py` → produces evaluation plots: ROC curve, confusion matrix, and feature importance charts.  
- `requirements.txt` → list of Python dependencies for reproducibility.  

## ▶️ How to Run
After cloning the repository and navigating to the project folder:

```bash
# 1. Install dependencies
pip install -r 02-predictive-maintenance/requirements.txt

# 2. Generate synthetic sensor data
python 02-predictive-maintenance/00_generate_sensor_data.py

# 3. Create features and labels
python 02-predictive-maintenance/01_create_features.py

# 4. Train the model and save metrics + model file
python 02-predictive-maintenance/02_train_model.py

# 5. Generate evaluation visualizations (ROC, confusion matrix, feature importances)
python 02-predictive-maintenance/03_evaluate_visualize.py

```

## 📊 Results & Outputs

The predictive maintenance pipeline produced the following outputs:

- **Performance Metrics** (see `reports/metrics.txt`):
  - ROC AUC: **0.9999**
  - Accuracy: **99.98%**
  - Precision/Recall (class 1 – failure):
    - Precision: 1.0000  
    - Recall: 0.9375  
    - F1-score: 0.9677  

- **Evaluation Visuals** (saved in `reports/visuals/`):
  - `roc_curve.png` — ROC curve showing excellent discrimination capability.
  - `confusion_matrix.png` — High accuracy with very few false negatives.
  - `feature_importances.png` — Highlights which sensors contribute most to predictions.

- **Model Artifact**:
  - Trained Random Forest model saved at `models/predictive_model.joblib`.

These results demonstrate that the model effectively predicts machine failures within the defined horizon, supporting proactive maintenance planning.

## 💡 Business Impact & Learnings

- **Proactive Maintenance**:  
  The model enables early detection of equipment issues, reducing unplanned downtime and costly emergency repairs.  

- **Resource Optimization**:  
  By predicting failures in advance, maintenance schedules can be better planned, saving labor hours and spare parts costs.  

- **Scalability**:  
  The pipeline is modular and can be adapted to real sensor streams from IoT devices in manufacturing plants.  

- **Key Learnings**:  
  - Synthetic data generation is useful to simulate industrial scenarios without relying on sensitive datasets.  
  - Rolling-window feature engineering significantly improves predictive power in time-series failure data.  
  - Model interpretability (via feature importance) is critical to gaining stakeholder trust in AI-driven solutions.  
## 🚀 Next Steps / Future Work

- Extend the dataset with additional synthetic sensors to simulate more complex machinery.  
- Experiment with advanced models (XGBoost, LSTMs for sequence data) to improve recall on rare failures.  
- Integrate real-time data ingestion and streaming prediction for live maintenance dashboards.  
- Deploy the trained model as an API service for integration with industrial systems.  
