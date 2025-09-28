# Data Science Portfolio

Welcome to my **Data Science Portfolio**.  
This repository showcases end-to-end data science projects demonstrating expertise in machine learning, predictive modeling, feature engineering, and data-driven decision-making.  

Each project is fully reproducible, with code, synthetic datasets, and reports included.  
The portfolio is structured so recruiters, hiring managers, and peers can explore real-world, production-style workflows.

## 👨‍💻 About Me
Data Science professional with 4+ years of software development and analytics experience.  
Currently pursuing **M.Sc. in Data Science** at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).  

I specialize in **machine learning, predictive modeling, and feature engineering** using Python, scikit-learn, and modern data science workflows.  
My portfolio highlights **end-to-end data pipelines**, from synthetic data generation to model deployment and visualization.  

## 🛠️ Technical Skills
- **Programming & Data Handling:** Python, SQL, R, Pandas, NumPy  
- **Machine Learning & AI:** Scikit-learn, Random Forest, Gradient Boosting, Neural Networks  
- **Visualization & Reporting:** Matplotlib, Seaborn, Plotly  
- **Data Engineering:** Feature Engineering, Synthetic Data Generation, Data Preprocessing  
- **Tools & Workflow:** Jupyter, Git, Docker, Virtual Environments, Model Serialization (joblib)  

## 📊 Featured Projects

### 1. [Customer Churn Prediction — Project 01](01-customer-churn/)
**End-to-end churn prediction pipeline (classification)**  
- **Tech Stack:** Python, scikit-learn, Pandas, Matplotlib, Seaborn  
- **Data:** public Telco CSV (recommended) or small `sample_data.csv` shipped in the repo (schema example). Large/full processed data and generated files are **not** committed.  
- **Pipeline:** Data acquisition → preprocessing → feature engineering → model training → evaluation & visualizations.  
- **Models:** Logistic Regression, Random Forest, Gradient Boosting (results & plots saved in `reports/`)  
- **Deliverables:** `reports/` (metrics + visuals), `models/` (model artifacts), `features/` (train/test files) — *generated locally, not committed*.

**Key Files**
- `01-customer-churn/00_get_data.py` — download/generate dataset  
- `01-customer-churn/02_preprocessing.py` — feature engineering & preprocessing  
- `01-customer-churn/03_train_model.py` — model training & metrics export  
- `01-customer-churn/04_evaluate_visualize.py` — ROC, confusion matrix, feature importance PNGs  
- `01-customer-churn/sample_data.csv` — small example schema for quick review on GitHub

**Quick highlight:** Example model metrics shown in project README (ROC_AUC ≈ 0.65 on the small example dataset). See project README for full metrics, interpretation and visuals.

---

### 2. [Predictive Maintenance — Industrial Equipment Failure Prediction  - Project 02](02-predictive-maintenance/)
**End-to-end machine learning pipeline for industrial predictive maintenance**  
- **Tech Stack:** Python, scikit-learn, Pandas, Matplotlib, Seaborn  
- **Data:** Synthetic sensor data generated programmatically (machines, sensor readings, failure logs)  
- **Pipeline:** Data generation → rolling-window feature engineering → model training → evaluation & visualization  
- **Models:** Random Forest classifier (evaluated with ROC, confusion matrix, feature importance)  
- **Results:** Example ROC AUC reported in project README; results, plots, and model artifact saved under `02-predictive-maintenance/reports/` and `02-predictive-maintenance/models/` (local artifacts may be large and are not tracked if gitignored).

**Key Files**
- `02-predictive-maintenance/00_generate_sensor_data.py`  
- `02-predictive-maintenance/01_create_features.py`  
- `02-predictive-maintenance/02_train_model.py`  
- `02-predictive-maintenance/03_evaluate_visualize.py`  
- `02-predictive-maintenance/reports/` — contains ROC, confusion matrix, metrics summary

---

### 3. [Text Sentiment Analysis — Multilingual Reviews  - Project 03(English + German)](03-text-sentiment/)
**End-to-end text classification pipeline (sentiment analysis)**  
- **Tech Stack:** Python, scikit-learn, Pandas, Matplotlib, Seaborn  
- **Data:** Synthetic multilingual reviews (English + German) generated if no real dataset provided.  
- **Pipeline:** Data acquisition → preprocessing (cleaning + train/test split) → TF-IDF vectorization → model training → evaluation & interpretability.  
- **Models:** Logistic Regression (baseline) and Random Forest (both scored perfectly on synthetic dataset).  
- **Deliverables:** `reports/metrics.csv`, `reports/predictions_test.csv`, `reports/visuals/` (ROC curves, confusion matrix, top influential words), `models/` (vectorizer + trained models).

**Key Files**
- `03-text-sentiment/00_get_data.py` — generate/load reviews dataset  
- `03-text-sentiment/01_preprocess.py` — text cleaning + train/test split  
- `03-text-sentiment/02_train_model.py` — TF-IDF + model training, metrics export  
- `03-text-sentiment/03_evaluate_visualize.py` — ROC, confusion matrix, top word features (saved as PNGs)

**Quick highlight:** Both models achieved **perfect metrics (AUC = 1.0)** on the synthetic dataset.  
This demonstrates pipeline correctness; with real-world reviews, results will be less perfect but more realistic.


### How to reproduce 
Each project contains a README explaining step-by-step how to install requirements and run the pipeline. Typical commands from repo root:

```bash
# example (from repo root)
pip install -r 01-customer-churn/requirements.txt
python 01-customer-churn/00_get_data.py
python 01-customer-churn/02_preprocessing.py
python 01-customer-churn/03_train_model.py
python 01-customer-churn/04_evaluate_visualize.py
```

---

### 🚀 Repository Status & Notes

This portfolio is **actively maintained**.  
To keep the repository lightweight and friendly:

- Large generated files (processed data, feature CSVs, trained models) are **not committed**.  
- Each project provides scripts to **recreate all outputs locally** (`reports/`, `features/`, `models/`).  
- Small sample CSVs and visuals (PNG) are included so results can be previewed directly on GitHub without running code.  

---

## 📫 Contact
- **Email:** iamkeshav0781@gmail.com  
- **Location:** Erlangen, Germany  
- **LinkedIn:** ([https://linkedin.com/in/keshavjha0781](https://www.linkedin.com/in/keshav-jha-11899b255/))  

