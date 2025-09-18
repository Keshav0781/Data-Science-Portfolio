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

### 1. [Predictive Maintenance — Industrial Equipment Failure Prediction](02-predictive-maintenance/)
**End-to-end machine learning pipeline for industrial predictive maintenance**

- **Tech Stack:** Python, Scikit-learn, Pandas, Matplotlib, Seaborn  
- **Data:** Synthetic sensor data generated programmatically (machines, sensor readings, failure logs)  
- **Pipeline:** Data generation → feature engineering → model training → evaluation & visualization  
- **Models:** Random Forest Classifier (evaluated with ROC, confusion matrix, feature importance)  
- **Results:** Achieved ROC AUC of **0.9999**, with high precision/recall for failure prediction  
- **Business Impact:** Enables proactive maintenance scheduling, reducing downtime and saving costs  

**Key Files:**
- [`00_generate_sensor_data.py`](02-predictive-maintenance/00_generate_sensor_data.py) – Synthetic data generation  
- [`01_create_features.py`](02-predictive-maintenance/01_create_features.py) – Rolling-window feature engineering  
- [`02_train_model.py`](02-predictive-maintenance/02_train_model.py) – Model training and metrics export  
- [`03_evaluate_visualize.py`](02-predictive-maintenance/03_evaluate_visualize.py) – Evaluation plots and reports  
- [`reports/`](02-predictive-maintenance/reports/) – Contains ROC, confusion matrix, metrics summary  
