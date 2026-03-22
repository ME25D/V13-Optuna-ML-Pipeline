# ⚙️ V13 Advanced ML Pipeline & Decoupled Optuna Optimization

## 🧠 Architecture Overview
An advanced, end-to-end machine learning pipeline designed to handle highly imbalanced tabular datasets. The core architecture relies on decoupled hyperparameter tuning and dynamic thresholding to maximize predictive performance (F1-Score/Recall).

## 🛠️ Tech Stack
* **Algorithms:** LightGBM, CatBoost, XGBoost
* **Optimization Framework:** Optuna
* **Data Engineering:** Scikit-Learn, Pandas, NumPy

## 🎯 Core Mechanisms
* **Decoupled Optimization:** Separates the feature engineering engine from the hyperparameter tuning loops, allowing isolated and rapid experimentation.
* **Dynamic Thresholding:** Automatically adjusts decision boundaries for heavily imbalanced classes.
* **Cross-Validation Strategy:** Stratified K-Fold implementation ensuring zero data leakage during training.
