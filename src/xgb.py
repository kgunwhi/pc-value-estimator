# src/xgb.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def xgboost_train_cpu(cpu_df, project_root):
    """
    Train a RandomForestRegressor to predict CPU Price from PassMark_Score.

    - cpu_df: DataFrame with at least these columns:
        "PassMark_Score" (numeric) and "Price" (numeric).
    - Saves the model to model/cpu_price_model.pkl
    - Prints out CPU RMSE on the hold‐out set.
    """
    # Features & target
    X = cpu_df[["PassMark_Score"]]
    y = cpu_df["Price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Define and fit the Random Forest
    cpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
    cpu_model.fit(X_train, y_train)

    # Evaluate on test set
    cpu_preds = cpu_model.predict(X_test)
    cpu_mse = mean_squared_error(y_test, cpu_preds)
    cpu_rmse = np.sqrt(cpu_mse)
    print(f"[Random Forest] CPU Price RMSE: ${cpu_rmse:.2f}")

    # Save CPU model
    cpu_model_path = os.path.join(project_root, 'model', 'cpu_price_model.pkl')
    joblib.dump(cpu_model, cpu_model_path)
    print("▶ Saved CPU model as cpu_price_model.pkl\n")


def xgboost_train_gpu(gpu_df,project_root):
    """
    Train a RandomForestRegressor to predict GPU Price from PassMark_Score.

    - gpu_df: DataFrame with at least these columns:
        "PassMark_Score" (numeric) and "Price" (numeric).
    - Saves the model to model/gpu_price_model.pkl
    - Prints out GPU RMSE on the hold‐out set.
    """
    # Features & target
    X = gpu_df[["PassMark_Score"]]
    y = gpu_df["Price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Define and fit the Random Forest
    gpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gpu_model.fit(X_train, y_train)

    # Evaluate on test set
    gpu_preds = gpu_model.predict(X_test)
    gpu_mse = mean_squared_error(y_test, gpu_preds)
    gpu_rmse = np.sqrt(gpu_mse)
    print(f"[Random Forest] GPU Price RMSE: ${gpu_rmse:.2f}")

    gpu_model_path = os.path.join(project_root, 'model', 'gpu_price_model.pkl')
    # Save GPU model
    joblib.dump(gpu_model, gpu_model_path)
    print("▶ Saved GPU model as gpu_price_model.pkl\n")

