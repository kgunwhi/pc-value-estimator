import os
import pandas as pd
import numpy as np

def remove_price_outliers(df, price_col="Price"):
    """
    Removes outliers in the Price column using IQR method.
    """
    Q1 = df[price_col].quantile(0.25)
    Q3 = df[price_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[price_col] >= lower) & (df[price_col] <= upper)]


def load_data(project_root):
    """
    Loads raw CPU and GPU benchmark CSVs.
    """
    raw_dir = os.path.join(project_root, "data")
    cpu_raw_path = os.path.join(raw_dir, "cpu_passmark.csv")
    gpu_raw_path = os.path.join(raw_dir, "gpu_passmark.csv")

    cpu_df = pd.read_csv(cpu_raw_path)
    gpu_df = pd.read_csv(gpu_raw_path)
    return cpu_df, gpu_df


def clean_data(cpu_df, gpu_df, project_root):
    """
    Cleans raw CPU and GPU DataFrames, removes outliers, and saves cleaned CSVs.
    """
    # Drop missing entries
    cpu_df.dropna(inplace=True)
    gpu_df.dropna(inplace=True)

    # Remove engineering/unknown parts
    cpu_df = cpu_df[~cpu_df["CPU"].str.contains("Engineering Sample|Unknown", case=False)]
    gpu_df = gpu_df[~gpu_df["GPU"].str.contains("Unknown", case=False)]

    # Remove price outliers
    if "Price" in cpu_df.columns:
        cpu_df = remove_price_outliers(cpu_df)
    if "Price" in gpu_df.columns:
        gpu_df = remove_price_outliers(gpu_df)

    # Extract brand info
    cpu_df["Brand"] = cpu_df["CPU"].str.extract(r"^(Intel|AMD)", expand=False)
    gpu_df["Brand"] = gpu_df["GPU"].str.extract(r"^(NVIDIA|AMD)", expand=False)

    # Save cleaned data
    clean_dir = os.path.join(project_root, "data")
    os.makedirs(clean_dir, exist_ok=True)

    cpu_clean_path = os.path.join(clean_dir, "cpu_clean.csv")
    gpu_clean_path = os.path.join(clean_dir, "gpu_clean.csv")

    cpu_df.to_csv(cpu_clean_path, index=False)
    gpu_df.to_csv(gpu_clean_path, index=False)

    print(f"✔ Saved cleaned CPU data to {cpu_clean_path}")
    print(f"✔ Saved cleaned GPU data to {gpu_clean_path}")

    return cpu_df, gpu_df


def preprocess_for_catboost(df):
    """
    Prepares features and target for CatBoost. Assumes 'Brand' column exists.
    """
    df = df.copy()
    df["Brand"] = df["Brand"].fillna("Unknown")

    features = ["PassMark_Score", "ValueScore", "Rank", "Brand"]
    X = df[features]
    y = np.log1p(df["Price"])  # log-transformed target
    cat_features = ["Brand"]

    return X, y, cat_features
