import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data():
    """
    Loads, cleans, and tags CPU and GPU PassMark benchmark data.

    Returns:
        cpu_df (DataFrame): Cleaned CPU benchmark data with Brand column.
        gpu_df (DataFrame): Cleaned GPU benchmark data with Brand column.
    """
    cpu_df = pd.read_csv("../data/cpu_passmark.csv")
    gpu_df = pd.read_csv("../data/gpu_passmark.csv")

    # Drop missing entries
    cpu_df.dropna(inplace=True)
    gpu_df.dropna(inplace=True)

    # Filter out engineering samples or 'Unknown' parts
    cpu_df = cpu_df[~cpu_df["CPU"].str.contains("Engineering Sample|Unknown", case=False)]
    gpu_df = gpu_df[~gpu_df["GPU"].str.contains("Unknown", case=False)]

    # Save cleaned datasets
    cpu_df.to_csv("../data/cpu_clean.csv", index=False)
    gpu_df.to_csv("../data/gpu_clean.csv", index=False)

    # Remove CPUs over $1000
    cpu_df = cpu_df[cpu_df["Price"] <= 1000]

    return cpu_df, gpu_df

