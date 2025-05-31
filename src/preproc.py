import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Loads, cleans, and tags CPU and GPU PassMark benchmark data.

    Returns:
        cpu_df (DataFrame): Cleaned CPU benchmark data with Brand column.
        gpu_df (DataFrame): Cleaned GPU benchmark data with Brand column.
    """
    cpu_df = pd.read_csv("data/cpu_passmark.csv")
    gpu_df = pd.read_csv("data/gpu_passmark.csv")

    # Drop missing entries
    cpu_df.dropna(inplace=True)
    gpu_df.dropna(inplace=True)

    # Filter out engineering samples or 'Unknown' parts
    cpu_df = cpu_df[~cpu_df["CPU"].str.contains("Engineering Sample|Unknown", case=False)]
    gpu_df = gpu_df[~gpu_df["GPU"].str.contains("Unknown", case=False)]

    # Save cleaned datasets
    cpu_df.to_csv("data/cpu_clean.csv", index=False)
    gpu_df.to_csv("data/gpu_clean.csv", index=False)


    return cpu_df, gpu_df

def plot_cpu_distribution(cpu_df):
    """
    Plots the distribution of CPU PassMark scores.
    """
    plt.figure(figsize=(10, 4))
    sns.histplot(cpu_df["PassMark_Score"], bins=50, kde=True)
    plt.title("CPU PassMark Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
