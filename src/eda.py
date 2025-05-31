import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure seaborn styling
sns.set(style="whitegrid")

def plot_score_and_price_distributions(cpu_df, gpu_df):
    """
    Plots distributions of PassMark scores and prices for CPUs and GPUs.
    """
    # CPU Score
    plt.figure(figsize=(8, 4))
    sns.histplot(cpu_df["PassMark_Score"], bins=50, kde=True)
    plt.title("CPU Score Distribution")
    plt.xlabel("PassMark Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # CPU Price
    plt.figure(figsize=(8, 4))
    sns.histplot(cpu_df["Price"], bins=40, kde=True)
    plt.title("CPU Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # GPU Score
    plt.figure(figsize=(8, 4))
    sns.histplot(gpu_df["PassMark_Score"], bins=50, kde=True)
    plt.title("GPU Score Distribution")
    plt.xlabel("PassMark Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # GPU Price
    plt.figure(figsize=(8, 4))
    sns.histplot(gpu_df["Price"], bins=40, kde=True)
    plt.title("GPU Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_price_vs_performance(cpu_df, gpu_df):
    """
    Plots price vs PassMark score for CPUs and GPUs.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=cpu_df, x="PassMark_Score", y="Price")
    plt.title("CPU: Price vs. Performance")
    plt.xlabel("PassMark Score")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=gpu_df, x="PassMark_Score", y="Price")
    plt.title("GPU: Price vs. Performance")
    plt.xlabel("PassMark Score")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.show()

def show_top_value_components(cpu_df, gpu_df, top_n=10):
    """
    Prints top N CPUs and GPUs sorted by value score.
    """
    print(f"\nTop {top_n} CPUs by Value:")
    print(cpu_df.sort_values("ValueScore", ascending=False)[["CPU", "PassMark_Score", "Price", "ValueScore"]].head(top_n))

    print(f"\nTop {top_n} GPUs by Value:")
    print(gpu_df.sort_values("ValueScore", ascending=False)[["GPU", "PassMark_Score", "Price", "ValueScore"]].head(top_n))

