import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

def ensure_plot_dir(project_root, label):
    plot_dir = os.path.join(project_root, "plots", label)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_score_and_price_distributions(cpu_df, gpu_df, plot_dir, label_prefix):
    """
    Distributions of PassMark scores and prices.
    """
    for label, df in [("CPU", cpu_df), ("GPU", gpu_df)]:
        # Score Distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(df["PassMark_Score"], bins=50, kde=True)
        plt.title(f"{label} Score Distribution")
        plt.xlabel("PassMark Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label_prefix.lower()}_{label}_score_dist.png"))
        plt.close()

        # Price Distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(df["Price"], bins=40, kde=True)
        plt.title(f"{label} Price Distribution")
        plt.xlabel("Price (USD)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label_prefix.lower()}_{label}_price_dist.png"))
        plt.close()

        # Boxplot
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df["Price"])
        plt.title(f"{label} Price Boxplot")
        plt.xlabel("Price (USD)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label_prefix.lower()}_{label}_price_boxplot.png"))
        plt.close()


def plot_price_vs_performance(cpu_df, gpu_df, plot_dir, label_prefix):
    """
    Price vs performance scatter plots, linear and log-scale.
    """
    for label, df in [("CPU", cpu_df), ("GPU", gpu_df)]:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="PassMark_Score", y="Price")
        plt.title(f"{label}: Price vs. Performance")
        plt.xlabel("PassMark Score")
        plt.ylabel("Price (USD)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label_prefix.lower()}_{label}_price_vs_score.png"))
        plt.close()

        # Log Price
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="PassMark_Score", y=np.log1p(df["Price"]))
        plt.title(f"{label}: Log Price vs. Performance")
        plt.xlabel("PassMark Score")
        plt.ylabel("Log(Price + 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label_prefix.lower()}_{label}_logprice_vs_score.png"))
        plt.close()


def plot_price_performance_ratio(cpu_df, gpu_df, plot_dir, label_prefix, top_n=20):
    """
    Bar plots of price-to-performance ratio (inverse of ValueScore).
    """
    for label, df, name_col in [("CPU", cpu_df, "CPU"), ("GPU", gpu_df, "GPU")]:
        df = df.copy()
        df["PricePerScore"] = df["Price"] / df["PassMark_Score"]
        top = df.sort_values("PricePerScore").head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="PricePerScore", y=name_col, data=top, hue=name_col, palette="viridis", legend=False)

        plt.title(f"Top {top_n} {label}s with Best Price-to-Performance")
        plt.xlabel("Price / Score")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label_prefix.lower()}_{label}_price_per_score.png"))
        plt.close()



def run_full_eda(cpu_df, gpu_df, project_root, label="clean"):
    """
    Runs all EDA visualizations and saves plots under a labeled subfolder.
    """
    plot_dir = ensure_plot_dir(project_root, label)
    plot_score_and_price_distributions(cpu_df, gpu_df, plot_dir, label)
    plot_price_vs_performance(cpu_df, gpu_df, plot_dir, label)
    plot_price_performance_ratio(cpu_df, gpu_df, plot_dir, label)

