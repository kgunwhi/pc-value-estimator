import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Create output directory
def gen_cat_analysis_plots(project_root):
    output_dir = os.path.join(project_root, "plots/cat_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    cpu_df = pd.read_csv("../data/cpu_clean.csv")
    gpu_df = pd.read_csv("../data/gpu_clean.csv")

    # Load models
    cpu_model = CatBoostRegressor()
    gpu_model = CatBoostRegressor()
    cpu_model.load_model("../model/cpu_price_model_catboost.cbm")
    gpu_model.load_model("../model/gpu_price_model_catboost.cbm")

    # Prepare features
    cpu_features = cpu_df[["PassMark_Score", "ValueScore", "Rank", "Brand"]].copy()
    gpu_features = gpu_df[["PassMark_Score", "ValueScore", "Rank", "Brand"]].copy()

    # True prices
    cpu_true = cpu_df["Price"]
    gpu_true = gpu_df["Price"]

    for df in [cpu_features, gpu_features]:
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].fillna("Unknown").astype(str)


    # Predict
    cpu_pred_log = cpu_model.predict(cpu_features)
    gpu_pred_log = gpu_model.predict(gpu_features)

    cpu_pred = np.expm1(cpu_pred_log)
    gpu_pred = np.expm1(gpu_pred_log)

    # Residuals
    cpu_residuals = cpu_true - cpu_pred
    gpu_residuals = gpu_true - gpu_pred


    # Actual vs predicted
    plt.scatter(cpu_true, cpu_pred, alpha=0.6)
    plt.plot([cpu_true.min(), cpu_true.max()], [cpu_true.min(), cpu_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("CPU: Actual vs Predicted")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cpu_actual_vs_pred.png"))
    plt.show()

    plt.scatter(gpu_true, gpu_pred, alpha=0.6)
    plt.plot([gpu_true.min(), gpu_true.max()], [gpu_true.min(), gpu_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("GPU: Actual vs Predicted")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "gpu_actual_vs_pred.png"))
    plt.show()



    # Residuals vs Predicted
    plt.scatter(cpu_pred, cpu_residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("CPU: Residuals vs Predicted")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cpu_Residual_vs_pred.png"))
    plt.show()


    plt.scatter(gpu_pred, gpu_residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("GPU: Residuals vs Predicted")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "gpu_Residual_vs_pred.png"))
    plt.show()


    # QQ Plot
    stats.probplot(cpu_residuals, dist="norm", plot=plt)
    plt.title("QQ Plot of CPU Residuals")
    plt.savefig(os.path.join(output_dir, "cpu_QQ.png"))
    plt.show()

    stats.probplot(gpu_residuals, dist="norm", plot=plt)
    plt.title("QQ Plot of GPU Residuals")
    plt.savefig(os.path.join(output_dir, "gpu_QQ.png"))
    plt.show()


    # Get importance and feature names
    importances_cpu = cpu_model.get_feature_importance()
    feature_names = cpu_features.columns

    # Plot
    plt.figure(figsize=(6, 3))
    plt.barh(feature_names, importances_cpu, color='skyblue')
    plt.xlabel("Importance Score")
    plt.title("CPU Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cpu_importance.png"))
    plt.show()


    importances_gpu = gpu_model.get_feature_importance()
    feature_names_gpu = gpu_features.columns

    plt.figure(figsize=(6, 3))
    plt.barh(feature_names_gpu, importances_gpu, color='salmon')
    plt.xlabel("Importance Score")
    plt.title("GPU Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_importance.png"))
    plt.show()

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("====== Generating Analysis Plots ======")
    gen_cat_analysis_plots(project_root)
    print("====== Complete ======")