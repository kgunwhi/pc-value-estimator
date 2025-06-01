# src/preproc.py

import os
import pandas as pd

def clean_data():
    """
    Loads raw PassMark CSVs, cleans/filter them, and writes out:
      - data/cpu_clean.csv
      - data/gpu_clean.csv

    Returns:
        cpu_df (DataFrame): cleaned CPU data
        gpu_df (DataFrame): cleaned GPU data
    """
    # ─── 1) Compute paths ────────────────────────────────────────────────────────
    # __file__ is ".../pc-value-estimator/src/preproc.py"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir       = os.path.join(project_root, "data")  # where cpu_passmark.csv / gpu_passmark.csv live
    clean_dir     = raw_dir  # we’ll save cpu_clean.csv & gpu_clean.csv here too
    print(project_root)
    print(raw_dir)
    print(clean_dir)

    cpu_raw_path = os.path.join(raw_dir, "cpu_passmark.csv")
    gpu_raw_path = os.path.join(raw_dir, "gpu_passmark.csv")

    # ─── 2) Load raw data ─────────────────────────────────────────────────────────
    cpu_df = pd.read_csv(cpu_raw_path)
    gpu_df = pd.read_csv(gpu_raw_path)

    # ─── 3) Drop missing entries ─────────────────────────────────────────────────
    cpu_df.dropna(inplace=True)
    gpu_df.dropna(inplace=True)

    # ─── 4) Filter out unwanted rows ──────────────────────────────────────────────
    #    e.g. engineering samples or “Unknown” parts
    cpu_df = cpu_df[~cpu_df["CPU"].str.contains("Engineering Sample|Unknown", case=False)]
    gpu_df = gpu_df[~gpu_df["GPU"].str.contains("Unknown", case=False)]

    # ─── 5) Example business rule: remove CPUs priced over $1000 (if “Price” exists)
    #    If your raw CPU DataFrame has a “Price” column, you can filter out expensive outliers:
    if "Price" in cpu_df.columns:
        cpu_df = cpu_df[cpu_df["Price"] <= 1000]

    # ─── 6) Extract “Brand” from the CPU/GPU name via regex ────────────────────────
    cpu_df["Brand"] = cpu_df["CPU"].str.extract(r"^(Intel|AMD)", expand=False)
    gpu_df["Brand"] = gpu_df["GPU"].str.extract(r"^(NVIDIA|AMD)", expand=False)

    # ─── 7) Ensure the data directory exists before writing clean CSVs ─────────────
    os.makedirs(clean_dir, exist_ok=True)

    # ─── 8) Write out the cleaned CSVs ────────────────────────────────────────────
    cpu_clean_path = os.path.join(clean_dir, "cpu_clean.csv")
    gpu_clean_path = os.path.join(clean_dir, "gpu_clean.csv")

    cpu_df.to_csv(cpu_clean_path, index=False)
    gpu_df.to_csv(gpu_clean_path, index=False)

    print(f"✔ Saved cleaned CPU data to {cpu_clean_path}")
    print(f"✔ Saved cleaned GPU data to {gpu_clean_path}")

    return cpu_df, gpu_df
