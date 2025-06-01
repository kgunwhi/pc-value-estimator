import threading
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
import numpy as np

flask_app = Flask(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

cpu_model_path = os.path.join(PROJECT_ROOT, "model", "cpu_price_model_catboost.cbm")
gpu_model_path = os.path.join(PROJECT_ROOT, "model", "gpu_price_model_catboost.cbm")

cpu_model = CatBoostRegressor()
gpu_model = CatBoostRegressor()
cpu_model.load_model(cpu_model_path)
gpu_model.load_model(gpu_model_path)


@flask_app.route("/predict_cpu", methods=["POST"])
def predict_cpu():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prediction = np.expm1(cpu_model.predict(input_df))[0]
        return jsonify({"estimated_price": round(prediction, 2)})
    except Exception as err:
        print(f"[CPU ERROR] {err}")
        return jsonify({"error": str(err)}), 500


@flask_app.route("/predict_gpu", methods=["POST"])
def predict_gpu():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prediction = np.expm1(gpu_model.predict(input_df))[0]
        return jsonify({"estimated_price": round(prediction, 2)})
    except Exception as err:
        print(f"[GPU ERROR] {err}")
        return jsonify({"error": str(err)}), 500


def run_flask():
    flask_app.run(host="127.0.0.1", port=5000)


flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(1)

st.set_page_config(page_title="PC Component Value Checker", layout="centered")
st.title("💻 PC Component Value Checker")
st.markdown("Estimate a fair price for your selected CPU and GPU based on benchmark scores.")


@st.cache_data
def load_cleaned_data():
    cpu_path = os.path.join(PROJECT_ROOT, "data", "cpu_clean.csv")
    gpu_path = os.path.join(PROJECT_ROOT, "data", "gpu_clean.csv")
    return pd.read_csv(cpu_path), pd.read_csv(gpu_path)


cpu_df, gpu_df = load_cleaned_data()
cpu_choice = st.selectbox("Choose CPU", cpu_df["CPU"].tolist())
cpu_row = cpu_df[cpu_df["CPU"] == cpu_choice].iloc[0]

gpu_choice = st.selectbox("Choose GPU", gpu_df["GPU"].tolist())
gpu_row = gpu_df[gpu_df["GPU"] == gpu_choice].iloc[0]

st.write("---")
st.write(f"**CPU:** {cpu_choice}  (Score: {cpu_row['PassMark_Score']})")
st.write(f"**GPU:** {gpu_choice}  (Score: {gpu_row['PassMark_Score']})")
st.write("---")


def plot_comparison_bar(df, selected_label, label_column, score_column, price_column, title, container):
    selected_price = df.loc[df[label_column] == selected_label, price_column].values[0]
    others = df.dropna(subset=[price_column]).copy()
    others = others[others[label_column] != selected_label]
    others["price_diff"] = (others[price_column] - selected_price).abs()
    closest = others.nsmallest(4, "price_diff")

    selected_row = pd.DataFrame([{
        "model": selected_label,
        "passmark": df.loc[df[label_column] == selected_label, score_column].values[0],
        "price": selected_price
    }])

    bar_df = pd.concat(
        [selected_row, closest.rename(columns={
            label_column: "model",
            score_column: "passmark",
            price_column: "price"
        })[["model", "passmark", "price"]]],
        ignore_index=True
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(
        bar_df["model"],
        bar_df["passmark"],
        color=["tab:blue" if m == selected_label else "lightgray" for m in bar_df["model"]]
    )
    ax.set_ylabel("PassMark Score")
    ax.set_title(title)
    ax.set_xticks(range(len(bar_df)))
    ax.set_xticklabels(bar_df["model"], rotation=45, ha="right", fontsize=8)
    container.pyplot(fig)


if st.button("Estimate Values"):
    cpu_est_price = None
    gpu_est_price = None

    try:
        cpu_payload = {
            "PassMark_Score": float(cpu_row["PassMark_Score"]),
            "ValueScore": float(cpu_row["ValueScore"]),
            "Rank": int(cpu_row["Rank"]),
            "Brand": str(cpu_row["Brand"])
        }

        gpu_payload = {
            "PassMark_Score": float(gpu_row["PassMark_Score"]),
            "ValueScore": float(gpu_row["ValueScore"]),
            "Rank": int(gpu_row["Rank"]),
            "Brand": str(gpu_row["Brand"])
        }

        cpu_response = requests.post("http://127.0.0.1:5000/predict_cpu", json=cpu_payload, timeout=3)
        gpu_response = requests.post("http://127.0.0.1:5000/predict_gpu", json=gpu_payload, timeout=3)

        if cpu_response.status_code == 200:
            cpu_est_price = cpu_response.json()["estimated_price"]
            st.subheader("CPU Price Estimate")
            st.success(f"Estimated price for **{cpu_choice}**:  **${cpu_est_price:.2f}**")
        else:
            st.error(f"CPU API error {cpu_response.status_code}")

        if gpu_response.status_code == 200:
            gpu_est_price = gpu_response.json()["estimated_price"]
            st.subheader("GPU Price Estimate")
            st.success(f"Estimated price for **{gpu_choice}**:  **${gpu_est_price:.2f}**")
        else:
            st.error(f"GPU API error {gpu_response.status_code}")

        if cpu_est_price is not None and gpu_est_price is not None:
            total = cpu_est_price + gpu_est_price
            st.markdown(f"**Combined Estimate: ${total:.2f}**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Similar CPUs (by price)**")
                plot_comparison_bar(cpu_df, cpu_choice, "CPU", "PassMark_Score", "Price",
                                    "Selected vs. Similar CPUs", st)
            with col2:
                st.markdown("**Similar GPUs (by price)**")
                plot_comparison_bar(gpu_df, gpu_choice, "GPU", "PassMark_Score", "Price",
                                    "Selected vs. Similar GPUs", st)

    except Exception as err:
        st.error(f"Request failed: {err}")
        print(f"[CLIENT ERROR] {err}")
