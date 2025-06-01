# app.py
import threading
import time
import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
import streamlit as st
import requests

# ─── PART 1: DEFINE & LAUNCH FLASK 💾────────────────────────────────────────────

flask_app = Flask(__name__)

# ─── Fix paths by computing the project root (one level above src/)
#    __file__ = ".../pc-value-estimator/src/app.py"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load trained CPU model from project_root/model/
cpu_model_path = os.path.join(PROJECT_ROOT, "model", "cpu_price_model.pkl")
cpu_model = joblib.load(cpu_model_path)

# Load trained GPU model from project_root/model/
gpu_model_path = os.path.join(PROJECT_ROOT, "model", "gpu_price_model.pkl")
gpu_model = joblib.load(gpu_model_path)

@flask_app.route("/predict_cpu", methods=["POST"])
def predict_cpu():
    try:
        data = request.get_json(force=True)
        if not data or "score" not in data:
            return jsonify({"error": "Missing 'score' in request"}), 400

        score = float(data["score"])
        prediction = cpu_model.predict([[score]])[0]
        return jsonify({
            "estimated_price": round(float(prediction), 2),
            "input_score": score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/predict_gpu", methods=["POST"])
def predict_gpu():
    try:
        data = request.get_json(force=True)
        if not data or "score" not in data:
            return jsonify({"error": "Missing 'score' in request"}), 400

        score = float(data["score"])
        prediction = gpu_model.predict([[score]])[0]
        return jsonify({
            "estimated_price": round(float(prediction), 2),
            "input_score": score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    flask_app.run(host="127.0.0.1", port=5000)

# Start Flask in a background thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(1)  # give Flask a moment to start up

# ─── PART 2: STREAMLIT UI 🖥────────────────────────────────────────────────────

st.set_page_config(page_title="CPU & GPU Value Checker", layout="centered")
st.title("💻 PC Component Value Checker")
st.markdown(
    """
    The app will look up each part’s PassMark score, then:
    1. Run the CPU’s PassMark score through the Random Forest model to estimate a “fair” CPU price.  
    2. Run the GPU’s PassMark score through the Random Forest model to estimate a “fair” GPU price.  
    3. Show the combined “fair value” (CPU + GPU).  
    """
)

@st.cache_data
def load_cleaned_data():
    cpu_path = os.path.join(PROJECT_ROOT, "data", "cpu_clean.csv")
    gpu_path = os.path.join(PROJECT_ROOT, "data", "gpu_clean.csv")

    cpu_df = pd.read_csv(cpu_path)
    gpu_df = pd.read_csv(gpu_path)
    return cpu_df, gpu_df

cpu_df, gpu_df = load_cleaned_data()

# ─── DROPDOWNS & LOOKUP ──────────────────────────────────────────────────────────

cpu_choice = st.selectbox("🔧 Choose Your CPU", cpu_df["CPU"].tolist())
cpu_score = cpu_df.loc[cpu_df["CPU"] == cpu_choice, "PassMark_Score"].values[0]

gpu_choice = st.selectbox("🎮 Choose Your GPU", gpu_df["GPU"].tolist())
gpu_score = gpu_df.loc[gpu_df["GPU"] == gpu_choice, "PassMark_Score"].values[0]

st.write("---")
st.write(f"**Selected CPU:** {cpu_choice} (PassMark: {cpu_score})")
st.write(f"**Selected GPU:** {gpu_choice} (PassMark: {gpu_score})")
st.write("---")

# ─── PART 4: BUTTON & FETCH RESULTS ───────────────────────────────────────────────

if st.button("Estimate Values"):
    try:
        # Call CPU endpoint
        cpu_response = requests.post(
            "http://127.0.0.1:5000/predict_cpu",
            json={"score": int(cpu_score)},
            timeout=3
        )
        # Call GPU endpoint
        gpu_response = requests.post(
            "http://127.0.0.1:5000/predict_gpu",
            json={"score": int(gpu_score)},
            timeout=3
        )

        if cpu_response.status_code == 200:
            cpu_result = cpu_response.json()
            cpu_est_price = cpu_result["estimated_price"]
            st.subheader("🔹 CPU Price Estimate")
            st.success(f"Predicted fair price for **{cpu_choice}**: **${cpu_est_price:.2f}**")
        else:
            st.error(f"CPU API returned status {cpu_response.status_code}")

        if gpu_response.status_code == 200:
            gpu_result = gpu_response.json()
            gpu_est_price = gpu_result["estimated_price"]
            st.subheader("🔹 GPU Price Estimate")
            st.success(f"Predicted fair price for **{gpu_choice}**: **${gpu_est_price:.2f}**")
        else:
            st.error(f"GPU API returned status {gpu_response.status_code}")

        if cpu_response.status_code == 200 and gpu_response.status_code == 200:
            total_fair = cpu_est_price + gpu_est_price
            st.markdown(f"> **Combined Fair Value (CPU + GPU): ${total_fair:.2f}**")

    except Exception as e:
        st.error(f"Connection failed: {e}")
