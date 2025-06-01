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

flask_app = Flask(__name__)

# Load trained CPU model
model_path = os.path.join(os.path.dirname(__file__), "./model/cpu_price_model.pkl")
model = joblib.load(model_path)

@flask_app.route("/predict_cpu", methods=["POST"])
def predict_cpu():
    """
    Expects JSON: {"score": <number>}
    Returns {"estimated_price": <float>, "input_score": <number>}
    """
    try:
        data = request.get_json(force=True)
        if not data or "score" not in data:
            return jsonify({"error": "Missing 'score' in request"}), 400

        score = float(data["score"])
        prediction = model.predict([[score]])[0]
        return jsonify({
            "estimated_price": round(prediction, 2),
            "input_score": score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    """
    Launch Flask on 127.0.0.1:5000 in a daemon thread,
    so Streamlit can call it directly.
    """
    # Note: Using 127.0.0.1 explicitly so the requests.post calls inside this process succeed.
    flask_app.run(host="127.0.0.1", port=5000)

# Start Flask in a background thread as soon as this module is imported
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(1)  # give Flask a moment to start up

# STREAMLIT UI

st.set_page_config(page_title="CPU & GPU Value Checker", layout="centered")
st.title("💻 PC Component Value Checker")
st.markdown(
    """
    The app will look up each part’s PassMark score, then:
    1. Run the CPU’s PassMark score through the Random Forest model to estimate a “fair” CPU price.  
    2. Show the GPU’s scraped market price (from your GPU dataset).  
    """
)

# Load cleaned CPU & GPU tables

@st.cache_data
def load_cleaned_data():
    """
    Returns two DataFrames: cpu_df and gpu_df, both loaded from data/*.csv
    Assumes columns at least:
      - cpu_df: "CPU", "PassMark_Score", "Price"
      - gpu_df: "GPU", "PassMark_Score", "Price"
    """
    cpu_df = pd.read_csv(os.path.join("./data", "cpu_clean.csv"))
    gpu_df = pd.read_csv(os.path.join("./data", "gpu_clean.csv"))
    return cpu_df, gpu_df

cpu_df, gpu_df = load_cleaned_data()

# DROPDOWNS & LOOKUP

# 1) CPU dropdown
cpu_choice = st.selectbox("🔧 Choose Your CPU", cpu_df["CPU"].tolist())
# Get CPU’s PassMark score
cpu_score = cpu_df.loc[cpu_df["CPU"] == cpu_choice, "PassMark_Score"].values[0]

# 2) GPU dropdown
gpu_choice = st.selectbox("🎮 Choose Your GPU", gpu_df["GPU"].tolist())
# Get GPU’s PassMark score and market price
gpu_score = gpu_df.loc[gpu_df["GPU"] == gpu_choice, "PassMark_Score"].values[0]
gpu_market_price = gpu_df.loc[gpu_df["GPU"] == gpu_choice, "Price"].values[0]

st.write("---")
st.write(f"**Selected CPU:** {cpu_choice} (PassMark: {cpu_score})")
st.write(f"**Selected GPU:** {gpu_choice} (PassMark: {gpu_score}), Market Price: ${gpu_market_price:.2f}")
st.write("---")

# PART 4: BUTTON & FETCH RESULTS

if st.button("Estimate Values"):
    # Call the Flask endpoint for CPU prediction
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict_cpu",
            json={"score": int(cpu_score)},
            timeout=3
        )

        if response.status_code == 200:
            cpu_result = response.json()
            cpu_est_price = cpu_result["estimated_price"]

            st.subheader("CPU Price Estimate")
            st.success(f"Predicted fair price for **{cpu_choice}**: **${cpu_est_price:.2f}**")

            st.subheader("GPU Market Price")
            st.info(f"Current scraped price for **{gpu_choice}**: **${gpu_market_price:.2f}**")

            total_fair = cpu_est_price + gpu_market_price
            st.markdown(
                f"> **Combined Fair Value (CPU + GPU): ${total_fair:.2f}**"
            )

        else:
            st.error(f"CPU API returned status {response.status_code}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
