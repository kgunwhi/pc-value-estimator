# pc-value-estimator

This project scrapes data on PC components (CPU, GPU, etc.) to build a model that evaluates performance-to-price ratios. Users can upload their system specs, and the app will estimate whether they overpaid based on market benchmarks.

## Project Goals
- Scrape PC part prices and performance benchmarks
- Build a simple regression model to estimate "fair price" using random forest or xgboost
- Deploy a web application using Shiny
- Deploy a backend API (Flask) for predictions
- Host everything via Google Cloud Run and shinyapps.io

## Repo Structure

pc-value-estimator/ ← Project root
├── .venv/ ← Python virtual environment (auto‐created by run_all.sh)
│ ├── bin/
│ ├── lib/
│ └── …
├── data/ ← Scraped & cleaned CSV files
│ ├── cpu_passmark.csv ← Raw CPU PassMark + price data
│ ├── gpu_passmark.csv ← Raw GPU PassMark + price data
│ ├── cpu_clean.csv ← Cleaned CPU data (filtered, brand tags, numeric)
│ └── gpu_clean.csv ← Cleaned GPU data (filtered, brand tags, numeric)
│
├── model/ ← Saved trained model artifacts
│ └── cpu_price_model.pkl ← Random Forest regressor for CPU price
│
├── src/ ← All Python source code
│ ├── app.py ← Combined Streamlit UI + embedded Flask API
│ ├── eda.py ← Exploratory data analysis scripts
│ ├── main.py ← Orchestrates scraper, preproc, training, etc.
│ ├── preproc.py ← Cleans raw CSVs → data/*_clean.csv
│ ├── scraper.py ← Scrapes CPU/GPU PassMark data → data/*.csv
│ ├── xgb.py ← (Or train_model.py) trains CPU Random Forest model → model/cpu_price_model.pkl
│ └── streamlit_app.py ← If you split app.py into a dedicated Streamlit file
│
├── requirements.txt ← All Python dependencies
├── run_all.sh ← One‐click shell script to scrape, clean, train, and launch the app
└── README.md ← This file

## Deployment Plan
- API: Flask + Google Cloud Run  
- App: Streamlit + Google Cloud Run or shinyapps.io  
