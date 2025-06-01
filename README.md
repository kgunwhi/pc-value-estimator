# pc-value-estimator

This project scrapes data on PC components (CPU, GPU, etc.) to build a model that evaluates performance-to-price ratios. Users can upload their system specs, and the app will estimate whether they overpaid based on market benchmarks.

## Project Goals
- Scrape PC part prices and performance benchmarks
- Build a simple regression model to estimate "fair price" using random forest or xgboost
- Deploy a web application using Shiny
- Deploy a backend API (Flask) for predictions
- Host everything via Google Cloud Run and shinyapps.io

## Repo Structure
-src
---app.py	: streamlit app with flask
---eda.py	: some eda
---main.py	: main calls all other functions
---preproc.py	: clean data
---scraper.py	: scrape cpu/gpu data from passmark
---xgb.py 	: train model
-README.md
-pc-value-estimator.sh : all in one shell script to scrape, train, and launch app
-requirements.txt : all dependencies required

## Deployment Plan
- API: Flask + Google Cloud Run  
- App: Streamlit + Google Cloud Run or shinyapps.io  
