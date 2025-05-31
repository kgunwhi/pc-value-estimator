# pc-value-estimator

This project scrapes data on PC components (CPU, GPU, etc.) to build a model that evaluates performance-to-price ratios. Users can upload their system specs, and the app will estimate whether they overpaid based on market benchmarks.

## Project Goals
- Scrape PC part prices and performance benchmarks
- Build a simple regression model to estimate "fair price" using random forest or xgboost
- Deploy a web application using Shiny
- Deploy a backend API (Flask) for predictions
- Host everything via Google Cloud Run and shinyapps.io

## Repo Structure (to come)

## Deployment Plan
- API: Flask + Google Cloud Run  
- App: Streamlit + Google Cloud Run or shinyapps.io  
