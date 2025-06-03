# Source Code Overview

This folder contains all source code for the PC Value Estimator application. The app combines data scraping, preprocessing, machine learning model, and a Streamlit + Flask web interface.

## Files

- `app.py`  
  Main entry point for the web app. It launches the Streamlit interface and includes an embedded Flask server to handle backend requests.

- `main.py`  
  Full pipeline: scraping benchmark data, cleaning it, training models, and saving the outputs to `model/` and `data/`.

- `scraper.py`  
  Scrapes CPU and GPU benchmark and price data from external sources and saves the raw files.

- `preproc.py`  
  Cleans and merges the scraped datasets, preparing them for model training and prediction.

- `cat.py`  
  Defines and trains the CatBoost regression models for CPU and GPU price estimation.

- `eda.py`  
  Generates exploratory visualizations, summary statistics, and feature plots, which are saved to the `plots/` directory.

## Usage

Run the entire pipeline locally in terminal:
```bash
./pc-value-estimator.sh
```
