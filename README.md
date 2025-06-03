# PC Value Estimator 💻💸

This project estimates the fair market price of a CPU and GPU based on performance benchmarks using machine learning. It includes a web app (Streamlit + Flask) deployed on Google Cloud Run.

## Features
- Scrapes CPU/GPU benchmark and pricing data
- Trains machine learning models (CatBoost, etc.)
- Predicts fair price from user input
- Fully containerized with Docker
- Live on Google Cloud Run 

## Deployment
This app is deployed to Google Cloud Run.
The Docker image is built using Google Cloud Build and automatically served via HTTPS.

## Demo
[Live App](https://pc-value-estimator-135418392758.us-central1.run.app)

## Getting Started

### 1. Clone the repo and run the app locally
```bash
git clone https://github.com/kgunwhi/pc-value-estimator.git
cd pc-value-estimator
./pc-value-estimator.sh
```

### 2. Run with Docker
```bash
docker build -t pc-value-estimator .
docker run -p 8080:8080 pc-value-estimator
```


## Project Structure
```
pc-value-estimator/
├── src/                  # All source code (Streamlit, Flask, scraping, catboost)
├── model/                # Trained models (generated at runtime)
├── data/                 # Processed benchmark CSVs (generated at runtime)
├── catboost_info/        # catboost training logs (generated at runtime)
├── plots/                # EDA (generated at runtime)
├── slides/               # Final presentation
├── Dockerfile            # Docker config
├── pc-value-estimator.sh # Run script (local or cloud)
├── requirements.txt      # Python dependencies
└── README.md             # You're here
```


## Author
Alexander Kim - STAT 418 - UCLA
