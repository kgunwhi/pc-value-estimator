#!/usr/bin/env bash
#
# run_all.sh
#   - Activates .venv
#   - Installs dependencies
#   - Scrapes data, cleans it, trains the model
#   - Launches Streamlit (with embedded Flask)
#

set -e

# Make sure we are running from the project root
cd "$(dirname "$0")"

# Activate the existing .venv 
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "==> Installing Python dependencies into .venv..."
    python -m pip install --upgrade pip
    python -m pip install --upgrade -r requirements.txt
else
    echo "Error: .venv not found. Running 'python3 -m venv .venv' first."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "==> Installing Python dependencies into .venv..."
    python -m pip install --upgrade pip
    python -m pip install --upgrade -r requirements.txt
    
fi

# Make sure data/ and model/ directories exist
mkdir -p data
mkdir -p model

# Run the scraper to create data/cpu_passmark.csv & data/gpu_passmark.csv
echo "==> Running scraper..."
python src/scraper.py
echo "    data/cpu_passmark.csv and data/gpu_passmark.csv generated"

# Run preprocessing (cleaning) to produce cpu_clean.csv & gpu_clean.csv
echo "==> Running preprocessing..."
python src/preproc.py
echo "    data/cpu_clean.csv and data/gpu_clean.csv generated"

# Train the CPU pricing model (writes model/cpu_price_model.pkl)
echo "==> Training CPU pricing model..."
python src/xgb.py
echo "    model/cpu_price_model.pkl saved"

# Launch the combined Flask + Streamlit app
echo "==> Launching Streamlit app (with embedded Flask)…"
streamlit run src/app.py
