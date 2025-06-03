#!/usr/bin/env bash
#
# pc-value-estimator.sh
#   - Activates .venv
#   - Installs dependencies
#   - Scrapes data, cleans it, trains the model
#   - Launches Streamlit (with embedded Flask)
#

set -e

# 1) Make sure we are running from the project root
cd "$(dirname "$0")"

# 2) Activate (or create) the .venv and install dependencies
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
    echo "==> Installing Python dependencies into .venv..."
    python -m pip install --upgrade pip
    python -m pip install --upgrade -r requirements.txt
else
    echo "==> .venv not found. Creating a new virtual environment..."
    python3 -m venv .venv
    . .venv/bin/activate
    echo "==> Installing Python dependencies into .venv..."
    python -m pip install --upgrade pip
    python -m pip install --upgrade -r requirements.txt
fi

# 3) Ensure data/ and model/ directories exist under project root
mkdir -p data
mkdir -p model
chmod +w data
chmod +w model

# 4) Run the scraper to create data/cpu_passmark.csv & data/gpu_passmark.csv
echo "==> Running scraper..."
python src/main.py

# 5) Launch the combined Flask + Streamlit app
echo "==> Launching Streamlit app (with embedded Flask)…"
which streamlit || echo "streamlit not found"
exec streamlit run src/app.py \
  --server.port=8080 \
  --server.address=0.0.0.0 \
  --browser.serverAddress=localhost






