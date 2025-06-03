# API Overview

The PC Value Estimator includes a lightweight Flask API that handles backend logic for predicting CPU and GPU prices.

## Base Configuration

- **Framework:** Flask (embedded inside `app.py`)
- **Runs on:** Port `5050` (locally), accessed internally from the Streamlit frontend
- **Deployment:** Runs inside the same container as Streamlit and is **not exposed publicly**

---

## Endpoints

### `POST /predict_cpu`

Predicts the price of a CPU based on its benchmark features.

#### Example:
Run the app in one terminal:
```bash
./pc-value-estimator.sh
```
In another terminal:
```bash
curl -X POST http://localhost:5050/predict_cpu \
  -H "Content-Type: application/json" \
  -d '{"PassMark_Score": 20588, "ValueScore": 12.8, "Rank": 37, "Brand": "Intel"}'
```
Sample Output: 
```bash
{"estimated_price": 1184.49}
```


### `POST /predict_gpu`

Predicts the price of a GPU based on its benchmark features.

#### Example:
Run the app in one terminal:
```bash
./pc-value-estimator.sh
```
In another terminal:
```bash
curl -X POST http://localhost:5050/predict_gpu \
  -H "Content-Type: application/json" \
  -d '{"PassMark_Score": 23888, "ValueScore": 10.4, "Rank": 12, "Brand": "NVIDIA"}'
```
Sample Output: 
```bash
{"estimated_price":781.88}
```

