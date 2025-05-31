from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def xgboost_train_cpu(cpu_df):
    X = cpu_df[["PassMark_Score"]]
    y = cpu_df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    print(f"[Random Forest] CPU Price RMSE: ${rmse:.2f}")

    # Save model
    joblib.dump(model, "model/cpu_price_model.pkl")
