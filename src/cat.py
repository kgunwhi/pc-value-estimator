import os
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preproc import preprocess_for_catboost


def train_catboost_model(X, y, cat_features, model_path, label="CPU"):
    """
    Trains a CatBoost model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        loss_function='RMSE',
        verbose=0,
        random_seed=42
    )
    model.fit(train_pool)

    preds = np.expm1(model.predict(test_pool))
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"[CatBoost] {label} Price RMSE: ${rmse:.2f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path, format="cbm")
    print(f"Saved {label} CatBoost model to {model_path}")


def catboost_train_cpu(cpu_df, project_root):
    X, y, cat_features = preprocess_for_catboost(cpu_df)
    model_path = os.path.join(project_root, "model", "cpu_price_model_catboost.cbm")
    train_catboost_model(X, y, cat_features, model_path, label="CPU")


def catboost_train_gpu(gpu_df, project_root):
    X, y, cat_features = preprocess_for_catboost(gpu_df)
    model_path = os.path.join(project_root, "model", "gpu_price_model_catboost.cbm")
    train_catboost_model(X, y, cat_features, model_path, label="GPU")
