import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Any, Dict, Tuple


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Tuple[LinearRegression, Dict[str, float]]:
    """Train a Linear Regression model and evaluate on validation set."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_val, y_val)

    return model, metrics


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 100,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train a Random Forest model and evaluate on validation set."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_val, y_val)

    return model, metrics


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a model and return RMSE, R², and MAE metrics."""
    predictions = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    mae = float(mean_absolute_error(y, predictions))

    return {"rmse": rmse, "r2": r2, "mae": mae}


def save_model(model: Any, path: str) -> None:
    """Save a model to the specified path using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load a model from the specified path using joblib."""
    return joblib.load(path)
