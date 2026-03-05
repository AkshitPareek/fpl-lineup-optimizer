import os
import numpy as np
import joblib
from typing import Dict, Any, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_estimators: int = 100,
) -> xgb.XGBRegressor:
    """Train XGBoost model."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    if X_val is not None and y_val is not None:
        params["eval_set"] = [(X_val, y_val)]
        params["early_stopping_rounds"] = 10

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    os.makedirs("models/xgboost", exist_ok=True)
    joblib.dump(model, "models/xgboost/model.pkl")

    return model


def train_xgboost_cv(
    X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5
) -> Dict[str, float]:
    """Train XGBoost with 5-fold cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    test_rmse_scores = []
    test_r2_scores = []
    test_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        predictions = model.predict(X_vl)

        rmse = np.sqrt(mean_squared_error(y_vl, predictions))
        r2 = r2_score(y_vl, predictions)
        mae = mean_absolute_error(y_vl, predictions)

        test_rmse_scores.append(rmse)
        test_r2_scores.append(r2)
        test_mae_scores.append(mae)

    return {
        "test_rmse_mean": np.mean(test_rmse_scores),
        "test_rmse_std": np.std(test_rmse_scores),
        "test_r2_mean": np.mean(test_r2_scores),
        "test_r2_std": np.std(test_r2_scores),
        "test_mae_mean": np.mean(test_mae_scores),
        "test_mae_std": np.std(test_mae_scores),
    }


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_estimators: int = 100,
) -> lgb.LGBMRegressor:
    """Train LightGBM model."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if X_val is not None and y_val is not None:
        params["eval_set"] = [(X_val, y_val)]
        params["callbacks"] = [lgb.early_stopping(10, verbose=False)]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    os.makedirs("models/lightgbm", exist_ok=True)
    joblib.dump(model, "models/lightgbm/model.pkl")

    return model


def train_lightgbm_cv(
    X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5
) -> Dict[str, float]:
    """Train LightGBM with 5-fold cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    test_rmse_scores = []
    test_r2_scores = []
    test_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)

        predictions = model.predict(X_vl)

        rmse = np.sqrt(mean_squared_error(y_vl, predictions))
        r2 = r2_score(y_vl, predictions)
        mae = mean_absolute_error(y_vl, predictions)

        test_rmse_scores.append(rmse)
        test_r2_scores.append(r2)
        test_mae_scores.append(mae)

    return {
        "test_rmse_mean": np.mean(test_rmse_scores),
        "test_rmse_std": np.std(test_rmse_scores),
        "test_r2_mean": np.mean(test_r2_scores),
        "test_r2_std": np.std(test_r2_scores),
        "test_mae_mean": np.mean(test_mae_scores),
        "test_mae_std": np.std(test_mae_scores),
    }


def evaluate_boosting_model(
    model: Any, X: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    """Evaluate a boosting model and return metrics."""
    predictions = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    mae = float(mean_absolute_error(y, predictions))

    return {"rmse": rmse, "r2": r2, "mae": mae}


def get_feature_importance(model: Any, feature_names: list = None) -> np.ndarray:
    """Extract feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    else:
        n_features = model.n_features_in_ if hasattr(model, "n_features_in_") else 1
        return np.ones(n_features) / n_features


def compute_shap_values(model: Any, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a trained model."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap package is required for SHAP analysis")

    if isinstance(model, xgb.XGBRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif isinstance(model, lgb.LGBMRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError("Model type not supported for SHAP analysis")

    return shap_values
