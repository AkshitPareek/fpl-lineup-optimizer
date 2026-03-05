import os
import numpy as np
import optuna
from typing import Dict, Any, Optional
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


def optimize_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50
) -> Dict[str, Any]:
    """Optimize XGBoost hyperparameters using Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
        )

        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def optimize_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50
) -> Dict[str, Any]:
    """Optimize LightGBM hyperparameters using Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMRegressor(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
        )

        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def get_feature_importance(
    model: Any, feature_names: Optional[list] = None
) -> Dict[str, float]:
    """Get feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        n_features = model.n_features_in_ if hasattr(model, "n_features_in_") else 1
        importances = np.ones(n_features) / n_features

    if feature_names is not None:
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
    else:
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}


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
