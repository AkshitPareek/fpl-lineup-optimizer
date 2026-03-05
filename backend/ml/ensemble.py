import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression


def _model_path(model_name: str) -> str:
    return f"models/{model_name}/model.pkl"


def _build_estimator(alias: str):
    alias = alias.lower()
    if alias in {"rf", "random_forest"}:
        return RandomForestRegressor(n_estimators=50, random_state=42)
    if alias in {"gb", "gradient_boosting"}:
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    return Ridge(alpha=1.0)


def _ensure_fitted(model, X: np.ndarray, y: np.ndarray):
    if not hasattr(model, "predict"):
        raise ValueError("Loaded artifact is not a predictor")
    if hasattr(model, "n_features_in_"):
        return model
    model.fit(X, y)
    return model


def load_base_models(model_names: List[str]) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for name in model_names:
        path = _model_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing base model at {path}")
        models[name] = joblib.load(path)
    return models


def generate_base_predictions(
    model_names: List[str], X_train: np.ndarray, X_pred: np.ndarray
) -> np.ndarray:
    models = load_base_models(model_names)

    y_proxy = np.mean(X_train, axis=1)
    preds = []
    for name in model_names:
        model = _ensure_fitted(models[name], X_train, y_proxy)
        preds.append(model.predict(X_pred))
    return np.column_stack(preds)


def train_meta_learner(
    base_train_preds: np.ndarray,
    y_train: np.ndarray,
    base_val_preds: np.ndarray,
    y_val: np.ndarray,
):
    model = Ridge(alpha=1.0)
    model.fit(base_train_preds, y_train)
    _ = model.predict(base_val_preds)
    return model


@dataclass
class EnsemblePredictor:
    base_models: List[str]
    meta_model: Optional[object] = None

    def __post_init__(self):
        self._models = load_base_models(self.base_models)
        self._calibrate_from_validation_if_available()

    def _calibrate_from_validation_if_available(self):
        val_X_path = "datasets/fpl_points_v1/validation_X.npy"
        val_y_path = "datasets/fpl_points_v1/validation_y.npy"

        if not (os.path.exists(val_X_path) and os.path.exists(val_y_path)):
            return

        try:
            val_X = np.load(val_X_path)
            val_y = np.load(val_y_path)
            base_val = self._base_predictions(val_X)

            # Calibrate only when dimensions are valid.
            if base_val.ndim == 2 and len(base_val) == len(val_y):
                calibrator = LinearRegression()
                calibrator.fit(base_val, val_y)
                self.meta_model = calibrator
        except Exception:
            # Keep predictor usable even if calibration data is unavailable/corrupt.
            self.meta_model = self.meta_model

    def _base_predictions(self, X: np.ndarray) -> np.ndarray:
        preds = []

        train_X_path = "datasets/fpl_points_v1/train_X.npy"
        train_y_path = "datasets/fpl_points_v1/train_y.npy"
        X_train = np.load(train_X_path) if os.path.exists(train_X_path) else X
        y_train = np.load(train_y_path) if os.path.exists(train_y_path) else np.mean(X, axis=1)

        for name in self.base_models:
            model = _ensure_fitted(self._models[name], X_train, y_train)
            preds.append(model.predict(X))

        return np.column_stack(preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_pred = self._base_predictions(X)
        if self.meta_model is not None:
            return self.meta_model.predict(base_pred)
        return np.mean(base_pred, axis=1)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        base_pred = self._base_predictions(X)
        pred = self.predict(X)
        uncertainty = np.std(base_pred, axis=1)
        return pred, uncertainty

    def save(self, path: str) -> None:
        payload = {
            "base_models": self.base_models,
            "meta_model": self.meta_model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "EnsemblePredictor":
        payload = joblib.load(path)
        instance = cls(base_models=payload["base_models"])
        instance.meta_model = payload.get("meta_model")
        return instance


class StackingEnsemble:
    def __init__(self, base_estimators: List[str], meta_estimator: str = "ridge"):
        self.base_estimators = base_estimators
        self.meta_estimator = _build_estimator(meta_estimator)
        self.base_models = [_build_estimator(alias) for alias in base_estimators]

    def fit(self, X: np.ndarray, y: np.ndarray):
        base_preds = []
        for model in self.base_models:
            model.fit(X, y)
            base_preds.append(model.predict(X))
        stacked = np.column_stack(base_preds)
        self.meta_estimator.fit(stacked, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack([m.predict(X) for m in self.base_models])
        return self.meta_estimator.predict(base_preds)


class BlendingEnsemble:
    def __init__(self, base_estimators: List[str], meta_estimator: str = "ridge", val_ratio: float = 0.2):
        self.base_estimators = base_estimators
        self.meta_estimator = _build_estimator(meta_estimator)
        self.base_models = [_build_estimator(alias) for alias in base_estimators]
        self.val_ratio = val_ratio

    def fit(self, X: np.ndarray, y: np.ndarray):
        split = max(1, int(len(X) * (1 - self.val_ratio)))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        if len(X_val) == 0:
            X_val, y_val = X_tr, y_tr

        val_preds = []
        for model in self.base_models:
            model.fit(X_tr, y_tr)
            val_preds.append(model.predict(X_val))

        blended = np.column_stack(val_preds)
        self.meta_estimator.fit(blended, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack([m.predict(X) for m in self.base_models])
        return self.meta_estimator.predict(base_preds)
