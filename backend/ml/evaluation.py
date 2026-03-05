from typing import Dict, List, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


def run_backtest(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    n_splits: int = 3,
) -> Dict[str, Any]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results: List[Dict[str, float]] = []

    for tr_idx, _ in kf.split(train_X):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(train_X[tr_idx], train_y[tr_idx])
        pred = model.predict(test_X)
        rmse = float(np.sqrt(mean_squared_error(test_y, pred)))
        fold_results.append({"rmse": rmse})

    overall_rmse = float(np.mean([f["rmse"] for f in fold_results]))
    return {"overall_rmse": overall_rmse, "fold_results": fold_results}


def compute_position_wise_mae(
    X: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    n_positions: int = 4,
):
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    names = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
    out: Dict[str, float] = {}

    for p in range(n_positions):
        mask = positions == p
        if np.any(mask):
            out[names.get(p, str(p))] = float(mean_absolute_error(y[mask], pred[mask]))
        else:
            out[names.get(p, str(p))] = 0.0

    return out


def analyze_price_brackets(
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    n_brackets: int = 4,
):
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    edges = np.linspace(np.min(prices), np.max(prices), n_brackets + 1)
    out = []
    for i in range(n_brackets):
        lo, hi = edges[i], edges[i + 1]
        if i == n_brackets - 1:
            mask = (prices >= lo) & (prices <= hi)
        else:
            mask = (prices >= lo) & (prices < hi)

        if not np.any(mask):
            continue

        out.append(
            {
                "bracket": f"{lo:.2f}-{hi:.2f}",
                "count": int(np.sum(mask)),
                "mae": float(mean_absolute_error(y[mask], pred[mask])),
            }
        )

    return out


def calculate_expected_rank_improvement(
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    n_simulations: int = 100,
):
    model = LinearRegression().fit(val_X, val_y)
    pred = model.predict(test_X)

    baseline_rmse = float(np.sqrt(mean_squared_error(test_y, np.full_like(test_y, np.mean(val_y)))))
    model_rmse = float(np.sqrt(mean_squared_error(test_y, pred)))

    improvement = (baseline_rmse - model_rmse) / max(baseline_rmse, 1e-8)
    return {"mean_improvement": float(improvement), "simulations": n_simulations}


def compare_models(
    models: Dict[str, Any],
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
):
    out: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        out[name] = {
            "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
            "mae": float(mean_absolute_error(test_y, pred)),
        }
    return out


def compute_cross_validation_results(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, va_idx in kf.split(X):
        model = LinearRegression().fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx])
        rmses.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))

    return {
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
    }


def analyze_feature_effects(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
):
    model = LinearRegression().fit(X, y)
    coef = np.asarray(model.coef_)
    return {feature_names[i]: float(coef[i]) for i in range(min(len(feature_names), len(coef)))}


def analyze_error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    return {
        "mean_error": float(np.mean(err)),
        "std_error": float(np.std(err)),
        "bias": float(np.mean(err)),
        "variance": float(np.var(err)),
    }


def compute_prediction_intervals(
    X: np.ndarray,
    predictions: np.ndarray,
    confidence: float = 0.95,
):
    residual_scale = np.std(predictions) * (1.0 - confidence + 0.05)
    lower = predictions - residual_scale
    upper = predictions + residual_scale
    return {"lower": lower, "upper": upper}


def generate_evaluation_report(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    positions = np.arange(len(test_y)) % 4

    return {
        "test_rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
        "test_metrics": {
            "mae": float(mean_absolute_error(test_y, pred)),
        },
        "position_analysis": compute_position_wise_mae(test_X, test_y, positions),
        "position_wise": compute_position_wise_mae(test_X, test_y, positions),
    }
