"""
ML Training Pipeline for FPL Lineup Optimizer

Trains XGBoost and Neural Network models to predict player expected points.

Usage:
    python ml_training.py --season 2024-25 --test-gw 38

Author: FPL ML Integration Team
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
import xgboost as xgb

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# Local imports
from fpl_service import FPLService
from historical_data_service import HistoricalDataService
from minutes_predictor import MinutesPredictor
from understat_service import UnderstatService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"


# ============================================================================
# CONFIGURATION (tunable hyperparameters)
# ============================================================================

XGB_PARAM_GRID = {
    "n_estimators": [300, 500, 700],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "min_child_weight": [3, 5, 7],
    "gamma": [0.0, 0.1, 0.2],
    "reg_alpha": [0.0, 1.0, 2.0],
    "reg_lambda": [1.0, 2.0, 3.0],
}

NN_HIDDEN_LAYERS = [128, 64, 32]
NN_DROPOUT_RATE = 0.3
NN_L2_REG = 1e-4
NN_BATCH_SIZE = 256
NN_EPOCHS = 200
NN_LEARNING_RATE = 0.001

CV_SPLITS = 5
TEST_GW = 38  # Last GW of season for testing


def load_data(seasons: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Load all historical data for training.

    Args:
        seasons: List of season strings like ['2023-24', '2024-25']

    Returns:
        players_df: DataFrame with all players
        history_data: Dict[player_id] -> List[per-gw data]
    """
    logger.info("Loading FPL data...")

    fpl_service = FPLService()
    latest_data = fpl_service.get_latest_data()
    static_data = latest_data["static"]

    players_df = pd.DataFrame(static_data["elements"])
    teams_df = pd.DataFrame(static_data["teams"])
    fixtures = latest_data["fixtures"]

    # Load historical service and cache all player history
    history_service = HistoricalDataService()
    history_service.fetch_all_player_history(static_data["elements"])

    # Build feature matrix
    logger.info("Extracting features...")
    features_list = []

    for _, player in players_df.iterrows():
        try:
            # Get player's history
            player_history = history_service._history_cache.get(
                str(player["id"]), {}
            ).get("history", [])

            # For each completed GW, create a training sample
            for gw_record in player_history:
                gw = gw_record["round"]
                # Only include if gw <= test cutoff (avoid leakage)
                if gw <= TEST_GW:
                    # Extract features for this player at this GW
                    features = extract_features_for_player(
                        player=player,
                        history=player_history,
                        target_gw=gw,
                        teams_df=teams_df,
                        fixtures=fixtures,
                    )
                    if features is not None:
                        features_list.append(features)
        except Exception as e:
            logger.warning(f"Failed to extract features for player {player['id']}: {e}")
            continue

    features_df = pd.DataFrame(features_list)

    # Drop rows with NaN target
    features_df = features_df.dropna(subset=["target_points"])

    logger.info(
        f"Loaded {len(features_df)} training samples from {len(players_df)} players"
    )

    return features_df, {"players": players_df, "teams": teams_df, "fixtures": fixtures}


def extract_features_for_player(
    player: pd.Series,
    history: List[Dict],
    target_gw: int,
    teams_df: pd.DataFrame,
    fixtures: List[Dict],
) -> Dict[str, Any]:
    """
    Extract feature vector for a single (player, GW) combination.

    This is the core feature engineering logic. See ML-MODELS-DESIGN.md Section 1.
    """
    try:
        # Find record for this player at this GW
        gw_record = next((r for r in history if r["round"] == target_gw), None)
        if not gw_record:
            return None  # No record for this GW

        # Target variable: actual points
        target_points = gw_record["total_points"]

        # Base features
        features = {
            "player_id": player["id"],
            "target_gw": target_gw,
            "target_points": target_points,
            "position": player["element_type"],
            "now_cost": player["now_cost"],
            "team": player["team"],
        }

        # Get past history (strictly before target_gw)
        past_history = [h for h in history if h["round"] < target_gw]

        if not past_history:
            # No prior data - use position baselines only
            position_baselines = {1: 1.5, 2: 2.5, 3: 4.0, 4: 5.0}
            features["form_points"] = position_baselines[player["element_type"]]
            features["recent_4_minutes_avg"] = 0
            # ... set other minimal features
            return features

        # Compute form features (weighted average last 6 GWs)
        recent_history = past_history[-6:] if len(past_history) >= 6 else past_history
        features.update(compute_form_features(recent_history))

        # Season aggregates
        total_minutes = sum(h["minutes"] for h in past_history)
        if total_minutes > 270:
            goals = sum(h["goals_scored"] for h in past_history)
            assists = sum(h["assists"] for h in past_history)
            features["goals_per_90"] = goals / (total_minutes / 90)
            features["assists_per_90"] = assists / (total_minutes / 90)
        else:
            # Position baselines
            pos_xg_baselines = {1: 0.0, 2: 0.03, 3: 0.08, 4: 0.35}
            pos_xa_baselines = {1: 0.0, 2: 0.04, 3: 0.08, 4: 0.12}
            features["goals_per_90"] = pos_xg_baselines[player["element_type"]]
            features["assists_per_90"] = pos_xa_baselines[player["element_type"]]

        # Fixture features for target GW
        team_id = player["team"]
        fixture_info = get_fixture_info(team_id, target_gw, fixtures)
        features["fixture_score"] = compute_fixture_score(
            team_id, target_gw, teams_df, fixtures
        )
        features["double_gw_flag"] = len(fixture_info) > 1
        features["blank_gw_flag"] = len(fixture_info) == 0

        # Team context (last 10 GWs average stats)
        team_stats = compute_team_stats(team_id, past_history[-10:])
        features.update(team_stats)

        # Minutes/Rotation features
        recent_minutes = [h["minutes"] for h in past_history[-5:]]
        features["recent_4_minutes_avg"] = (
            np.mean(recent_minutes[-4:]) if len(recent_minutes) >= 4 else 0
        )
        features["start_probability"] = compute_start_probability(recent_minutes)

        # ICT and advanced metrics
        influence = sum(float(h.get("influence", 0)) for h in past_history)
        creativity = sum(float(h.get("creativity", 0)) for h in past_history)
        threat = sum(float(h.get("threat", 0)) for h in past_history)
        minutes_total = sum(h["minutes"] for h in past_history)
        if minutes_total > 0:
            features["ict_per_90"] = (influence + creativity + threat) / (
                minutes_total / 90
            )
        else:
            features["ict_per_90"] = 0.0

        # Understat (if available) - placeholder, will be enriched separately
        features["understat_matched"] = 0
        features["understat_xG_per_90"] = 0.0
        features["understat_xA_per_90"] = 0.0

        # Injury status (from current player record)
        status_code = encode_status(player.get("status", "a"))
        features["injury_status"] = status_code
        availability = player.get("chance_of_playing_next_round", 100)
        features["availability_factor"] = (
            float(availability) / 100.0 if pd.notna(availability) else 1.0
        )

        return features

    except Exception as e:
        logger.debug(f"Error extracting features: {e}")
        return None


def compute_form_features(recent_history: List[Dict]) -> Dict[str, float]:
    """Compute weighted form metrics from recent match history."""
    if not recent_history:
        return {}

    # Exponential decay weights: most recent gets highest weight
    n = len(recent_history)
    weights = np.exp(np.linspace(-1, 0, n))
    weights = weights / weights.sum()

    points = np.array([h["total_points"] for h in recent_history])
    minutes = np.array([h["minutes"] for h in recent_history])
    xg = np.array([float(h.get("expected_goals", 0)) for h in recent_history])
    xa = np.array([float(h.get("expected_assists", 0)) for h in recent_history])
    ict = np.array([float(h.get("ict_index", 0)) for h in recent_history])

    # Weighted averages
    form_points = np.dot(points, weights[-n:])
    form_minutes = np.dot(minutes, weights[-n:])

    # Per-90 scaling for xG/xA/ICT
    total_mins = max(minutes.sum(), 1)
    form_xg = (xg * weights[-n:]).sum() / (total_mins / 90) if total_mins > 0 else 0
    form_xa = (xa * weights[-n:]).sum() / (total_mins / 90) if total_mins > 0 else 0
    form_ict = (ict * weights[-n:]).sum() / (total_mins / 90) if total_mins > 0 else 0

    return {
        "form_points": float(form_points),
        "form_minutes": float(form_minutes),
        "form_xg": float(form_xg),
        "form_xa": float(form_xa),
        "form_ict": float(form_ict),
    }


def get_fixture_info(team_id: int, gw: int, fixtures: List[Dict]) -> List[Dict]:
    """Get fixtures for team in target GW."""
    return [
        f
        for f in fixtures
        if f["event"] == gw and (f["team_h"] == team_id or f["team_a"] == team_id)
    ]


def compute_fixture_score(
    team_id: int, gw: int, teams_df: pd.DataFrame, fixtures: List[Dict]
) -> float:
    """
    Compute fixture quality score (0-1 scale, 1=easiest).
    Uses team attack and opponent defense strength.
    """
    team_fixtures = get_fixture_info(team_id, gw, fixtures)
    if not team_fixtures:
        return 0.5  # Neutral default

    scores = []
    team_strength = teams_df.loc[
        teams_df["id"] == team_id, "strength_attack_home"
    ].values[0]

    for f in team_fixtures:
        is_home = f["team_h"] == team_id
        opponent_id = f["team_a"] if is_home else f["team_h"]
        opponent_defense = teams_df.loc[
            teams_df["id"] == opponent_id,
            "strength_defence_home" if is_home else "strength_defence_away",
        ].values[0]

        # Normalize to 0-1
        attack_factor = team_strength / 1000
        defense_factor = (1000 - opponent_defense) / 1000
        xg_env = attack_factor * defense_factor
        scores.append(xg_env)

    return float(np.mean(scores))


def compute_team_stats(team_id: int, recent_history: List[Dict]) -> Dict[str, float]:
    """Compute team-level stats from recent matches."""
    if not recent_history:
        return {"team_clean_sheet_rate": 0.25, "team_avg_points": 1.0}

    # This is simplified - would need opponent data for proper team stats
    # Placeholder: return zeros for now
    return {}


def compute_start_probability(recent_minutes: List[int]) -> float:
    """Estimate probability of starting based on recent minutes."""
    if not recent_minutes:
        return 0.5

    recent_4 = recent_minutes[-4:] if len(recent_minutes) >= 4 else recent_minutes
    avg_mins = np.mean(recent_4)
    return min(1.0, avg_mins / 90.0)


def encode_status(status: str) -> int:
    """Encode injury status to integer."""
    status_map = {"a": 0, "d": 1, "i": 2, "u": 2, "s": 3}
    return status_map.get(status[0].lower() if status else "a", 0)


def prepare_training_data(
    features_df: pd.DataFrame,
    feature_columns: List[str],
    transformed_columns: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix X and target y.

    Args:
        features_df: DataFrame with all features
        feature_columns: List of column names to use as features

    Returns:
        X: Feature matrix (n_samples x n_features)
        y: Target vector (n_samples)
        transformed_columns: Final transformed feature columns used by model/scaler
    """
    logger.info("Preparing training data...")

    # Select features
    X = features_df[feature_columns].copy()
    y = features_df["target_points"].copy()

    # Handle missing values
    X = X.fillna(0)  # Simple fill for now - TODO: more sophisticated imputation

    # Position one-hot encoding
    position_dummies = pd.get_dummies(X["position"], prefix="pos")
    X = pd.concat([X.drop("position", axis=1), position_dummies], axis=1)

    if transformed_columns is None:
        transformed_columns = X.columns.tolist()
    else:
        X = X.reindex(columns=transformed_columns, fill_value=0)

    # Team one-hot encoding (optional - might be too many)
    # Consider using team strength features instead

    return X.values, y.values, transformed_columns


def tune_xgb_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict,
) -> Dict:
    """
    Tune XGBoost hyperparameters using Bayesian optimization or grid search.

    Returns: Best parameters dict
    """
    logger.info("Tuning XGBoost hyperparameters...")

    # Simple random search for skeleton
    # In practice: use BayesSearchCV or Optuna
    best_score = float("inf")
    best_params = None

    base_params = {
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "mae",
    }

    # Sample random combinations (10 iterations for skeleton)
    param_combinations = []
    keys = list(param_grid.keys())
    for _ in range(10):
        combo = {k: np.random.choice(param_grid[k]) for k in keys}
        param_combinations.append(combo)

    for params in param_combinations:
        full_params = {**base_params, **params}
        model = xgb.XGBRegressor(**full_params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_score:
            best_score = val_mae
            best_params = full_params

        logger.info(f"Params: {params}, Val MAE: {val_mae:.4f}")

    logger.info(f"Best XGB params: {best_params}, Best MAE: {best_score:.4f}")
    return best_params


def train_xgb_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict,
) -> xgb.XGBRegressor:
    """Train final XGBoost model with best hyperparameters."""
    logger.info("Training final XGBoost model...")

    model = xgb.XGBRegressor(**best_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
        early_stopping_rounds=50,
    )

    return model


def build_nn_model(
    input_dim: int, hidden_layers: List[int], dropout_rate: float, l2_reg: float
) -> tf.keras.Model:
    """
    Build feedforward neural network architecture.

    See ML-MODELS-DESIGN.md Section 3.1 for architecture details.
    """
    inputs = tf.keras.Input(shape=(input_dim,))

    # Normalization
    x = layers.Normalization()(inputs)

    # Hidden layers
    for units in hidden_layers:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_nn_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tf.keras.Model:
    """Train neural network with early stopping."""
    logger.info("Training Neural Network...")

    model = build_nn_model(
        input_dim=input_dim,
        hidden_layers=NN_HIDDEN_LAYERS,
        dropout_rate=NN_DROPOUT_RATE,
        l2_reg=NN_L2_REG,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_mae", patience=30, restore_best_weights=True, mode="min"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5, patience=10, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )

    return model


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if isinstance(model, xgb.XGBRegressor):
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test, verbose=0).flatten()

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "median_ae": np.median(np.abs(y_pred - y_test)),
        "within_2pts": (np.abs(y_pred - y_test) <= 2).mean(),
        "within_4pts": (np.abs(y_pred - y_test) <= 4).mean(),
    }

    logger.info(f"{model_name} metrics: {metrics}")
    return metrics


def save_models(
    xgb_model: xgb.XGBRegressor,
    nn_model: tf.keras.Model,
    scaler: StandardScaler,
    feature_columns: List[str],
    metrics: Dict[str, Dict],
    version: str,
):
    """Save models, scaler, and metadata to disk."""
    model_dir = MODELS_DIR / version
    xgb_dir = model_dir / "xgb"
    nn_dir = model_dir / "nn"
    model_dir.mkdir(parents=True, exist_ok=True)
    xgb_dir.mkdir(parents=True, exist_ok=True)
    nn_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving models to {model_dir}")

    # Save XGBoost
    joblib.dump(xgb_model, xgb_dir / "model.pkl")
    xgb_model.save_model(str(xgb_dir / "model.json"))

    # Save NN
    nn_model.save(nn_dir / "model.keras")

    # Save scaler
    joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save feature columns
    pd.DataFrame({"feature": feature_columns}).to_csv(
        xgb_dir / "feature_columns.csv", index=False
    )
    pd.DataFrame({"feature": feature_columns}).to_csv(
        nn_dir / "feature_columns.csv", index=False
    )

    # Save metadata
    metadata = {
        "version": version,
        "training_date": datetime.now().isoformat(),
        "features": feature_columns,
        "metrics": metrics,
        "xgb_params": xgb_model.get_params(),
        "nn_architecture": {
            "hidden_layers": NN_HIDDEN_LAYERS,
            "dropout": NN_DROPOUT_RATE,
            "l2_reg": NN_L2_REG,
        },
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Update 'current' symlink
    current_link = MODELS_DIR / "current"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(model_dir)

    logger.info(f"Models saved successfully as version {version}")


def create_version_tag() -> str:
    """Create version string based on date and counter."""
    today = datetime.now().strftime("%Y_%m_%d")
    version = f"2025_v1_{today}"
    return version


def main():
    parser = argparse.ArgumentParser(description="Train ML models for FPL")
    parser.add_argument(
        "--season", type=str, default="2024-25", help="Season to train on"
    )
    parser.add_argument("--test-gw", type=int, default=38, help="Final test gameweek")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version tag (auto-generated if None)",
    )
    args = parser.parse_args()

    # Set test cutoff
    global TEST_GW
    TEST_GW = args.test_gw

    # 1. Load data
    features_df, context_data = load_data([args.season])

    if len(features_df) < 10000:
        logger.warning(f"Only {len(features_df)} samples - may need more data")

    # 2. Define feature columns
    # Start with base features (excluding ID, target, and derived interactions)
    base_features = [
        "position",
        "now_cost",
        "team",
        "form_points",
        "form_minutes",
        "form_xg",
        "form_xa",
        "form_ict",
        "goals_per_90",
        "assists_per_90",
        "fixture_score",
        "double_gw_flag",
        "blank_gw_flag",
        "recent_4_minutes_avg",
        "start_probability",
        "ict_per_90",
        "understat_matched",
        "understat_xG_per_90",
        "understat_xA_per_90",
        "injury_status",
        "availability_factor",
    ]

    # TODO: Add interaction features, team stats, etc.
    feature_columns = base_features.copy()

    # 3. Time-based split
    logger.info("Splitting data by time...")
    train_mask = features_df["target_gw"] < (TEST_GW - 3)
    val_mask = (features_df["target_gw"] >= (TEST_GW - 3)) & (
        features_df["target_gw"] < TEST_GW
    )
    test_mask = features_df["target_gw"] >= TEST_GW

    train_df = features_df[train_mask]
    val_df = features_df[val_mask]
    test_df = features_df[test_mask]

    logger.info(
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} samples"
    )

    # 4. Prepare feature matrices
    X_train, y_train, transformed_feature_columns = prepare_training_data(
        train_df, feature_columns
    )
    X_val, y_val, _ = prepare_training_data(
        val_df, feature_columns, transformed_feature_columns
    )
    X_test, y_test, _ = prepare_training_data(
        test_df, feature_columns, transformed_feature_columns
    )

    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 6. Tune and train XGBoost
    best_xgb_params = tune_xgb_hyperparameters(
        X_train_scaled, y_train, X_val_scaled, y_val, XGB_PARAM_GRID
    )
    xgb_model = train_xgb_final(
        X_train_scaled, y_train, X_val_scaled, y_val, best_xgb_params
    )

    # 7. Train Neural Network
    nn_model = train_nn_final(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        input_dim=X_train_scaled.shape[1],
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        learning_rate=NN_LEARNING_RATE,
    )

    # 8. Evaluate all models
    metrics = {}
    metrics["xgb"] = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    metrics["nn"] = evaluate_model(nn_model, X_test_scaled, y_test, "Neural Network")

    # Ensemble evaluation
    xgb_pred = xgb_model.predict(X_test_scaled)
    nn_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
    ensemble_pred = 0.6 * xgb_pred + 0.4 * nn_pred  # Simple average - tune weights
    metrics["ensemble"] = {
        "mae": mean_absolute_error(y_test, ensemble_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        "r2": r2_score(y_test, ensemble_pred),
    }
    logger.info(f"Ensemble metrics: {metrics['ensemble']}")

    # 9. Save models
    version = args.version or create_version_tag()
    save_models(
        xgb_model,
        nn_model,
        scaler,
        transformed_feature_columns,
        metrics,
        version,
    )

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
