"""
ML Predictor Module for FPL Lineup Optimizer

Production inference module that loads trained models and provides
point predictions with confidence scores for FPL players.

Usage:
    from ml_predictor import MLPredictor

    predictor = MLPredictor(model_version='current')
    pred, conf = predictor.predict_player(player_id, gameweek)

Author: FPL ML Integration Team
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf

# Local imports (adjust based on project structure)
try:
    from fpl_service import FPLService
    from historical_data_service import HistoricalDataService
    from minutes_predictor import MinutesPredictor
except ImportError as e:
    logging.warning(f"Could not import FPL modules: {e}")

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Main class for ML-based FPL point predictions.

    Loads trained XGBoost and NN models, performs feature extraction,
    and generates ensemble predictions with confidence scores.
    """

    def __init__(
        self,
        model_version: str = "current",
        data_dir: str = "data",
        models_dir: str = "models",
    ):
        """
        Initialize ML Predictor.

        Args:
            model_version: Version name or 'current' (symlink)
            data_dir: Directory for cached data
            models_dir: Directory containing model versions
        """
        self.model_version = model_version
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

        # Will be loaded on demand
        self._xgb_model = None
        self._nn_model = None
        self._scaler = None
        self._feature_columns = None
        self._ensemble_weights = None
        self._metadata = None

        logger.info(f"MLPredictor initialized (version: {model_version})")

    @property
    def model_path(self) -> Path:
        """Get path to model version directory."""
        if self.model_version == "current":
            return self.models_dir / "current"
        return self.models_dir / self.model_version

    def load_models(self):
        """Load all models and scaler from disk."""
        if self._xgb_model is not None:
            logger.debug("Models already loaded")
            return

        model_path = self.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model version not found: {model_path}")

        logger.info(f"Loading models from {model_path}")

        # Load XGBoost
        xgb_path = model_path / "xgb" / "model.pkl"
        if xgb_path.exists():
            self._xgb_model = joblib.load(xgb_path)
        else:
            # Try JSON format
            xgb_json = model_path / "xgb" / "model.json"
            if xgb_json.exists():
                self._xgb_model = xgb.XGBRegressor()
                self._xgb_model.load_model(str(xgb_json))

        # Load Neural Network
        nn_path = model_path / "nn" / "model.keras"
        if nn_path.exists():
            self._nn_model = tf.keras.models.load_model(str(nn_path))
        else:
            logger.warning("NN model file not found")

        # Load scaler
        scaler_path = model_path / "scaler.pkl"
        self._scaler = joblib.load(scaler_path)

        # Load feature columns
        feat_path = model_path / "xgb" / "feature_columns.csv"
        self._feature_columns = pd.read_csv(feat_path)["feature"].tolist()

        # Load metadata
        meta_path = model_path / "metadata.json"
        with open(meta_path) as f:
            self._metadata = json.load(f)

        # Ensemble weights (from metadata or default)
        self._ensemble_weights = self._metadata.get(
            "ensemble_weights", {"xgb": 0.6, "nn": 0.4}
        )

        logger.info(
            f"Models loaded successfully. Features: {len(self._feature_columns)}"
        )

    def extract_features(
        self,
        player_id: int,
        target_gw: int,
        players_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        fixtures: List[Dict],
        history_df: pd.DataFrame,
        use_understat: bool = True,
        minutes_predictor: Optional[MinutesPredictor] = None,
    ) -> pd.DataFrame:
        """
        Extract complete feature vector for a player.

        Mirrors the training pipeline feature extraction.
        See ML-MODELS-DESIGN.md Section 1.2.

        Returns: DataFrame with one row, columns matching training features
        """
        # Get player record
        player = players_df[players_df["id"] == player_id].iloc[0]

        # Get player's historical data up to target_gw
        player_history = history_df[
            (history_df["player_id"] == player_id) & (history_df["round"] < target_gw)
        ].sort_values("round")

        # Build feature dict
        features = {
            "player_id": player_id,
            "target_gw": target_gw,
            "position": player["element_type"],
            "now_cost": player["now_cost"],
            "team": player["team"],
        }

        if len(player_history) == 0:
            # No history - use baselines
            pos_baselines = {1: 1.5, 2: 2.5, 3: 4.0, 4: 5.0}
            features["form_points"] = pos_baselines[player["element_type"]]
            features["recent_4_minutes_avg"] = 0.0
            # ... fill other baselines
            return pd.DataFrame([features]).reindex(
                columns=self._feature_columns, fill_value=0
            )

        # Compute form features (weighted last 6 GWs)
        recent_history = player_history.tail(6)
        features.update(self._compute_form_features(recent_history))

        # Season aggregates
        total_minutes = player_history["minutes"].sum()
        if total_minutes > 270:
            goals = player_history["goals_scored"].sum()
            assists = player_history["assists"].sum()
            features["goals_per_90"] = goals / (total_minutes / 90)
            features["assists_per_90"] = assists / (total_minutes / 90)
        else:
            pos_xg_baselines = {1: 0.0, 2: 0.03, 3: 0.08, 4: 0.35}
            pos_xa_baselines = {1: 0.0, 2: 0.04, 3: 0.08, 4: 0.12}
            features["goals_per_90"] = pos_xg_baselines[player["element_type"]]
            features["assists_per_90"] = pos_xa_baselines[player["element_type"]]

        # Fixture features
        team_id = player["team"]
        features["fixture_score"] = self._compute_fixture_score(
            team_id, target_gw, teams_df, fixtures
        )
        team_fixtures = self._get_fixtures_for_team(team_id, target_gw, fixtures)
        features["double_gw_flag"] = len(team_fixtures) > 1
        features["blank_gw_flag"] = len(team_fixtures) == 0

        # Minutes/Rotation
        recent_mins = player_history.tail(5)["minutes"].tolist()
        features["recent_4_minutes_avg"] = (
            np.mean(recent_mins[-4:]) if len(recent_mins) >= 4 else 0.0
        )
        features["start_probability"] = self._compute_start_prob(recent_mins)

        # ICT values from FPL history can be strings; coerce before aggregation.
        ict_total = pd.to_numeric(player_history["ict_index"], errors="coerce").fillna(0).sum()
        if total_minutes > 0:
            features["ict_per_90"] = ict_total / (total_minutes / 90)
        else:
            features["ict_per_90"] = 0.0

        # Understat (placeholder - would integrate with UnderstatService)
        features["understat_matched"] = 0
        features["understat_xG_per_90"] = 0.0
        features["understat_xA_per_90"] = 0.0

        # Injury/Availability
        status_code = self._encode_status(player.get("status", "a"))
        features["injury_status"] = status_code
        availability = player.get("chance_of_playing_next_round", 100)
        features["availability_factor"] = (
            float(availability) / 100.0 if pd.notna(availability) else 1.0
        )

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # Reindex to match training feature order, fill missing with 0
        features_df = features_df.reindex(columns=self._feature_columns, fill_value=0)

        # Some training variants use numeric `position`, others use one-hot `pos_*`.
        need_pos_dummies = any(col.startswith("pos_") for col in self._feature_columns)
        if need_pos_dummies and "position" in features_df.columns:
            pos_dummies = pd.get_dummies(features_df["position"], prefix="pos")
            features_df = pd.concat(
                [features_df.drop("position", axis=1), pos_dummies], axis=1
            )
            for pos in [1, 2, 3, 4]:
                col = f"pos_{pos}"
                if col not in features_df.columns:
                    features_df[col] = 0

        # Final alignment to model contract.
        features_df = features_df.reindex(columns=self._feature_columns, fill_value=0)
        return features_df

    def _compute_form_features(self, recent_history: pd.DataFrame) -> Dict[str, float]:
        """Compute weighted form features from recent match history."""
        if len(recent_history) == 0:
            return {}

        n = len(recent_history)
        weights = np.exp(np.linspace(-1, 0, n))
        weights = weights / weights.sum()

        points = recent_history["total_points"].values
        minutes = recent_history["minutes"].values
        xg = recent_history["expected_goals"].astype(float).fillna(0).values
        xa = recent_history["expected_assists"].astype(float).fillna(0).values
        ict = recent_history["ict_index"].astype(float).fillna(0).values

        # Weighted averages
        form_points = np.dot(points, weights[-n:])
        form_minutes = np.dot(minutes, weights[-n:])

        total_mins = max(minutes.sum(), 1)
        form_xg = (xg * weights[-n:]).sum() / (total_mins / 90)
        form_xa = (xa * weights[-n:]).sum() / (total_mins / 90)
        form_ict = (ict * weights[-n:]).sum() / (total_mins / 90)

        return {
            "form_points": float(form_points),
            "form_minutes": float(form_minutes),
            "form_xg": float(form_xg),
            "form_xa": float(form_xa),
            "form_ict": float(form_ict),
        }

    def _compute_fixture_score(
        self, team_id: int, gw: int, teams_df: pd.DataFrame, fixtures: List[Dict]
    ) -> float:
        """Compute fixture quality score (0-1)."""
        team_fixtures = [
            f
            for f in fixtures
            if f["event"] == gw and (f["team_h"] == team_id or f["team_a"] == team_id)
        ]
        if not team_fixtures:
            return 0.5

        scores = []
        team_row = teams_df[teams_df["id"] == team_id]
        if len(team_row) == 0:
            return 0.5

        team_attack_home = team_row.iloc[0]["strength_attack_home"]

        for f in team_fixtures:
            is_home = f["team_h"] == team_id
            opponent_id = f["team_a"] if is_home else f["team_h"]
            opponent_row = teams_df[teams_df["id"] == opponent_id]
            if len(opponent_row) == 0:
                continue
            opponent_defense = opponent_row.iloc[0][
                "strength_defence_home" if is_home else "strength_defence_away"
            ]

            attack_factor = team_attack_home / 1000
            defense_factor = (1000 - opponent_defense) / 1000
            xg_env = attack_factor * defense_factor
            scores.append(xg_env)

        return float(np.mean(scores)) if scores else 0.5

    def _get_fixtures_for_team(
        self, team_id: int, gw: int, fixtures: List[Dict]
    ) -> List[Dict]:
        """Helper to get fixtures for team in a GW."""
        return [
            f
            for f in fixtures
            if f["event"] == gw and (f["team_h"] == team_id or f["team_a"] == team_id)
        ]

    def _compute_start_prob(self, recent_mins: List[int]) -> float:
        """Compute start probability from recent minutes."""
        if not recent_mins:
            return 0.5
        recent_4 = recent_mins[-4:] if len(recent_mins) >= 4 else recent_mins
        avg_mins = np.mean(recent_4)
        return min(1.0, avg_mins / 90.0)

    def _encode_status(self, status: str) -> int:
        """Encode injury status."""
        status_map = {"a": 0, "d": 1, "i": 2, "u": 2, "s": 3}
        return status_map.get(status[0].lower() if status else "a", 0)

    def predict_player(
        self,
        player_id: int,
        target_gw: int,
        context_data: Optional[Dict] = None,
        use_ml: bool = True,
    ) -> Tuple[float, float]:
        """
        Predict expected points for a single player.

        Args:
            player_id: FPL player element ID
            target_gw: Target gameweek
            context_data: Optional pre-loaded context (players_df, teams_df, fixtures, history_df)
            use_ml: If False, raises error (this is ML predictor only)

        Returns:
            predicted_points: Expected points (float)
            confidence: Confidence score 0-100
        """
        if not use_ml:
            raise ValueError(
                "MLPredictor only provides ML predictions. Use PointPredictor for rule-based."
            )

        self.load_models()

        # Get context data if not provided
        if context_data is None:
            context_data = self._load_context_data()

        players_df = context_data["players"]
        teams_df = context_data["teams"]
        fixtures = context_data["fixtures"]
        history_df = context_data["history"]

        # Extract features
        features_df = self.extract_features(
            player_id=player_id,
            target_gw=target_gw,
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures,
            history_df=history_df,
        )

        # Scale features
        X_scaled = self._scaler.transform(features_df.values)

        # Get predictions from both models
        xgb_pred = self._xgb_model.predict(X_scaled)[0]
        nn_pred = self._nn_model.predict(X_scaled, verbose=0)[0][0]

        # Ensemble
        w_xgb = self._ensemble_weights["xgb"]
        w_nn = self._ensemble_weights["nn"]
        ensemble_pred = w_xgb * xgb_pred + w_nn * nn_pred

        # Confidence score (simple: model agreement + data quality)
        model_diff = abs(xgb_pred - nn_pred)
        max_pred = max(abs(xgb_pred), abs(nn_pred), 1.0)
        agreement = max(0.0, 1.0 - (model_diff / max_pred))

        # Data quality factor (based on availability, minutes)
        availability = features_df["availability_factor"].iloc[0]
        start_prob = features_df["start_probability"].iloc[0]
        data_quality = (availability + start_prob) / 2.0

        confidence = (agreement * 0.6 + data_quality * 0.4) * 100.0
        confidence = max(0.0, min(100.0, confidence))

        return float(ensemble_pred), float(confidence)

    def predict_batch(
        self, player_ids: List[int], target_gw: int, context_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple players.

        Returns: DataFrame with columns [player_id, predicted_points, confidence]
        """
        self.load_models()

        if context_data is None:
            context_data = self._load_context_data()

        players_df = context_data["players"]
        teams_df = context_data["teams"]
        fixtures = context_data["fixtures"]
        history_df = context_data["history"]

        predictions = []

        for player_id in player_ids:
            try:
                pred, conf = self.predict_player(
                    player_id=player_id, target_gw=target_gw, context_data=context_data
                )
                predictions.append(
                    {
                        "player_id": player_id,
                        "predicted_points": pred,
                        "confidence": conf,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to predict player {player_id}: {e}")
                # Fallback: position baseline
                pos = players_df[players_df["id"] == player_id].iloc[0]["element_type"]
                baseline = {1: 1.5, 2: 2.5, 3: 4.0, 4: 5.0}.get(pos, 3.0)
                predictions.append(
                    {
                        "player_id": player_id,
                        "predicted_points": baseline,
                        "confidence": 30.0,  # Low confidence
                    }
                )

        return pd.DataFrame(predictions)

    def _load_context_data(self) -> Dict[str, Any]:
        """Load current FPL context data."""
        logger.debug("Loading FPL context data...")

        fpl_service = FPLService()
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures = data["fixtures"]

        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])

        # Load history cache
        history_service = HistoricalDataService()
        # Build history DataFrame
        history_records = []
        for _, player in players_df.iterrows():
            pid = str(player["id"])
            player_history = history_service._history_cache.get(pid, {}).get(
                "history", []
            )
            for record in player_history:
                record["player_id"] = player["id"]
                history_records.append(record)

        history_df = pd.DataFrame(history_records)

        return {
            "players": players_df,
            "teams": teams_df,
            "fixtures": fixtures,
            "history": history_df,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about loaded model."""
        if self._metadata is None:
            self.load_models()

        return {
            "version": self.model_version,
            "training_date": self._metadata.get("training_date"),
            "features": len(self._feature_columns),
            "metrics": self._metadata.get("metrics"),
            "ensemble_weights": self._ensemble_weights,
        }


# ============================================================================
# Convenience functions
# ============================================================================


def predict_for_gameweek(
    gameweek: int, use_ml: bool = True, model_version: str = "current"
) -> pd.DataFrame:
    """
    Convenience function: predict all players for a gameweek.

    Returns: DataFrame with player info + predictions + confidence
    """
    predictor = MLPredictor(model_version=model_version)

    context = predictor._load_context_data()
    player_ids = context["players"]["id"].tolist()

    predictions = predictor.predict_batch(player_ids, gameweek, context_data=context)

    # Merge with player info
    result = context["players"][
        ["id", "web_name", "team", "element_type", "now_cost"]
    ].copy()
    result = result.merge(predictions, left_on="id", right_on="player_id", how="left")

    return result


def get_predictions_with_threshold(
    predictions_df: pd.DataFrame, confidence_threshold: float = 50.0
) -> pd.DataFrame:
    """
    Filter predictions by confidence and add adjusted points.

    For optimizer: use adjusted_points = predicted_points * (confidence/100)
    """
    df = predictions_df.copy()
    df["adjusted_points"] = df["predicted_points"] * (df["confidence"] / 100.0)
    df["above_threshold"] = df["confidence"] >= confidence_threshold
    return df


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    try:
        predictor = MLPredictor()

        # Predict for next gameweek
        next_gw = 30  # TODO: get from FPL API
        results = predict_for_gameweek(next_gw, model_version="current")

        print(f"\nTop 20 predictions for GW {next_gw}:")
        print(
            results.sort_values("predicted_points", ascending=False)[
                ["web_name", "team", "element_type", "predicted_points", "confidence"]
            ].head(20)
        )

        print(f"\nModel info: {predictor.get_model_info()}")

    except Exception as e:
        logger.error(f"Error running prediction: {e}")
