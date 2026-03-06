# ML Prediction Service Integration Design

## Overview

This document outlines the production-ready integration of an ML prediction service into the FPL Lineup Optimizer. The design ensures optional ML usage with seamless fallback to the existing rule-based system, maintaining 100% backward compatibility while enabling future ML enhancements.

---

## 1. Architecture Principles

- **Optional & Gradual**: ML service is opt-in via configuration; system works perfectly without it
- **Graceful Degradation**: ML failures automatically fall back to rule-based predictions
- **Lazy Loading**: ML models load only when needed and first called
- **Performance First**: Batch predictions, caching, and async operations
- **Observable**: Comprehensive logging of ML vs rule-based usage and confidence scores
- **No Circular Dependencies**: Clean separation between ML service and point predictor

---

## 2. New Files Structure

```
backend/
├── ml_predictor.py          # ML prediction service wrapper (NEW)
├── ml_config.py            # Configuration management (NEW)
├── point_predictor.py      # MODIFIED - adds ML integration
├── main.py                 # MODIFIED - new endpoints and ML usage
└── ml_cache.py            # Optional - separate caching layer
```

---

## 3. Configuration Management

### 3.1 Environment Variables

Add these to `.env` or Render configuration:

```bash
# ML Service Configuration
ML_ENABLED=true                    # Enable ML predictions (default: false)
ML_MODEL_PATH=/app/models/ml_fpl_predictor.pkl  # Model file path
ML_MODEL_URL=https://ml-api.example.com/predict   # Remote API endpoint
ML_USE_REMOTE=false                # Use remote API instead of local model
ML_BATCH_SIZE=100                  # Batch prediction size (default: 50)
ML_TIMEOUT_SECONDS=30             # Prediction timeout (default: 10)
ML_CACHE_TTL_MINUTES=60           # Cache TTL (default: 30)
ML_CONFIDENCE_THRESHOLD=0.3       # Min confidence to use ML (default: 0.2)
ML_FALLBACK_ON_ERROR=true         # Fallback to rule-based on ML failure
ML_LOG_PREDICTIONS=false          # Log individual predictions (debug only)
```

### 3.2 Configuration Class (`ml_config.py`)

```python
"""
ML Service Configuration Management
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLConfig:
    """Configuration for ML prediction service."""
    enabled: bool = False
    model_path: Optional[str] = None
    model_url: Optional[str] = None
    use_remote: bool = False
    batch_size: int = 50
    timeout_seconds: int = 10
    cache_ttl_minutes: int = 30
    confidence_threshold: float = 0.2
    fallback_on_error: bool = True
    log_predictions: bool = False
    
    @classmethod
    def from_env(cls) -> "MLConfig":
        """Load configuration from environment variables."""
        def get_bool(var: str, default: bool) -> bool:
            val = os.getenv(var, str(default)).lower()
            return val in ("true", "1", "yes", "on")
        
        def get_int(var: str, default: int) -> int:
            try:
                return int(os.getenv(var, default))
            except (ValueError, TypeError):
                return default
        
        def get_float(var: str, default: float) -> float:
            try:
                return float(os.getenv(var, default))
            except (ValueError, TypeError):
                return default
        
        return cls(
            enabled=get_bool("ML_ENABLED", False),
            model_path=os.getenv("ML_MODEL_PATH"),
            model_url=os.getenv("ML_MODEL_URL"),
            use_remote=get_bool("ML_USE_REMOTE", False),
            batch_size=get_int("ML_BATCH_SIZE", 50),
            timeout_seconds=get_int("ML_TIMEOUT_SECONDS", 10),
            cache_ttl_minutes=get_int("ML_CACHE_TTL_MINUTES", 30),
            confidence_threshold=get_float("ML_CONFIDENCE_THRESHOLD", 0.2),
            fallback_on_error=get_bool("ML_FALLBACK_ON_ERROR", True),
            log_predictions=get_bool("ML_LOG_PREDICTIONS", False)
        )
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.enabled:
            if not self.use_remote and not self.model_path:
                errors.append("ML_MODEL_PATH is required when ML_ENABLED=true and ML_USE_REMOTE=false")
            if self.use_remote and not self.model_url:
                errors.append("ML_MODEL_URL is required when ML_USE_REMOTE=true")
            if self.batch_size <= 0:
                errors.append("ML_BATCH_SIZE must be positive")
            if self.timeout_seconds <= 0:
                errors.append("ML_TIMEOUT_SECONDS must be positive")
            if not (0 <= self.confidence_threshold <= 1):
                errors.append("ML_CONFIDENCE_THRESHOLD must be between 0 and 1")
        return errors
```

---

## 4. ML Prediction Service Wrapper

### 4.1 `ml_predictor.py` - Full Implementation

```python
"""
ML Prediction Service for FPL

Provides a unified interface for ML-based point predictions with:
- Lazy model loading
- Batch prediction support
- Prediction caching
- Graceful error handling
- Observability and logging
"""
import logging
import time
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import lru_cache

from .ml_config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Container for ML prediction results."""
    player_id: int
    gameweek: int
    predicted_points: float
    confidence: float  # 0-1 confidence score from model
    features: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    prediction_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "player_id": self.player_id,
            "gameweek": self.gameweek,
            "predicted_points": round(self.predicted_points, 2),
            "confidence": round(self.confidence, 3),
            "features": self.features,
            "model_version": self.model_version,
            "timestamp": self.prediction_time
        }


class MLPredictionService:
    """
    ML-based point prediction service with lazy loading and caching.
    
    Features:
    - Lazy model loading (only on first prediction)
    - Batch prediction support
    - In-memory caching with TTL
    - Automatic fallback to rule-based
    - Comprehensive logging
    - Async-compatible (if needed)
    """
    
    def __init__(self, config: MLConfig, model_version: str = "1.0.0"):
        """
        Initialize ML prediction service.
        
        Args:
            config: ML configuration
            model_version: Version identifier for model tracking
        """
        self.config = config
        self.model_version = model_version
        self._model = None
        self._model_loaded = False
        self._cache: Dict[Tuple[int, int], MLPrediction] = {}
        self._stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "ml_used": 0,
            "fallback_count": 0,
            "errors": 0,
            "batch_predictions": 0,
            "avg_prediction_time_ms": 0
        }
        self._last_cache_cleanup = time.time()
        
        logger.info(f"ML Prediction Service initialized (enabled={config.enabled})")
    
    def is_available(self) -> bool:
        """Check if ML service is available and ready."""
        if not self.config.enabled:
            return False
        
        if self._model_loaded:
            return True
        
        # Try to load model if not already loaded
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"ML model not available: {e}")
            return False
    
    def _load_model(self) -> None:
        """Lazy load the ML model."""
        if self._model_loaded:
            return
        
        if self.config.use_remote:
            logger.info("Using remote ML API (no local model loading required)")
            self._model = {"type": "remote", "url": self.config.model_url}
        else:
            logger.info(f"Loading local ML model from {self.config.model_path}")
            with open(self.config.model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            # Validate model has predict method
            if not hasattr(self._model, 'predict'):
                raise ValueError("Loaded model does not have 'predict' method")
        
        self._model_loaded = True
        logger.info(f"ML model loaded successfully (version={self.model_version})")
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        ttl_seconds = self.config.cache_ttl_minutes * 60
        
        expired_keys = []
        for key, pred in self._cache.items():
            if now - pred.prediction_time > ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self._last_cache_cleanup = now
    
    def _get_cache_key(self, player_id: int, gameweek: int) -> Tuple[int, int]:
        """Generate cache key for prediction."""
        return (player_id, gameweek)
    
    def _get_from_cache(self, player_id: int, gameweek: int) -> Optional[MLPrediction]:
        """Retrieve prediction from cache if available and not expired."""
        # Periodic cache cleanup (every 100 requests)
        if self._stats["total_predictions"] % 100 == 0:
            self._cleanup_cache()
        
        key = self._get_cache_key(player_id, gameweek)
        pred = self._cache.get(key)
        
        if pred:
            ttl_seconds = self.config.cache_ttl_minutes * 60
            if time.time() - pred.prediction_time <= ttl_seconds:
                self._stats["cache_hits"] += 1
                return pred
            else:
                del self._cache[key]
        
        return None
    
    def _store_in_cache(self, prediction: MLPrediction) -> None:
        """Store prediction in cache."""
        key = self._get_cache_key(prediction.player_id, prediction.gameweek)
        self._cache[key] = prediction
    
    def _extract_features(self, player: pd.Series, gameweek: int, 
                         team_strengths: Dict, fixtures: List[Dict]) -> Dict[str, float]:
        """
        Extract features for ML model from player data.
        
        This adapts FPL data to match model's expected feature set.
        Modify based on your actual model requirements.
        """
        features = {
            # Basic features
            "element_type": float(player.get('element_type', 0)),
            "now_cost": float(player.get('now_cost', 0)),
            "form": float(player.get('form', 0) or 0),
            
            # xG/xA metrics (Understat if available, otherwise FPL)
            "expected_goals": float(player.get('expected_goals', 0) or 0),
            "expected_assists": float(player.get('expected_assists', 0) or 0),
            
            # Minutes/playing time
            "minutes": float(player.get('minutes', 0) or 0),
            "starts": float(player.get('starts', 0) or 0),
            
            # ICT index
            "ict_index": float(player.get('ict_index', 0) or 0),
            "influence": float(player.get('influence', 0) or 0),
            "creativity": float(player.get('creativity', 0) or 0),
            "threat": float(player.get('threat', 0) or 0),
            
            # Team/fixture features
            "fixture_difficulty": self._calculate_avg_fixture_difficulty(player, fixtures),
            "team_attack_strength": self._get_team_attack_strength(player, team_strengths),
            "team_defense_strength": self._get_team_defense_strength(player, team_strengths),
            
            # Context
            "gameweek": float(gameweek),
            "is_home_game": 1.0 if fixtures and fixtures[0].get('is_home', False) else 0.0,
            
            # Historical performance (if available)
            "points_per_game": float(player.get('points_per_game', 0) or 0),
            "total_points": float(player.get('total_points', 0) or 0),
        }
        
        # Add derived features
        features["xG_per_90"] = (features["expected_goals"] / max(features["minutes"], 1)) * 90
        features["xA_per_90"] = (features["expected_assists"] / max(features["minutes"], 1)) * 90
        
        return features
    
    def _calculate_avg_fixture_difficulty(self, player: pd.Series, fixtures: List[Dict]) -> float:
        """Calculate average fixture difficulty for player's team."""
        if not fixtures:
            return 3.0  # Average difficulty
        
        difficulties = [f.get('difficulty', 3) for f in fixtures]
        return float(np.mean(difficulties))
    
    def _get_team_attack_strength(self, player: pd.Series, team_strengths: Dict) -> float:
        """Get team attack strength rating."""
        team_id = player.get('team')
        if not team_id:
            return 1000.0
        return float(team_strengths.get('attack', {}).get(team_id, 1000))
    
    def _get_team_defense_strength(self, player: pd.Series, team_strengths: Dict) -> float:
        """Get team defense strength rating."""
        team_id = player.get('team')
        if not team_id:
            return 1000.0
        return float(team_strengths.get('defense', {}).get(team_id, 1000))
    
    def predict(self, player: pd.Series, gameweek: int, 
                team_strengths: Optional[Dict] = None,
                fixtures: Optional[List[Dict]] = None) -> MLPrediction:
        """
        Predict points for a single player.
        
        Args:
            player: Player data Series
            gameweek: Target gameweek
            team_strengths: Precomputed team strength dict (optional)
            fixtures: Fixture list for player's team (optional)
            
        Returns:
            MLPrediction object with points and confidence
            
        Raises:
            RuntimeError: If ML service unavailable and fallback disabled
        """
        start_time = time.time()
        self._stats["total_predictions"] += 1
        
        # Check cache first
        cached = self._get_from_cache(player['id'], gameweek)
        if cached:
            logger.debug(f"Cache hit for player {player['id']} GW{gameweek}")
            return cached
        
        try:
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model()
            
            # Extract features
            features = self._extract_features(player, gameweek, team_strengths or {}, fixtures or [])
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            if self.config.use_remote:
                predicted_points = self._predict_remote(feature_df)
                confidence = 0.7  # Default for remote API
            else:
                predicted_points = float(self._model.predict(feature_df)[0])
                # Get confidence if model supports it
                confidence = getattr(self._model, 'confidence', 0.7)
            
            # Sanity check - cap extreme values
            predicted_points = max(0, min(predicted_points, 12.0))
            confidence = max(0, min(confidence, 1.0))
            
            # Create prediction object
            prediction = MLPrediction(
                player_id=int(player['id']),
                gameweek=gameweek,
                predicted_points=predicted_points,
                confidence=confidence,
                features=features,
                model_version=self.model_version,
                prediction_time=time.time()
            )
            
            # Store in cache
            self._store_in_cache(prediction)
            
            # Logging
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats["ml_used"] += 1
            self._stats["avg_prediction_time_ms"] = (
                (self._stats["avg_prediction_time_ms"] * (self._stats["ml_used"] - 1) + elapsed_ms) 
                / self._stats["ml_used"]
            )
            
            if self.config.log_predictions:
                logger.info(f"ML prediction: player={player['id']}, gw={gameweek}, "
                          f"points={predicted_points:.2f}, confidence={confidence:.3f}, "
                          f"time={elapsed_ms:.1f}ms")
            
            return prediction
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"ML prediction failed for player {player['id']} GW{gameweek}: {e}")
            
            if self.config.fallback_on_error:
                self._stats["fallback_count"] += 1
                logger.warning(f"Falling back to rule-based for player {player['id']}")
                raise MLPredictionError(f"ML prediction failed: {e}") from e
            else:
                raise
    
    def _predict_remote(self, feature_df: pd.DataFrame) -> float:
        """
        Call remote ML API for prediction.
        
        Implement based on your remote API specifications.
        """
        import requests
        from requests.exceptions import RequestException
        
        try:
            response = requests.post(
                self.config.model_url,
                json=feature_df.to_dict(orient="records")[0],
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            result = response.json()
            return float(result.get("prediction", 0.0))
        except RequestException as e:
            raise RuntimeError(f"Remote API request failed: {e}") from e
    
    def predict_batch(self, players_df: pd.DataFrame, gameweek: int,
                     team_strengths: Optional[Dict] = None,
                     fixtures_map: Optional[Dict] = None) -> List[MLPrediction]:
        """
        Batch prediction for multiple players.
        
        Args:
            players_df: DataFrame with player data
            gameweek: Target gameweek
            team_strengths: Precomputed team strength dict
            fixtures_map: Mapping of team_id -> fixtures list
            
        Returns:
            List of MLPrediction objects
        """
        start_time = time.time()
        predictions = []
        uncached = []
        uncached_indices = []
        
        # Check cache for each player
        for idx, (_, player) in enumerate(players_df.iterrows()):
            cached = self._get_from_cache(player['id'], gameweek)
            if cached:
                predictions.append(cached)
            else:
                predictions.append(None)  # Placeholder
                uncached.append(player)
                uncached_indices.append(idx)
        
        # Batch predict uncached players
        if uncached:
            logger.info(f"Batch predicting {len(uncached)} players (GW{gameweek})")
            
            try:
                # Extract features for all uncached players
                features_list = []
                for player in uncached:
                    team_fixtures = []
                    if fixtures_map:
                        team_fixtures = fixtures_map.get(player['team'], [])
                    
                    features = self._extract_features(
                        player, gameweek, team_strengths or {}, team_fixtures
                    )
                    features_list.append(features)
                
                feature_df = pd.DataFrame(features_list)
                
                # Call model (local or remote)
                if self.config.use_remote:
                    batch_predictions = []
                    for _, row in feature_df.iterrows():
                        pred = self._predict_remote(pd.DataFrame([row]))
                        batch_predictions.append(pred)
                    point_preds = np.array(batch_predictions)
                    confidences = np.full(len(uncached), 0.7)
                else:
                    point_preds = self._model.predict(feature_df)
                    confidences = getattr(self._model, 'predict_with_confidence', 
                                        lambda X: (point_preds, np.full(len(X), 0.7)))(feature_df)[1]
                
                # Create MLPrediction objects
                for i, (player, points, confidence) in enumerate(zip(uncached, point_preds, confidences)):
                    idx = uncached_indices[i]
                    
                    pred = MLPrediction(
                        player_id=int(player['id']),
                        gameweek=gameweek,
                        predicted_points=max(0, min(float(points), 12.0)),
                        confidence=max(0, min(float(confidence), 1.0)),
                        features=features_list[i],
                        model_version=self.model_version
                    )
                    
                    predictions[idx] = pred
                    self._store_in_cache(pred)
                
                self._stats["batch_predictions"] += 1
                logger.info(f"Batch prediction completed: {len(uncached)} players, "
                          f"time={((time.time() - start_time) * 1000):.1f}ms")
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                self._stats["errors"] += 1
                # Don't fill with None - caller must handle missing predictions
                raise
        
        # Filter out any None (cached ones are always filled)
        return [p for p in predictions if p is not None]
    
    def get_stats(self) -> Dict:
        """Get service statistics for monitoring."""
        total = self._stats["total_predictions"]
        cache_hit_rate = (self._stats["cache_hits"] / total) if total > 0 else 0
        ml_usage_rate = (self._stats["ml_used"] / total) if total > 0 else 0
        error_rate = (self._stats["errors"] / total) if total > 0 else 0
        
        return {
            "enabled": self.config.enabled,
            "model_loaded": self._model_loaded,
            "model_version": self.model_version,
            "total_predictions": total,
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": round(cache_hit_rate, 3),
            "ml_used": self._stats["ml_used"],
            "ml_usage_rate": round(ml_usage_rate, 3),
            "fallback_count": self._stats["fallback_count"],
            "errors": self._stats["errors"],
            "error_rate": round(error_rate, 3),
            "batch_predictions": self._stats["batch_predictions"],
            "avg_prediction_time_ms": round(self._stats["avg_prediction_time_ms"], 1),
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._cache.clear()
        logger.info("ML prediction cache cleared")


class MLPredictionError(Exception):
    """Raised when ML prediction fails and fallback is disabled."""
    pass
```

---

## 5. Point Predictor Modifications

### 5.1 Changes to `point_predictor.py`

The `PointPredictor` class will integrate ML as an optional enhancement while preserving all existing rule-based logic.

```python
# Additional imports at top of file
from typing import Optional
import logging

try:
    from .ml_predictor import MLPredictionService
    from .ml_config import MLConfig
    ML_SERVICE_AVAILABLE = True
except ImportError:
    ML_SERVICE_AVAILABLE = False
    logging.info("ML prediction service not available")


class PointPredictor:
    """
    Advanced expected points calculator for FPL 2025/26.
    
    Now supports optional ML augmentation:
    - ML predictions can replace or supplement rule-based calculations
    - Automatic fallback to pure rule-based if ML unavailable
    - Configurable via environment variables
    """
    
    def __init__(self, players_df: pd.DataFrame, teams_df: pd.DataFrame,
                 fixtures: List[Dict], current_gameweek: int,
                 use_understat: bool = True,
                 use_ml: bool = None,  # New parameter
                 ml_config: Optional[MLConfig] = None):
        """
        Initialize predictor with FPL data.
        
        Args:
            players_df: DataFrame of all players
            teams_df: DataFrame of all teams
            fixtures: List of fixture dicts
            current_gameweek: Current or next gameweek
            use_understat: Whether to use Understat data
            use_ml: Override ML usage (None = use config/env)
            ml_config: ML configuration object (optional)
        """
        self.players = players_df.copy()
        self.teams = teams_df.copy()
        self.fixtures = fixtures
        self.current_gw = current_gameweek
        self.use_understat = use_understat and UNDERSTAT_AVAILABLE
        
        # Initialize ML service if configured
        self.ml_service = None
        self.use_ml = use_ml if use_ml is not None else self._should_use_ml()
        
        if self.use_ml and ML_SERVICE_AVAILABLE:
            try:
                config = ml_config or MLConfig.from_env()
                if config.enabled:
                    self.ml_service = MLPredictionService(config)
                    # Test ML availability (doesn't load model until first use)
                    if self.ml_service.is_available():
                        logging.info("ML prediction service enabled")
                    else:
                        logging.warning("ML service configured but not available, "
                                      "falling back to rule-based")
                        self.ml_service = None
                        self.use_ml = False
                else:
                    self.use_ml = False
            except Exception as e:
                logging.warning(f"Failed to initialize ML service: {e}")
                self.ml_service = None
                self.use_ml = False
        
        # Precompute team strengths
        self._compute_team_strengths()
        
        # Parse fixture data
        self._build_fixture_map()
        
        # Enrich with Understat data if available
        self._understat_enriched = False
        if self.use_understat:
            self._enrich_with_understat()
        
        # Log configuration summary
        self._log_config_summary()
    
    def _should_use_ml(self) -> bool:
        """Determine if ML should be used based on env/config."""
        if not ML_SERVICE_AVAILABLE:
            return False
        return MLConfig.from_env().enabled
    
    def _log_config_summary(self):
        """Log configuration summary on initialization."""
        summary = {
            "use_understat": self.use_understat,
            "use_ml": self.use_ml,
            "understat_enriched": self._understat_enriched,
        }
        logging.info(f"PointPredictor config: {summary}")
    
    def _get_team_strengths_dict(self) -> Dict:
        """Build team strengths dict for ML service."""
        return {
            "attack": {
                "home": dict(zip(self.teams['id'], self.teams['strength_attack_home'])),
                "away": dict(zip(self.teams['id'], self.teams['strength_attack_away']))
            },
            "defense": {
                "home": dict(zip(self.teams['id'], self.teams['strength_defence_home'])),
                "away": dict(zip(self.teams['id'], self.teams['strength_defence_away']))
            }
        }
    
    def _get_fixtures_for_team(self, team_id: int, gameweek: int) -> List[Dict]:
        """Get fixtures for specific team and gameweek."""
        fixtures = []
        key = (team_id, gameweek)
        if key in self.fixture_map:
            for fix in self.fixture_map[key]:
                fix_with_gw = fix.copy()
                fix_with_gw['gameweek'] = gameweek
                fixtures.append(fix_with_gw)
        return fixtures
    
    def predict_gameweek(self, player: pd.Series, gameweek: int) -> Tuple[float, Dict[str, float]]:
        """
        Predict total expected points for a player in a specific gameweek.
        
        Strategy:
        1. Try ML prediction first (if enabled and available)
        2. If ML fails or is disabled, use rule-based calculation
        3. Optionally blend ML and rule-based based on confidence
        
        Returns:
            Tuple of (total_points, breakdown_dict)
        """
        player_id = player['id']
        
        # ML prediction attempt
        if self.use_ml and self.ml_service:
            try:
                # Check if ML is confident enough
                ml_pred = self.ml_service.predict(
                    player=player,
                    gameweek=gameweek,
                    team_strengths=self._get_team_strengths_dict(),
                    fixtures=self._get_fixtures_for_team(player['team'], gameweek)
                )
                
                # Check confidence threshold
                if ml_pred.confidence >= self.ml_service.config.confidence_threshold:
                    logging.debug(f"Using ML prediction for player {player_id} "
                                f"(confidence={ml_pred.confidence:.3f})")
                    
                    breakdown = {
                        'ml_points': ml_pred.predicted_points,
                        'ml_confidence': ml_pred.confidence,
                        'prediction_source': 'ml',
                        'model_version': ml_pred.model_version
                    }
                    
                    # Optionally: blend with rule-based for stability
                    # total = 0.7 * ml_pred.predicted_points + 0.3 * rule_based
                    # But for now, pure ML when confidence is sufficient
                    
                    return ml_pred.predicted_points, breakdown
                else:
                    logging.info(f"ML confidence too low for player {player_id} "
                               f"({ml_pred.confidence:.3f} < "
                               f"{self.ml_service.config.confidence_threshold}), "
                               "using rule-based")
                    
            except Exception as e:
                logging.warning(f"ML prediction failed for player {player_id}: {e}")
                # Continue to rule-based fallback
        
        # Rule-based prediction (existing logic)
        return self._predict_rule_based(player, gameweek)
    
    def _predict_rule_based(self, player: pd.Series, gameweek: int) -> Tuple[float, Dict[str, float]]:
        """
        Original rule-based prediction logic (extracted from existing code).
        This ensures complete backward compatibility.
        """
        # Existing implementation from line 505-566 of point_predictor.py
        # (Original predict_gameweek logic moved here)
        # ... [copy all the existing logic] ...
        
        # For this design doc, we'll show the structure:
        regression_factor = self.calculate_regression_factor(player)
        
        breakdown = {
            'xg_points': self.calculate_xg_points(player, gameweek) * regression_factor,
            'xa_points': self.calculate_xa_points(player, gameweek) * regression_factor,
            'cs_points': self.calculate_clean_sheet_prob(player, gameweek),
            'cbit_bonus': self.calculate_cbit_bonus(player, gameweek),
            'appearance': self.calculate_appearance_points(player),
            'bonus': self.calculate_bonus_points(player, gameweek),
            'regression_factor': regression_factor,
            'prediction_source': 'rule-based'
        }
        
        total = sum(v for k, v in breakdown.items() if k != 'regression_factor')
        
        # [Rest of existing logic...]
        # dynamic_calibration, form_factor, minutes_decay, availability_adj, etc.
        
        return total, breakdown
    
    def predict_multi_gameweek(self, player_id: int, gameweeks: int = 5) -> PointPrediction:
        """
        Generate multi-gameweek point predictions for a player.
        
        Uses ML if available for each gameweek, with per-GW caching.
        """
        player = self.players[self.players['id'] == player_id].iloc[0]
        
        gw_predictions = {}
        all_breakdowns = {}
        
        for gw in range(self.current_gw, self.current_gw + gameweeks):
            total, breakdown = self.predict_gameweek(player, gw)
            gw_predictions[gw] = total
            all_breakdowns[gw] = breakdown
        
        # Aggregate breakdown
        agg_breakdown = {}
        for key in ['xg_points', 'xa_points', 'cs_points', 'cbit_bonus', 
                    'appearance', 'bonus', 'ml_points']:
            agg_breakdown[key] = sum(b.get(key, 0) for b in all_breakdowns.values())
        
        # Add source info
        ml_used = any(b.get('prediction_source') == 'ml' for b in all_breakdowns.values())
        agg_breakdown['prediction_source'] = 'ml' if ml_used else 'rule-based'
        
        return PointPrediction(
            player_id=player_id,
            player_name=player['web_name'],
            position=player.get('position', 'UNK'),
            team=self.team_names.get(player['team'], 'Unknown'),
            gameweek_predictions=gw_predictions,
            total_expected=sum(gw_predictions.values()),
            breakdown=agg_breakdown
        )
    
    def predict_all_players(self, gameweeks: int = 5) -> pd.DataFrame:
        """
        Generate predictions for all players.
        
        Optimizes by using batch ML predictions when enabled.
        """
        # If ML is enabled, use batch prediction
        if self.use_ml and self.ml_service:
            try:
                predictions = self._predict_all_players_ml(gameweeks)
                return predictions
            except Exception as e:
                logging.warning(f"ML batch prediction failed, falling back to "
                              f"sequential rule-based: {e}")
                # Fall through to rule-based
        
        # Sequential rule-based (existing logic)
        return self._predict_all_players_rule_based(gameweeks)
    
    def _predict_all_players_ml(self, gameweeks: int) -> pd.DataFrame:
        """Batch ML prediction for all players."""
        predictions = []
        
        # Build fixtures map: team_id -> fixtures for each gameweek
        fixtures_by_team_and_gw = {}
        for gw in range(self.current_gw, self.current_gw + gameweeks):
            for team_id in self.teams['id']:
                key = (team_id, gw)
                if key in self.fixture_map:
                    if team_id not in fixtures_by_team_and_gw:
                        fixtures_by_team_and_gw[team_id] = {}
                    fixtures_by_team_and_gw[team_id][gw] = [
                        {**f, 'gameweek': gw} for f in self.fixture_map[key]
                    ]
        
        # Get team strengths
        team_strengths = self._get_team_strengths_dict()
        
        # Predict for each gameweek
        for gw in range(self.current_gw, self.current_gw + gameweeks):
            gw_col = f'xp_gw{gw}'
            
            # Get players needing prediction (not cached)
            players_to_predict = []
            for _, player in self.players.iterrows():
                # Check cache first
                if self.ml_service._get_from_cache(player['id'], gw):
                    # Already cached, skip in batch
                    continue
                players_to_predict.append(player)
            
            if players_to_predict:
                players_df = pd.DataFrame(players_to_predict)
                batch_predictions = self.ml_service.predict_batch(
                    players_df=players_df,
                    gameweek=gw,
                    team_strengths=team_strengths,
                    fixtures_map=fixtures_by_team_and_gw
                )
            
            # Build results for all players
            for _, player in self.players.iterrows():
                # Try to get from cache (populated by batch or previous)
                cached = self.ml_service._get_from_cache(player['id'], gw)
                if cached:
                    xp = cached.predicted_points
                else:
                    # Fallback to rule-based
                    xp, _ = self._predict_rule_based(player, gw)
                
                # Find or create player entry
                player_entry = next((p for p in predictions if p['id'] == player['id']), None)
                if player_entry is None:
                    player_entry = {
                        'id': player['id'],
                        'web_name': player['web_name'],
                        'team': player['team'],
                        'team_name': self.team_names.get(player['team'], 'Unknown'),
                        'element_type': player['element_type'],
                        'position': player.get('position', 'UNK'),
                        'now_cost': player['now_cost'],
                        'total_xp': 0
                    }
                    predictions.append(player_entry)
                
                player_entry[gw_col] = xp
                player_entry['total_xp'] = player_entry.get('total_xp', 0) + xp
        
        return pd.DataFrame(predictions)
    
    def _predict_all_players_rule_based(self, gameweeks: int) -> pd.DataFrame:
        """Original sequential rule-based prediction (unchanged)."""
        predictions = []
        
        for _, player in self.players.iterrows():
            player_id = player['id']
            
            gw_cols = {}
            total_xp = 0
            
            for gw in range(self.current_gw, self.current_gw + gameweeks):
                xp, _ = self._predict_rule_based(player, gw)
                gw_cols[f'xp_gw{gw}'] = xp
                total_xp += xp
            
            predictions.append({
                'id': player_id,
                'web_name': player['web_name'],
                'team': player['team'],
                'team_name': self.team_names.get(player['team'], 'Unknown'),
                'element_type': player['element_type'],
                'position': player.get('position', 'UNK'),
                'now_cost': player['now_cost'],
                'total_xp': total_xp,
                **gw_cols
            })
        
        return pd.DataFrame(predictions)
```

---

## 6. API Endpoint Modifications

### 6.1 Changes to `main.py`

Add ML-specific endpoints and optionally enhance existing ones.

```python
# Additional imports at top
from ml_predictor import MLPredictionService
from ml_config import MLConfig

# Global ML service instance (initialized on startup)
ml_service: Optional[MLPredictionService] = None
ml_enabled: bool = False


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ml_service, ml_enabled
    
    print(f"🚀 FPL Optimizer API starting...")
    print(f"📋 CORS allowed origins: {ALLOWED_ORIGINS}")
    
    # Initialize ML service if configured
    try:
        config = MLConfig.from_env()
        if config.enabled:
            ml_service = MLPredictionService(config)
            if ml_service.is_available():
                ml_enabled = True
                print(f"🤖 ML prediction service enabled (version: {ml_service.model_version})")
            else:
                print("⚠️  ML service configured but unavailable - using rule-based only")
        else:
            print("ℹ️  ML prediction disabled (ML_ENABLED=false)")
    except Exception as e:
        print(f"⚠️  Failed to initialize ML service: {e}")


# =============================================================================
# NEW ML-SPECIFIC ENDPOINTS
# =============================================================================

@app.get("/api/ml-status")
async def get_ml_status():
    """
    Get ML service health and statistics.
    
    Returns:
        Service status, version, cache stats, and configuration
    """
    if not ml_service:
        return {
            "enabled": False,
            "available": False,
            "message": "ML service not initialized"
        }
    
    stats = ml_service.get_stats()
    
    return {
        "enabled": ml_enabled,
        "available": ml_service.is_available(),
        "model_version": ml_service.model_version,
        "stats": stats,
        "config": {
            "batch_size": ml_service.config.batch_size,
            "timeout_seconds": ml_service.config.timeout_seconds,
            "cache_ttl_minutes": ml_service.config.cache_ttl_minutes,
            "confidence_threshold": ml_service.config.confidence_threshold
        }
    }


@app.post("/api/ml-predictions")
async def get_ml_predictions(
    player_ids: Optional[List[int]] = None,
    gameweek: Optional[int] = None,
    batch: bool = True
):
    """
    Get ML predictions for specific players.
    
    Debug/testing endpoint to compare ML vs rule-based predictions.
    
    Args:
        player_ids: List of player IDs (empty = all players)
        gameweek: Target gameweek (default = current)
        batch: Use batch prediction mode
        
    Returns:
        Dictionary with ML predictions, confidence scores, and optional comparison
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = get_active_gameweek(events)
        target_gw = gameweek or current_gw
        
        # Filter players if specified
        if player_ids:
            players_df = players_df[players_df['id'].isin(player_ids)]
        
        if len(players_df) == 0:
            raise HTTPException(status_code=404, detail="No players found")
        
        # Initialize point predictor with ML enabled
        predictor = PointPredictor(
            players_df=players_df.copy(),
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw,
            use_ml=True  # Force ML for this endpoint
        )
        
        # Get predictions
        results = []
        for _, player in players_df.iterrows():
            try:
                xp, breakdown = predictor.predict_gameweek(player, target_gw)
                
                result = {
                    "id": int(player['id']),
                    "name": player['web_name'],
                    "team": teams_df[teams_df['id'] == player['team']]['name'].values[0] 
                           if len(teams_df[teams_df['id'] == player['team']]) > 0 else 'Unknown',
                    "position": ['GK', 'DEF', 'MID', 'FWD'][int(player['element_type']) - 1],
                    "expected_points": round(xp, 2),
                    "prediction_source": breakdown.get('prediction_source', 'unknown'),
                    "confidence": breakdown.get('ml_confidence'),
                    "breakdown": {k: round(v, 2) for k, v in breakdown.items() 
                                 if isinstance(v, (int, float)) and k != 'prediction_source'}
                }
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to predict for player {player['id']}: {e}")
                results.append({
                    "id": int(player['id']),
                    "name": player['web_name'],
                    "error": str(e)
                })
        
        return {
            "gameweek": target_gw,
            "current_gw": current_gw,
            "ml_enabled": predictor.use_ml,
            "total_players": len(results),
            "predictions": results,
            "generated_at": time.time()
        }
        
    except Exception as e:
        print(f"ML predictions error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
async def get_predictions(gameweeks: int = 5):
    """
    Get expected point predictions for all players over multiple gameweeks.
    
    ENHANCED: Now uses ML predictions if enabled, otherwise rule-based.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        
        # Get current gameweek
        events = static_data.get("events", [])
        current_gw = get_active_gameweek(events)
        
        # Initialize predictor (ML auto-enabled if configured)
        predictor = PointPredictor(
            players_df=players_df,
            teams_df=teams_df,
            fixtures=fixtures_data,
            current_gameweek=current_gw,
            use_ml=None  # Use default from config
        )
        
        # Get predictions (uses ML if available)
        predictions = predictor.predict_all_players(gameweeks=gameweeks)
        
        # Return top players by expected points
        top_players = predictions.nlargest(50, 'total_xp')
        
        return {
            "current_gameweek": current_gw,
            "horizon": gameweeks,
            "ml_used": predictor.use_ml,
            "predictions": top_players.to_dict(orient="records")
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENHANCED OPTIMIZATION ENDPOINTS (OPTIONAL ML INTEGRATION)
# =============================================================================

# For endpoints that already use predictions (optimize, optimize/multi-period, etc.),
# they already instantiate PointPredictor internally. The ML integration is
# automatic based on configuration. Optionally, we can add query parameter to
# override ML usage:

class OptimizationRequestWithML(OptimizationRequest):
    """Extended request with ML override option."""
    use_ml: Optional[bool] = Field(
        default=None, 
        description="Override ML setting (null = use config default)"
    )


@app.post("/api/optimize")
async def optimize_team(request: OptimizationRequestWithML):
    """
    Generates the optimal lineup based on constraints.
    
    ML predictions are used if enabled in configuration.
    """
    try:
        data = fpl_service.get_latest_data()
        static_data = data["static"]
        fixtures_data = data["fixtures"]
        
        players_df = pd.DataFrame(static_data["elements"])
        teams_df = pd.DataFrame(static_data["teams"])
        events = static_data.get("events", [])
        current_gw = get_active_gameweek(events)
        
        # Determine whether to use ML
        use_ml = request.use_ml if request.use_ml is not None else (
            ml_enabled if ml_service else False
        )
        
        # For multi-period and dream-team endpoints, same pattern:
        # Instantiate PointPredictor with appropriate use_ml flag
        
        # ... rest of optimization logic ...
        
    except Exception as e:
        print(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Keep existing endpoints unchanged - they continue to work with rule-based only
# Only the ones we modify will use ML automatically
```

---

## 7. Performance Optimizations

### 7.1 Batch Processing

- `MLPredictionService.predict_batch()` processes multiple players in a single model call
- Used by `PointPredictor.predict_all_players()` when ML enabled
- Batch size configurable via `ML_BATCH_SIZE` (default 50)

### 7.2 Caching Strategy

- **Memory cache** with TTL (default 60 minutes)
- Cache key: `(player_id, gameweek)`
- Automatic cleanup every 100 requests
- Separate from ML model memory (predictions only)

### 7.3 Lazy Model Loading

- Model not loaded until first prediction
- `ml_service.is_available()` triggers load
- Pre-warm on startup if desired (optional)

### 7.4 Timeout Handling

```python
# In MLPredictionService.predict()
try:
    result = asyncio.wait_for(
        asyncio.to_thread(self._model.predict, features),
        timeout=self.config.timeout_seconds
    )
except asyncio.TimeoutError:
    raise MLServiceTimeoutError(f"Prediction timed out after {self.config.timeout_seconds}s")
```

### 7.5 Async Considerations

For high-concurrency production deployment:

```python
# Add async methods
async def predict_async(self, player: pd.Series, gameweek: int) -> MLPrediction:
    """Async prediction using thread pool to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: self.predict(player, gameweek)
    )

async def predict_batch_async(self, players_df: pd.DataFrame, gameweek: int) -> List[MLPrediction]:
    """Async batch prediction."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.predict_batch(players_df, gameweek)
    )
```

---

## 8. Logging and Observability

### 8.1 Structured Logging

```python
import json
from datetime import datetime

def log_prediction_structured(prediction: MLPrediction, source: str):
    """Log prediction in structured JSON format."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "player_id": prediction.player_id,
        "gameweek": prediction.gameweek,
        "points": round(prediction.predicted_points, 2),
        "confidence": round(prediction.confidence, 3),
        "source": source,
        "model_version": prediction.model_version,
        "features_sample": {
            k: v for k, v in list(prediction.features.items())[:5]  # Sample
        }
    }
    logger.info(json.dumps(log_entry))
```

### 8.2 Metrics to Track

- **Usage metrics**: ML vs rule-based prediction count
- **Performance**: Prediction latency (p50, p95, p99)
- **Cache effectiveness**: Hit rate, size
- **Errors**: Error rate by type (model, timeout, validation)
- **Confidence distribution**: Histogram of confidence scores
- **Fallback triggers**: Count of low-confidence and failed predictions

### 8.3 Monitoring Endpoints

```python
@app.get("/api/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint."""
    if not ml_service:
        return {"ml_service": "unavailable"}
    
    stats = ml_service.get_stats()
    
    return {
        # Counter metrics
        "ml_predictions_total": stats["total_predictions"],
        "ml_cache_hits_total": stats["cache_hits"],
        "ml_errors_total": stats["errors"],
        "ml_fallbacks_total": stats["fallback_count"],
        
        # Gauge metrics
        "ml_cache_hit_rate": stats["cache_hit_rate"],
        "ml_usage_rate": stats["ml_usage_rate"],
        "ml_error_rate": stats["error_rate"],
        "ml_avg_latency_ms": stats["avg_prediction_time_ms"],
        "ml_cache_size": stats["cache_size"],
        
        # Service info
        "ml_model_version": stats["model_version"],
        "ml_enabled": stats["enabled"],
        "ml_model_loaded": stats["model_loaded"]
    }
```

---

## 9. Backward Compatibility Guarantees

### 9.1 No Breaking API Changes

- All existing API endpoints continue to work identically without ML enabled
- No changes to request/response schemas
- Default behavior exactly as before unless `ML_ENABLED=true`

### 9.2 Feature Flag Control

```
Environment: ML_ENABLED=false (default)
Result: System operates exactly as before - no ML code executed
```

### 9.3 Gradual Rollout Path

1. **Phase 1**: Deploy with `ML_ENABLED=false`
   - Zero impact on existing functionality
   - ML service initializes but stays dormant
   
2. **Phase 2**: Enable for specific endpoints only via code changes
   - Modify only `/api/predictions` to use ML
   - Keep others on rule-based
   
3. **Phase 3**: Enable all endpoints via `ML_ENABLED=true`
   - Full ML integration
   
4. **Phase 4**: Tune confidence thresholds, batch sizes, timeouts

### 9.4 Testing Strategy

```python
# In tests/ directory (existing tests remain unchanged)

def test_rule_based_unchanged():
    """Ensure rule-based predictions identical when ML disabled."""
    with patch.dict(os.environ, {"ML_ENABLED": "false"}):
        predictor = PointPredictor(...)
        result1 = predictor.predict_gameweek(player, 5)
        result2 = predictor.predict_gameweek(player, 5)
        assert result1 == result2  # Deterministic

def test_ml_fallback():
    """Ensure fallback to rule-based when ML fails."""
    config = MLConfig(enabled=True, model_path="nonexistent.pkl")
    service = MLPredictionService(config)
    
    with pytest.raises(MLPredictionError):
        service.predict(player, 5)

def test_ml_cache_works():
    """Test ML caching mechanism."""
    service = MLPredictionService(MLConfig.from_env())
    pred1 = service.predict(player, 5)
    pred2 = service.predict(player, 5)
    assert pred1.prediction_time == pred2.prediction_time  # From cache
```

---

## 10. Error Handling Strategy

### 10.1 ML Service Errors

```python
class MLServiceError(Exception):
    """Base ML service error."""
    pass

class MLModelNotFoundError(MLServiceError):
    """Model file not found."""
    pass

class MLPredictionError(MLServiceError):
    """Prediction failed."""
    pass

class MLServiceTimeoutError(MLServiceError):
    """Prediction timed out."""
    pass
```

### 10.2 Fallback Flowchart

```
predict_gameweek(player, gw)
  |
  |-- ML enabled? NO → return rule_based()
  |
  YES
  |
  |-- ML service available? NO → log, return rule_based()
  |
  YES
  |
  |-- Cache hit? YES → return cached ML prediction
  |
  NO
  |
  |-- Extract features
  |
  |-- Predict via ML
  |   |
  |   |-- Success? YES → check confidence
  |                        |
  |                        |-- Confidence >= threshold? YES → return ML
  |                        |
  |                        NO → log low confidence, raise error → fallback
  |   |
  |   |-- Error? → log, raise error → fallback
  |
  Fallback: return rule_based()
```

### 10.3 Partial Failures

For batch predictions, individual failures don't stop the batch:

```python
def predict_batch_safe(self, players_df, gameweek):
    results = []
    for _, player in players_df.iterrows():
        try:
            pred = self.predict(player, gameweek)
            results.append(pred)
        except Exception as e:
            logger.error(f"Skipping player {player['id']}: {e}")
            results.append(None)  # Or create placeholder with error info
    return results
```

---

## 11. Implementation Plan

### Phase 1: Foundation (Week 1)

1. Create `backend/ml_config.py`
   - Implement `MLConfig` dataclass
   - Add environment variable loading
   - Add validation
   
2. Create `backend/ml_predictor.py`
   - Implement `MLPredictionService` skeleton
   - Add lazy loading, caching, stats
   - Add mock model for testing

3. Add basic unit tests for ML service
   - Test config loading
   - Test caching
   - Test error handling

### Phase 2: Integration (Week 2)

4. Modify `backend/point_predictor.py`
   - Add ML service initialization
   - Inject ML into `predict_gameweek()`
   - Add fallback logic
   - Add `predict_all_players()` optimization
   
5. Add ML-specific endpoints to `backend/main.py`
   - `/api/ml-status`
   - `/api/ml-predictions`
   
6. Add comprehensive logging
   - Structured logs for predictions
   - Source tracking (ML vs rule-based)

### Phase 3: Testing & Optimization (Week 3)

7. Integration testing
   - Test with mock ML model
   - Test fallback scenarios
   - Test batch predictions
   - Verify backward compatibility
   
8. Performance tuning
   - Tune batch size
   - Tune cache TTL
   - Add async methods if needed
   
9. Add monitoring/metrics endpoint

### Phase 4: Production Readiness (Week 4)

10. Documentation
    - Update README with ML configuration
    - Add troubleshooting guide
    - Document model format requirements
    
11. Production deployment prep
    - Add Docker support for model volume
    - Add health check integration
    - Add alerting thresholds
    
12. Gradual rollout
    - Deploy with ML disabled
    - Enable for testing
    - Monitor metrics
    - Increase confidence threshold
    - Full enablement

---

## 12. Model Format Requirements

### 12.1 Expected Model Interface

```python
class MLModel:
    """Required interface for ML models."""
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict expected points for players.
        
        Args:
            X: DataFrame with features matching training data
            
        Returns:
            Array of predicted points (one per row)
        """
        raise NotImplementedError()
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional: Predict with confidence intervals.
        
        Returns:
            (points_array, confidence_array)
        """
        points = self.predict(X)
        confidences = self.get_default_confidence(points)
        return points, confidences
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names model expects."""
        raise NotImplementedError()
```

### 12.2 Feature Alignment

The `MLPredictionService._extract_features()` method must match your model's training features.

Common features used (adjust as needed):
- `element_type`, `now_cost`, `form`
- `expected_goals`, `expected_assists`
- `minutes`, `starts`
- `ict_index`, `influence`, `creativity`, `threat`
- `fixture_difficulty`, `team_attack_strength`, `team_defense_strength`
- `gameweek`, `is_home_game`
- Derived: `xG_per_90`, `xA_per_90`

**Action**: Train model with these features or adjust `_extract_features()` to match your training pipeline.

---

## 13. Deployment Considerations

### 13.1 Render Configuration

Add to `render.yaml`:

```yaml
envVars:
  - key: ML_ENABLED
    value: false  # Change to true after testing
  
  - key: ML_MODEL_PATH
    value: /app/models/ml_fpl_predictor.pkl
    # Mount as volume: https://render.com/docs/mount-persistent-disk
  
  - key: ML_BATCH_SIZE
    value: "50"
  
  - key: ML_TIMEOUT_SECONDS
    value: "10"
  
  - key: ML_CACHE_TTL_MINUTES
    value: "60"
```

### 13.2 Model Update Strategy

1. Store model in cloud storage (S3, GCS, Azure Blob)
2. Download on startup if version changed
3. Version via filename: `ml_fpl_predictor_v1.2.0.pkl`
4. Zero-downtime: Load new model in separate process, swap reference
5. Old model stays in memory until all predictions complete

### 13.3 Scaling

- ML predictions are CPU-bound → scale horizontally with more instances
- Cache is per-instance → consider Redis for shared caching if needed
- Batch size tuning based on instance memory
- Monitor prediction latency; if exceeding timeout, reduce batch size

---

## 14. rollback Procedure

If ML service causes issues:

1. **Immediate**: Set `ML_ENABLED=false` and restart
   - Instant fallback to rule-based
   - No data loss or corruption

2. **Debug**: Check logs for error patterns
   - High error rate → model issue
   - High fallback rate → confidence threshold too high
   - High latency → batch size too large

3. **Adjust**: Tune configuration without code changes
   - Increase `ML_TIMEOUT_SECONDS`
   - Decrease `ML_BATCH_SIZE`
   - Lower `ML_CONFIDENCE_THRESHOLD`
   - Enable `ML_LOG_PREDICTIONS` for debugging

4. **Remediate**: Fix model or feature extraction issues
   - Redeploy model
   - Update feature extraction to match training
   - Restart service

---

## 15. Success Metrics

### 15.1 Prediction Quality

- Compare top-10 player ranking correlation between ML and rule-based
- Track backtest performance improvement
- Monitor user feedback on transferred players

### 15.2 System Health

- ML service availability > 99.5%
- Prediction error rate < 0.1%
- Cache hit rate > 80% after warmup
- P95 prediction latency < 500ms

### 15.3 Adoption

- % of predictions using ML (target: 100% when enabled)
- Confidence distribution (should be spread, not all 0 or 1)
- Fallback rate (should be < 5% unless issues)

---

## 16. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model quality poor | Medium | High | A/B testing, gradual rollout, easy rollback |
| High latency | Medium | Medium | Tune batch size, increase timeout, async |
| Memory leaks | Low | Medium | Cache limits, periodic cleanup |
| Feature drift | Medium | High | Monitor feature distributions, retrain monthly |
| Model loading failure | Low | Medium | Validate on startup, clear error messages |
| Cache stampede | Low | Low | Per-key locking not needed for TTL-based |

---

## 17. Future Enhancements

- **A/B testing framework**: Compare ML vs rule-based with controlled traffic splitting
- **Model ensemble**: Combine multiple models with weighted averaging
- **Dynamic confidence**: Adjust threshold based on recent accuracy
- **Online learning**: Update model with new gameweek data
- **Explainability**: SHAP values for individual predictions
- **Multi-task learning**: Predict points, minutes, injury risk jointly
- **Player embeddings**: Learn player similarity for better generalization
- **Temporal features**: Explicit gameweek-aware model inputs

---

## Conclusion

This design provides a production-ready, backward-compatible ML integration that:

- ✅ Can be deployed with zero risk (disabled by default)
- ✅ Falls back gracefully on any failure
- ✅ Scales with batching and caching
- ✅ Is fully observable with metrics and logging
- ✅ Allows gradual rollout and tuning
- ✅ Maintains clean separation of concerns
- ✅ Requires minimal changes to existing code

The Integration Developer can implement this module-by-module with confidence that each piece works independently and the system degrades gracefully if ML service has issues.