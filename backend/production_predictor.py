"""
Production Predictor for FPL Lineup Optimizer

Uses the champion model (EXP-030 - Weighted Ensemble) for point predictions.
Replaces the old MLPredictor with the improved ensemble model.

Usage:
    from production_predictor import ProductionPredictor
    
    predictor = ProductionPredictor()
    predictions = predictor.predict_batch(player_features)
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionPredictor:
    """
    Production-grade predictor using the champion ensemble model (EXP-030).
    
    This model achieved 3.32% RMSE improvement over baseline through
    optimized weighted ensemble with negative weighting.
    
    Model Configuration:
        - XGBoost: -0.75 (hedge)
        - LightGBM: 1.28 (positive predictor)
        - Gradient Boosting: -0.35 (hedge)
        - Random Forest: -2.05 (strong hedge)
        - Ridge Regression: 2.86 (primary predictor)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the production predictor.
        
        Args:
            model_path: Path to model pickle file. If None, uses default production model.
        """
        self.model_path = model_path or self._get_default_model_path()
        self.ensemble_data = None
        self.models = None
        self.weights = None
        self.metadata = None
        
        self._load_model()
        
    def _get_default_model_path(self) -> str:
        """Get the default production model path."""
        backend_dir = Path(__file__).parent
        return str(backend_dir.parent / "models" / "production" / "model.pkl")
    
    def _load_model(self):
        """Load the ensemble model from disk."""
        try:
            logger.info(f"Loading production model from {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                self.ensemble_data = pickle.load(f)
            
            self.models = self.ensemble_data['models']
            self.weights = self.ensemble_data['weights']
            
            # Load metadata if available
            metadata_path = Path(self.model_path).parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    logger.info(f"Loaded model: {self.metadata.get('name', 'Unknown')}")
                    logger.info(f"RMSE: {self.metadata.get('rmse', 'Unknown')}")
            
            logger.info(f"Model loaded successfully with {len(self.models)} base models")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted points of shape (n_samples,)
        """
        if self.models is None or self.weights is None:
            raise RuntimeError("Model not loaded")
        
        X = np.asarray(X)
        predictions = np.zeros(len(X))
        
        # Weighted ensemble prediction
        for (name, model), weight in zip(self.models.items(), self.weights):
            try:
                pred = model.predict(X)
                predictions += pred * weight
            except Exception as e:
                logger.warning(f"Model {name} failed to predict: {e}")
                continue
        
        return predictions
    
    def predict_with_confidence(
        self, 
        X: np.ndarray,
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[Dict]]]:
        """
        Generate predictions with confidence intervals.
        
        Confidence is estimated from disagreement between base models.
        
        Args:
            X: Feature matrix
            return_individual: If True, also return individual model predictions
            
        Returns:
            predictions: Weighted ensemble predictions
            confidence: Estimated confidence (higher = more agreement)
            individual_preds: (optional) Dict of individual model predictions
        """
        X = np.asarray(X)
        n_samples = len(X)
        
        # Collect individual predictions
        individual_preds = {}
        all_preds = np.zeros((n_samples, len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            try:
                pred = model.predict(X)
                individual_preds[name] = pred
                all_preds[:, i] = pred
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                all_preds[:, i] = np.nan
        
        # Weighted prediction
        weights = np.array(self.weights).reshape(1, -1)
        predictions = np.nansum(all_preds * weights, axis=1)
        
        # Confidence = inverse of std across models (normalized)
        pred_std = np.nanstd(all_preds, axis=1)
        confidence = 1 / (1 + pred_std)  # Higher confidence = lower std
        
        if return_individual:
            return predictions, confidence, individual_preds
        return predictions, confidence
    
    def predict_player(
        self,
        player_features: Dict[str, float],
        feature_order: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """
        Predict points for a single player.
        
        Args:
            player_features: Dict of feature names to values
            feature_order: Optional list specifying feature order
            
        Returns:
            (predicted_points, confidence)
        """
        # Convert dict to array
        if feature_order is None:
            feature_order = list(player_features.keys())
        
        X = np.array([[player_features.get(f, 0.0) for f in feature_order]])
        
        predictions, confidence = self.predict_with_confidence(X)
        return float(predictions[0]), float(confidence[0])
    
    def predict_batch(
        self,
        players_df,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Predict points for a batch of players.
        
        Args:
            players_df: DataFrame with player features
            feature_columns: List of feature column names (if None, uses all numeric)
            
        Returns:
            DataFrame with added 'predicted_points' and 'confidence' columns
        """
        import pandas as pd
        
        if feature_columns is None:
            feature_columns = players_df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = players_df[feature_columns].values
        predictions, confidence = self.predict_with_confidence(X)
        
        result_df = players_df.copy()
        result_df['predicted_points'] = predictions
        result_df['confidence'] = confidence
        
        return result_df
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            "model_path": self.model_path,
            "n_base_models": len(self.models) if self.models else 0,
            "base_models": list(self.models.keys()) if self.models else [],
            "weights": self.weights,
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info
    
    def validate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Validate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict of performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import spearmanr
        
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        spearman = spearmanr(y_test, predictions)[0]
        
        return {
            "rmse": rmse,
            "mae": mae,
            "spearman_correlation": spearman,
            "mean_prediction": np.mean(predictions),
            "std_prediction": np.std(predictions),
            "validated_at": datetime.now().isoformat()
        }


# Convenience function for quick predictions
def predict_points(features: np.ndarray) -> np.ndarray:
    """
    Quick prediction function.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        
    Returns:
        Predicted points
    """
    predictor = ProductionPredictor()
    return predictor.predict(features)


if __name__ == "__main__":
    # Test the predictor
    print("Testing Production Predictor...")
    
    predictor = ProductionPredictor()
    print(f"\nModel Info:")
    info = predictor.get_model_info()
    print(f"  Name: {info.get('name', 'Unknown')}")
    print(f"  RMSE: {info.get('rmse', 'Unknown')}")
    print(f"  Improvement: {info.get('improvement', 'Unknown')}%")
    print(f"  Base Models: {info['n_base_models']}")
    
    # Test prediction
    test_input = np.random.randn(5, 30)
    predictions = predictor.predict(test_input)
    print(f"\nTest Predictions: {predictions}")
    
    # Test with confidence
    preds, conf, indiv = predictor.predict_with_confidence(test_input, return_individual=True)
    print(f"\nConfidence: {conf}")
    print(f"Individual predictions:")
    for name, vals in indiv.items():
        print(f"  {name}: {vals}")
    
    print("\n✅ Production Predictor ready!")
