"""
Regression Testing Suite for FPL ML Models

Ensures that new model versions don't break existing functionality
and maintain or improve performance.

Usage:
    pytest backend/tests/test_model_regression.py -v
    
Or run specific test categories:
    pytest backend/tests/test_model_regression.py::TestModelPerformanceRegression -v
    pytest backend/tests/test_model_regression.py::TestPredictionBehavior -v
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import joblib

# Import benchmark framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.model_benchmark import ModelBenchmark, BenchmarkResult


# ============== Fixtures ==============

@pytest.fixture
def data_paths():
    """Return paths to training data."""
    base_path = "datasets/fpl_points_v1"
    return {
        "train_X": os.path.join(base_path, "train_X.npy"),
        "train_y": os.path.join(base_path, "train_y.npy"),
        "val_X": os.path.join(base_path, "validation_X.npy"),
        "val_y": os.path.join(base_path, "validation_y.npy"),
        "test_X": os.path.join(base_path, "test_X.npy"),
        "test_y": os.path.join(base_path, "test_y.npy"),
    }


@pytest.fixture
def test_data(data_paths):
    """Load test data."""
    return {
        'X': np.load(data_paths["test_X"]),
        'y': np.load(data_paths["test_y"]),
    }


@pytest.fixture
def benchmark():
    """Create benchmark instance."""
    return ModelBenchmark(results_dir='benchmark_results/regression')


@pytest.fixture
def baseline_metrics():
    """Load baseline performance metrics."""
    baseline_path = Path('benchmark_results/baseline_metrics.json')
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return None


# ============== Performance Regression Tests ==============

class TestModelPerformanceRegression:
    """
    Test that model performance doesn't regress below baseline.
    
    These tests compare current model performance against stored
    baseline metrics and fail if performance degrades significantly.
    """
    
    def test_xgboost_performance_not_regressed(self, test_data, baseline_metrics):
        """XGBoost model should not regress from baseline."""
        if baseline_metrics is None:
            pytest.skip("No baseline metrics found. Run establish_baseline() first.")
        
        model_path = "models/xgboost/model.pkl"
        if not os.path.exists(model_path):
            pytest.skip("XGBoost model not found")
        
        model = joblib.load(model_path)
        predictions = model.predict(test_data['X'])
        
        current_rmse = np.sqrt(np.mean((test_data['y'] - predictions) ** 2))
        baseline_rmse = baseline_metrics.get('xgboost', {}).get('rmse')
        
        if baseline_rmse:
            # Allow 5% regression tolerance
            assert current_rmse <= baseline_rmse * 1.05, \
                f"XGBoost RMSE regressed: {current_rmse:.4f} > {baseline_rmse * 1.05:.4f}"
    
    def test_lightgbm_performance_not_regressed(self, test_data, baseline_metrics):
        """LightGBM model should not regress from baseline."""
        if baseline_metrics is None:
            pytest.skip("No baseline metrics found")
        
        model_path = "models/lightgbm/model.pkl"
        if not os.path.exists(model_path):
            pytest.skip("LightGBM model not found")
        
        model = joblib.load(model_path)
        predictions = model.predict(test_data['X'])
        
        current_rmse = np.sqrt(np.mean((test_data['y'] - predictions) ** 2))
        baseline_rmse = baseline_metrics.get('lightgbm', {}).get('rmse')
        
        if baseline_rmse:
            assert current_rmse <= baseline_rmse * 1.05, \
                f"LightGBM RMSE regressed: {current_rmse:.4f} > {baseline_rmse * 1.05:.4f}"
    
    def test_ensemble_performance_not_regressed(self, test_data, baseline_metrics):
        """Ensemble model should not regress from baseline."""
        if baseline_metrics is None:
            pytest.skip("No baseline metrics found")
        
        from ml.ensemble import EnsemblePredictor
        
        if not os.path.exists("models/ensemble/ensemble.pkl"):
            pytest.skip("Ensemble model not found")
        
        ensemble = EnsemblePredictor.load("models/ensemble/ensemble.pkl")
        predictions = ensemble.predict(test_data['X'])
        
        current_rmse = np.sqrt(np.mean((test_data['y'] - predictions) ** 2))
        baseline_rmse = baseline_metrics.get('ensemble', {}).get('rmse')
        
        if baseline_rmse:
            assert current_rmse <= baseline_rmse * 1.05, \
                f"Ensemble RMSE regressed: {current_rmse:.4f} > {baseline_rmse * 1.05:.4f}"


class TestPredictionBehavior:
    """
    Test that model predictions behave reasonably.
    
    These tests check prediction ranges, monotonicity, and other
    behavioral properties that should hold regardless of model version.
    """
    
    def test_predictions_within_reasonable_range(self, test_data):
        """All model predictions should be within reasonable FPL point ranges."""
        models = self._load_all_models()
        
        for name, model in models.items():
            predictions = self._get_predictions(model, test_data['X'])
            
            # No negative predictions
            assert np.all(predictions >= -5), \
                f"{name}: Found predictions < -5 (min: {predictions.min():.2f})"
            
            # No extremely high predictions (>20 for single GW)
            assert np.all(predictions <= 20), \
                f"{name}: Found predictions > 20 (max: {predictions.max():.2f})"
    
    def test_predictions_not_all_identical(self, test_data):
        """Models should produce varied predictions, not all identical."""
        models = self._load_all_models()
        
        for name, model in models.items():
            predictions = self._get_predictions(model, test_data['X'])
            
            # Standard deviation should be > 0.5 (reasonable spread)
            assert np.std(predictions) > 0.5, \
                f"{name}: Predictions have very low variance ({np.std(predictions):.4f})"
    
    def test_rank_correlation_positive(self, test_data):
        """Models should have positive rank correlation with actuals."""
        from scipy.stats import spearmanr
        
        models = self._load_all_models()
        
        for name, model in models.items():
            predictions = self._get_predictions(model, test_data['X'])
            corr, _ = spearmanr(test_data['y'], predictions)
            
            assert corr > 0.1, \
                f"{name}: Rank correlation too low ({corr:.4f})"
    
    def test_higher_scoring_players_ranked_higher_on_average(self, test_data):
        """Models should generally rank high-scoring players higher."""
        models = self._load_all_models()
        
        # Get top 20% actual scorers
        n_top = int(len(test_data['y']) * 0.2)
        top_actual_indices = np.argsort(test_data['y'])[-n_top:]
        
        for name, model in models.items():
            predictions = self._get_predictions(model, test_data['X'])
            top_pred_indices = np.argsort(predictions)[-n_top:]
            
            # At least 30% overlap between top actual and top predicted
            overlap = len(set(top_actual_indices) & set(top_pred_indices))
            overlap_pct = overlap / n_top
            
            assert overlap_pct >= 0.3, \
                f"{name}: Top player overlap only {overlap_pct:.1%}"
    
    def test_extreme_predictions_rare(self, test_data):
        """Extreme predictions (>15 or <-2) should be rare (<5%)."""
        models = self._load_all_models()
        
        for name, model in models.items():
            predictions = self._get_predictions(model, test_data['X'])
            
            extreme_pct = np.mean((predictions > 15) | (predictions < -2))
            
            assert extreme_pct < 0.05, \
                f"{name}: {extreme_pct:.1%} extreme predictions (should be <5%)"
    
    def _load_all_models(self) -> Dict[str, Any]:
        """Load all available models."""
        models = {}
        
        model_paths = [
            ("xgboost", "models/xgboost/model.pkl"),
            ("lightgbm", "models/lightgbm/model.pkl"),
            ("random_forest", "models/random_forest/model.pkl"),
            ("gradient_boosting", "models/gradient_boosting/model.pkl"),
            ("ridge", "models/ridge/model.pkl"),
        ]
        
        for name, path in model_paths:
            if os.path.exists(path):
                models[name] = joblib.load(path)
        
        # Load ensemble if available
        if os.path.exists("models/ensemble/ensemble.pkl"):
            from ml.ensemble import EnsemblePredictor
            models["ensemble"] = EnsemblePredictor.load("models/ensemble/ensemble.pkl")
        
        return models
    
    def _get_predictions(self, model, X):
        """Safely get predictions."""
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif hasattr(model, 'forward_pass'):
            return model.forward_pass(X)
        else:
            raise ValueError("Model has no predict method")


class TestModelRobustness:
    """
    Test model robustness to input variations.
    
    Models should handle edge cases gracefully.
    """
    
    def test_model_handles_zero_features(self, test_data):
        """Models should handle zero-valued features without crashing."""
        models = TestPredictionBehavior()._load_all_models()
        
        # Create input with all zeros
        X_zero = np.zeros((1, test_data['X'].shape[1]))
        
        for name, model in models.items():
            try:
                pred = TestPredictionBehavior()._get_predictions(model, X_zero)
                assert np.isfinite(pred).all(), f"{name}: Non-finite prediction for zero input"
            except Exception as e:
                pytest.fail(f"{name}: Failed on zero input: {e}")
    
    def test_model_handles_very_large_values(self, test_data):
        """Models should handle large feature values gracefully."""
        models = TestPredictionBehavior()._load_all_models()
        
        # Create input with large values
        X_large = np.ones((1, test_data['X'].shape[1])) * 100
        
        for name, model in models.items():
            try:
                pred = TestPredictionBehavior()._get_predictions(model, X_large)
                assert np.isfinite(pred).all(), f"{name}: Non-finite prediction for large input"
                # Predictions should still be reasonable
                assert pred[0] < 100, f"{name}: Prediction too large for large input"
            except Exception as e:
                pytest.fail(f"{name}: Failed on large input: {e}")
    
    def test_model_output_shape_consistency(self, test_data):
        """Model output shape should match input batch size."""
        models = TestPredictionBehavior()._load_all_models()
        
        batch_sizes = [1, 5, 10, 50]
        
        for name, model in models.items():
            for batch_size in batch_sizes:
                if batch_size > len(test_data['X']):
                    continue
                    
                X_batch = test_data['X'][:batch_size]
                pred = TestPredictionBehavior()._get_predictions(model, X_batch)
                
                assert len(pred) == batch_size, \
                    f"{name}: Output shape mismatch for batch size {batch_size}"


class TestFeatureEngineeringRegression:
    """
    Test that feature engineering produces consistent outputs.
    """
    
    def test_feature_generation_deterministic(self, test_data):
        """Feature generation should be deterministic (same input = same output)."""
        from ml.advanced_features import generate_all_features
        
        X = test_data['X']
        
        # Generate features twice
        feat1 = generate_all_features(X)
        feat2 = generate_all_features(X)
        
        np.testing.assert_array_equal(feat1, feat2, 
            "Feature generation is not deterministic")
    
    def test_feature_generation_preserves_sample_count(self, test_data):
        """Feature generation should preserve number of samples."""
        from ml.advanced_features import generate_all_features
        
        X = test_data['X']
        X_enhanced = generate_all_features(X)
        
        assert X_enhanced.shape[0] == X.shape[0], \
            "Feature generation changed sample count"
    
    def test_feature_generation_increases_feature_count(self, test_data):
        """Feature generation should add new features."""
        from ml.advanced_features import generate_all_features
        
        X = test_data['X']
        X_enhanced = generate_all_features(X)
        
        assert X_enhanced.shape[1] > X.shape[1], \
            "Feature generation did not add new features"


class TestInferencePerformance:
    """
    Test that model inference meets performance requirements.
    """
    
    def test_single_prediction_latency(self, test_data):
        """Single prediction should complete within 50ms."""
        import time
        
        models = TestPredictionBehavior()._load_all_models()
        
        for name, model in models.items():
            X_single = test_data['X'][:1]
            
            # Warmup
            _ = TestPredictionBehavior()._get_predictions(model, X_single)
            
            # Time 10 predictions
            start = time.perf_counter()
            for _ in range(10):
                _ = TestPredictionBehavior()._get_predictions(model, X_single)
            elapsed = (time.perf_counter() - start) / 10 * 1000  # ms
            
            assert elapsed < 50, \
                f"{name}: Single prediction too slow ({elapsed:.2f}ms)"
    
    def test_batch_prediction_latency(self, test_data):
        """Batch prediction for 100 players should complete within 500ms."""
        import time
        
        models = TestPredictionBehavior()._load_all_models()
        batch_size = min(100, len(test_data['X']))
        X_batch = test_data['X'][:batch_size]
        
        for name, model in models.items():
            # Warmup
            _ = TestPredictionBehavior()._get_predictions(model, X_batch)
            
            start = time.perf_counter()
            _ = TestPredictionBehavior()._get_predictions(model, X_batch)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            assert elapsed < 500, \
                f"{name}: Batch prediction too slow ({elapsed:.2f}ms for {batch_size} players)"


# ============== Utility Functions ==============

def establish_baseline(test_data: Dict[str, np.ndarray], output_path: str = 'benchmark_results/baseline_metrics.json'):
    """
    Establish baseline performance metrics for all models.
    
    Run this once before making changes to create a reference point.
    """
    models = TestPredictionBehavior()._load_all_models()
    baseline = {}
    
    for name, model in models.items():
        predictions = TestPredictionBehavior()._get_predictions(model, test_data['X'])
        
        baseline[name] = {
            'rmse': float(np.sqrt(np.mean((test_data['y'] - predictions) ** 2))),
            'mae': float(np.mean(np.abs(test_data['y'] - predictions))),
            'r2': float(1 - np.sum((test_data['y'] - predictions) ** 2) / np.sum((test_data['y'] - np.mean(test_data['y'])) ** 2)),
            'timestamp': datetime.now().isoformat(),
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"Baseline metrics saved to {output_path}")
    return baseline


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
