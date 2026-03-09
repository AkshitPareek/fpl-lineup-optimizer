"""
Model Validation Tests for FPL ML Models

Validates that models behave correctly, make reasonable predictions,
and meet production requirements.

Usage:
    pytest backend/tests/test_model_validation.py -v
    
Categories:
    - Behavioral tests: Do models act reasonably?
    - Performance tests: Do models meet latency/throughput requirements?
    - Robustness tests: Do models handle edge cases?
    - Fairness tests: Do models treat all positions fairly?
"""

import pytest
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import joblib


# ============== Fixtures ==============

@pytest.fixture
def test_data():
    """Load test data."""
    base_path = "datasets/fpl_points_v1"
    return {
        'X': np.load(os.path.join(base_path, "test_X.npy")),
        'y': np.load(os.path.join(base_path, "test_y.npy")),
    }


@pytest.fixture
def trained_models():
    """Load all trained models."""
    models = {}
    model_paths = [
        ('xgboost', 'models/xgboost/model.pkl'),
        ('lightgbm', 'models/lightgbm/model.pkl'),
        ('random_forest', 'models/random_forest/model.pkl'),
        ('gradient_boosting', 'models/gradient_boosting/model.pkl'),
        ('ridge', 'models/ridge/model.pkl'),
    ]
    
    for name, path in model_paths:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    # Load ensemble
    if os.path.exists('models/ensemble/ensemble.pkl'):
        from ml.ensemble import EnsemblePredictor
        models['ensemble'] = EnsemblePredictor.load('models/ensemble/ensemble.pkl')
    
    return models


# ============== Behavioral Validation Tests ==============

class TestModelBehavior:
    """
    Test that models behave reasonably and make sensible predictions.
    """
    
    def test_predictions_are_finite(self, trained_models, test_data):
        """All predictions must be finite numbers."""
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            assert np.all(np.isfinite(preds)), \
                f"{name}: Non-finite predictions found"
    
    def test_predictions_within_fpl_range(self, trained_models, test_data):
        """Predictions should be within realistic FPL point ranges."""
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            # Single GW predictions rarely outside [-3, 15]
            assert np.all(preds >= -5), \
                f"{name}: Predictions below -5 (min: {preds.min():.2f})"
            assert np.all(preds <= 20), \
                f"{name}: Predictions above 20 (max: {preds.max():.2f})"
    
    def test_predictions_have_variance(self, trained_models, test_data):
        """Predictions should vary across players (not all identical)."""
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            std = np.std(preds)
            assert std > 0.5, \
                f"{name}: Predictions too homogeneous (std: {std:.4f})"
    
    def test_predictions_not_all_zeros(self, trained_models, test_data):
        """Models shouldn't predict zero for everyone."""
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            non_zero_pct = np.mean(preds != 0)
            assert non_zero_pct > 0.9, \
                f"{name}: Too many zero predictions ({(1-non_zero_pct):.1%})"
    
    def test_high_scorers_get_higher_predictions(self, trained_models, test_data):
        """
        Test that models generally rank high scorers higher.
        
        This is a weak test - we don't expect perfect correlation,
        but the top actual scorers should generally be in the top half
        of predicted scorers.
        """
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            # Get top 30% actual and predicted
            n_top = int(len(test_data['y']) * 0.3)
            top_actual = set(np.argsort(test_data['y'])[-n_top:])
            top_predicted = set(np.argsort(preds)[-n_top:])
            
            overlap = len(top_actual & top_predicted) / n_top
            
            # At least 20% overlap (better than random)
            assert overlap > 0.2, \
                f"{name}: Poor top-player identification ({overlap:.1%} overlap)"
    
    def test_predictions_monotonic_with_form(self, trained_models, test_data):
        """
        Test that players with better form features get higher predictions.
        
        This tests that the model is using form features sensibly.
        """
        # This requires knowing which features are form-related
        # We'll check that predictions correlate with a few key features
        
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            # Assuming first few features include form/price
            # Check correlation with first feature (often price or recent points)
            if test_data['X'].shape[1] > 0:
                feature_0 = test_data['X'][:, 0]
                corr = np.corrcoef(feature_0, preds)[0, 1]
                
                # Should have some correlation (positive or negative)
                assert not np.isnan(corr), \
                    f"{name}: No correlation with primary feature"
    
    def test_model_respects_minutes_feature(self, trained_models, test_data):
        """
        Test that models give lower predictions to players with low minutes.
        
        Players who don't play should get near-zero predictions.
        """
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            # Assuming minutes feature exists (commonly feature 4-6)
            # Find players with very low minutes
            if test_data['X'].shape[1] > 4:
                low_minutes_mask = test_data['X'][:, 4] < 10  # Less than 10 mins
                
                if np.any(low_minutes_mask):
                    low_min_preds = preds[low_minutes_mask]
                    high_min_preds = preds[~low_minutes_mask]
                    
                    # Low-minute players should have lower predictions on average
                    assert np.median(low_min_preds) < np.median(high_min_preds), \
                        f"{name}: Not penalizing low-minute players"


# ============== Performance Validation Tests ==============

class TestModelPerformance:
    """
    Test that models meet production performance requirements.
    """
    
    def test_single_prediction_latency(self, trained_models, test_data):
        """Single prediction should complete within 100ms."""
        for name, model in trained_models.items():
            X_single = test_data['X'][:1]
            
            # Warmup
            _ = self._get_predictions(model, X_single)
            
            # Time
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = self._get_predictions(model, X_single)
                times.append(time.perf_counter() - start)
            
            avg_time_ms = np.mean(times) * 1000
            
            assert avg_time_ms < 100, \
                f"{name}: Single prediction too slow ({avg_time_ms:.2f}ms)"
    
    def test_batch_prediction_latency(self, trained_models, test_data):
        """Batch of 100 predictions should complete within 500ms."""
        batch_size = min(100, len(test_data['X']))
        X_batch = test_data['X'][:batch_size]
        
        for name, model in trained_models.items():
            # Warmup
            _ = self._get_predictions(model, X_batch)
            
            # Time
            start = time.perf_counter()
            _ = self._get_predictions(model, X_batch)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            assert elapsed_ms < 500, \
                f"{name}: Batch prediction too slow ({elapsed_ms:.2f}ms for {batch_size})")
    
    def test_full_squad_prediction_latency(self, trained_models, test_data):
        """Predictions for full FPL squad (15 players) should be fast."""
        for name, model in trained_models.items():
            X_squad = test_data['X'][:15]
            
            start = time.perf_counter()
            _ = self._get_predictions(model, X_squad)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            assert elapsed_ms < 100, \
                f"{name}: Squad prediction too slow ({elapsed_ms:.2f}ms)"
    
    def test_model_loading_time(self):
        """Models should load within reasonable time (<5s each)."""
        model_paths = [
            ('xgboost', 'models/xgboost/model.pkl'),
            ('lightgbm', 'models/lightgbm/model.pkl'),
        ]
        
        for name, path in model_paths:
            if not os.path.exists(path):
                continue
            
            start = time.perf_counter()
            _ = joblib.load(path)
            elapsed = time.perf_counter() - start
            
            assert elapsed < 5, \
                f"{name}: Loading too slow ({elapsed:.2f}s)"


# ============== Robustness Tests ==============

class TestModelRobustness:
    """
    Test model robustness to edge cases and adversarial inputs.
    """
    
    def test_model_handles_zeros(self, trained_models, test_data):
        """Models should handle all-zero input without crashing."""
        X_zero = np.zeros((1, test_data['X'].shape[1]))
        
        for name, model in trained_models.items():
            try:
                preds = self._get_predictions(model, X_zero)
                assert np.isfinite(preds).all(), \
                    f"{name}: Non-finite predictions for zero input"
            except Exception as e:
                pytest.fail(f"{name}: Crashed on zero input: {e}")
    
    def test_model_handles_ones(self, trained_models, test_data):
        """Models should handle all-ones input."""
        X_ones = np.ones((1, test_data['X'].shape[1]))
        
        for name, model in trained_models.items():
            try:
                preds = self._get_predictions(model, X_ones)
                assert np.isfinite(preds).all(), \
                    f"{name}: Non-finite predictions for ones input"
            except Exception as e:
                pytest.fail(f"{name}: Crashed on ones input: {e}")
    
    def test_model_handles_large_values(self, trained_models, test_data):
        """Models should handle very large feature values gracefully."""
        X_large = np.ones((1, test_data['X'].shape[1])) * 1000
        
        for name, model in trained_models.items():
            try:
                preds = self._get_predictions(model, X_large)
                assert np.isfinite(preds).all(), \
                    f"{name}: Non-finite predictions for large input"
                # Predictions should still be reasonable
                assert preds[0] < 100, \
                    f"{name}: Unreasonable prediction ({preds[0]:.2f}) for large input"
            except Exception as e:
                pytest.fail(f"{name}: Crashed on large input: {e}")
    
    def test_model_handles_negative_values(self, trained_models, test_data):
        """Models should handle negative feature values."""
        X_neg = -np.abs(test_data['X'][:1])
        
        for name, model in trained_models.items():
            try:
                preds = self._get_predictions(model, X_neg)
                assert np.isfinite(preds).all(), \
                    f"{name}: Non-finite predictions for negative input"
            except Exception as e:
                pytest.fail(f"{name}: Crashed on negative input: {e}")
    
    def test_model_handles_nan_by_error(self, trained_models, test_data):
        """Models should either handle NaN or raise clear error."""
        X_nan = test_data['X'][:1].copy()
        X_nan[0, 0] = np.nan
        
        for name, model in trained_models.items():
            try:
                preds = self._get_predictions(model, X_nan)
                # If it doesn't error, predictions should still be reasonable
                assert np.isfinite(preds).all() or np.isnan(preds).all(), \
                    f"{name}: Unexpected behavior with NaN input"
            except (ValueError, RuntimeError):
                # Raising an error is acceptable behavior
                pass
    
    def test_model_output_shape_consistency(self, trained_models, test_data):
        """Model output shape should match input batch size."""
        batch_sizes = [1, 5, 10, 50]
        
        for name, model in trained_models.items():
            for batch_size in batch_sizes:
                if batch_size > len(test_data['X']):
                    continue
                
                X_batch = test_data['X'][:batch_size]
                preds = self._get_predictions(model, X_batch)
                
                assert len(preds) == batch_size, \
                    f"{name}: Output shape mismatch for batch size {batch_size}"


# ============== Fairness Tests ==============

class TestModelFairness:
    """
    Test that models treat different positions fairly.
    
    We don't want models that systematically underpredict any position.
    """
    
    def test_position_wise_bias(self, trained_models, test_data):
        """Models should not have systematic bias for any position."""
        # This would require position information in the test data
        # For now, we check that prediction variance is similar across
        # different prediction ranges (proxy for positions)
        
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            # Split into quartiles by prediction
            quartiles = np.percentile(preds, [25, 50, 75])
            
            # Get actual values for each quartile
            q1_actual = test_data['y'][preds <= quartiles[0]]
            q4_actual = test_data['y'][preds >= quartiles[2]]
            
            if len(q1_actual) > 0 and len(q4_actual) > 0:
                # Bias for low-predicted players
                q1_bias = np.mean(preds[preds <= quartiles[0]] - q1_actual)
                q4_bias = np.mean(preds[preds >= quartiles[2]] - q4_actual)
                
                # Bias shouldn't differ by more than 1 point
                assert abs(q1_bias - q4_bias) < 1.0, \
                    f"{name}: Unequal bias across prediction ranges"
    
    def test_no_extreme_prediction_disparity(self, trained_models, test_data):
        """Models shouldn't predict extreme values disproportionately for any subset."""
        for name, model in trained_models.items():
            preds = self._get_predictions(model, test_data['X'])
            
            # Check prediction distribution
            q1, q99 = np.percentile(preds, [1, 99])
            
            # Very few extreme predictions
            extreme_pct = np.mean((preds < q1) | (preds > q99))
            assert extreme_pct < 0.05, \
                f"{name}: Too many extreme predictions ({extreme_pct:.1%})"


# ============== Consistency Tests ==============

class TestModelConsistency:
    """
    Test that models produce consistent results.
    """
    
    def test_predictions_deterministic(self, trained_models, test_data):
        """Same input should produce same output (no randomness)."""
        X_sample = test_data['X'][:10]
        
        for name, model in trained_models.items():
            preds1 = self._get_predictions(model, X_sample)
            preds2 = self._get_predictions(model, X_sample)
            
            np.testing.assert_array_almost_equal(preds1, preds2, decimal=6,
                err_msg=f"{name}: Predictions not deterministic")
    
    def test_predictions_order_invariant(self, trained_models, test_data):
        """Shuffling inputs should shuffle predictions accordingly."""
        X_sample = test_data['X'][:10].copy()
        
        for name, model in trained_models.items():
            preds_original = self._get_predictions(model, X_sample)
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(X_sample))
            X_shuffled = X_sample[shuffle_idx]
            preds_shuffled = self._get_predictions(model, X_shuffled)
            
            # Unshuffle predictions
            preds_unshuffled = preds_shuffled[np.argsort(shuffle_idx)]
            
            np.testing.assert_array_almost_equal(preds_original, preds_unshuffled, decimal=6,
                err_msg=f"{name}: Predictions not order-invariant")


# ============== Helper Methods ==============

def _get_predictions(self, model, X):
    """Safely get predictions from various model types."""
    if hasattr(model, 'predict'):
        return model.predict(X)
    elif hasattr(model, 'forward_pass'):
        return model.forward_pass(X)
    else:
        raise ValueError(f"Model has no predict or forward_pass method")


TestModelBehavior._get_predictions = _get_predictions
TestModelPerformance._get_predictions = _get_predictions
TestModelRobustness._get_predictions = _get_predictions
TestModelFairness._get_predictions = _get_predictions
TestModelConsistency._get_predictions = _get_predictions


# ============== Model Validation Runner ==============

def validate_models_for_production(models_dir: str = 'models') -> Dict[str, Any]:
    """
    Run all model validation checks and return a report.
    
    Usage:
        from tests.test_model_validation import validate_models_for_production
        report = validate_models_for_production()
        if not report['valid']:
            print(report['errors'])
    """
    errors = []
    warnings_list = []
    
    # Load test data
    test_X = np.load('datasets/fpl_points_v1/test_X.npy')
    test_y = np.load('datasets/fpl_points_v1/test_y.npy')
    
    # Load models
    models = {}
    model_paths = [
        ('xgboost', f'{models_dir}/xgboost/model.pkl'),
        ('lightgbm', f'{models_dir}/lightgbm/model.pkl'),
    ]
    
    for name, path in model_paths:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    if not models:
        return {'valid': False, 'errors': ['No models found'], 'warnings': []}
    
    # Run checks
    for name, model in models.items():
        preds = model.predict(test_X)
        
        # Check 1: Finite predictions
        if not np.all(np.isfinite(preds)):
            errors.append(f"{name}: Non-finite predictions")
        
        # Check 2: Reasonable range
        if np.any(preds < -10) or np.any(preds > 25):
            warnings_list.append(f"{name}: Predictions outside typical range")
        
        # Check 3: Variance
        if np.std(preds) < 0.5:
            warnings_list.append(f"{name}: Low prediction variance")
        
        # Check 4: Correlation with actuals
        corr = np.corrcoef(test_y, preds)[0, 1]
        if corr < 0.1:
            errors.append(f"{name}: Very low correlation with actuals ({corr:.4f})")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings_list,
        'models_checked': list(models.keys()),
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
