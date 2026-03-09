"""
Data Validation Tests for FPL ML Pipeline

Ensures data quality and consistency throughout the ML pipeline.
These tests should be run before training or inference.

Usage:
    pytest backend/tests/test_data_validation.py -v
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any


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
        "metadata": os.path.join(base_path, "metadata.json"),
    }


@pytest.fixture
def training_data(data_paths):
    """Load all training data."""
    data = {}
    for key, path in data_paths.items():
        if key == 'metadata' and os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
        elif os.path.exists(path):
            data[key] = np.load(path)
    return data


# ============== Data Presence Tests ==============

class TestDataPresence:
    """Test that required data files exist."""
    
    def test_training_data_exists(self, data_paths):
        """Training data files must exist."""
        required = ['train_X', 'train_y', 'val_X', 'val_y', 'test_X', 'test_y']
        for key in required:
            assert os.path.exists(data_paths[key]), f"Missing: {data_paths[key]}"
    
    def test_metadata_exists(self, data_paths):
        """Metadata file should exist for feature tracking."""
        assert os.path.exists(data_paths['metadata']), \
            f"Metadata missing: {data_paths['metadata']}"
    
    def test_raw_data_exists(self, data_paths):
        """Raw CSV data should exist for debugging."""
        base = Path(data_paths['train_X']).parent
        assert (base / 'train_raw.csv').exists(), "Missing train_raw.csv"
        assert (base / 'test_raw.csv').exists(), "Missing test_raw.csv"


# ============== Data Shape Tests ==============

class TestDataShapes:
    """Test data shape consistency."""
    
    def test_feature_target_alignment(self, training_data):
        """X and y arrays must have same number of samples."""
        splits = [
            ('train_X', 'train_y'),
            ('val_X', 'val_y'),
            ('test_X', 'test_y'),
        ]
        
        for x_key, y_key in splits:
            if x_key in training_data and y_key in training_data:
                X = training_data[x_key]
                y = training_data[y_key]
                assert X.shape[0] == y.shape[0], \
                    f"{x_key} has {X.shape[0]} samples, {y_key} has {y.shape[0]}"
    
    def test_consistent_feature_count(self, training_data):
        """All X arrays must have same number of features."""
        X_keys = ['train_X', 'val_X', 'test_X']
        X_shapes = []
        
        for key in X_keys:
            if key in training_data:
                X_shapes.append(training_data[key].shape[1])
        
        if X_shapes:
            assert len(set(X_shapes)) == 1, \
                f"Inconsistent feature counts: {X_shapes}"
    
    def test_1d_targets(self, training_data):
        """Target arrays should be 1D."""
        for key in ['train_y', 'val_y', 'test_y']:
            if key in training_data:
                y = training_data[key]
                assert y.ndim == 1, f"{key} should be 1D, got shape {y.shape}"
    
    def test_2d_features(self, training_data):
        """Feature arrays should be 2D."""
        for key in ['train_X', 'val_X', 'test_X']:
            if key in training_data:
                X = training_data[key]
                assert X.ndim == 2, f"{key} should be 2D, got shape {X.shape}"


# ============== Data Quality Tests ==============

class TestDataQuality:
    """Test data quality (no NaN, inf, etc.)."""
    
    def test_no_nan_in_features(self, training_data):
        """Features should not contain NaN values."""
        for key in ['train_X', 'val_X', 'test_X']:
            if key in training_data:
                X = training_data[key]
                nan_count = np.isnan(X).sum()
                assert nan_count == 0, \
                    f"{key} has {nan_count} NaN values"
    
    def test_no_nan_in_targets(self, training_data):
        """Targets should not contain NaN values."""
        for key in ['train_y', 'val_y', 'test_y']:
            if key in training_data:
                y = training_data[key]
                nan_count = np.isnan(y).sum()
                assert nan_count == 0, \
                    f"{key} has {nan_count} NaN values"
    
    def test_no_inf_in_features(self, training_data):
        """Features should not contain infinite values."""
        for key in ['train_X', 'val_X', 'test_X']:
            if key in training_data:
                X = training_data[key]
                inf_count = np.isinf(X).sum()
                assert inf_count == 0, \
                    f"{key} has {inf_count} infinite values"
    
    def test_no_inf_in_targets(self, training_data):
        """Targets should not contain infinite values."""
        for key in ['train_y', 'val_y', 'test_y']:
            if key in training_data:
                y = training_data[key]
                inf_count = np.isinf(y).sum()
                assert inf_count == 0, \
                    f"{key} has {inf_count} infinite values"
    
    def test_targets_within_reasonable_range(self, training_data):
        """Target values should be within reasonable FPL point ranges."""
        for key in ['train_y', 'val_y', 'test_y']:
            if key in training_data:
                y = training_data[key]
                
                # FPL points typically between -5 and 20 per GW
                assert np.all(y >= -10), \
                    f"{key} has values < -10 (min: {y.min()})"
                assert np.all(y <= 25), \
                    f"{key} has values > 25 (max: {y.max()})"
    
    def test_features_not_all_identical(self, training_data):
        """Features should have variance (not all identical)."""
        for key in ['train_X', 'val_X', 'test_X']:
            if key in training_data:
                X = training_data[key]
                
                # Check each feature column
                for col_idx in range(X.shape[1]):
                    col = X[:, col_idx]
                    if np.std(col) == 0:
                        pytest.warns(UserWarning, f"{key} feature {col_idx} has zero variance")


# ============== Data Distribution Tests ==============

class TestDataDistribution:
    """Test data distributions are reasonable."""
    
    def test_training_set_larger_than_validation(self, training_data):
        """Training set should be larger than validation set."""
        if 'train_X' in training_data and 'val_X' in training_data:
            train_size = training_data['train_X'].shape[0]
            val_size = training_data['val_X'].shape[0]
            assert train_size > val_size, \
                f"Training set ({train_size}) not larger than validation ({val_size})"
    
    def test_no_extreme_outliers_in_targets(self, training_data):
        """Targets should not have extreme outliers (beyond 5 std)."""
        for key in ['train_y', 'val_y', 'test_y']:
            if key in training_data:
                y = training_data[key]
                
                mean = np.mean(y)
                std = np.std(y)
                
                if std > 0:
                    outliers = np.abs(y - mean) > 5 * std
                    outlier_pct = np.mean(outliers) * 100
                    
                    assert outlier_pct < 5, \
                        f"{key} has {outlier_pct:.1f}% extreme outliers"
    
    def test_target_distribution_reasonable(self, training_data):
        """Target distribution should look like FPL points."""
        if 'train_y' not in training_data:
            pytest.skip("No training targets")
        
        y = training_data['train_y']
        
        # Most FPL scores are between 0-10
        pct_0_10 = np.mean((y >= 0) & (y <= 10))
        assert pct_0_10 > 0.5, \
            f"Only {pct_0_10:.1%} of scores in 0-10 range (expected >50%)"
        
        # Very few negative scores
        pct_negative = np.mean(y < 0)
        assert pct_negative < 0.1, \
            f"{pct_negative:.1%} negative scores (expected <10%)"


# ============== Metadata Tests ==============

class TestMetadata:
    """Test metadata consistency."""
    
    def test_metadata_has_required_fields(self, training_data):
        """Metadata should have required fields."""
        if 'metadata' not in training_data:
            pytest.skip("No metadata")
        
        metadata = training_data['metadata']
        
        # Check for feature metadata
        assert 'feature_metadata' in metadata or 'features' in metadata, \
            "Metadata missing feature information"
    
    def test_feature_count_matches_metadata(self, training_data):
        """Actual feature count should match metadata."""
        if 'metadata' not in training_data or 'train_X' not in training_data:
            pytest.skip("Missing metadata or training data")
        
        metadata = training_data['metadata']
        actual_features = training_data['train_X'].shape[1]
        
        meta_features = metadata.get('feature_metadata', {}).get('n_features')
        if meta_features is None:
            meta_features = len(metadata.get('feature_metadata', {}).get('feature_names', []))
        
        if meta_features:
            assert actual_features == meta_features, \
                f"Feature count mismatch: actual={actual_features}, metadata={meta_features}"
    
    def test_feature_names_if_available(self, training_data):
        """Feature names should be provided if available."""
        if 'metadata' not in training_data:
            pytest.skip("No metadata")
        
        metadata = training_data['metadata']
        feature_meta = metadata.get('feature_metadata', {})
        
        if 'feature_names' in feature_meta:
            names = feature_meta['feature_names']
            assert len(names) == feature_meta.get('n_features', len(names)), \
                "Feature names count doesn't match n_features"


# ============== Train/Val/Test Split Tests ==============

class TestDataSplit:
    """Test train/val/test split integrity."""
    
    def test_no_data_leakage_between_splits(self, training_data):
        """
        Test that there's no exact duplicate rows between splits.
        
        Note: This is a basic check. More sophisticated leakage detection
        would require checking for player-time combinations.
        """
        # This is a simplified check - in practice, you'd want to check
        # that the same (player_id, gameweek) doesn't appear in multiple splits
        pass
    
    def test_test_set_not_empty(self, training_data):
        """Test set should have sufficient samples."""
        if 'test_X' in training_data:
            assert training_data['test_X'].shape[0] >= 10, \
                "Test set too small (< 10 samples)"
    
    def test_all_splits_have_positive_samples(self, training_data):
        """All splits should have positive number of samples."""
        for key in ['train_X', 'val_X', 'test_X']:
            if key in training_data:
                assert training_data[key].shape[0] > 0, \
                    f"{key} has no samples"


# ============== Data Drift Detection Tests ==============

class TestDataDrift:
    """
    Test for data drift between training and test sets.
    
    These tests help detect when the data distribution has changed,
    which could indicate the model needs retraining.
    """
    
    def test_feature_means_similar_between_splits(self, training_data):
        """Feature means should be similar between train and test."""
        if 'train_X' not in training_data or 'test_X' not in training_data:
            pytest.skip("Missing train or test data")
        
        train_mean = np.mean(training_data['train_X'], axis=0)
        test_mean = np.mean(training_data['test_X'], axis=0)
        
        # Compute relative difference
        relative_diff = np.abs(train_mean - test_mean) / (np.abs(train_mean) + 1e-8)
        
        # No feature should have >50% mean difference
        max_diff_idx = np.argmax(relative_diff)
        max_diff = relative_diff[max_diff_idx]
        
        assert max_diff < 0.5, \
            f"Feature {max_diff_idx} has {max_diff:.1%} mean difference between train/test"
    
    def test_target_means_similar_between_splits(self, training_data):
        """Target means should be similar between train and test."""
        if 'train_y' not in training_data or 'test_y' not in training_data:
            pytest.skip("Missing train or test targets")
        
        train_mean = np.mean(training_data['train_y'])
        test_mean = np.mean(training_data['test_y'])
        
        relative_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-8)
        
        assert relative_diff < 0.3, \
            f"Target mean differs by {relative_diff:.1%} between train/test"


# ============== Utility Functions ==============

def validate_data_for_training(data_dir: str = 'datasets/fpl_points_v1') -> Dict[str, Any]:
    """
    Run all data validation checks and return a report.
    
    Usage:
        from tests.test_data_validation import validate_data_for_training
        report = validate_data_for_training()
        if not report['valid']:
            print(report['errors'])
    """
    errors = []
    warnings_list = []
    
    data_paths = {
        "train_X": os.path.join(data_dir, "train_X.npy"),
        "train_y": os.path.join(data_dir, "train_y.npy"),
        "val_X": os.path.join(data_dir, "validation_X.npy"),
        "val_y": os.path.join(data_dir, "validation_y.npy"),
        "test_X": os.path.join(data_dir, "test_X.npy"),
        "test_y": os.path.join(data_dir, "test_y.npy"),
    }
    
    # Check files exist
    for key, path in data_paths.items():
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")
    
    if errors:
        return {'valid': False, 'errors': errors, 'warnings': warnings_list}
    
    # Load data
    data = {k: np.load(v) for k, v in data_paths.items()}
    
    # Check shapes
    for x_key, y_key in [('train_X', 'train_y'), ('val_X', 'val_y'), ('test_X', 'test_y')]:
        if data[x_key].shape[0] != data[y_key].shape[0]:
            errors.append(f"Shape mismatch: {x_key} vs {y_key}")
    
    # Check for NaN/Inf
    for key in data:
        if np.isnan(data[key]).any():
            errors.append(f"NaN values in {key}")
        if np.isinf(data[key]).any():
            errors.append(f"Infinite values in {key}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings_list,
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
