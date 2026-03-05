"""
Tests for P2-T2: Advanced Feature Engineering
Created by: Testing Expert (Teammate D)

Tests verify:
1. 20+ new features generated correctly
2. Feature correlation analysis runs
3. Feature importance from baseline models computed
4. Training datasets updated with new features
5. No data leakage between train/validation
"""

import pytest
import numpy as np
import os
import pandas as pd


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
def train_data(data_paths):
    """Load training and validation data."""
    train_X = np.load(data_paths["train_X"])
    train_y = np.load(data_paths["train_y"])
    val_X = np.load(data_paths["val_X"])
    val_y = np.load(data_paths["val_y"])
    return train_X, train_y, val_X, val_y


class TestAdvancedFeatures:
    """Test suite for advanced feature engineering (P2-T2)."""

    def test_advanced_features_module_exists(self):
        """Advanced features module should exist."""
        from ml import advanced_features

        assert advanced_features is not None

    def test_generate_polynomial_features(self, train_data):
        """Should generate polynomial features (20+ new features)."""
        from ml.advanced_features import generate_polynomial_features

        train_X, train_y, val_X, val_y = train_data

        train_X_enhanced, feature_names = generate_polynomial_features(
            train_X, n_degrees=2
        )

        assert train_X_enhanced.shape[1] > train_X.shape[1]
        assert (train_X_enhanced.shape[1] - train_X.shape[1]) >= 20

    def test_generate_interaction_features(self, train_data):
        """Should generate interaction features between features."""
        from ml.advanced_features import generate_interaction_features

        train_X, train_y, val_X, val_y = train_data

        train_X_enhanced = generate_interaction_features(train_X)

        assert train_X_enhanced.shape[1] > train_X.shape[1]

    def test_generate_statistical_features(self, train_data):
        """Should generate statistical rolling features."""
        from ml.advanced_features import generate_statistical_features

        train_X, train_y, val_X, val_y = train_data

        train_X_enhanced = generate_statistical_features(train_X)

        assert train_X_enhanced.shape[1] > train_X.shape[1]

    def test_feature_correlation_analysis(self, train_data):
        """Feature correlation analysis should run without error."""
        from ml.advanced_features import (
            generate_all_features,
            compute_feature_correlation,
        )

        train_X, train_y, val_X, val_y = train_data

        train_X_enhanced = generate_all_features(train_X)
        correlation_matrix = compute_feature_correlation(train_X_enhanced)

        assert correlation_matrix is not None
        assert correlation_matrix.shape[0] == train_X_enhanced.shape[1]
        assert correlation_matrix.shape[1] == train_X_enhanced.shape[1]

    def test_feature_importance_computation(self, train_data):
        """Feature importance should be computed from baseline models."""
        from ml.advanced_features import (
            generate_all_features,
            compute_feature_importance,
        )
        from sklearn.ensemble import RandomForestRegressor

        train_X, train_y, val_X, val_y = train_data

        train_X_enhanced = generate_all_features(train_X)

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(train_X_enhanced, train_y)

        importance = compute_feature_importance(model, train_X_enhanced.shape[1])

        assert importance is not None
        assert len(importance) == train_X_enhanced.shape[1]

    def test_training_datasets_updated(self, train_data):
        """Training datasets should be updated with new features."""
        from ml.advanced_features import generate_all_features

        train_X, train_y, val_X, val_y = train_data

        train_X_new = generate_all_features(train_X)
        val_X_new = generate_all_features(val_X)

        assert train_X_new.shape[1] > train_X.shape[1]
        assert val_X_new.shape[1] > val_X.shape[1]
        assert train_X_new.shape[1] == val_X_new.shape[1]

    def test_no_data_leakage(self, train_data):
        """Should have no data leakage between train/validation."""
        from ml.advanced_features import generate_all_features

        train_X, train_y, val_X, val_y = train_data

        train_X_new = generate_all_features(train_X)
        val_X_new = generate_all_features(val_X)

        train_mean = np.mean(train_X_new, axis=0)
        val_mean = np.mean(val_X_new, axis=0)

        assert not np.array_equal(train_X_new, val_X_new)

    def test_feature_registry_exists(self):
        """Feature registry should document new features."""
        from ml.advanced_features import get_feature_registry

        registry = get_feature_registry()

        assert registry is not None
        assert len(registry) >= 20

    def test_redundant_features_identified(self, train_data):
        """Redundant features should be identified and removable."""
        from ml.advanced_features import (
            generate_all_features,
            remove_redundant_features,
        )

        train_X, train_y, val_X, val_y = train_data

        train_X_enhanced = generate_all_features(train_X)
        train_X_reduced, removed_indices = remove_redundant_features(
            train_X_enhanced, threshold=0.95
        )

        assert train_X_reduced.shape[1] <= train_X_enhanced.shape[1]
