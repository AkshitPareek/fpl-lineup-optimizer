"""
Tests for P2-T1: Baseline Models
Created by: Testing Expert (Teammate D)

Tests verify:
1. LinearRegression model trains without error
2. RandomForest model trains without error
3. Models produce predictions on validation set
4. Metrics computed: RMSE, R², MAE
5. Models save/load correctly (joblib)
"""

import pytest
import numpy as np
import os
import tempfile
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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
def train_data(data_paths):
    """Load training and validation data."""
    train_X = np.load(data_paths["train_X"])
    train_y = np.load(data_paths["train_y"])
    val_X = np.load(data_paths["val_X"])
    val_y = np.load(data_paths["val_y"])
    return train_X, train_y, val_X, val_y


@pytest.fixture
def test_data(data_paths):
    """Load test data."""
    test_X = np.load(data_paths["test_X"])
    test_y = np.load(data_paths["test_y"])
    return test_X, test_y


class TestBaselineModels:
    """Test suite for baseline models (P2-T1)."""

    def test_linear_regression_trains(self, train_data):
        """LinearRegression should train without error."""
        train_X, train_y, val_X, val_y = train_data

        model = LinearRegression()
        model.fit(train_X, train_y)

        assert model is not None
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")

    def test_random_forest_trains(self, train_data):
        """RandomForest should train without error."""
        train_X, train_y, val_X, val_y = train_data

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_X, train_y)

        assert model is not None
        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == 100

    def test_linear_regression_predictions(self, train_data):
        """LinearRegression should produce predictions."""
        train_X, train_y, val_X, val_y = train_data

        model = LinearRegression()
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)
        assert predictions.shape == val_y.shape

    def test_random_forest_predictions(self, train_data):
        """RandomForest should produce predictions."""
        train_X, train_y, val_X, val_y = train_data

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)
        assert predictions.shape == val_y.shape

    def test_linear_regression_metrics(self, train_data):
        """LinearRegression should compute RMSE, R², MAE."""
        train_X, train_y, val_X, val_y = train_data

        model = LinearRegression()
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)

        rmse = np.sqrt(mean_squared_error(val_y, predictions))
        r2 = r2_score(val_y, predictions)
        mae = mean_absolute_error(val_y, predictions)

        assert rmse is not None
        assert r2 is not None
        assert mae is not None
        assert rmse >= 0
        assert mae >= 0

    def test_random_forest_metrics(self, train_data):
        """RandomForest should compute RMSE, R², MAE."""
        train_X, train_y, val_X, val_y = train_data

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)

        rmse = np.sqrt(mean_squared_error(val_y, predictions))
        r2 = r2_score(val_y, predictions)
        mae = mean_absolute_error(val_y, predictions)

        assert rmse is not None
        assert r2 is not None
        assert mae is not None
        assert rmse >= 0
        assert mae >= 0

    def test_linear_regression_save_load(self, train_data):
        """LinearRegression should save and load correctly."""
        train_X, train_y, val_X, val_y = train_data

        model = LinearRegression()
        model.fit(train_X, train_y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            joblib.dump(model, temp_path)
            loaded_model = joblib.load(temp_path)

            predictions_original = model.predict(val_X)
            predictions_loaded = loaded_model.predict(val_X)

            np.testing.assert_array_almost_equal(
                predictions_original, predictions_loaded
            )
        finally:
            os.unlink(temp_path)

    def test_random_forest_save_load(self, train_data):
        """RandomForest should save and load correctly."""
        train_X, train_y, val_X, val_y = train_data

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(train_X, train_y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            joblib.dump(model, temp_path)
            loaded_model = joblib.load(temp_path)

            predictions_original = model.predict(val_X)
            predictions_loaded = loaded_model.predict(val_X)

            np.testing.assert_array_almost_equal(
                predictions_original, predictions_loaded
            )
        finally:
            os.unlink(temp_path)

    def test_models_in_models_directory_structure(self):
        """Models should be saveable to expected directory structure."""
        expected_dirs = [
            "models/baseline",
            "models/baseline/linear_regression",
            "models/baseline/random_forest",
        ]

        for dir_path in expected_dirs:
            assert os.path.exists(dir_path) or os.path.isabs(dir_path)
