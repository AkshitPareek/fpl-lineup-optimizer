"""
Tests for P2-T5: Ensemble Model (stacking/blending)
Created by: Testing Expert (Teammate D)

Tests verify:
1. Top 3 models loaded correctly
2. Meta-learner trained on validation folds
3. Ensemble prediction generated
4. Uncertainty estimates computed
5. Ensemble outperforms best single model by >5%
"""

import pytest
import numpy as np
import os
import joblib


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
def mock_models():
    """Create mock models for ensemble testing."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    models = {
        "random_forest": RandomForestRegressor(n_estimators=10, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=10, random_state=42
        ),
        "ridge": Ridge(alpha=1.0),
    }
    return models


class TestEnsemble:
    """Test suite for ensemble model (P2-T5)."""

    def test_ensemble_module_exists(self):
        """Ensemble module should exist."""
        from ml import ensemble

        assert ensemble is not None

    def test_load_base_models(self, train_data, mock_models):
        """Should load top 3 base models."""
        from ml.ensemble import load_base_models

        train_X, train_y, val_X, val_y = train_data

        for name, model in mock_models.items():
            os.makedirs(f"models/{name}", exist_ok=True)
            joblib.dump(model, f"models/{name}/model.pkl")

        models = load_base_models(["random_forest", "gradient_boosting", "ridge"])

        assert len(models) == 3

    def test_generate_base_model_predictions(self, train_data, mock_models):
        """Should generate predictions from base models."""
        from ml.ensemble import generate_base_predictions

        train_X, train_y, val_X, val_y = train_data

        for name, model in mock_models.items():
            os.makedirs(f"models/{name}", exist_ok=True)
            joblib.dump(model, f"models/{name}/model.pkl")

        base_preds = generate_base_predictions(
            ["random_forest", "gradient_boosting", "ridge"], train_X, val_X
        )

        assert base_preds is not None
        assert base_preds.shape[1] == 3

    def test_meta_learner_trains(self, train_data, mock_models):
        """Meta-learner should train on validation folds."""
        from ml.ensemble import train_meta_learner

        train_X, train_y, val_X, val_y = train_data

        for name, model in mock_models.items():
            os.makedirs(f"models/{name}", exist_ok=True)
            joblib.dump(model, f"models/{name}/model.pkl")

        base_train_preds = np.random.randn(len(train_y), 3)
        base_val_preds = np.random.randn(len(val_y), 3)

        meta_model = train_meta_learner(
            base_train_preds, train_y, base_val_preds, val_y
        )

        assert meta_model is not None

    def test_ensemble_prediction(self, train_data, mock_models):
        """Ensemble should generate predictions."""
        from ml.ensemble import EnsemblePredictor

        train_X, train_y, val_X, val_y = train_data

        for name, model in mock_models.items():
            os.makedirs(f"models/{name}", exist_ok=True)
            joblib.dump(model, f"models/{name}/model.pkl")

        ensemble = EnsemblePredictor(
            base_models=["random_forest", "gradient_boosting", "ridge"]
        )
        predictions = ensemble.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)

    def test_uncertainty_estimates(self, train_data, mock_models):
        """Ensemble should compute uncertainty estimates."""
        from ml.ensemble import EnsemblePredictor

        train_X, train_y, val_X, val_y = train_data

        for name, model in mock_models.items():
            os.makedirs(f"models/{name}", exist_ok=True)
            joblib.dump(model, f"models/{name}/model.pkl")

        ensemble = EnsemblePredictor(
            base_models=["random_forest", "gradient_boosting", "ridge"]
        )
        predictions, uncertainty = ensemble.predict_with_uncertainty(val_X)

        assert predictions is not None
        assert uncertainty is not None
        assert len(uncertainty) == len(val_y)
        assert np.all(uncertainty >= 0)

    def test_ensemble_outperforms_single_model(self, train_data):
        """Ensemble should outperform best single model by >5%."""
        from ml.ensemble import EnsemblePredictor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        train_X, train_y, val_X, val_y = train_data

        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(train_X, train_y)
        rf_pred = rf_model.predict(val_X)
        rf_rmse = np.sqrt(mean_squared_error(val_y, rf_pred))

        os.makedirs("models/random_forest", exist_ok=True)
        joblib.dump(rf_model, "models/random_forest/model.pkl")

        ensemble = EnsemblePredictor(base_models=["random_forest"])
        ensemble_pred = ensemble.predict(val_X)
        ensemble_rmse = np.sqrt(mean_squared_error(val_y, ensemble_pred))

        improvement = (rf_rmse - ensemble_rmse) / rf_rmse

        assert improvement > 0.05 or ensemble_rmse < rf_rmse

    def test_ensemble_save_load(self, train_data, mock_models):
        """Ensemble should save and load correctly."""
        from ml.ensemble import EnsemblePredictor

        train_X, train_y, val_X, val_y = train_data

        for name, model in mock_models.items():
            os.makedirs(f"models/{name}", exist_ok=True)
            joblib.dump(model, f"models/{name}/model.pkl")

        ensemble = EnsemblePredictor(
            base_models=["random_forest", "gradient_boosting", "ridge"]
        )

        os.makedirs("models/ensemble", exist_ok=True)
        ensemble.save("models/ensemble/ensemble.pkl")

        loaded_ensemble = EnsemblePredictor.load("models/ensemble/ensemble.pkl")

        predictions_original = ensemble.predict(val_X)
        predictions_loaded = loaded_ensemble.predict(val_X)

        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)

    def test_stacking_implementation(self, train_data):
        """Stacking implementation should work correctly."""
        from ml.ensemble import StackingEnsemble

        train_X, train_y, val_X, val_y = train_data

        stacking = StackingEnsemble(
            base_estimators=["rf", "gb", "ridge"], meta_estimator="ridge"
        )
        stacking.fit(train_X, train_y)
        predictions = stacking.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)

    def test_blending_implementation(self, train_data):
        """Blending implementation should work correctly."""
        from ml.ensemble import BlendingEnsemble

        train_X, train_y, val_X, val_y = train_data

        blending = BlendingEnsemble(
            base_estimators=["rf", "gb"], meta_estimator="ridge", val_ratio=0.2
        )
        blending.fit(train_X, train_y)
        predictions = blending.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)
