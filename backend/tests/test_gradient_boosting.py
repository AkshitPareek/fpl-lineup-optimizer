"""
Tests for P2-T3: Gradient Boosting (XGBoost/LightGBM + Optuna)
Created by: Testing Expert (Teammate D)

Tests verify:
1. XGBoost model trains with 5-fold CV
2. LightGBM model trains with 5-fold CV
3. Optuna runs 50+ trials
4. Feature importance extracted
5. SHAP values computed
6. Model performance >15% over baseline
"""

import pytest
import numpy as np
import os


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


class TestGradientBoosting:
    """Test suite for gradient boosting models (P2-T3)."""

    def test_xgboost_module_exists(self):
        """XGBoost module should exist."""
        from ml import gradient_boosting

        assert gradient_boosting is not None

    def test_xgboost_5fold_cv(self, train_data):
        """XGBoost should train with 5-fold cross-validation."""
        from ml.gradient_boosting import train_xgboost_cv

        train_X, train_y, val_X, val_y = train_data

        cv_results = train_xgboost_cv(train_X, train_y, n_folds=5)

        assert cv_results is not None
        assert "test_rmse_mean" in cv_results
        assert "test_rmse_std" in cv_results

    def test_lightgbm_5fold_cv(self, train_data):
        """LightGBM should train with 5-fold cross-validation."""
        from ml.gradient_boosting import train_lightgbm_cv

        train_X, train_y, val_X, val_y = train_data

        cv_results = train_lightgbm_cv(train_X, train_y, n_folds=5)

        assert cv_results is not None
        assert "test_rmse_mean" in cv_results
        assert "test_rmse_std" in cv_results

    def test_xgboost_model_predicts(self, train_data):
        """XGBoost model should produce predictions."""
        from ml.gradient_boosting import train_xgboost

        train_X, train_y, val_X, val_y = train_data

        model = train_xgboost(train_X, train_y)
        predictions = model.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)

    def test_lightgbm_model_predicts(self, train_data):
        """LightGBM model should produce predictions."""
        from ml.gradient_boosting import train_lightgbm

        train_X, train_y, val_X, val_y = train_data

        model = train_lightgbm(train_X, train_y)
        predictions = model.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)

    def test_optuna_optimization_runs(self, train_data):
        """Optuna should run optimization with 50+ trials."""
        from ml.hyperparameter_tuning import optimize_xgboost, optimize_lightgbm

        train_X, train_y, val_X, val_y = train_data

        best_params_xgb = optimize_xgboost(train_X, train_y, n_trials=50)

        assert best_params_xgb is not None
        assert len(best_params_xgb) > 0

    def test_optuna_optimization_lightgbm(self, train_data):
        """Optuna should optimize LightGBM hyperparameters."""
        from ml.hyperparameter_tuning import optimize_lightgbm

        train_X, train_y, val_X, val_y = train_data

        best_params_lgb = optimize_lightgbm(train_X, train_y, n_trials=50)

        assert best_params_lgb is not None
        assert len(best_params_lgb) > 0

    def test_xgboost_feature_importance(self, train_data):
        """XGBoost feature importance should be extracted."""
        from ml.gradient_boosting import train_xgboost, get_feature_importance

        train_X, train_y, val_X, val_y = train_data

        model = train_xgboost(train_X, train_y)
        importance = get_feature_importance(model)

        assert importance is not None
        assert len(importance) == train_X.shape[1]

    def test_lightgbm_feature_importance(self, train_data):
        """LightGBM feature importance should be extracted."""
        from ml.gradient_boosting import train_lightgbm, get_feature_importance

        train_X, train_y, val_X, val_y = train_data

        model = train_lightgbm(train_X, train_y)
        importance = get_feature_importance(model)

        assert importance is not None
        assert len(importance) == train_X.shape[1]

    def test_shap_values_computed(self, train_data):
        """SHAP values should be computed."""
        from ml.gradient_boosting import train_xgboost, compute_shap_values

        train_X, train_y, val_X, val_y = train_data

        model = train_xgboost(train_X, train_y)
        shap_values = compute_shap_values(model, val_X)

        assert shap_values is not None
        assert shap_values.shape == val_X.shape

    def test_model_performance_improvement(self, train_data):
        """XGBoost should achieve >15% improvement over baseline."""
        from ml.gradient_boosting import train_xgboost_cv
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        train_X, train_y, val_X, val_y = train_data

        baseline_model = LinearRegression()
        baseline_model.fit(train_X, train_y)
        baseline_pred = baseline_model.predict(val_X)
        baseline_rmse = np.sqrt(mean_squared_error(val_y, baseline_pred))

        cv_results = train_xgboost_cv(train_X, train_y, n_folds=5)
        xgb_rmse = cv_results["test_rmse_mean"]

        improvement = (baseline_rmse - xgb_rmse) / baseline_rmse

        assert improvement > 0.15

    def test_models_save_to_directory(self, train_data):
        """Models should save to expected directory structure."""
        from ml.gradient_boosting import train_xgboost, train_lightgbm
        import joblib

        train_X, train_y, val_X, val_y = train_data

        xgb_model = train_xgboost(train_X, train_y)
        lgb_model = train_lightgbm(train_X, train_y)

        os.makedirs("models/xgboost", exist_ok=True)
        os.makedirs("models/lightgbm", exist_ok=True)

        joblib.dump(xgb_model, "models/xgboost/model.pkl")
        joblib.dump(lgb_model, "models/lightgbm/model.pkl")

        assert os.path.exists("models/xgboost/model.pkl")
        assert os.path.exists("models/lightgbm/model.pkl")
