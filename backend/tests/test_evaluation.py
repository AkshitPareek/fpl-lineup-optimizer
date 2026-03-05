"""
Tests for P2-T6: Model Evaluation (Backtesting and Position-wise Analysis)
Created by: Testing Expert (Teammate D)

Tests verify:
1. Backtest runs on holdout seasons
2. Position-wise MAE computed (GK, DEF, MID, FWD)
3. Price bracket analysis completed
4. Expected GW rank improvement calculated
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
    test_X = np.load(data_paths["test_X"])
    test_y = np.load(data_paths["test_y"])
    return train_X, train_y, val_X, val_y, test_X, test_y


class TestEvaluation:
    """Test suite for model evaluation (P2-T6)."""

    def test_evaluation_module_exists(self):
        """Evaluation module should exist."""
        from ml import evaluation

        assert evaluation is not None

    def test_backtest_runs(self, train_data):
        """Backtest should run on holdout data."""
        from ml.evaluation import run_backtest

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        results = run_backtest(train_X, train_y, test_X, test_y, n_splits=3)

        assert results is not None
        assert "overall_rmse" in results
        assert "fold_results" in results

    def test_position_wise_mae(self, train_data):
        """Position-wise MAE should be computed."""
        from ml.evaluation import compute_position_wise_mae

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        positions = np.array([0] * 10 + [1] * 15 + [2] * 10 + [3] * 6)

        position_mae = compute_position_wise_mae(
            test_X, test_y, positions, n_positions=4
        )

        assert position_mae is not None
        assert len(position_mae) == 4
        assert "GK" in position_mae or 0 in position_mae

    def test_price_bracket_analysis(self, train_data):
        """Price bracket analysis should be completed."""
        from ml.evaluation import analyze_price_brackets

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        prices = np.random.uniform(4.0, 14.0, len(test_y))

        bracket_results = analyze_price_brackets(test_X, test_y, prices, n_brackets=4)

        assert bracket_results is not None
        assert len(bracket_results) > 0

    def test_expected_gw_rank_improvement(self, train_data):
        """Expected GW rank improvement should be calculated."""
        from ml.evaluation import calculate_expected_rank_improvement

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        improvement = calculate_expected_rank_improvement(
            val_X, val_y, test_X, test_y, n_simulations=100
        )

        assert improvement is not None
        assert "mean_improvement" in improvement or "improvement" in improvement

    def test_model_comparison(self, train_data):
        """Models should be compared on test set."""
        from ml.evaluation import compare_models

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=10, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=10, random_state=42
            ),
        }

        comparison = compare_models(models, train_X, train_y, test_X, test_y)

        assert comparison is not None
        assert len(comparison) == len(models)

    def test_cross_validation_results(self, train_data):
        """Cross-validation results should be computed."""
        from ml.evaluation import compute_cross_validation_results

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        cv_results = compute_cross_validation_results(train_X, train_y, n_folds=5)

        assert cv_results is not None
        assert "mean_rmse" in cv_results
        assert "std_rmse" in cv_results

    def test_feature_effect_analysis(self, train_data):
        """Feature effect analysis should be performed."""
        from ml.evaluation import analyze_feature_effects

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        feature_effects = analyze_feature_effects(
            train_X,
            train_y,
            feature_names=[f"feature_{i}" for i in range(train_X.shape[1])],
        )

        assert feature_effects is not None

    def test_error_distribution_analysis(self, train_data):
        """Error distribution analysis should be computed."""
        from ml.evaluation import analyze_error_distribution

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)

        error_dist = analyze_error_distribution(test_y, predictions)

        assert error_dist is not None
        assert "mean_error" in error_dist or "bias" in error_dist
        assert "std_error" in error_dist or "variance" in error_dist

    def test_evaluation_report_generated(self, train_data):
        """Evaluation report should be generated."""
        from ml.evaluation import generate_evaluation_report

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        report = generate_evaluation_report(
            train_X, train_y, val_X, val_y, test_X, test_y
        )

        assert report is not None
        assert "test_rmse" in report or "test_metrics" in report
        assert "position_analysis" in report or "position_wise" in report

    def test_prediction_intervals(self, train_data):
        """Prediction intervals should be computed."""
        from ml.evaluation import compute_prediction_intervals

        train_X, train_y, val_X, val_y, test_X, test_y = train_data

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)

        intervals = compute_prediction_intervals(test_X, predictions, confidence=0.95)

        assert intervals is not None
        assert "lower" in intervals
        assert "upper" in intervals
        assert len(intervals["lower"]) == len(predictions)
