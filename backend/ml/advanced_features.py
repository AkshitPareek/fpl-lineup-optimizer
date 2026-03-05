import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Any


FEATURE_REGISTRY: Dict[str, str] = {}


def _register_feature(name: str, description: str) -> None:
    """Register a new feature with its description."""
    FEATURE_REGISTRY[name] = description


def get_feature_registry() -> Dict[str, str]:
    """Return the feature registry."""
    return FEATURE_REGISTRY.copy()


ORIGINAL_FEATURE_NAMES = [
    "now_cost",
    "total_points_rolling_1",
    "total_points_rolling_3",
    "total_points_rolling_6",
    "minutes_rolling_1",
    "minutes_rolling_3",
    "minutes_rolling_6",
    "goals_scored_rolling_1",
    "goals_scored_rolling_3",
    "assists_rolling_1",
    "assists_rolling_3",
    "clean_sheets_rolling_1",
    "clean_sheets_rolling_3",
    "avg_fdr_next_3",
    "avg_fdr_next_6",
    "dgw_next_3",
    "dgw_next_6",
    "bgw_next_3",
    "team_points_rolling_3",
    "team_goals_rolling_3",
    "team_conceded_rolling_3",
    "injury_risk_score",
    "form_trend",
    "points_per_minute",
    "value_ratio",
    "ownership_factor",
    "opp_strength",
]


def generate_polynomial_features(
    X: np.ndarray, n_degrees: int = 2
) -> Tuple[np.ndarray, List[str]]:
    """Generate polynomial features (squared, cubed terms for key numeric features)."""
    global FEATURE_REGISTRY

    key_feature_indices = [0, 13, 14, 21, 22, 25]
    key_feature_indices = [i for i in key_feature_indices if i < X.shape[1]]

    poly = PolynomialFeatures(
        degree=n_degrees, include_bias=False, interaction_only=False
    )

    key_features = X[:, key_feature_indices]
    poly_features = poly.fit_transform(key_features)

    poly_feature_names = poly.get_feature_names_out(
        [ORIGINAL_FEATURE_NAMES[i] for i in key_feature_indices]
    )

    new_features = poly_features[:, len(key_feature_indices) :]

    for i, name in enumerate(poly_feature_names[len(key_feature_indices) :]):
        _register_feature(f"poly_{name}", f"Polynomial feature: {name}")

    result = np.hstack([X, new_features])

    new_names = [
        f"poly_{name}" for name in poly_feature_names[len(key_feature_indices) :]
    ]
    all_names = ORIGINAL_FEATURE_NAMES + new_names

    return result, all_names


def generate_interaction_features(X: np.ndarray) -> np.ndarray:
    """Generate interaction features (multiplication of related features)."""
    global FEATURE_REGISTRY

    interactions = []
    interaction_names = []

    if X.shape[1] > 13:
        interactions.append(X[:, 1] * X[:, 13])
        interaction_names.append("total_points_rolling_3_x_avg_fdr_next_3")
        _register_feature(
            "total_points_rolling_3_x_avg_fdr_next_3",
            "Interaction: total_points_rolling_3 * avg_fdr_next_3",
        )

    if X.shape[1] > 14:
        interactions.append(X[:, 2] * X[:, 14])
        interaction_names.append("total_points_rolling_6_x_avg_fdr_next_6")
        _register_feature(
            "total_points_rolling_6_x_avg_fdr_next_6",
            "Interaction: total_points_rolling_6 * avg_fdr_next_6",
        )

    if X.shape[1] > 21:
        interactions.append(X[:, 21] * X[:, 22])
        interaction_names.append("injury_risk_score_x_form_trend")
        _register_feature(
            "injury_risk_score_x_form_trend",
            "Interaction: injury_risk_score * form_trend",
        )

    if X.shape[1] > 0:
        interactions.append(X[:, 0] * X[:, 22])
        interaction_names.append("now_cost_x_form_trend")
        _register_feature("now_cost_x_form_trend", "Interaction: now_cost * form_trend")

    if X.shape[1] > 4:
        interactions.append(X[:, 4] * X[:, 7])
        interaction_names.append("minutes_rolling_1_x_goals_scored_rolling_1")
        _register_feature(
            "minutes_rolling_1_x_goals_scored_rolling_1",
            "Interaction: minutes_rolling_1 * goals_scored_rolling_1",
        )

    if X.shape[1] > 5:
        interactions.append(X[:, 5] * X[:, 8])
        interaction_names.append("minutes_rolling_3_x_goals_scored_rolling_3")
        _register_feature(
            "minutes_rolling_3_x_goals_scored_rolling_3",
            "Interaction: minutes_rolling_3 * goals_scored_rolling_3",
        )

    if X.shape[1] > 18:
        interactions.append(X[:, 18] * X[:, 19])
        interaction_names.append("team_points_x_team_goals")
        _register_feature(
            "team_points_x_team_goals",
            "Interaction: team_points_rolling_3 * team_goals_rolling_3",
        )

    if X.shape[1] > 20:
        interactions.append(X[:, 0] * X[:, 20])
        interaction_names.append("now_cost_x_opp_strength")
        _register_feature(
            "now_cost_x_opp_strength", "Interaction: now_cost * opp_strength"
        )

    if X.shape[1] > 24:
        interactions.append(X[:, 22] * X[:, 24])
        interaction_names.append("form_trend_x_value_ratio")
        _register_feature(
            "form_trend_x_value_ratio", "Interaction: form_trend * value_ratio"
        )

    if X.shape[1] > 23:
        interactions.append(X[:, 1] * X[:, 23])
        interaction_names.append("total_points_x_ownership_factor")
        _register_feature(
            "total_points_x_ownership_factor",
            "Interaction: total_points_rolling_1 * ownership_factor",
        )

    if interactions:
        interaction_array = np.column_stack(interactions)
        return np.hstack([X, interaction_array])

    return X


def generate_ratio_features(X: np.ndarray) -> np.ndarray:
    """Generate ratio features (divisions that create meaningful ratios)."""
    global FEATURE_REGISTRY

    ratios = []
    ratio_names = []

    if X.shape[1] > 2 and X.shape[1] > 4:
        mask = X[:, 4] != 0
        ratio = np.zeros(X.shape[0])
        ratio[mask] = X[mask, 1] / (X[mask, 4] + 1e-8)
        ratios.append(ratio)
        ratio_names.append("points_per_minute_rolling_3")
        _register_feature(
            "points_per_minute_rolling_3",
            "Ratio: total_points_rolling_3 / minutes_rolling_3",
        )

    if X.shape[1] > 3 and X.shape[1] > 5:
        mask = X[:, 5] != 0
        ratio = np.zeros(X.shape[0])
        ratio[mask] = X[mask, 2] / (X[mask, 5] + 1e-8)
        ratios.append(ratio)
        ratio_names.append("points_per_minute_rolling_6")
        _register_feature(
            "points_per_minute_rolling_6",
            "Ratio: total_points_rolling_6 / minutes_rolling_6",
        )

    if X.shape[1] > 18 and X.shape[1] > 20:
        mask = X[:, 20] != 0
        ratio = np.zeros(X.shape[0])
        ratio[mask] = X[mask, 18] / (X[mask, 20] + 1e-8)
        ratios.append(ratio)
        ratio_names.append("team_points_per_conceded")
        _register_feature(
            "team_points_per_conceded",
            "Ratio: team_points_rolling_3 / team_conceded_rolling_3",
        )

    if X.shape[1] > 19 and X.shape[1] > 20:
        mask = X[:, 20] != 0
        ratio = np.zeros(X.shape[0])
        ratio[mask] = X[mask, 19] / (X[mask, 20] + 1e-8)
        ratios.append(ratio)
        ratio_names.append("team_goal_diff")
        _register_feature(
            "team_goal_diff", "Ratio: team_goals_rolling_3 / team_conceded_rolling_3"
        )

    if X.shape[1] > 8 and X.shape[1] > 10:
        mask = X[:, 10] != 0
        ratio = np.zeros(X.shape[0])
        ratio[mask] = X[mask, 8] / (X[mask, 10] + 1e-8)
        ratios.append(ratio)
        ratio_names.append("goals_to_assists_ratio_3")
        _register_feature(
            "goals_to_assists_ratio_3",
            "Ratio: goals_scored_rolling_3 / assists_rolling_3",
        )

    if X.shape[1] > 14 and X.shape[1] > 13:
        mask = X[:, 13] != 0
        ratio = np.zeros(X.shape[0])
        ratio[mask] = X[mask, 14] / (X[mask, 13] + 1e-8)
        ratios.append(ratio)
        ratio_names.append("fdr_improvement")
        _register_feature("fdr_improvement", "Ratio: avg_fdr_next_6 / avg_fdr_next_3")

    if ratios:
        ratio_array = np.column_stack(ratios)
        return np.hstack([X, ratio_array])

    return X


def generate_statistical_features(X: np.ndarray) -> np.ndarray:
    """Generate statistical features (rolling means, std from recent gameweeks)."""
    global FEATURE_REGISTRY

    stats = []
    stat_names = []

    if X.shape[1] > 1:
        stats.append(
            np.std(X[:, 1:4], axis=1) if X.shape[1] >= 4 else np.zeros(X.shape[0])
        )
        stat_names.append("points_std_rolling")
        _register_feature("points_std_rolling", "Statistical: std of rolling points")

    if X.shape[1] > 4:
        stats.append(
            np.std(X[:, 4:7], axis=1) if X.shape[1] >= 7 else np.zeros(X.shape[0])
        )
        stat_names.append("minutes_std_rolling")
        _register_feature("minutes_std_rolling", "Statistical: std of rolling minutes")

    if X.shape[1] > 7:
        stats.append(
            np.max(X[:, 7:9], axis=1) if X.shape[1] >= 9 else np.zeros(X.shape[0])
        )
        stat_names.append("max_goals_rolling")
        _register_feature("max_goals_rolling", "Statistical: max of rolling goals")

    if X.shape[1] > 1 and X.shape[1] > 13:
        stats.append(X[:, 1] - X[:, 13])
        stat_names.append("points_vs_fdr_3")
        _register_feature(
            "points_vs_fdr_3", "Statistical: total_points_rolling_1 - avg_fdr_next_3"
        )

    if X.shape[1] > 18:
        stats.append(np.sum(X[:, 18:21], axis=1) if X.shape[1] >= 21 else X[:, 18])
        stat_names.append("team_offensive_score")
        _register_feature(
            "team_offensive_score", "Statistical: sum of team points + goals"
        )

    if X.shape[1] > 22:
        stats.append(np.abs(X[:, 22]))
        stat_names.append("form_trend_abs")
        _register_feature("form_trend_abs", "Statistical: absolute form_trend")

    if stats:
        stat_array = np.column_stack(stats)
        return np.hstack([X, stat_array])

    return X


def generate_log_features(X: np.ndarray) -> np.ndarray:
    """Generate log transforms for skewed distributions."""
    global FEATURE_REGISTRY

    log_features = []
    log_names = []

    if X.shape[1] > 0:
        log_now_cost = np.log1p(np.abs(X[:, 0]))
        log_features.append(log_now_cost)
        log_names.append("log_now_cost")
        _register_feature("log_now_cost", "Log transform: log(1 + now_cost)")

    if X.shape[1] > 1:
        log_points_1 = np.log1p(np.abs(X[:, 1]))
        log_features.append(log_points_1)
        log_names.append("log_total_points_rolling_1")
        _register_feature(
            "log_total_points_rolling_1",
            "Log transform: log(1 + total_points_rolling_1)",
        )

    if X.shape[1] > 2:
        log_points_3 = np.log1p(np.abs(X[:, 2]))
        log_features.append(log_points_3)
        log_names.append("log_total_points_rolling_3")
        _register_feature(
            "log_total_points_rolling_3",
            "Log transform: log(1 + total_points_rolling_3)",
        )

    if X.shape[1] > 13:
        log_fdr = np.log1p(np.abs(X[:, 13]))
        log_features.append(log_fdr)
        log_names.append("log_avg_fdr_next_3")
        _register_feature(
            "log_avg_fdr_next_3", "Log transform: log(1 + avg_fdr_next_3)"
        )

    if log_features:
        log_array = np.column_stack(log_features)
        return np.hstack([X, log_array])

    return X


def generate_advanced_features(
    X: np.ndarray, feature_names: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate advanced features (20+ new features).

    Returns new X with additional features and updated feature names.
    """
    global FEATURE_REGISTRY, ORIGINAL_FEATURE_NAMES

    if feature_names is None:
        feature_names = ORIGINAL_FEATURE_NAMES.copy()
    else:
        ORIGINAL_FEATURE_NAMES = feature_names.copy()

    X_poly, poly_names = generate_polynomial_features(X.copy(), n_degrees=2)

    X_interact = generate_interaction_features(X_poly)

    X_ratio = generate_ratio_features(X_interact)

    X_stats = generate_statistical_features(X_ratio)

    X_log = generate_log_features(X_stats)

    new_feature_count = X_log.shape[1] - X.shape[1]

    all_names = feature_names + [
        name for name in FEATURE_REGISTRY.keys() if name not in feature_names
    ]

    return X_log, all_names


def generate_all_features(X: np.ndarray) -> np.ndarray:
    """Generate all advanced features, returning only the enhanced X array."""
    X_enhanced, _ = generate_advanced_features(X)
    return X_enhanced


def compute_feature_correlations(X: np.ndarray) -> np.ndarray:
    """Compute correlation matrix for all features."""
    return np.corrcoef(X.T)


def compute_feature_correlation(X: np.ndarray) -> np.ndarray:
    """Alias for compute_feature_correlations for test compatibility."""
    return compute_feature_correlations(X)


def compute_feature_importance(model: Any, n_features: int) -> np.ndarray:
    """Compute feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        return np.abs(model.coef_)
    else:
        return np.ones(n_features) / n_features


def get_top_features_by_importance(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], n: int = 20
) -> List[Tuple[str, float]]:
    """Returns top N features by importance from Random Forest."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = model.feature_importances_

    if len(feature_names) < len(importances):
        feature_names = feature_names + [
            f"feature_{i}" for i in range(len(importances) - len(feature_names))
        ]

    paired = list(zip(feature_names[: len(importances)], importances))
    paired.sort(key=lambda x: x[1], reverse=True)

    return paired[:n]


def remove_redundant_features(
    X: np.ndarray, feature_names: List[str] = None, threshold: float = 0.95
) -> Tuple[np.ndarray, List[int]]:
    """Remove redundant features based on correlation threshold."""
    if X.shape[1] <= 1:
        return X, []

    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.abs(corr_matrix)

    n_features = X.shape[1]
    to_remove = set()

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if corr_matrix[i, j] > threshold:
                to_remove.add(j)

    keep_indices = [i for i in range(n_features) if i not in to_remove]

    X_reduced = X[:, keep_indices] if keep_indices else X

    return X_reduced, sorted(list(to_remove))


def update_training_datasets(
    new_X: np.ndarray = None,
    new_feature_names: List[str] = None,
    output_dir: str = "datasets/fpl_points_v1",
) -> None:
    """Update training datasets with new features and save to numpy files."""
    train_X_path = os.path.join(output_dir, "train_X.npy")
    val_X_path = os.path.join(output_dir, "validation_X.npy")
    test_X_path = os.path.join(output_dir, "test_X.npy")

    if os.path.exists(train_X_path):
        original_train_X = np.load(train_X_path)

        train_X_enhanced, all_feature_names = generate_advanced_features(
            original_train_X
        )
        np.save(train_X_path, train_X_enhanced)

        original_val_X = np.load(val_X_path)
        val_X_enhanced = generate_all_features(original_val_X)
        np.save(val_X_path, val_X_enhanced)

        original_test_X = np.load(test_X_path)
        test_X_enhanced = generate_all_features(original_test_X)
        np.save(test_X_path, test_X_enhanced)

        final_feature_names = all_feature_names
        final_n_features = train_X_enhanced.shape[1]
    else:
        final_feature_names = new_feature_names if new_feature_names else []
        final_n_features = new_X.shape[1] if new_X is not None else 0

    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata["feature_metadata"]["n_features"] = final_n_features
        metadata["feature_metadata"]["feature_names"] = final_feature_names
        metadata["advanced_features"] = list(FEATURE_REGISTRY.keys())

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Updated training datasets with {final_n_features} features")


def save_feature_metadata(
    feature_names: List[str], output_path: str = "datasets/fpl_points_v1/metadata.json"
) -> None:
    """Save feature names to metadata.json."""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"feature_metadata": {}}

    metadata["feature_metadata"]["feature_names"] = feature_names
    metadata["feature_metadata"]["n_features"] = len(feature_names)
    metadata["advanced_features"] = list(FEATURE_REGISTRY.keys())

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
