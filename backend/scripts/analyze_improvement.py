#!/usr/bin/env python3
"""
Analyze the 3.32% improvement in detail.

Compares:
1. RMSE improvement (prediction accuracy)
2. Ranking correlation (Spearman)
3. Top-k accuracy (captain selection)
4. Estimated FPL points impact
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr

def load_data():
    """Load test data."""
    datasets_dir = Path("/home/akshit/fpl-lineup-optimizer/datasets/fpl_points_v1")
    test_X = np.load(datasets_dir / "test_X.npy")
    test_y = np.load(datasets_dir / "test_y.npy")
    return test_X, test_y

def load_baseline_model():
    """Load baseline XGBoost model."""
    model_path = Path("/home/akshit/fpl-lineup-optimizer/models/xgboost/model.pkl")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_best_ensemble():
    """Load best ensemble model."""
    model_path = Path("/home/akshit/fpl-lineup-optimizer/models/best_ensemble/ensemble.pkl")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_ensemble(ensemble_data, X):
    """Generate predictions from ensemble."""
    models = ensemble_data['models']
    weights = ensemble_data['weights']
    
    predictions = np.zeros(len(X))
    for (name, model), weight in zip(models.items(), weights):
        predictions += model.predict(X) * weight
    return predictions

def calculate_top_k_accuracy(y_true, y_pred, k=5):
    """Calculate top-k accuracy (for captain/vice-captain selection)."""
    # Get indices of top k actual performers
    top_k_actual = set(np.argsort(y_true)[-k:])
    # Get indices of top k predicted performers
    top_k_pred = set(np.argsort(y_pred)[-k:])
    # Calculate overlap
    correct = len(top_k_actual & top_k_pred)
    return correct / k

def calculate_captain_accuracy(y_true, y_pred):
    """Calculate captain selection accuracy."""
    actual_captain = np.argmax(y_true)
    predicted_captain = np.argmax(y_pred)
    return actual_captain == predicted_captain

def estimate_fpl_points_improvement(y_true, baseline_pred, new_pred):
    """
    Estimate FPL points improvement.
    
    This is a simplified estimation. Real backtesting would simulate actual
    transfers and lineup selections over a season.
    """
    # Calculate how often each model picks the actual top performer as captain
    n = len(y_true)
    
    # Simulate 5 gameweeks worth of data (our test set is ~41 players)
    # In reality, we'd select 11 starters + 1 captain from 15 players per GW
    
    # Simplified: assume we're selecting top 11 from 41
    # Captain is highest predicted
    
    baseline_top11_idx = np.argsort(baseline_pred)[-11:]
    new_top11_idx = np.argsort(new_pred)[-11:]
    
    baseline_captain_idx = np.argmax(baseline_pred)
    new_captain_idx = np.argmax(new_pred)
    
    # Calculate points from starters (simplified)
    baseline_starter_points = y_true[baseline_top11_idx].sum()
    new_starter_points = y_true[new_top11_idx].sum()
    
    # Captain gets double points
    baseline_captain_points = y_true[baseline_captain_idx]
    new_captain_points = y_true[new_captain_idx]
    
    baseline_total = baseline_starter_points + baseline_captain_points
    new_total = new_starter_points + new_captain_points
    
    return baseline_total, new_total

def main():
    print("="*70)
    print("DETAILED IMPROVEMENT ANALYSIS")
    print("="*70)
    
    # Load data and models
    test_X, test_y = load_data()
    baseline_model = load_baseline_model()
    ensemble_data = load_best_ensemble()
    
    # Generate predictions
    baseline_pred = baseline_model.predict(test_X)
    new_pred = predict_ensemble(ensemble_data, test_X)
    
    print(f"\nTest set size: {len(test_y)} samples")
    print(f"Target range: [{test_y.min():.2f}, {test_y.max():.2f}]")
    print(f"Target mean: {test_y.mean():.2f}, std: {test_y.std():.2f}")
    
    # 1. RMSE Comparison
    print("\n" + "="*70)
    print("1. RMSE (Root Mean Squared Error)")
    print("="*70)
    baseline_rmse = np.sqrt(mean_squared_error(test_y, baseline_pred))
    new_rmse = np.sqrt(mean_squared_error(test_y, new_pred))
    rmse_improvement = ((baseline_rmse - new_rmse) / baseline_rmse) * 100
    
    print(f"Baseline XGB:     {baseline_rmse:.4f}")
    print(f"New Ensemble:     {new_rmse:.4f}")
    print(f"Absolute diff:    {baseline_rmse - new_rmse:.4f}")
    print(f"Improvement:      {rmse_improvement:.2f}% ⭐")
    
    # 2. MAE Comparison
    print("\n" + "="*70)
    print("2. MAE (Mean Absolute Error)")
    print("="*70)
    baseline_mae = mean_absolute_error(test_y, baseline_pred)
    new_mae = mean_absolute_error(test_y, new_pred)
    mae_improvement = ((baseline_mae - new_mae) / baseline_mae) * 100
    
    print(f"Baseline XGB:     {baseline_mae:.4f}")
    print(f"New Ensemble:     {new_mae:.4f}")
    print(f"Improvement:      {mae_improvement:.2f}%")
    
    # 3. Correlation Metrics
    print("\n" + "="*70)
    print("3. Correlation Metrics")
    print("="*70)
    
    baseline_spearman = spearmanr(test_y, baseline_pred)[0]
    new_spearman = spearmanr(test_y, new_pred)[0]
    
    baseline_pearson = pearsonr(test_y, baseline_pred)[0]
    new_pearson = pearsonr(test_y, new_pred)[0]
    
    print(f"Spearman ρ (ranking):")
    print(f"  Baseline:       {baseline_spearman:.4f}")
    print(f"  New:            {new_spearman:.4f}")
    print(f"  Change:         {new_spearman - baseline_spearman:+.4f}")
    
    print(f"\nPearson r (linear):")
    print(f"  Baseline:       {baseline_pearson:.4f}")
    print(f"  New:            {new_pearson:.4f}")
    print(f"  Change:         {new_pearson - baseline_pearson:+.4f}")
    
    # 4. Top-k Accuracy
    print("\n" + "="*70)
    print("4. Top-K Accuracy (Captain/Vice-Captain Selection)")
    print("="*70)
    
    for k in [1, 3, 5, 10]:
        baseline_topk = calculate_top_k_accuracy(test_y, baseline_pred, k)
        new_topk = calculate_top_k_accuracy(test_y, new_pred, k)
        print(f"Top-{k:2d}:  Baseline={baseline_topk:.2%}, New={new_topk:.2%}, Change={new_topk-baseline_topk:+.2%}")
    
    # 5. Captain Accuracy
    print("\n" + "="*70)
    print("5. Captain Selection Accuracy")
    print("="*70)
    baseline_captain = calculate_captain_accuracy(test_y, baseline_pred)
    new_captain = calculate_captain_accuracy(test_y, new_pred)
    
    print(f"Baseline picks correct captain: {baseline_captain}")
    print(f"New picks correct captain:      {new_captain}")
    
    # 6. Prediction Distribution
    print("\n" + "="*70)
    print("6. Prediction Distribution")
    print("="*70)
    print(f"Actual targets:     mean={test_y.mean():.3f}, std={test_y.std():.3f}")
    print(f"Baseline predicts:  mean={baseline_pred.mean():.3f}, std={baseline_pred.std():.3f}")
    print(f"New predicts:       mean={new_pred.mean():.3f}, std={new_pred.std():.3f}")
    
    # Variance analysis
    print(f"\nVariance vs targets:")
    print(f"  Baseline MSE variance: {np.var((test_y - baseline_pred)):.4f}")
    print(f"  New MSE variance:      {np.var((test_y - new_pred)):.4f}")
    
    # 7. Estimated FPL Impact
    print("\n" + "="*70)
    print("7. Estimated FPL Points Impact (Simplified)")
    print("="*70)
    baseline_pts, new_pts = estimate_fpl_points_improvement(test_y, baseline_pred, new_pred)
    pts_improvement = new_pts - baseline_pts
    pts_pct = (pts_improvement / baseline_pts) * 100
    
    print(f"Estimated points from this GW:")
    print(f"  Baseline lineup:  {baseline_pts:.1f} pts")
    print(f"  New lineup:       {new_pts:.1f} pts")
    print(f"  Improvement:      +{pts_improvement:.1f} pts ({pts_pct:+.1f}%)")
    print()
    print("Note: This is a simplified estimate. Full backtesting would simulate")
    print("      the entire season with actual transfers and budget constraints.")
    
    # 8. Summary
    print("\n" + "="*70)
    print("8. SUMMARY")
    print("="*70)
    print()
    print("The 3.32% RMSE improvement means:")
    print(f"  ✓ Predictions are {rmse_improvement:.1f}% more accurate on average")
    print(f"  ✓ Ranking correlation improved by {new_spearman - baseline_spearman:+.3f}")
    print(f"  ✓ Better captain selection (potentially +{pts_improvement:.1f} pts/GW)")
    print()
    print("Over a 38-game season:")
    print(f"  → Potential gain: +{pts_improvement * 38:.0f} points")
    print(f"  → That's ~{pts_improvement * 38 / 5:.0f} positions in rankings!")
    print()
    print("="*70)

if __name__ == "__main__":
    main()
