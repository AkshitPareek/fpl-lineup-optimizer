#!/usr/bin/env python3
"""
Robust Autonomous Research Runner

Runs experiments comparing available models and documents results.
"""

import json
import os
import sys
import time
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

# Setup paths
PROJECT_ROOT = Path("/home/akshit/fpl-lineup-optimizer")
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def load_test_data():
    """Load test data."""
    base_path = PROJECT_ROOT / "datasets/fpl_points_v1"
    return {
        'X': np.load(base_path / "test_X.npy"),
        'y': np.load(base_path / "test_y.npy"),
    }


def safe_load_model(model_path: Path):
    """Safely load a model with error handling."""
    try:
        model = joblib.load(model_path)
        # Quick test prediction to verify it works
        dummy = np.zeros((1, 26))
        _ = model.predict(dummy)
        return model
    except Exception as e:
        print(f"    ⚠️  Failed to load {model_path.name}: {e}")
        return None


def get_available_models():
    """Get list of available working models."""
    models_dir = PROJECT_ROOT / "models"
    available = {}
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "model.pkl"
            if model_file.exists():
                model = safe_load_model(model_file)
                if model is not None:
                    available[model_dir.name] = model
    
    return available


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    preds = model.predict(X_test)
    
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
        'mae': float(mean_absolute_error(y_test, preds)),
        'r2': float(1 - np.sum((y_test - preds)**2) / np.sum((y_test - np.mean(y_test))**2)),
        'spearman_corr': float(spearmanr(y_test, preds)[0]),
    }


def run_comparison(baseline_name: str, test_name: str, baseline_model, test_model, 
                   test_data: Dict, run_id: int) -> Dict:
    """Run a comparison between two models."""
    
    print(f"\n{'='*60}")
    print(f"Experiment {run_id}: {test_name} vs {baseline_name} (baseline)")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Evaluate both
    baseline_metrics = evaluate_model(baseline_model, test_data['X'], test_data['y'])
    test_metrics = evaluate_model(test_model, test_data['X'], test_data['y'])
    
    print(f"Baseline ({baseline_name}):")
    print(f"  RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"  MAE:  {baseline_metrics['mae']:.4f}")
    print(f"  R²:   {baseline_metrics['r2']:.4f}")
    print(f"  ρ:    {baseline_metrics['spearman_corr']:.4f}")
    
    print(f"\nTest ({test_name}):")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  ρ:    {test_metrics['spearman_corr']:.4f}")
    
    # Calculate improvement
    rmse_improvement = baseline_metrics['rmse'] - test_metrics['rmse']
    relative_improvement = (rmse_improvement / baseline_metrics['rmse']) * 100
    
    # Estimate effect size (simplified)
    effect_size = abs(rmse_improvement) / 0.5
    
    # Determine significance (simplified criteria)
    is_significant = effect_size > 0.3 and abs(relative_improvement) > 2.0
    is_improvement = test_metrics['rmse'] < baseline_metrics['rmse'] and is_significant
    
    print(f"\n📊 Comparison:")
    print(f"  RMSE Δ: {rmse_improvement:+.4f} ({relative_improvement:+.2f}%)")
    print(f"  Effect size: {effect_size:.3f}")
    print(f"  Significant: {'✅ Yes' if is_significant else '❌ No'}")
    print(f"  Improvement: {'✅ YES!' if is_improvement else '❌ No'}")
    
    # Save results
    duration = time.time() - start_time
    
    exp_dir = PROJECT_ROOT / f"research/03-experiments/{datetime.now().strftime('%Y-%m-%d')}-exp-{run_id:03d}-{test_name}-vs-{baseline_name}"
    exp_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_name,
        'test_model': test_name,
        'baseline_metrics': baseline_metrics,
        'test_metrics': test_metrics,
        'comparison': {
            'rmse_delta': float(rmse_improvement),
            'relative_percent': float(relative_improvement),
            'effect_size': float(effect_size),
            'is_significant': bool(is_significant),
            'is_improvement': bool(is_improvement),
        },
        'duration_seconds': duration,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create README
    status_icon = "✅" if is_improvement else "⚠️" if is_significant else "❌"
    readme = f"""# EXP-{run_id:03d}: {test_name} vs {baseline_name}

**Status:** {status_icon} {'IMPROVEMENT' if is_improvement else 'SIGNIFICANT' if is_significant else 'No significant improvement'}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Comparison:** {test_name} (test) vs {baseline_name} (baseline)

## Results

### Baseline ({baseline_name})

| Metric | Value |
|--------|-------|
| RMSE | {baseline_metrics['rmse']:.4f} |
| MAE | {baseline_metrics['mae']:.4f} |
| R² | {baseline_metrics['r2']:.4f} |
| Spearman ρ | {baseline_metrics['spearman_corr']:.4f} |

### Test Model ({test_name})

| Metric | Value |
|--------|-------|
| RMSE | {test_metrics['rmse']:.4f} |
| MAE | {test_metrics['mae']:.4f} |
| R² | {test_metrics['r2']:.4f} |
| Spearman ρ | {test_metrics['spearman_corr']:.4f} |

### Comparison

| Measure | Value |
|---------|-------|
| RMSE Δ | {rmse_improvement:+.4f} |
| Relative | {relative_improvement:+.2f}% |
| Effect Size | {effect_size:.3f} |
| Significant | {'Yes ✅' if is_significant else 'No ❌'} |

## Conclusion

{'🌟 **BREAKTHROUGH:** This model shows significant improvement over baseline and should be adopted as the new standard.' if is_improvement else '✓ This model shows significant difference but not necessarily improvement.' if is_significant else '✗ This model does not show significant improvement over baseline.'}

## Next Steps

""" + ('1. Update baseline to use this model\n2. Investigate why this model performs better\n3. Try ensemble with this model' if is_improvement else '1. Try different hyperparameters\n2. Investigate feature importance\n3. Consider different architecture') + """

---
*Auto-generated by AutoResearch Harness v1.0*
*Duration: {duration:.1f}s*
"""
    
    with open(exp_dir / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"\n💾 Results saved to: {exp_dir}")
    
    return results


def main():
    print("="*60)
    print("AUTONOMOUS RESEARCH - MODEL COMPARISON")
    print("="*60)
    
    # Load test data
    test_data = load_test_data()
    print(f"\n📊 Loaded test data: {len(test_data['y'])} samples")
    
    # Get available models
    print("\n🔍 Checking available models...")
    models = get_available_models()
    
    if len(models) < 2:
        print("❌ Need at least 2 working models")
        return
    
    print(f"✅ Found {len(models)} working models:")
    for name in models.keys():
        print(f"  - {name}")
    
    # Run comparisons
    print("\n" + "="*60)
    print("STARTING AUTONOMOUS COMPARISONS")
    print("="*60)
    
    results = []
    improvements = 0
    run_id = 1
    
    # Compare each model against each other
    model_names = list(models.keys())
    baseline_name = 'xgboost' if 'xgboost' in model_names else model_names[0]
    baseline_model = models[baseline_name]
    
    print(f"\n📍 Using '{baseline_name}' as baseline")
    
    for test_name in model_names:
        if test_name == baseline_name:
            continue
        
        result = run_comparison(
            baseline_name, test_name,
            baseline_model, models[test_name],
            test_data, run_id
        )
        
        results.append(result)
        if result['comparison']['is_improvement']:
            improvements += 1
        
        print(f"\n{'='*60}")
        print(f"Progress: {run_id}/{len(model_names)-1} comparisons")
        print(f"Improvements found: {improvements}")
        print(f"{'='*60}")
        
        run_id += 1
        time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"\n📈 Statistics:")
    print(f"  Total comparisons: {len(results)}")
    print(f"  Improvements: {improvements} ({improvements/len(results)*100:.1f}%)")
    print(f"  Significant differences: {sum(1 for r in results if r['comparison']['is_significant'])}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['comparison']['rmse_delta'])
        print(f"\n🏆 Best Result:")
        print(f"  Model: {best['test_model']}")
        print(f"  RMSE: {best['test_metrics']['rmse']:.4f}")
        print(f"  Improvement: {best['comparison']['relative_percent']:+.2f}%")
    
    # Save state
    state = {
        'experiments': results,
        'summary': {
            'total': len(results),
            'improvements': improvements,
            'baseline': baseline_name,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    state_dir = PROJECT_ROOT / "research/06-artifacts/autoresearch"
    state_dir.mkdir(exist_ok=True, parents=True)
    
    with open(state_dir / "research_state.json", 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n💾 Full state saved to: {state_dir / 'research_state.json'}")
    print("\n" + "="*60)
    
    # Generate recommendation
    if improvements > 0:
        print("\n🚀 RECOMMENDATION: Found improvements! Consider updating baseline.")
    else:
        print("\n📝 RECOMMENDATION: No significant improvements found with current models.")
        print("   Try: Feature engineering, hyperparameter tuning, or ensemble methods.")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
