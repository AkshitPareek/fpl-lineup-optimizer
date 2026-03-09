#!/usr/bin/env python3
"""
A/B Testing CLI for FPL Model Comparison

Provides command-line interface for comparing two model versions
with statistical rigor.

Usage:
    # Compare two models
    python ab_testing_cli.py compare --model-a xgboost --model-b lightgbm
    
    # Compare with specific test data
    python ab_testing_cli.py compare --model-a xgboost --model-b ensemble --test-data datasets/fpl_points_v1
    
    # Run full benchmark suite
    python ab_testing_cli.py benchmark --models xgboost lightgbm ensemble
    
    # Establish baseline
    python ab_testing_cli.py establish-baseline --output baseline_metrics.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.model_benchmark import ModelBenchmark, quick_benchmark, run_ab_test


def load_model(model_name: str):
    """Load a model by name."""
    import joblib
    
    # Map common names to paths
    model_paths = {
        'xgboost': 'models/xgboost/model.pkl',
        'lightgbm': 'models/lightgbm/model.pkl',
        'random_forest': 'models/random_forest/model.pkl',
        'gradient_boosting': 'models/gradient_boosting/model.pkl',
        'ridge': 'models/ridge/model.pkl',
        'ensemble': 'models/ensemble/ensemble.pkl',
    }
    
    path = model_paths.get(model_name, f'models/{model_name}/model.pkl')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    if model_name == 'ensemble':
        from ml.ensemble import EnsemblePredictor
        return EnsemblePredictor.load(path)
    
    return joblib.load(path)


def load_data(data_dir: str = 'datasets/fpl_points_v1'):
    """Load test data."""
    X_test = np.load(os.path.join(data_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_y.npy'))
    return X_test, y_test


def format_metric(value: float, metric: str) -> str:
    """Format metric value with appropriate precision."""
    if metric in ['rmse', 'mae']:
        return f"{value:.4f}"
    elif metric in ['r2', 'spearman_corr', 'kendall_tau']:
        return f"{value:.4f}"
    elif 'accuracy' in metric or 'rate' in metric:
        return f"{value:.2%}"
    else:
        return f"{value:.4f}"


def print_comparison_table(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    models = list(results.keys())
    
    if len(models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    # Get all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    # Priority metrics first
    priority = ['rmse', 'mae', 'r2', 'spearman_corr', 'within_2_points', 'top_10_accuracy']
    metrics = [m for m in priority if m in all_metrics]
    metrics += [m for m in sorted(all_metrics) if m not in priority]
    
    # Print header
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Print table
    col_width = 20
    print(f"{'Metric':<{col_width}}", end="")
    for model in models:
        print(f"{model:>{col_width}}", end="")
    print()
    print("-" * (col_width * (len(models) + 1)))
    
    for metric in metrics:
        if not any(metric in results[m] for m in models):
            continue
            
        print(f"{metric:<{col_width}}", end="")
        
        values = []
        for model in models:
            val = results[model].get(metric, float('nan'))
            values.append(val)
            print(f"{format_metric(val, metric):>{col_width}}", end="")
        print()
        
        # Mark best for this metric
        if metric in ['rmse', 'mae']:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
    
    print("="*80)


def cmd_compare(args):
    """Compare two models command."""
    print(f"\n🔬 A/B Testing: {args.model_a} vs {args.model_b}")
    print("-" * 50)
    
    # Load models
    print(f"Loading {args.model_a}...")
    model_a = load_model(args.model_a)
    
    print(f"Loading {args.model_b}...")
    model_b = load_model(args.model_b)
    
    # Load data
    print(f"Loading test data from {args.test_data}...")
    X_test, y_test = load_data(args.test_data)
    
    # Run A/B test
    print("\nRunning statistical comparison...")
    result = run_ab_test(
        model_a, model_b, X_test, y_test,
        name_a=args.model_a,
        name_b=args.model_b,
    )
    
    # Print results
    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)
    
    mae = result['mae']
    print(f"\n📊 Mean Absolute Error (MAE):")
    print(f"  {args.model_a}: {mae['model_a']:.4f}")
    print(f"  {args.model_b}: {mae['model_b']:.4f}")
    print(f"  Difference: {mae['difference']:+.4f}")
    print(f"  95% CI: [{mae['ci_lower']:+.4f}, {mae['ci_upper']:+.4f}]")
    
    rmse = result['rmse']
    print(f"\n📊 Root Mean Squared Error (RMSE):")
    print(f"  {args.model_a}: {rmse['model_a']:.4f}")
    print(f"  {args.model_b}: {rmse['model_b']:.4f}")
    
    effect = result['effect_size']
    print(f"\n📈 Effect Size (Cohen's d): {effect['cohens_d']:.4f}")
    print(f"  Interpretation: {effect['interpretation']}")
    
    win_rates = result['win_rates']
    print(f"\n🏆 Win Rates (lower error = win):")
    print(f"  {args.model_a}: {win_rates[args.model_a]:.1%}")
    print(f"  {args.model_b}: {win_rates[args.model_b]:.1%}")
    print(f"  Ties: {win_rates['ties']:.1%}")
    
    print(f"\n📝 Statistical Significance:")
    sig = "✅ Significant" if mae['significant'] else "❌ Not Significant"
    print(f"  p-value: {mae['p_value']:.4f} ({sig})")
    
    print(f"\n💡 Recommendation:")
    print(f"  {result['recommendation']}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    return result


def cmd_benchmark(args):
    """Run full benchmark command."""
    print(f"\n🏁 Running Full Benchmark Suite")
    print("-" * 50)
    
    # Load models
    models = {}
    for model_name in args.models:
        print(f"Loading {model_name}...")
        try:
            models[model_name] = load_model(model_name)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping: {e}")
    
    if len(models) < 1:
        print("❌ No models loaded. Exiting.")
        return
    
    # Load data
    print(f"\nLoading test data from {args.test_data}...")
    X_test, y_test = load_data(args.test_data)
    
    # Load position and price data if available
    position_ids = None
    price_brackets = None
    
    metadata_path = os.path.join(args.test_data, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            # Try to extract position info if available
        except:
            pass
    
    # Run benchmark
    print(f"\nEvaluating {len(models)} models...")
    benchmark = ModelBenchmark(results_dir=args.results_dir)
    comparison = benchmark.compare_models(
        models, X_test, y_test,
        position_ids=position_ids,
        price_brackets=price_brackets,
        statistical_test=True,
    )
    
    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n🏆 Winner: {comparison.winner}")
    
    # Leaderboard
    print("\n📊 Leaderboard (by RMSE):")
    print(comparison.get_leaderboard('rmse').to_string(index=False))
    
    # Statistical tests
    if comparison.statistical_tests:
        print("\n📈 Statistical Significance (p-values):")
        for comp, stats in comparison.statistical_tests.items():
            sig = "✅" if stats.get('significant_95') else "❌"
            print(f"  {comp}: p={stats['t_test_pvalue']:.4f} {sig}")
    
    # Generate report
    if args.report:
        report_path = benchmark.generate_report(comparison)
        print(f"\n📄 HTML report saved to: {report_path}")
    
    return comparison


def cmd_establish_baseline(args):
    """Establish baseline metrics command."""
    print(f"\n📏 Establishing Baseline Metrics")
    print("-" * 50)
    
    # Load all available models
    models = {}
    for model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'ridge', 'ensemble']:
        try:
            print(f"Loading {model_name}...")
            models[model_name] = load_model(model_name)
        except FileNotFoundError:
            print(f"  ⚠️  {model_name} not found, skipping")
    
    # Load data
    print(f"\nLoading test data...")
    X_test, y_test = load_data(args.test_data)
    
    # Compute metrics
    baseline = {}
    for name, model in models.items():
        print(f"Computing metrics for {name}...")
        
        if hasattr(model, 'predict'):
            preds = model.predict(X_test)
        elif hasattr(model, 'forward_pass'):
            preds = model.forward_pass(X_test)
        else:
            continue
        
        baseline[name] = {
            'rmse': float(np.sqrt(np.mean((y_test - preds) ** 2))),
            'mae': float(np.mean(np.abs(y_test - preds))),
            'r2': float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)),
        }
    
    # Save baseline
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\n💾 Baseline metrics saved to {args.output}")
    print("\nBaseline Summary:")
    for name, metrics in baseline.items():
        print(f"  {name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
    
    return baseline


def cmd_track_metrics(args):
    """Track metrics over time."""
    print(f"\n📈 Tracking Metrics for {args.model}")
    print("-" * 50)
    
    # Load model
    model = load_model(args.model)
    
    # Load data
    X_test, y_test = load_data(args.test_data)
    
    # Compute metrics
    if hasattr(model, 'predict'):
        preds = model.predict(X_test)
    else:
        preds = model.forward_pass(X_test)
    
    metrics = {
        'rmse': float(np.sqrt(np.mean((y_test - preds) ** 2))),
        'mae': float(np.mean(np.abs(y_test - preds))),
        'r2': float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)),
    }
    
    # Track
    benchmark = ModelBenchmark()
    benchmark.track_metrics_over_time(
        args.model,
        metrics,
        metadata={'version': args.version, 'note': args.note},
    )
    
    print(f"✅ Metrics tracked for {args.model}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='A/B Testing CLI for FPL Model Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific models
  python ab_testing_cli.py compare --model-a xgboost --model-b lightgbm
  
  # Run full benchmark on all models
  python ab_testing_cli.py benchmark --models xgboost lightgbm ensemble --report
  
  # Establish baseline for regression testing
  python ab_testing_cli.py establish-baseline
  
  # Track metrics for a model version
  python ab_testing_cli.py track --model xgboost --version v2.1 --note "Added new features"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models with A/B testing')
    compare_parser.add_argument('--model-a', required=True, help='First model name')
    compare_parser.add_argument('--model-b', required=True, help='Second model name')
    compare_parser.add_argument('--test-data', default='datasets/fpl_points_v1', help='Test data directory')
    compare_parser.add_argument('--output', '-o', help='Output JSON file for results')
    compare_parser.set_defaults(func=cmd_compare)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run full benchmark suite')
    benchmark_parser.add_argument('--models', nargs='+', required=True, help='Model names to benchmark')
    benchmark_parser.add_argument('--test-data', default='datasets/fpl_points_v1', help='Test data directory')
    benchmark_parser.add_argument('--results-dir', default='benchmark_results', help='Results directory')
    benchmark_parser.add_argument('--report', action='store_true', help='Generate HTML report')
    benchmark_parser.set_defaults(func=cmd_benchmark)
    
    # Establish baseline command
    baseline_parser = subparsers.add_parser('establish-baseline', help='Establish baseline metrics')
    baseline_parser.add_argument('--test-data', default='datasets/fpl_points_v1', help='Test data directory')
    baseline_parser.add_argument('--output', default='benchmark_results/baseline_metrics.json', help='Output file')
    baseline_parser.set_defaults(func=cmd_establish_baseline)
    
    # Track metrics command
    track_parser = subparsers.add_parser('track', help='Track metrics over time')
    track_parser.add_argument('--model', required=True, help='Model name')
    track_parser.add_argument('--version', help='Model version')
    track_parser.add_argument('--note', help='Optional note')
    track_parser.add_argument('--test-data', default='datasets/fpl_points_v1', help='Test data directory')
    track_parser.set_defaults(func=cmd_track_metrics)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
