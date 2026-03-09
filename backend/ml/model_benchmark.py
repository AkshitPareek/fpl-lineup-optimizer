"""
Model Benchmarking Framework for FPL Lineup Optimizer

Provides comprehensive benchmarking capabilities to compare and validate
model performance across multiple dimensions.

Usage:
    from ml.model_benchmark import ModelBenchmark
    
    benchmark = ModelBenchmark()
    results = benchmark.compare_models(
        models=['xgboost', 'lightgbm', 'lstm', 'ensemble'],
        metrics=['rmse', 'mae', 'r2', 'rank_correlation']
    )
    benchmark.generate_report(results, output_path='benchmark_report.html')
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import joblib
import warnings
from collections import defaultdict

# ML metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)
from scipy.stats import pearsonr, spearmanr, kendalltau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BenchmarkResult:
    """Container for a single model's benchmark results."""
    model_name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    inference_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'inference_time_ms': self.inference_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }


@dataclass
class ComparisonResult:
    """Container for model comparison results."""
    results: List[BenchmarkResult]
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[str] = None
    
    def get_leaderboard(self, metric: str = 'rmse') -> pd.DataFrame:
        """Get ranked leaderboard by metric."""
        data = []
        for r in self.results:
            data.append({
                'model': r.model_name,
                metric: r.metrics.get(metric, float('inf')),
                'inference_ms': r.inference_time_ms,
            })
        df = pd.DataFrame(data)
        ascending = metric not in ['r2', 'explained_variance', 'rank_correlation']
        return df.sort_values(metric, ascending=ascending)


class ModelBenchmark:
    """
    Comprehensive model benchmarking framework.
    
    Supports:
    - Standard regression metrics (RMSE, MAE, R²)
    - Rank correlation (important for FPL - we care about relative ordering)
    - Position-wise performance analysis
    - Price bracket analysis
    - Statistical significance testing
    - Inference time benchmarking
    """
    
    def __init__(self, results_dir: str = 'benchmark_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.metrics_history: List[Dict] = []
        
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        position_ids: Optional[np.ndarray] = None,
        price_brackets: Optional[np.ndarray] = None,
        n_bootstrap: int = 100,
    ) -> BenchmarkResult:
        """
        Comprehensive model evaluation with confidence intervals.
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            model_name: Name identifier for the model
            position_ids: Array of position IDs (1=GK, 2=DEF, 3=MID, 4=FWD)
            price_brackets: Array of price brackets for analysis
            n_bootstrap: Number of bootstrap samples for confidence intervals
        """
        # Inference time measurement
        start = time.perf_counter()
        predictions = self._get_predictions(model, X_test)
        inference_time = (time.perf_counter() - start) * 1000  # ms
        
        # Standard metrics
        metrics = self._compute_standard_metrics(y_test, predictions)
        
        # Rank correlation (important for FPL - we need to rank players correctly)
        metrics['spearman_corr'] = spearmanr(y_test, predictions)[0]
        metrics['kendall_tau'] = kendalltau(y_test, predictions)[0]
        
        # Top-k accuracy (how often do we correctly identify top performers?)
        metrics.update(self._compute_top_k_accuracy(y_test, predictions, k=[5, 10, 20]))
        
        # Bootstrap confidence intervals
        metrics.update(self._bootstrap_metrics(y_test, predictions, n_bootstrap))
        
        # Position-wise metrics
        if position_ids is not None:
            pos_metrics = self._compute_position_wise_metrics(
                y_test, predictions, position_ids
            )
            metrics['position_wise'] = pos_metrics
        
        # Price bracket metrics
        if price_brackets is not None:
            price_metrics = self._compute_price_bracket_metrics(
                y_test, predictions, price_brackets
            )
            metrics['price_wise'] = price_metrics
        
        # FPL-specific metrics
        metrics.update(self._compute_fpl_specific_metrics(y_test, predictions))
        
        return BenchmarkResult(
            model_name=model_name,
            metrics=metrics,
            predictions=predictions,
            inference_time_ms=inference_time,
            metadata={
                'n_samples': len(y_test),
                'n_features': X_test.shape[1] if len(X_test.shape) > 1 else 1,
            }
        )
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        position_ids: Optional[np.ndarray] = None,
        price_brackets: Optional[np.ndarray] = None,
        statistical_test: bool = True,
    ) -> ComparisonResult:
        """
        Compare multiple models with statistical significance testing.
        
        Args:
            models: Dict of {model_name: model_object}
            X_test: Test features
            y_test: Test targets
            position_ids: Position identifiers for position-wise analysis
            price_brackets: Price brackets for price-wise analysis
            statistical_test: Whether to run paired t-tests
        """
        results = []
        all_predictions = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            result = self.evaluate_model(
                model, X_test, y_test, name,
                position_ids=position_ids,
                price_brackets=price_brackets,
            )
            results.append(result)
            all_predictions[name] = result.predictions
        
        # Statistical significance tests
        statistical_tests = {}
        if statistical_test and len(results) > 1:
            statistical_tests = self._paired_significance_tests(
                y_test, all_predictions
            )
        
        # Determine winner
        winner = self._determine_winner(results)
        
        comparison = ComparisonResult(
            results=results,
            statistical_tests=statistical_tests,
            winner=winner,
        )
        
        # Store in history
        self._save_comparison(comparison)
        
        return comparison
    
    def ab_test_models(
        self,
        model_a: Any,
        model_b: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_a_name: str = 'Model A',
        model_b_name: str = 'Model B',
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Perform rigorous A/B testing between two models.
        
        Returns detailed statistical comparison including:
        - Effect size (Cohen's d)
        - Confidence intervals for difference
        - p-values for various metrics
        - Recommendation
        """
        pred_a = self._get_predictions(model_a, X_test)
        pred_b = self._get_predictions(model_b, X_test)
        
        # Errors
        error_a = np.abs(y_test - pred_a)
        error_b = np.abs(y_test - pred_b)
        squared_error_a = (y_test - pred_a) ** 2
        squared_error_b = (y_test - pred_b) ** 2
        
        # Paired t-tests
        from scipy.stats import ttest_rel
        
        mae_tstat, mae_pvalue = ttest_rel(error_a, error_b)
        rmse_tstat, rmse_pvalue = ttest_rel(
            np.sqrt(squared_error_a), np.sqrt(squared_error_b)
        )
        
        # Effect size (Cohen's d)
        cohens_d = self._cohens_d(error_a, error_b)
        
        # Win rates
        a_wins = np.sum(error_a < error_b)
        b_wins = np.sum(error_b < error_a)
        ties = len(error_a) - a_wins - b_wins
        
        # Bootstrap confidence interval for difference in MAE
        mae_diff_ci = self._bootstrap_difference_ci(error_a, error_b, confidence_level)
        
        result = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'mae': {
                'model_a': float(np.mean(error_a)),
                'model_b': float(np.mean(error_b)),
                'difference': float(np.mean(error_a) - np.mean(error_b)),
                'ci_lower': mae_diff_ci[0],
                'ci_upper': mae_diff_ci[1],
                'p_value': float(mae_pvalue),
                'significant': mae_pvalue < (1 - confidence_level),
            },
            'rmse': {
                'model_a': float(np.sqrt(np.mean(squared_error_a))),
                'model_b': float(np.sqrt(np.mean(squared_error_b))),
                'p_value': float(rmse_pvalue),
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d),
            },
            'win_rates': {
                model_a_name: a_wins / len(error_a),
                model_b_name: b_wins / len(error_a),
                'ties': ties / len(error_a),
            },
            'recommendation': self._ab_test_recommendation(
                mae_pvalue, cohens_d, np.mean(error_a), np.mean(error_b), model_a_name, model_b_name
            ),
        }
        
        return result
    
    def generate_report(
        self,
        comparison: ComparisonResult,
        output_path: Optional[str] = None,
        include_plots: bool = True,
    ) -> str:
        """Generate comprehensive HTML report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.results_dir / f'benchmark_report_{timestamp}.html'
        
        html = self._generate_html_report(comparison, include_plots)
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return str(output_path)
    
    def track_metrics_over_time(
        self,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None,
    ):
        """Track model metrics over time for monitoring."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'metadata': metadata or {},
        }
        self.metrics_history.append(entry)
        
        # Save to file
        history_path = self.results_dir / 'metrics_history.jsonl'
        with open(history_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def detect_performance_regression(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        metric: str = 'rmse',
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect if current model performance has regressed.
        
        Returns:
            Dict with regression status, severity, and details
        """
        # Load historical metrics for this model
        model_history = [
            h for h in self.metrics_history
            if h['model_name'] == model_name
        ]
        
        if not model_history:
            return {'status': 'no_history', 'message': 'No historical data available'}
        
        # Compare to best historical performance
        best_metric = min(h['metrics'][metric] for h in model_history)
        current_value = current_metrics[metric]
        
        degradation = (current_value - best_metric) / best_metric
        
        if degradation > threshold:
            return {
                'status': 'regression_detected',
                'severity': 'high' if degradation > 0.1 else 'medium',
                'metric': metric,
                'best_historical': best_metric,
                'current': current_value,
                'degradation_pct': degradation * 100,
                'message': f'Performance degraded by {degradation*100:.1f}%',
            }
        
        return {
            'status': 'ok',
            'metric': metric,
            'current': current_value,
            'best_historical': best_metric,
            'improvement_pct': abs(degradation) * 100 if degradation < 0 else 0,
        }
    
    # ============== Private Helper Methods ==============
    
    def _get_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Safely get predictions from various model types."""
        if hasattr(model, 'predict'):
            preds = model.predict(X)
        elif hasattr(model, 'forward_pass'):
            preds = model.forward_pass(X)
        else:
            raise ValueError(f"Model has no predict or forward_pass method")
        
        return np.array(preds).flatten()
    
    def _compute_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics."""
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred + 1e-8)),
            'explained_variance': float(explained_variance_score(y_true, y_pred)),
            'mean_bias': float(np.mean(y_pred - y_true)),
        }
    
    def _compute_top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: List[int] = [5, 10, 20],
    ) -> Dict[str, float]:
        """Compute accuracy of identifying top-k performers."""
        results = {}
        
        for k_val in k:
            # Get top-k by true values
            top_k_true = set(np.argsort(y_true)[-k_val:])
            # Get top-k by predictions
            top_k_pred = set(np.argsort(y_pred)[-k_val:])
            # Intersection
            overlap = len(top_k_true & top_k_pred)
            results[f'top_{k_val}_accuracy'] = overlap / k_val
            results[f'top_{k_val}_overlap'] = overlap
        
        return results
    
    def _bootstrap_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 100,
    ) -> Dict[str, float]:
        """Compute bootstrap confidence intervals for metrics."""
        n = len(y_true)
        rmse_boot = []
        mae_boot = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            y_t = y_true[idx]
            y_p = y_pred[idx]
            rmse_boot.append(np.sqrt(mean_squared_error(y_t, y_p)))
            mae_boot.append(mean_absolute_error(y_t, y_p))
        
        return {
            'rmse_ci_lower': float(np.percentile(rmse_boot, 2.5)),
            'rmse_ci_upper': float(np.percentile(rmse_boot, 97.5)),
            'mae_ci_lower': float(np.percentile(mae_boot, 2.5)),
            'mae_ci_upper': float(np.percentile(mae_boot, 97.5)),
        }
    
    def _compute_position_wise_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        position_ids: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by position."""
        position_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        results = {}
        
        for pos_id, pos_name in position_names.items():
            mask = position_ids == pos_id
            if np.sum(mask) > 0:
                results[pos_name] = {
                    'rmse': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'n_samples': int(np.sum(mask)),
                }
        
        return results
    
    def _compute_price_bracket_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        price_brackets: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by price bracket."""
        unique_brackets = np.unique(price_brackets)
        results = {}
        
        for bracket in unique_brackets:
            mask = price_brackets == bracket
            if np.sum(mask) > 0:
                results[f'price_{bracket}'] = {
                    'rmse': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'n_samples': int(np.sum(mask)),
                }
        
        return results
    
    def _compute_fpl_specific_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute FPL-specific evaluation metrics."""
        # Captain pick accuracy (if we pick top predictor as captain)
        captain_correct = int(np.argmax(y_pred) == np.argmax(y_true))
        
        # Points within 1, 2, 3 of actual
        within_1 = np.mean(np.abs(y_pred - y_true) <= 1)
        within_2 = np.mean(np.abs(y_pred - y_true) <= 2)
        within_3 = np.mean(np.abs(y_pred - y_true) <= 3)
        
        # Over/under estimation rates
        overestimate = np.mean(y_pred > y_true)
        underestimate = np.mean(y_pred < y_true)
        
        return {
            'captain_accuracy': float(captain_correct),
            'within_1_point': float(within_1),
            'within_2_points': float(within_2),
            'within_3_points': float(within_3),
            'overestimate_rate': float(overestimate),
            'underestimate_rate': float(underestimate),
        }
    
    def _paired_significance_tests(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run paired statistical tests between models."""
        from scipy.stats import ttest_rel, wilcoxon
        
        model_names = list(predictions.keys())
        results = {}
        
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                error_a = np.abs(y_true - predictions[model_a])
                error_b = np.abs(y_true - predictions[model_b])
                
                # Paired t-test
                t_stat, p_val = ttest_rel(error_a, error_b)
                
                # Wilcoxon signed-rank test (non-parametric)
                w_stat, w_pval = wilcoxon(error_a, error_b)
                
                results[f'{model_a}_vs_{model_b}'] = {
                    't_test_pvalue': float(p_val),
                    'wilcoxon_pvalue': float(w_pval),
                    'significant_95': p_val < 0.05,
                    'mean_diff': float(np.mean(error_a) - np.mean(error_b)),
                }
        
        return results
    
    def _determine_winner(self, results: List[BenchmarkResult]) -> Optional[str]:
        """Determine the best performing model."""
        if not results:
            return None
        
        # Use composite score: RMSE + (1 - rank_correlation)
        scores = []
        for r in results:
            rmse = r.metrics.get('rmse', float('inf'))
            rank_corr = r.metrics.get('spearman_corr', 0)
            # Lower is better
            composite = rmse + (1 - rank_corr) * 5  # Weight rank correlation
            scores.append((r.model_name, composite))
        
        scores.sort(key=lambda x: x[1])
        return scores[0][0]
    
    def _cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
        if pooled_std == 0:
            return 0
        return (np.mean(x) - np.mean(y)) / pooled_std
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _bootstrap_difference_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for difference in means."""
        n = len(x)
        diffs = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            diff = np.mean(x[idx]) - np.mean(y[idx])
            diffs.append(diff)
        
        alpha = (1 - confidence) / 2
        return (
            float(np.percentile(diffs, alpha * 100)),
            float(np.percentile(diffs, (1 - alpha) * 100)),
        )
    
    def _ab_test_recommendation(
        self,
        p_value: float,
        cohens_d: float,
        mae_a: float,
        mae_b: float,
        name_a: str,
        name_b: str,
    ) -> str:
        """Generate A/B test recommendation."""
        if p_value >= 0.05:
            return f"No statistically significant difference. Keep {name_a} (status quo)."
        
        better = name_a if mae_a < mae_b else name_b
        worse = name_b if mae_a < mae_b else name_a
        
        effect = self._interpret_cohens_d(cohens_d)
        
        return f"Recommend {better} over {worse}. Effect size: {effect} (Cohen's d={abs(cohens_d):.3f})"
    
    def _save_comparison(self, comparison: ComparisonResult):
        """Save comparison results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.results_dir / f'comparison_{timestamp}.json'
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in comparison.results],
            'statistical_tests': comparison.statistical_tests,
            'winner': comparison.winner,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_html_report(
        self,
        comparison: ComparisonResult,
        include_plots: bool,
    ) -> str:
        """Generate HTML report content."""
        # Create leaderboard
        leaderboard = comparison.get_leaderboard('rmse')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FPL Model Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .winner {{ background-color: #d4edda !important; font-weight: bold; }}
                .metric {{ font-family: monospace; }}
                .significant {{ color: green; font-weight: bold; }}
                .not-significant {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>🏆 FPL Model Benchmark Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Leaderboard (by RMSE)</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R²</th>
                    <th>Spearman</th>
                    <th>Inference (ms)</th>
                </tr>
        """
        
        for idx, row in leaderboard.iterrows():
            model_name = row['model']
            is_winner = model_name == comparison.winner
            winner_class = 'winner' if is_winner else ''
            
            # Find full metrics
            full_metrics = next(
                (r.metrics for r in comparison.results if r.model_name == model_name),
                {}
            )
            
            html += f"""
                <tr class="{winner_class}">
                    <td>{idx + 1}</td>
                    <td>{model_name} {'🏆' if is_winner else ''}</td>
                    <td class="metric">{row['rmse']:.4f}</td>
                    <td class="metric">{full_metrics.get('mae', 0):.4f}</td>
                    <td class="metric">{full_metrics.get('r2', 0):.4f}</td>
                    <td class="metric">{full_metrics.get('spearman_corr', 0):.4f}</td>
                    <td class="metric">{row['inference_ms']:.2f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Statistical Significance Tests</h2>
            <table>
                <tr>
                    <th>Comparison</th>
                    <th>Mean MAE Diff</th>
                    <th>t-test p-value</th>
                    <th>Significant (α=0.05)</th>
                </tr>
        """
        
        for comp_name, test_results in comparison.statistical_tests.items():
            sig_class = 'significant' if test_results.get('significant_95') else 'not-significant'
            sig_text = 'Yes ✓' if test_results.get('significant_95') else 'No'
            
            html += f"""
                <tr>
                    <td>{comp_name}</td>
                    <td class="metric">{test_results.get('mean_diff', 0):.4f}</td>
                    <td class="metric">{test_results.get('t_test_pvalue', 1):.4f}</td>
                    <td class="{sig_class}">{sig_text}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>FPL-Specific Metrics</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Top-5 Acc</th>
                    <th>Top-10 Acc</th>
                    <th>Within 1pt</th>
                    <th>Within 2pt</th>
                    <th>Captain Acc</th>
                </tr>
        """
        
        for result in comparison.results:
            m = result.metrics
            html += f"""
                <tr>
                    <td>{result.model_name}</td>
                    <td class="metric">{m.get('top_5_accuracy', 0):.2%}</td>
                    <td class="metric">{m.get('top_10_accuracy', 0):.2%}</td>
                    <td class="metric">{m.get('within_1_point', 0):.2%}</td>
                    <td class="metric">{m.get('within_2_points', 0):.2%}</td>
                    <td class="metric">{m.get('captain_accuracy', 0):.0%}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


# ============== Convenience Functions ==============

def quick_benchmark(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_report: bool = True,
) -> ComparisonResult:
    """Quick benchmark function for convenience."""
    benchmark = ModelBenchmark()
    comparison = benchmark.compare_models(models, X_test, y_test)
    
    if save_report:
        report_path = benchmark.generate_report(comparison)
        print(f"Report saved to: {report_path}")
    
    return comparison


def run_ab_test(
    model_a: Any,
    model_b: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name_a: str = 'Model A',
    name_b: str = 'Model B',
) -> Dict[str, Any]:
    """Quick A/B test function for convenience."""
    benchmark = ModelBenchmark()
    return benchmark.ab_test_models(
        model_a, model_b, X_test, y_test, name_a, name_b
    )


if __name__ == '__main__':
    # Example usage
    print("Model Benchmark Framework")
    print("Import this module to use:")
    print("  from ml.model_benchmark import ModelBenchmark, quick_benchmark")
