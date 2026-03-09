#!/usr/bin/env python3
"""
Improvement Tracking & Visualization System

Tracks model improvement trajectory and provides insights on:
- Are we actually improving?
- Which strategies work best?
- When should we stop?
- What's the improvement rate?

Usage:
    python improvement_tracker.py visualize
    python improvement_tracker.py analyze
    python improvement_tracker.py dashboard

Author: AutoFPL Research System
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


@dataclass
class ImprovementMetrics:
    """Metrics for tracking improvement."""
    
    # Overall progress
    total_experiments: int
    successful_improvements: int
    failed_experiments: int
    improvement_rate: float
    
    # Trajectory
    initial_rmse: float
    current_rmse: float
    total_improvement: float
    improvement_per_run: float
    
    # Statistical
    avg_effect_size: float
    avg_p_value: float
    significant_rate: float
    
    # Strategy performance
    best_strategy: str
    strategy_success_rates: Dict[str, float]
    
    # Trends
    recent_trend: str  # 'improving', 'plateau', 'degrading'
    trend_confidence: float
    
    # Recommendations
    should_continue: bool
    reason: str


class ImprovementTracker:
    """Tracks and analyzes model improvement over time."""
    
    def __init__(self, state_file: str = "research/06-artifacts/autoresearch/state.json"):
        self.state_file = state_file
        self.data = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load experiment data."""
        if not os.path.exists(self.state_file):
            print(f"No state file found at {self.state_file}")
            return pd.DataFrame()
        
        with open(self.state_file) as f:
            state = json.load(f)
        
        experiments = state.get('experiments', [])
        
        if not experiments:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for exp in experiments:
            data.append({
                'run_id': exp['run_id'],
                'timestamp': exp['timestamp'],
                'strategy': exp['strategy'],
                'status': exp['status'],
                'is_improvement': exp.get('is_improvement', False),
                'is_significant': exp.get('is_significant', False),
                'rmse_before': exp.get('metrics_before', {}).get('rmse', np.nan),
                'rmse_after': exp.get('metrics_after', {}).get('rmse', np.nan),
                'mae_after': exp.get('metrics_after', {}).get('mae', np.nan),
                'spearman_after': exp.get('metrics_after', {}).get('spearman_corr', np.nan),
                'p_value': exp.get('p_value', np.nan),
                'effect_size': exp.get('effect_size', np.nan),
                'duration': exp.get('duration_seconds', 0) / 60,  # minutes
            })
        
        df = pd.DataFrame(data)
        
        # Calculate improvement metrics
        if not df.empty and 'rmse_after' in df.columns:
            df['rmse_change'] = df['rmse_after'] - df['rmse_before']
            df['relative_improvement'] = -df['rmse_change'] / df['rmse_before'] * 100
        
        return df
    
    def calculate_metrics(self) -> ImprovementMetrics:
        """Calculate comprehensive improvement metrics."""
        
        if self.data.empty:
            return ImprovementMetrics(
                total_experiments=0,
                successful_improvements=0,
                failed_experiments=0,
                improvement_rate=0.0,
                initial_rmse=0.0,
                current_rmse=0.0,
                total_improvement=0.0,
                improvement_per_run=0.0,
                avg_effect_size=0.0,
                avg_p_value=1.0,
                significant_rate=0.0,
                best_strategy='',
                strategy_success_rates={},
                recent_trend='unknown',
                trend_confidence=0.0,
                should_continue=True,
                reason="No data yet"
            )
        
        df = self.data
        
        # Basic counts
        total = len(df)
        improvements = df['is_improvement'].sum()
        failures = (df['status'] == 'failed').sum()
        improvement_rate = improvements / total if total > 0 else 0
        
        # RMSE trajectory
        initial_rmse = df.iloc[0]['rmse_before'] if not df.empty else 0
        current_rmse = df[df['is_improvement']]['rmse_after'].min() if improvements > 0 else initial_rmse
        total_improvement = (initial_rmse - current_rmse) / initial_rmse * 100 if initial_rmse > 0 else 0
        improvement_per_run = total_improvement / total if total > 0 else 0
        
        # Statistical metrics
        sig_exps = df[df['is_significant'] == True]
        avg_effect_size = sig_exps['effect_size'].mean() if not sig_exps.empty else 0
        avg_p_value = df['p_value'].mean()
        significant_rate = len(sig_exps) / total if total > 0 else 0
        
        # Strategy analysis
        strategy_success = {}
        for strategy in df['strategy'].unique():
            strat_df = df[df['strategy'] == strategy]
            success_rate = strat_df['is_improvement'].mean()
            strategy_success[strategy] = success_rate
        
        best_strategy = max(strategy_success.items(), key=lambda x: x[1])[0] if strategy_success else ''
        
        # Trend analysis (last 5 runs)
        recent_trend = 'unknown'
        trend_confidence = 0.0
        
        if len(df) >= 5:
            recent = df.tail(5)
            recent_improvements = recent['is_improvement'].sum()
            
            if recent_improvements >= 3:
                recent_trend = 'improving'
                trend_confidence = recent_improvements / 5
            elif recent_improvements == 0:
                recent_trend = 'plateau'
                trend_confidence = 1.0
            elif recent['rmse_after'].is_monotonic_decreasing:
                recent_trend = 'improving'
                trend_confidence = 0.6
            else:
                recent_trend = 'uncertain'
                trend_confidence = 0.5
        
        # Recommendation
        should_continue, reason = self._should_continue(df, improvement_rate, recent_trend)
        
        return ImprovementMetrics(
            total_experiments=total,
            successful_improvements=int(improvements),
            failed_experiments=int(failures),
            improvement_rate=improvement_rate,
            initial_rmse=initial_rmse,
            current_rmse=current_rmse,
            total_improvement=total_improvement,
            improvement_per_run=improvement_per_run,
            avg_effect_size=avg_effect_size,
            avg_p_value=avg_p_value,
            significant_rate=significant_rate,
            best_strategy=best_strategy,
            strategy_success_rates=strategy_success,
            recent_trend=recent_trend,
            trend_confidence=trend_confidence,
            should_continue=should_continue,
            reason=reason
        )
    
    def _should_continue(self, df: pd.DataFrame, improvement_rate: float, trend: str) -> Tuple[bool, str]:
        """Determine if research should continue."""
        
        n = len(df)
        
        # Minimum runs
        if n < 3:
            return True, f"Minimum runs not reached ({n}/3)"
        
        # Check for convergence
        if n >= 5:
            recent_rmse = df.tail(5)['rmse_after'].dropna()
            if len(recent_rmse) >= 3:
                cv = recent_rmse.std() / recent_rmse.mean()
                if cv < 0.005:  # Very low variance = converged
                    return False, f"Converged (CV={cv:.4f})"
        
        # Check for sustained failure
        recent_failures = (df.tail(3)['is_improvement'] == False).sum()
        if n >= 5 and recent_failures == 3:
            return False, "3 consecutive failures - strategy exhaustion likely"
        
        # Check improvement rate
        if n >= 10 and improvement_rate < 0.1:
            return False, f"Low improvement rate ({improvement_rate:.1%}) after {n} runs"
        
        return True, f"Continue (trend: {trend})"
    
    def visualize(self, output_dir: str = "research/06-artifacts/autoresearch/plots"):
        """Create comprehensive visualizations."""
        
        if self.data.empty:
            print("No data to visualize")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = self.calculate_metrics()
        df = self.data
        
        # Figure 1: RMSE Trajectory
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: RMSE over time
        ax = axes[0, 0]
        ax.plot(df['run_id'], df['rmse_after'], 'o-', label='RMSE', color='steelblue')
        
        # Highlight improvements
        improvements = df[df['is_improvement'] == True]
        if not improvements.empty:
            ax.scatter(improvements['run_id'], improvements['rmse_after'], 
                      color='green', s=100, marker='*', label='Improvement', zorder=5)
        
        # Show baseline
        if not df.empty and 'rmse_before' in df.columns:
            baseline = df.iloc[0]['rmse_before']
            ax.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Experiment Run')
        ax.set_ylabel('RMSE (lower is better)')
        ax.set_title('Model Performance Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Improvement Rate by Strategy
        ax = axes[0, 1]
        if metrics.strategy_success_rates:
            strategies = list(metrics.strategy_success_rates.keys())
            rates = [metrics.strategy_success_rates[s] * 100 for s in strategies]
            
            colors = ['green' if r > 50 else 'orange' if r > 20 else 'red' for r in rates]
            ax.barh(strategies, rates, color=colors, alpha=0.7)
            ax.set_xlabel('Success Rate (%)')
            ax.set_title('Strategy Success Rates')
            ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
            ax.legend()
        
        # Plot 3: Effect Size Distribution
        ax = axes[1, 0]
        sig_exps = df[df['is_significant'] == True]['effect_size'].dropna()
        if not sig_exps.empty:
            ax.hist(sig_exps, bins=10, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(x=metrics.avg_effect_size, color='red', linestyle='--', 
                      label=f'Mean: {metrics.avg_effect_size:.3f}')
            ax.set_xlabel("Effect Size (Cohen's d)")
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Effect Sizes')
            ax.legend()
        
        # Plot 4: Cumulative Improvement
        ax = axes[1, 1]
        df_sorted = df.sort_values('run_id')
        cumulative_best = df_sorted['rmse_after'].expanding().min()
        
        ax.plot(df_sorted['run_id'], cumulative_best, 'o-', color='green', linewidth=2)
        ax.fill_between(df_sorted['run_id'], cumulative_best, alpha=0.3, color='green')
        
        # Show improvements as steps
        for i, (run_id, is_imp) in enumerate(zip(df_sorted['run_id'], df_sorted['is_improvement'])):
            if is_imp:
                ax.axvline(x=run_id, color='green', alpha=0.3, linestyle='--')
        
        ax.set_xlabel('Experiment Run')
        ax.set_ylabel('Best RMSE (cumulative)')
        ax.set_title('Cumulative Best Performance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'improvement_trajectory.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to: {plot_path}")
        plt.close()
        
        # Figure 2: Detailed Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: P-value distribution
        ax = axes[0, 0]
        p_values = df['p_value'].dropna()
        if not p_values.empty:
            ax.hist(p_values, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05 threshold')
            ax.set_xlabel('p-value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of P-values')
            ax.legend()
        
        # Plot 2: Relative Improvement Distribution
        ax = axes[0, 1]
        rel_imp = df['relative_improvement'].dropna()
        if not rel_imp.empty:
            ax.hist(rel_imp, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', label='No change')
            ax.axvline(x=rel_imp.mean(), color='blue', linestyle='--', 
                      label=f'Mean: {rel_imp.mean():.2f}%')
            ax.set_xlabel('Relative Improvement (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Improvements')
            ax.legend()
        
        # Plot 3: Duration vs Improvement
        ax = axes[1, 0]
        if 'duration' in df.columns and not df['duration'].isna().all():
            scatter = ax.scatter(df['duration'], df['relative_improvement'], 
                               c=df['is_improvement'], cmap='RdYlGn', 
                               s=100, alpha=0.6, edgecolors='black')
            ax.set_xlabel('Experiment Duration (minutes)')
            ax.set_ylabel('Relative Improvement (%)')
            ax.set_title('Duration vs Improvement')
            plt.colorbar(scatter, ax=ax, label='Is Improvement')
        
        # Plot 4: Rolling Success Rate
        ax = axes[1, 1]
        if len(df) >= 3:
            window = min(5, len(df))
            rolling_success = df['is_improvement'].rolling(window=window).mean() * 100
            ax.plot(df['run_id'], rolling_success, 'o-', color='purple', linewidth=2)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% target')
            ax.set_xlabel('Experiment Run')
            ax.set_ylabel(f'Success Rate (% in last {window} runs)')
            ax.set_title('Rolling Success Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'detailed_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved detailed analysis to: {plot_path}")
        plt.close()
    
    def analyze(self):
        """Print comprehensive analysis."""
        
        metrics = self.calculate_metrics()
        
        print("\n" + "="*70)
        print("IMPROVEMENT ANALYSIS REPORT")
        print("="*70)
        
        print("\n📊 OVERALL PROGRESS")
        print(f"  Total Experiments: {metrics.total_experiments}")
        print(f"  Successful Improvements: {metrics.successful_improvements}")
        print(f"  Failed Experiments: {metrics.failed_experiments}")
        print(f"  Improvement Rate: {metrics.improvement_rate:.1%}")
        
        print("\n📈 PERFORMANCE TRAJECTORY")
        print(f"  Initial RMSE: {metrics.initial_rmse:.4f}")
        print(f"  Current Best RMSE: {metrics.current_rmse:.4f}")
        print(f"  Total Improvement: {metrics.total_improvement:.2f}%")
        print(f"  Improvement per Run: {metrics.improvement_per_run:.3f}%")
        
        print("\n📉 STATISTICAL METRICS")
        print(f"  Average Effect Size: {metrics.avg_effect_size:.3f}")
        print(f"  Average P-value: {metrics.avg_p_value:.4f}")
        print(f"  Significance Rate: {metrics.significant_rate:.1%}")
        
        print("\n🎯 STRATEGY PERFORMANCE")
        print(f"  Best Strategy: {metrics.best_strategy}")
        print("  Success Rates by Strategy:")
        for strategy, rate in sorted(metrics.strategy_success_rates.items(), 
                                     key=lambda x: x[1], reverse=True):
            status = "✅" if rate > 0.5 else "⚠️" if rate > 0.2 else "❌"
            print(f"    {status} {strategy}: {rate:.1%}")
        
        print("\n📊 RECENT TREND")
        trend_icon = {"improving": "📈", "plateau": "➡️", "degrading": "📉", "unknown": "❓"}
        print(f"  Trend: {trend_icon.get(metrics.recent_trend, '❓')} {metrics.recent_trend}")
        print(f"  Confidence: {metrics.trend_confidence:.1%}")
        
        print("\n🤖 RECOMMENDATION")
        if metrics.should_continue:
            print(f"  ✅ CONTINUE: {metrics.reason}")
        else:
            print(f"  ⏹️  STOP: {metrics.reason}")
        
        print("\n" + "="*70)
    
    def dashboard(self):
        """Generate interactive-style dashboard output."""
        
        metrics = self.calculate_metrics()
        
        # Clear screen effect
        print("\n" * 2)
        
        # Header
        print("╔" + "═"*68 + "╗")
        print("║" + " AUTOFPL RESEARCH DASHBOARD ".center(68) + "║")
        print("╠" + "═"*68 + "╣")
        
        # Key Metrics
        print("║  📊 KEY METRICS" + " "*52 + "║")
        print(f"║     Runs: {metrics.total_experiments:3d}    Improvements: {metrics.successful_improvements:3d}    Rate: {metrics.improvement_rate:5.1%}     ║")
        print("╠" + "═"*68 + "╣")
        
        # Performance
        print("║  📈 PERFORMANCE" + " "*52 + "║")
        print(f"║     Initial RMSE:  {metrics.initial_rmse:.4f}                                ║")
        print(f"║     Current RMSE:  {metrics.current_rmse:.4f}    ({metrics.total_improvement:+.2f}%)               ║")
        print("╠" + "═"*68 + "╣")
        
        # Status
        print("║  📊 STATUS" + " "*57 + "║")
        trend_symbols = {"improving": "📈", "plateau": "➡️", "degrading": "📉", "unknown": "❓"}
        trend_sym = trend_symbols.get(metrics.recent_trend, "❓")
        print(f"║     Trend: {trend_sym} {metrics.recent_trend:12s}  Confidence: {metrics.trend_confidence:5.1%}              ║")
        print("╠" + "═"*68 + "╣")
        
        # Recommendation
        print("║  🤖 RECOMMENDATION" + " "*49 + "║")
        if metrics.should_continue:
            rec = "CONTINUE: " + metrics.reason[:48]
        else:
            rec = "STOP: " + metrics.reason[:52]
        print(f"║     {rec:64s} ║")
        print("╚" + "═"*68 + "╝")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Improvement Tracking & Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Print detailed analysis')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--output', default='research/06-artifacts/autoresearch/plots',
                           help='Output directory for plots')
    
    # Dashboard command
    subparsers.add_parser('dashboard', help='Show dashboard view')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tracker = ImprovementTracker()
    
    if args.command == 'analyze':
        tracker.analyze()
    
    elif args.command == 'visualize':
        tracker.visualize(args.output)
    
    elif args.command == 'dashboard':
        tracker.dashboard()


if __name__ == '__main__':
    main()
