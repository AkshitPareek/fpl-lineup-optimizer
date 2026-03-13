#!/usr/bin/env python3
"""
Quick Backtest - Uses existing data to validate champion model.

This backtest uses the training/test data we already have rather than
fetching live FPL data (which has reliability issues).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')

from production_predictor import ProductionPredictor


class QuickBacktest:
    """Simple backtest using existing datasets."""
    
    def __init__(self):
        self.datasets_dir = Path("/home/akshit/fpl-lineup-optimizer/datasets/fpl_points_v1")
        self.predictor = ProductionPredictor()
        
    def run(self):
        """Run backtest simulation."""
        print("="*70)
        print("QUICK BACKTEST - Champion Model EXP-030")
        print("="*70)
        print()
        
        # Load data
        print("📊 Loading datasets...")
        train_X = np.load(self.datasets_dir / "train_X.npy")
        train_y = np.load(self.datasets_dir / "train_y.npy")
        test_X = np.load(self.datasets_dir / "test_X.npy")
        test_y = np.load(self.datasets_dir / "test_y.npy")
        
        print(f"   Training samples: {len(train_X)}")
        print(f"   Test samples: {len(test_y)}")
        print()
        
        # Simulate gameweeks
        results = []
        
        # Use test set as "next 4 gameweeks"
        # We have limited test data (41 samples)
        # Simulate 2 gameweeks with available players
        gws = 2
        players_per_gw = min(15, len(test_y) // gws)  # 15-20 players per GW
        
        print(f"🎮 Simulating {gws} gameweeks...")
        print(f"   Players per GW: ~{players_per_gw}")
        print()
        
        baseline_total = 0
        champion_total = 0
        hindsight_total = 0
        
        for gw in range(1, gws + 1):
            print(f"📅 Gameweek {gw}")
            print("-"*70)
            
            # Get players for this GW
            start_idx = (gw - 1) * players_per_gw
            end_idx = gw * players_per_gw
            
            gw_X = test_X[start_idx:end_idx]
            gw_actual = test_y[start_idx:end_idx]
            
            # Generate predictions
            champion_preds = self.predictor.predict(gw_X)
            
            # Baseline: Use mean (simple heuristic)
            baseline_preds = np.full_like(champion_preds, train_y.mean())
            
            # Simulate lineup selection (top N by predicted points)
            # Use fewer players since we have limited data
            n_lineup = min(8, len(gw_actual))  # 8 players (simplified squad)
            
            # Champion lineup
            champion_indices = np.argsort(champion_preds)[-n_lineup:]
            champion_points = gw_actual[champion_indices].sum()
            champion_captain_idx = champion_indices[np.argmax(champion_preds[champion_indices])]
            champion_points += gw_actual[champion_captain_idx]  # Captain double
            
            # Baseline lineup (mean-based selection)
            np.random.seed(gw)
            n_select = min(n_lineup, len(gw_actual))
            baseline_indices = np.random.choice(len(gw_actual), n_select, replace=False)
            baseline_points = gw_actual[baseline_indices].sum()
            baseline_captain_idx = baseline_indices[np.argmax(baseline_preds[baseline_indices])]
            baseline_points += gw_actual[baseline_captain_idx]
            
            # Hindsight optimal (knowing actual points)
            hindsight_indices = np.argsort(gw_actual)[-n_lineup:]
            hindsight_points = gw_actual[hindsight_indices].sum()
            hindsight_captain_idx = hindsight_indices[np.argmax(gw_actual[hindsight_indices])]
            hindsight_points += gw_actual[hindsight_captain_idx]
            
            # Accumulate
            baseline_total += baseline_points
            champion_total += champion_points
            hindsight_total += hindsight_points
            
            # Print GW results
            print(f"   Champion Points:  {champion_points:6.1f} (Captain: +{gw_actual[champion_captain_idx]:.1f})")
            print(f"   Baseline Points:  {baseline_points:6.1f}")
            print(f"   Hindsight Opt:    {hindsight_points:6.1f}")
            print(f"   vs Baseline:      {champion_points - baseline_points:+6.1f}")
            print()
            
            results.append({
                'gw': gw,
                'champion': champion_points,
                'baseline': baseline_points,
                'hindsight': hindsight_points,
                'improvement': champion_points - baseline_points
            })
        
        # Summary
        print("="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        print()
        
        total_improvement = champion_total - baseline_total
        improvement_pct = (total_improvement / baseline_total) * 100 if baseline_total > 0 else 0
        
        print(f"Total Points:")
        print(f"   Champion Model:   {champion_total:6.1f}")
        print(f"   Baseline:         {baseline_total:6.1f}")
        print(f"   Hindsight Opt:    {hindsight_total:6.1f}")
        print()
        print(f"Improvement:")
        print(f"   Absolute:         {total_improvement:+6.1f} points")
        print(f"   Percentage:       {improvement_pct:+.1f}%")
        print(f"   Per GW Average:   {total_improvement/gws:+.1f} points")
        print()
        
        # Compare to hindsight
        champion_vs_optimal = (champion_total / hindsight_total) * 100 if hindsight_total > 0 else 0
        print(f"Champion vs Hindsight Optimal: {champion_vs_optimal:.1f}%")
        print()
        
        # GW by GW breakdown
        print("Gameweek Breakdown:")
        print(f"{'GW':<5} {'Champion':<10} {'Baseline':<10} {'Diff':<8} {'% of Opt':<10}")
        print("-"*50)
        for r in results:
            pct_opt = (r['champion'] / r['hindsight'] * 100) if r['hindsight'] > 0 else 0
            print(f"{r['gw']:<5} {r['champion']:<10.1f} {r['baseline']:<10.1f} {r['improvement']:+7.1f} {pct_opt:<9.1f}%")
        
        print()
        print("="*70)
        
        if improvement_pct > 0:
            print("✅ CHAMPION MODEL OUTPERFORMS BASELINE")
        else:
            print("⚠️  CHAMPION MODEL UNDERPERFORMS (needs investigation)")
        
        print("="*70)
        
        return {
            'champion_total': champion_total,
            'baseline_total': baseline_total,
            'hindsight_total': hindsight_total,
            'improvement': total_improvement,
            'improvement_pct': improvement_pct,
            'results': results
        }


def main():
    backtest = QuickBacktest()
    results = backtest.run()
    
    # Save results
    output_dir = Path("/home/akshit/fpl-lineup-optimizer/research/backtests")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return 0 if results['improvement_pct'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
