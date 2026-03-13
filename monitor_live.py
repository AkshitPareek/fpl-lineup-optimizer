#!/usr/bin/env python3
"""
Monitor live FPL performance for team 9777842.

Run this before GW30 deadline to save predictions,
then after GW30 to compare with actual results.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')
from run_team_prediction_v2 import FPLTeamAnalyzer


def save_predictions(team_id=9777842, gw=30):
    """Save predictions before gameweek."""
    print("="*70)
    print(f"SAVING GW{gw} PREDICTIONS")
    print("="*70)
    
    analyzer = FPLTeamAnalyzer(team_id)
    analyzer.fetch_static_data()
    analyzer.fetch_team_data()
    
    squad = analyzer.get_squad_details()
    
    # Save key predictions
    predictions = {
        'timestamp': datetime.now().isoformat(),
        'gameweek': gw,
        'team_id': team_id,
        'players': []
    }
    
    for p in sorted(squad, key=lambda x: x['predicted_points'], reverse=True):
        predictions['players'].append({
            'id': p['id'],
            'name': p['name'],
            'position': p['position'],
            'predicted_points': round(p['predicted_points'], 2),
            'base_prediction': round(p.get('base_prediction', p['predicted_points']), 2),
            'availability': p['availability_status'],
            'is_captain': p['is_captain'],
            'is_vice_captain': p['is_vice_captain']
        })
    
    # Save to file
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f'gw{gw}_predictions_{datetime.now().strftime("%Y%m%d")}.json'
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✅ Predictions saved to: {filename}")
    
    # Show summary
    print(f"\nTop 5 Predicted Scorers:")
    for i, p in enumerate(predictions['players'][:5], 1):
        captain_marker = " (C)" if p['is_captain'] else " (V)" if p['is_vice_captain'] else ""
        print(f"{i}. {p['name']}: {p['predicted_points']} pts{captain_marker}")
    
    return predictions


def compare_with_actual(predictions_file, actual_points):
    """Compare predictions with actual GW results."""
    print("="*70)
    print("COMPARING PREDICTIONS VS ACTUAL")
    print("="*70)
    
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    print(f"\nGameweek: {predictions['gameweek']}")
    print(f"Predicted on: {predictions['timestamp']}")
    print()
    
    total_pred = 0
    total_actual = 0
    errors = []
    
    print(f"{'Player':<25} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
    print("-"*60)
    
    for p in predictions['players']:
        name = p['name']
        pred = p['predicted_points']
        actual = actual_points.get(name, 0)
        error = pred - actual
        
        total_pred += pred
        total_actual += actual
        errors.append(abs(error))
        
        print(f"{name:<25} {pred:<12.1f} {actual:<12.1f} {error:<+10.1f}")
    
    # Summary stats
    mae = sum(errors) / len(errors) if errors else 0
    rmse = (sum(e**2 for e in errors) / len(errors))**0.5 if errors else 0
    
    print()
    print(f"Total Predicted: {total_pred:.1f}")
    print(f"Total Actual:    {total_actual:.1f}")
    print(f"Difference:      {total_pred - total_actual:+.1f}")
    print(f"MAE:             {mae:.2f}")
    print(f"RMSE:            {rmse:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['predict', 'compare'], default='predict')
    parser.add_argument('--team-id', type=int, default=9777842)
    parser.add_argument('--gw', type=int, default=30)
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        save_predictions(args.team_id, args.gw)
    else:
        # Example usage for compare mode
        print("To compare, run:")
        print(f"python monitor_live.py --mode=compare --gw={args.gw}")
        print("\nThen enter actual points when prompted.")
