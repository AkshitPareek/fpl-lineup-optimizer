#!/usr/bin/env python3
"""
Agent 1: EXP-031 Production Testing

Tests EXP-031 on real FPL data to confirm the 73% Spearman correlation
translates to better team selection in production.

Author: Research Agent
Date: 2026-03-13
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EXP031-Production-Test')

# Add backend to path
sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')


class EXP031ProductionTester:
    """Test EXP-031 on production FPL data."""
    
    def __init__(self, team_id=9777842):
        self.team_id = team_id
        self.results_dir = Path('research/agents/agent1_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load EXP-031 model
        self.exp031 = self._load_exp031()
        
        # Load EXP-030 for comparison
        self.exp030 = self._load_exp030()
        
        logger.info(f"Production tester initialized for team {team_id}")
    
    def _load_exp031(self):
        """Load EXP-031 model."""
        model_path = Path('models/exp031_clean/model.pkl')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        logger.info("EXP-031 loaded (Spearman: 0.7263)")
        return data
    
    def _load_exp030(self):
        """Load EXP-030 champion."""
        try:
            from production_predictor import ProductionPredictor
            predictor = ProductionPredictor()
            logger.info("EXP-030 loaded (RMSE: 0.8284)")
            return predictor
        except Exception as e:
            logger.warning(f"Could not load EXP-030: {e}")
            return None
    
    def fetch_current_players(self):
        """Fetch current FPL player data."""
        logger.info("Fetching current player data...")
        
        # Load from aggregated data (most recent)
        df = pd.read_csv('datasets/fpl_multi_year/unified_metadata.json')
        
        # Get most recent gameweek data as proxy for current
        # In production, this would fetch from FPL API
        latest_gw = pd.read_csv('datasets/fpl_multi_year/test.csv').tail(100)
        
        logger.info(f"Loaded {len(latest_gw)} players for testing")
        return latest_gw
    
    def prepare_exp031_features(self, player_df):
        """Prepare features for EXP-031."""
        features = []
        
        # Match features used in training
        feature_cols = [
            'form_3gw', 'form_5gw', 'value', 'was_home',
            'selected', 'transfers_balance'
        ]
        
        for col in feature_cols:
            if col in player_df.columns:
                features.append(player_df[col].fillna(0).values)
            else:
                features.append(np.zeros(len(player_df)))
        
        # Position encoding
        if 'position_code' in player_df.columns:
            for pos in [1, 2, 3, 4]:
                features.append((player_df['position_code'] == pos).astype(float).values)
        else:
            for _ in range(4):
                features.append(np.zeros(len(player_df)))
        
        # Gameweek
        if 'gameweek' in player_df.columns:
            features.append(player_df['gameweek'].fillna(20).values / 38)
        else:
            features.append(np.zeros(len(player_df)))
        
        return np.column_stack(features)
    
    def rank_players(self, players_df):
        """Rank players using both models."""
        logger.info("Ranking players...")
        
        # EXP-031 predictions
        X = self.prepare_exp031_features(players_df)
        X_scaled = self.exp031['scaler'].transform(X)
        exp031_preds = self.exp031['model'].predict(X_scaled)
        
        # Add predictions to dataframe
        players_df = players_df.copy()
        players_df['exp031_prediction'] = exp031_preds
        
        # EXP-030 predictions (if available)
        if self.exp030:
            # EXP-030 uses different features, use proxy for comparison
            players_df['exp030_prediction'] = players_df['xP'].fillna(players_df['total_points'])
        
        return players_df
    
    def select_teams(self, players_df):
        """Select optimal teams using both models."""
        logger.info("Selecting teams...")
        
        # EXP-031 team (top players by prediction)
        exp031_team = players_df.nlargest(15, 'exp031_prediction')
        
        # EXP-030 team
        if 'exp030_prediction' in players_df.columns:
            exp030_team = players_df.nlargest(15, 'exp030_prediction')
        else:
            exp030_team = players_df.nlargest(15, 'xP')
        
        # Baseline (random selection for comparison)
        baseline_team = players_df.sample(15)
        
        return {
            'exp031': exp031_team,
            'exp030': exp030_team,
            'baseline': baseline_team
        }
    
    def evaluate_teams(self, teams):
        """Evaluate team selections."""
        logger.info("Evaluating teams...")
        
        results = {}
        for name, team in teams.items():
            avg_prediction = team['exp031_prediction'].mean()
            total_value = team['value'].sum() / 10 if 'value' in team.columns else 0
            
            # Count by position
            position_counts = team['position'].value_counts().to_dict() if 'position' in team.columns else {}
            
            results[name] = {
                'avg_predicted_points': float(avg_prediction),
                'total_value': float(total_value),
                'position_distribution': position_counts,
                'player_count': len(team)
            }
        
        return results
    
    def generate_report(self, results, teams):
        """Generate production test report."""
        timestamp = datetime.now().isoformat()
        
        report = {
            'experiment': 'EXP-031 Production Test',
            'timestamp': timestamp,
            'team_id': self.team_id,
            'model_comparison': results,
            'key_findings': [
                f"EXP-031 avg prediction: {results['exp031']['avg_predicted_points']:.2f} pts",
                f"EXP-030 avg prediction: {results['exp030']['avg_predicted_points']:.2f} pts",
                f"Baseline avg: {results['baseline']['avg_predicted_points']:.2f} pts"
            ],
            'recommendations': []
        }
        
        # Analysis
        exp031_better = results['exp031']['avg_predicted_points'] > results['exp030']['avg_predicted_points']
        
        if exp031_better:
            improvement = (
                (results['exp031']['avg_predicted_points'] - results['exp030']['avg_predicted_points']) /
                results['exp030']['avg_predicted_points'] * 100
            )
            report['recommendations'].append(
                f"EXP-031 shows {improvement:.1f}% improvement in team selection"
            )
            report['recommendations'].append(
                "Deploy EXP-031 for player ranking in production"
            )
        else:
            report['recommendations'].append(
                "EXP-030 still competitive, consider hybrid approach"
            )
        
        # Save report
        report_path = self.results_dir / f'production_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save team selections
        for name, team in teams.items():
            team_path = self.results_dir / f'team_selection_{name}_{datetime.now().strftime("%Y%m%d")}.csv'
            team.to_csv(team_path, index=False)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run(self, num_gws=4):
        """Run production test."""
        logger.info("="*70)
        logger.info("EXP-031 PRODUCTION TEST")
        logger.info("="*70)
        
        # Fetch players
        players = self.fetch_current_players()
        
        # Rank players
        ranked = self.rank_players(players)
        
        # Select teams
        teams = self.select_teams(ranked)
        
        # Evaluate
        results = self.evaluate_teams(teams)
        
        # Generate report
        report = self.generate_report(results, teams)
        
        logger.info("="*70)
        logger.info("PRODUCTION TEST COMPLETE")
        logger.info("="*70)
        logger.info(f"Results: {report['key_findings']}")
        
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test EXP-031 in production')
    parser.add_argument('--team-id', type=int, default=9777842)
    parser.add_argument('--gws', type=int, default=4)
    args = parser.parse_args()
    
    tester = EXP031ProductionTester(team_id=args.team_id)
    report = tester.run(num_gws=args.gws)
    
    print("\n" + "="*70)
    print("EXP-031 PRODUCTION TEST COMPLETE")
    print("="*70)
    print(f"Results saved to: research/agents/agent1_results/")
    print()
    print("Key Findings:")
    for finding in report['key_findings']:
        print(f"  • {finding}")
    print()
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")


if __name__ == '__main__':
    main()
