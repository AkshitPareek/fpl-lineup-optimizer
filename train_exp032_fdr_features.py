#!/usr/bin/env python3
"""
EXP-032: Training with Real FDR (Fixture Difficulty Rating) Features

This script:
1. Fetches real FDR data from FPL API
2. Calculates opponent strength features
3. Adds team attack/defense ratings
4. Trains model with enhanced features
5. Evaluates against EXP-031 v2 champion

Target: Beat Spearman 0.7630
"""

import sys
import json
import pickle
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EXP-032-FDR')


class FDRFeatureEngineer:
    """Engineer features using real FDR data from FPL API."""
    
    def __init__(self):
        self.team_strength = {}
        self.fdr_cache = {}
        self._load_fpl_data()
    
    def _load_fpl_data(self):
        """Load team strength and fixture data from FPL API."""
        logger.info("Fetching FPL data for FDR features...")
        
        # Get bootstrap data
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        
        # Extract team strength
        for team in data.get('teams', []):
            self.team_strength[team['id']] = {
                'name': team['name'],
                'strength': team['strength'],
                'strength_attack_home': team['strength_attack_home'],
                'strength_attack_away': team['strength_attack_away'],
                'strength_defence_home': team['strength_defence_home'],
                'strength_defence_away': team['strength_defence_away']
            }
        
        # Get fixtures
        fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
        fixtures_resp = requests.get(fixtures_url, timeout=30)
        fixtures = fixtures_resp.json()
        
        # Create FDR lookup
        for f in fixtures:
            gw = f.get('event')
            if gw:
                key = (f['team_h'], gw)  # Home team
                self.fdr_cache[key] = {
                    'fdr': f.get('team_h_difficulty', 3),
                    'opponent': f['team_a'],
                    'was_home': True
                }
                key = (f['team_a'], gw)  # Away team
                self.fdr_cache[key] = {
                    'fdr': f.get('team_a_difficulty', 3),
                    'opponent': f['team_h'],
                    'was_home': False
                }
        
        logger.info(f"Loaded {len(self.team_strength)} teams, {len(self.fdr_cache)} fixture ratings")
    
    def get_fdr_features(self, team_id, gameweek):
        """Get FDR features for a team/gameweek."""
        # Get fixture difficulty
        key = (team_id, gameweek)
        fdr_data = self.fdr_cache.get(key, {'fdr': 3, 'opponent': 0, 'was_home': True})
        
        # Get opponent strength
        opponent_id = fdr_data['opponent']
        opp_strength = self.team_strength.get(opponent_id, {})
        
        # Get own team strength
        own_strength = self.team_strength.get(team_id, {})
        
        # Calculate features
        is_home = fdr_data['was_home']
        
        if is_home:
            opp_attack = opp_strength.get('strength_attack_away', 1000)
            opp_defense = opp_strength.get('strength_defence_away', 1000)
            own_attack = own_strength.get('strength_attack_home', 1000)
            own_defense = own_strength.get('strength_defence_home', 1000)
        else:
            opp_attack = opp_strength.get('strength_attack_home', 1000)
            opp_defense = opp_strength.get('strength_defence_home', 1000)
            own_attack = own_strength.get('strength_attack_away', 1000)
            own_defense = own_strength.get('strength_defence_away', 1000)
        
        return {
            'fdr': fdr_data['fdr'],
            'opp_attack_strength': opp_attack / 1000,  # Normalize
            'opp_defense_strength': opp_defense / 1000,
            'own_attack_strength': own_attack / 1000,
            'own_defense_strength': own_defense / 1000,
            'strength_diff': (own_attack - opp_defense) / 1000,
            'relative_difficulty': fdr_data['fdr'] * (opp_attack / 1200)  # Weighted FDR
        }


class EXP032Trainer:
    """Train EXP-032 with FDR features."""
    
    def __init__(self):
        self.models_dir = Path('/home/akshit/fpl-lineup-optimizer/models')
        self.fdr_engineer = FDRFeatureEngineer()
        
        # Load data
        logger.info("Loading training data...")
        self.train_df = pd.read_csv('/home/akshit/fpl-lineup-optimizer/datasets/fpl_multi_year/train.csv', low_memory=False)
        self.test_df = pd.read_csv('/home/akshit/fpl-lineup-optimizer/datasets/fpl_multi_year/test.csv', low_memory=False)
        
        # Map team names to IDs
        self._build_team_mapping()
        
        logger.info(f"Data loaded: {len(self.train_df)} train, {len(self.test_df)} test")
    
    def _build_team_mapping(self):
        """Build mapping from team names to IDs."""
        # Common team name mappings
        name_to_id = {
            'Arsenal': 1, 'Aston Villa': 2, 'Bournemouth': 3, 'Brentford': 4,
            'Brighton': 5, 'Burnley': 6, 'Chelsea': 7, 'Crystal Palace': 8,
            'Everton': 9, 'Fulham': 10, 'Liverpool': 11, 'Man City': 12,
            'Man Utd': 13, 'Newcastle': 14, "Nott'm Forest": 15, 'Spurs': 16,
            'West Ham': 17, 'Wolves': 18, 'Luton': 19, 'Sheffield Utd': 20,
            'Leeds': 21, 'Leicester': 22, 'Southampton': 23, 'Watford': 24,
            'Norwich': 25, 'Cardiff': 26, 'Huddersfield': 27, 'Fulham': 28
        }
        self.team_name_to_id = name_to_id
    
    def prepare_features_with_fdr(self, df):
        """Prepare features including FDR."""
        features_list = []
        feature_names = []
        
        # Base features (from EXP-031)
        base_features = {
            'form_3gw': df.get('form_3gw', df.get('form_3gw', pd.Series([0]*len(df)))),
            'form_5gw': df.get('form_5gw', df.get('form_5gw', pd.Series([0]*len(df)))),
            'value': df.get('value', pd.Series([50]*len(df))) / 10,
            'was_home': df.get('was_home', pd.Series([1]*len(df))),
            'selected': np.log1p(df.get('selected', pd.Series([10000]*len(df)))),
            'transfers_balance': df.get('transfers_balance', pd.Series([0]*len(df))) / 1000
        }
        
        for name, values in base_features.items():
            features_list.append(values.fillna(0).values)
            feature_names.append(name)
        
        # Position encoding
        if 'position_code' in df.columns:
            for pos in [1, 2, 3, 4]:
                features_list.append((df['position_code'] == pos).astype(float).values)
                feature_names.append(f'pos_{pos}')
        
        # FDR features (NEW!)
        logger.info("Calculating FDR features...")
        fdr_features = {k: [] for k in ['fdr', 'opp_attack', 'opp_defense', 'own_attack', 'own_defense', 'strength_diff', 'rel_difficulty']}
        
        for idx, row in df.iterrows():
            team = row.get('team', '')
            gw = row.get('gameweek', 1)
            
            # Map team name to ID
            team_id = self.team_name_to_id.get(str(team), 0)
            
            if team_id > 0:
                fdr_data = self.fdr_engineer.get_fdr_features(team_id, int(gw))
                fdr_features['fdr'].append(fdr_data['fdr'])
                fdr_features['opp_attack'].append(fdr_data['opp_attack_strength'])
                fdr_features['opp_defense'].append(fdr_data['opp_defense_strength'])
                fdr_features['own_attack'].append(fdr_data['own_attack_strength'])
                fdr_features['own_defense'].append(fdr_data['own_defense_strength'])
                fdr_features['strength_diff'].append(fdr_data['strength_diff'])
                fdr_features['rel_difficulty'].append(fdr_data['relative_difficulty'])
            else:
                # Default values
                for k in fdr_features:
                    fdr_features[k].append(3.0 if k == 'fdr' else 1.0)
        
        # Add FDR features
        for name, values in fdr_features.items():
            features_list.append(np.array(values))
            feature_names.append(f'fdr_{name}')
        
        X = np.column_stack(features_list)
        y = df['total_points'].fillna(0).values if 'total_points' in df.columns else np.zeros(len(df))
        
        logger.info(f"Feature matrix: {X.shape}, Names: {feature_names}")
        return X, y, feature_names
    
    def train(self):
        """Train models with FDR features."""
        logger.info("="*70)
        logger.info("TRAINING EXP-032 WITH FDR FEATURES")
        logger.info("="*70)
        logger.info(f"Target: Beat EXP-031 v2 (Spearman 0.7630)")
        
        # Prepare features
        logger.info("\nPreparing features...")
        train_X, train_y, feature_names = self.prepare_features_with_fdr(self.train_df)
        test_X, test_y, _ = self.prepare_features_with_fdr(self.test_df)
        
        # Scale
        scaler = StandardScaler()
        train_X_s = scaler.fit_transform(train_X)
        test_X_s = scaler.transform(test_X)
        
        # Train models
        logger.info("\n" + "="*70)
        logger.info("TRAINING MODELS")
        logger.info("="*70)
        
        models = {
            'ridge': Ridge(alpha=1.0),
            'gb': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            model.fit(train_X_s, train_y)
            
            train_pred = model.predict(train_X_s)
            test_pred = model.predict(test_X_s)
            
            train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
            test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
            spearman = spearmanr(test_y, test_pred)[0]
            
            results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'spearman': spearman,
                'model': model
            }
            
            logger.info(f"  Test RMSE: {test_rmse:.4f}")
            logger.info(f"  Spearman: {spearman:.4f}")
            
            # Check if beats champion
            if spearman > 0.7630:
                logger.info(f"  🎉 BEATS CHAMPION! (+{(spearman-0.7630)/0.7630*100:.1f}%)")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("RESULTS SUMMARY")
        logger.info("="*70)
        logger.info(f"{'Model':<15} {'RMSE':<10} {'Spearman':<10} {'vs 0.7630':<12}")
        logger.info("-"*70)
        
        best_model = None
        best_spearman = 0.7630  # Champion benchmark
        
        for name, res in results.items():
            improvement = (res['spearman'] - 0.7630) / 0.7630 * 100
            status = "✅ NEW CHAMPION" if res['spearman'] > 0.7630 else "❌"
            logger.info(f"{name:<15} {res['test_rmse']:<10.4f} {res['spearman']:<10.4f} {improvement:>+6.2f}% {status}")
            
            if res['spearman'] > best_spearman:
                best_spearman = res['spearman']
                best_model = name
        
        if best_model:
            logger.info(f"\n🏆 NEW CHAMPION: {best_model} (Spearman: {best_spearman:.4f})")
            
            # Save model
            output_dir = self.models_dir / 'exp032_fdr'
            output_dir.mkdir(exist_ok=True)
            
            model_data = {
                'model': results[best_model]['model'],
                'scaler': scaler,
                'feature_names': feature_names,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_dir / 'model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Saved to: {output_dir}/model.pkl")
            return True
        else:
            logger.info("\n✗ No model beat EXP-031 v2 (0.7630)")
            logger.info("FDR features didn't provide improvement. Try different approach.")
            return False


def main():
    trainer = EXP032Trainer()
    success = trainer.train()
    
    if success:
        print("\n🎉 EXP-032 DISCOVERED! New champion with FDR features!")
    else:
        print("\n⏳ EXP-032 not found yet. EXP-031 v2 remains champion.")
        print("Next: Try position-specific models or LSTM.")


if __name__ == '__main__':
    main()
