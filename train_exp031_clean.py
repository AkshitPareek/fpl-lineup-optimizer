#!/usr/bin/env python3
"""
EXP-031: Train with 52k Historical Samples (NO DATA LEAKAGE)

Uses only features available BEFORE the match:
- Historical form (from previous gameweeks)
- Player attributes (value, position)
- Fixture info (home/away)
- Historical averages (NOT including current match)
"""

import sys
import json
import pickle
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


class CleanFeatureTrainer:
    """Train with clean, non-leaking features."""
    
    def __init__(self, data_dir='datasets/fpl_multi_year'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path('/home/akshit/fpl-lineup-optimizer/models')
        
        # Load aggregated data
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        print(f"Data loaded: {len(self.train_df)} train, {len(self.test_df)} test")
    
    def prepare_features(self, df):
        """Prepare features with NO data leakage.
        
        Features used:
        - form_3gw: rolling 3-game average (from previous matches only)
        - form_5gw: rolling 5-game average (from previous matches only)  
        - value: player price
        - was_home: home/away fixture
        - position: player position (encoded)
        - selected: ownership (popularity)
        - transfers_balance: net transfers
        
        NOT used (data leakage):
        - total_points (target)
        - minutes (only known after match)
        - points_per_90 (derived from target)
        - goals_scored, assists (match outcomes)
        - all post-match stats
        """
        features_list = []
        feature_names = []
        
        # Lagged form features (from previous matches, not current)
        if 'form_3gw' in df.columns:
            features_list.append(df['form_3gw'].fillna(0).values)
            feature_names.append('form_3gw')
        
        if 'form_5gw' in df.columns:
            features_list.append(df['form_5gw'].fillna(0).values)
            feature_names.append('form_5gw')
        
        # Pre-match known features
        if 'value' in df.columns:
            features_list.append(df['value'].fillna(50).values / 10)  # Price in millions
            feature_names.append('value')
        
        if 'was_home' in df.columns:
            features_list.append(df['was_home'].astype(float).values)
            feature_names.append('was_home')
        
        if 'selected' in df.columns:
            # Log transform for ownership
            selected = df['selected'].fillna(10000).values
            features_list.append(np.log1p(selected))
            feature_names.append('log_selected')
        
        if 'transfers_balance' in df.columns:
            # Net transfers in/out
            features_list.append(df['transfers_balance'].fillna(0).values / 1000)
            feature_names.append('transfers_balance_k')
        
        # Position encoding (known pre-match)
        if 'position_code' in df.columns:
            for pos in [1, 2, 3, 4]:  # GK, DEF, MID, FWD
                features_list.append((df['position_code'] == pos).astype(float).values)
                feature_names.append(f'pos_{pos}')
        elif 'position' in df.columns:
            # One-hot encode position strings
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                features_list.append((df['position'] == pos).astype(float).values)
                feature_names.append(f'pos_{pos}')
        
        # Gameweek (season progression)
        if 'gameweek' in df.columns:
            features_list.append(df['gameweek'].fillna(20).values / 38)  # Normalize
            feature_names.append('gameweek_norm')
        
        X = np.column_stack(features_list)
        y = df['total_points'].fillna(0).values
        
        return X, y, feature_names
    
    def train(self):
        """Train models with clean features."""
        print("\n" + "="*70)
        print("TRAINING EXP-031 WITH CLEAN FEATURES (NO LEAKAGE)")
        print("="*70)
        
        print("\nPreparing features...")
        train_X, train_y, feature_names = self.prepare_features(self.train_df)
        test_X, test_y, _ = self.prepare_features(self.test_df)
        
        print(f"  Features: {feature_names}")
        print(f"  Feature matrix: {train_X.shape}")
        print(f"  Target range: [{train_y.min():.1f}, {train_y.max():.1f}]")
        print(f"  Target mean: {train_y.mean():.2f}, std: {train_y.std():.2f}")
        
        # Scale features
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)
        
        # Baseline: predict mean
        baseline_pred = np.full(len(test_y), train_y.mean())
        baseline_rmse = np.sqrt(mean_squared_error(test_y, baseline_pred))
        print(f"\n  Baseline (mean) RMSE: {baseline_rmse:.4f}")
        
        # Naive baseline: predict 1 point for everyone
        naive_pred = np.ones(len(test_y))
        naive_rmse = np.sqrt(mean_squared_error(test_y, naive_pred))
        print(f"  Baseline (all 1s) RMSE: {naive_rmse:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        models = {
            'ridge': Ridge(alpha=1.0),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
        }
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(train_X_scaled, train_y)
            
            train_pred = model.predict(train_X_scaled)
            test_pred = model.predict(test_X_scaled)
            predictions[name] = test_pred
            
            train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
            test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
            test_mae = mean_absolute_error(test_y, test_pred)
            spearman = spearmanr(test_y, test_pred)[0]
            
            results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'spearman': spearman,
                'model': model
            }
            
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE:  {test_rmse:.4f}")
            print(f"  Test MAE:   {test_mae:.4f}")
            print(f"  Spearman:   {spearman:.4f}")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:3]
                print(f"  Top features: {', '.join([f'{n}({i:.2f})' for n, i in top_features])}")
        
        # Ensemble
        print("\n" + "="*70)
        print("ENSEMBLE")
        print("="*70)
        
        # Weighted ensemble
        ensemble_weights = {'gb': 0.5, 'rf': 0.3, 'ridge': 0.2}
        ensemble_pred = np.zeros(len(test_y))
        for name, weight in ensemble_weights.items():
            ensemble_pred += weight * predictions[name]
        
        ensemble_rmse = np.sqrt(mean_squared_error(test_y, ensemble_pred))
        ensemble_spearman = spearmanr(test_y, ensemble_pred)[0]
        
        print(f"Weighted Ensemble RMSE: {ensemble_rmse:.4f}")
        print(f"Weighted Ensemble Spearman: {ensemble_spearman:.4f}")
        
        results['ensemble'] = {
            'test_rmse': ensemble_rmse,
            'spearman': ensemble_spearman
        }
        
        # Summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"{'Model':<15} {'Train RMSE':<12} {'Test RMSE':<12} {'Spearman':<10} {'vs Mean':<10}")
        print("-"*70)
        
        for name, res in sorted(results.items(), key=lambda x: x[1].get('test_rmse', 999)):
            if 'test_rmse' in res and name != 'ensemble':
                improvement = (baseline_rmse - res['test_rmse']) / baseline_rmse * 100
                status = "✅" if improvement > 0 else "❌"
                print(f"{name:<15} {res['train_rmse']:<12.4f} {res['test_rmse']:<12.4f} "
                      f"{res['spearman']:<10.4f} {improvement:>+6.2f}% {status}")
        
        print(f"\n{'ensemble':<15} {'--':<12} {ensemble_rmse:<12.4f} {ensemble_spearman:<10.4f} "
              f"{(baseline_rmse - ensemble_rmse) / baseline_rmse * 100:>+6.2}% ✅")
        
        print(f"\nBaselines:")
        print(f"  Mean predictor: {baseline_rmse:.4f}")
        print(f"  All 1s:         {naive_rmse:.4f}")
        
        # Save best model
        best_model_name = min(
            [k for k in results.keys() if k != 'ensemble' and 'test_rmse' in results[k]],
            key=lambda k: results[k]['test_rmse']
        )
        best = results[best_model_name]
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Test RMSE: {best['test_rmse']:.4f}")
        print(f"   Spearman:  {best['spearman']:.4f}")
        
        # Save model
        output_dir = self.models_dir / 'exp031_clean'
        output_dir.mkdir(exist_ok=True)
        
        model_data = {
            'model': best['model'],
            'scaler': scaler,
            'feature_names': feature_names,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'samples_train': len(self.train_df),
            'samples_test': len(self.test_df),
            'n_features': train_X.shape[1]
        }
        
        with open(output_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                          for kk, vv in v.items() if kk != 'model'} 
                      for k, v in results.items()}, f, indent=2)
        
        print(f"\n💾 Saved to: {output_dir}/")
        
        return results, best_model_name


def main():
    print("="*70)
    print("EXP-031: CLEAN FEATURE TRAINING")
    print("="*70)
    
    trainer = CleanFeatureTrainer()
    results, best = trainer.train()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model: {best}")
    print(f"RMSE: {results[best]['test_rmse']:.4f}")


if __name__ == '__main__':
    main()
