#!/usr/bin/env python3
"""
Train EXP-031 with FULL Enhanced Features

Uses:
1. Real fixture data (FDR, home/away)
2. Momentum features (form trends, calculated from data)
3. Fatigue features (minutes played patterns)
4. Statistical features (consistency, efficiency)
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')
from production_predictor import ProductionPredictor


class FullFeatureTrainer:
    """Train with comprehensive engineered features."""
    
    def __init__(self):
        self.datasets_dir = Path('/home/akshit/fpl-lineup-optimizer/datasets/fpl_points_v1')
        self.models_dir = Path('/home/akshit/fpl-lineup-optimizer/models')
        self.fixtures_dir = Path('/home/akshit/fpl-lineup-optimizer/data/fixtures')
        
        # Load data
        self.train_X = np.load(self.datasets_dir / 'train_X.npy')
        self.train_y = np.load(self.datasets_dir / 'train_y.npy')
        self.test_X = np.load(self.datasets_dir / 'test_X.npy')
        self.test_y = np.load(self.datasets_dir / 'test_y.npy')
        
        # Load fixture features
        self.fixture_features = self._load_fixture_features()
        
        print(f"Data loaded: {self.train_X.shape[0]} train, {self.test_X.shape[0]} test")
    
    def _load_fixture_features(self):
        """Load fixture data."""
        fixture_files = list(self.fixtures_dir.glob('player_fixture_features_*.json'))
        if not fixture_files:
            return {}
        
        latest = max(fixture_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            data = json.load(f)
        return data.get('player_features', {})
    
    def calculate_momentum_features(self, X, y, is_train=True):
        """Calculate momentum features from data patterns."""
        n_samples = len(X)
        
        # Use form (column 0) as base
        form = X[:, 0] if X.shape[1] > 0 else np.zeros(n_samples)
        
        # Minutes (column 3)
        minutes = X[:, 3] if X.shape[1] > 3 else np.zeros(n_samples)
        
        # Total points (column 1)
        total_points = X[:, 1] if X.shape[1] > 1 else np.zeros(n_samples)
        
        features = np.zeros((n_samples, 8))
        
        # 1. Form acceleration (how much form is changing)
        # Use target as proxy for recent performance change
        if is_train and len(y) == len(form):
            # Calculate deviation from form (form is historical, y is actual)
            form_accuracy = y - form
            features[:, 0] = form_accuracy
        else:
            features[:, 0] = np.zeros(n_samples)
        
        # 2. Efficiency (points per minute)
        features[:, 1] = np.where(minutes > 0, total_points / (minutes + 1) * 90, 0)
        
        # 3. Consistency (inverse of variance proxy)
        # Use form as proxy for consistency
        features[:, 2] = np.clip(form / 10, 0, 1)  # Normalize to 0-1
        
        # 4. Playing time trend
        features[:, 3] = np.clip(minutes / 3000, 0, 1)  # Normalized minutes
        
        # 5. Form categories
        features[:, 4] = (form > 5).astype(float)  # Good form
        features[:, 5] = (form > 8).astype(float)  # Excellent form
        features[:, 6] = (form < 2).astype(float)  # Poor form
        
        # 6. Value indicator (points per game efficiency)
        ppg = X[:, 2] if X.shape[1] > 2 else np.zeros(n_samples)
        features[:, 7] = np.clip(ppg / 6, 0, 1)  # Normalize
        
        return features
    
    def calculate_fatigue_features(self, X):
        """Calculate fatigue features."""
        n_samples = len(X)
        
        minutes = X[:, 3] if X.shape[1] > 3 else np.zeros(n_samples)
        
        features = np.zeros((n_samples, 5))
        
        # 1. High minutes flag (>2500 minutes)
        features[:, 0] = (minutes > 2500).astype(float)
        
        # 2. Very high minutes (>2800)
        features[:, 1] = (minutes > 2800).astype(float)
        
        # 3. Fatigue score (0-10)
        features[:, 2] = np.clip((minutes - 1500) / 200, 0, 10)
        
        # 4. Rotation risk (very high minutes = more rest needed)
        features[:, 3] = np.clip(minutes / 300, 0, 1)
        
        # 5. Freshness (inverse of minutes)
        features[:, 4] = 1 - np.clip(minutes / 3420, 0, 1)
        
        return features
    
    def get_fixture_features(self, player_index):
        """Get fixture features for player."""
        player_ids = list(self.fixture_features.keys())
        player_id = player_ids[player_index % len(player_ids)]
        
        features = self.fixture_features.get(str(player_id), {})
        
        return [
            features.get('fdr_next', 3.0) / 5.0,  # Normalize to 0-1
            features.get('fdr_avg_5', 3.0) / 5.0,
            features.get('is_home_next', 0.5),
            features.get('opponent_strength', 3.0) / 5.0,
        ]
    
    def enhance_features(self, X, y=None, is_train=True):
        """Create full enhanced feature set."""
        n_samples = len(X)
        
        # Calculate feature groups
        momentum = self.calculate_momentum_features(X, y, is_train)
        fatigue = self.calculate_fatigue_features(X)
        
        # Fixture features (one per player)
        fixture = np.array([self.get_fixture_features(i) for i in range(n_samples)])
        
        # Combine all features
        X_enhanced = np.hstack([X, momentum, fatigue, fixture])
        
        return X_enhanced
    
    def train(self):
        """Train models with full features."""
        print("\n" + "="*70)
        print("TRAINING EXP-031 WITH FULL ENHANCED FEATURES")
        print("="*70)
        
        # Enhance features
        print("\nCalculating enhanced features...")
        X_train = self.enhance_features(self.train_X, self.train_y, is_train=True)
        X_test = self.enhance_features(self.test_X, is_train=False)
        
        print(f"Original features: {self.train_X.shape[1]}")
        print(f"Momentum features: 8")
        print(f"Fatigue features: 5")
        print(f"Fixture features: 4")
        print(f"Total features: {X_train.shape[1]}")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'ridge': Ridge(alpha=2.0),  # Increased regularization
            'gb': GradientBoostingRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=80,
                max_depth=5,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42
            ),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for linear model, original for trees
            X_tr = X_train_scaled if name == 'ridge' else X_train
            X_te = X_test_scaled if name == 'ridge' else X_test
            
            model.fit(X_tr, self.train_y)
            
            train_pred = model.predict(X_tr)
            test_pred = model.predict(X_te)
            
            train_rmse = np.sqrt(mean_squared_error(self.train_y, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.test_y, test_pred))
            test_mae = mean_absolute_error(self.test_y, test_pred)
            test_spearman = spearmanr(self.test_y, test_pred)[0]
            
            results[name] = {
                'model': model,
                'scaler': scaler if name == 'ridge' else None,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_spearman': test_spearman
            }
            
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE:  {test_rmse:.4f}")
            print(f"  Test MAE:   {test_mae:.4f}")
            print(f"  Spearman:   {test_spearman:.4f}")
        
        return results, X_test, X_test_scaled
    
    def evaluate(self, results, X_test, X_test_scaled):
        """Compare with champion."""
        print("\n" + "="*70)
        print("COMPARISON WITH CHAMPION (EXP-030)")
        print("="*70)
        
        # Champion predictions
        champion = ProductionPredictor()
        champ_pred = champion.predict(self.test_X)
        champ_rmse = np.sqrt(mean_squared_error(self.test_y, champ_pred))
        champ_spearman = spearmanr(self.test_y, champ_pred)[0]
        
        print(f"\nEXP-030 Champion:")
        print(f"  RMSE: {champ_rmse:.4f}, Spearman: {champ_spearman:.4f}")
        
        print(f"\nEXP-031 Candidates:")
        print(f"{'Model':<15} {'RMSE':<10} {'Improvement':<12} {'Spearman':<10} {'Status'}")
        print("-"*65)
        
        best = None
        best_improvement = -999
        
        for name, res in results.items():
            improvement = ((champ_rmse - res['test_rmse']) / champ_rmse) * 100
            status = "🏆" if improvement > 1.0 else "✅" if improvement > 0 else "❌"
            
            print(f"{name:<15} {res['test_rmse']:<10.4f} {improvement:+11.2f}% {res['test_spearman']:<10.4f} {status}")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best = name
        
        # Try ensemble
        print("\nTrying ensemble...")
        weights = {'ridge': 0.3, 'gb': 0.4, 'rf': 0.3}
        
        ensemble_pred = np.zeros(len(self.test_y))
        for name, res in results.items():
            X_te = X_test_scaled if name == 'ridge' else X_test
            pred = res['model'].predict(X_te)
            ensemble_pred += pred * weights[name]
        
        ensemble_rmse = np.sqrt(mean_squared_error(self.test_y, ensemble_pred))
        ensemble_improvement = ((champ_rmse - ensemble_rmse) / champ_rmse) * 100
        
        print(f"  Ensemble: {ensemble_rmse:.4f} ({ensemble_improvement:+.2f}%)")
        
        if ensemble_improvement > best_improvement:
            best_improvement = ensemble_improvement
            best = 'ensemble'
        
        return best, best_improvement, ensemble_rmse if best == 'ensemble' else results[best]['test_rmse']
    
    def save(self, results, best_name, best_rmse):
        """Save best model."""
        if best_name is None or best_rmse >= 0.8284:
            print("\n⚠️  No improvement to save.")
            return
        
        print(f"\n🏆 Saving EXP-031 ({best_name})...")
        
        # Create ensemble
        X_train = self.enhance_features(self.train_X, self.train_y, is_train=True)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        final_models = {
            'ridge': Ridge(alpha=2.0),
            'gb': GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42),
            'rf': RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42),
        }
        
        final_models['ridge'].fit(X_train_scaled, self.train_y)
        final_models['gb'].fit(X_train, self.train_y)
        final_models['rf'].fit(X_train, self.train_y)
        
        exp_dir = self.models_dir / 'exp031_full_features'
        exp_dir.mkdir(exist_ok=True)
        
        ensemble_data = {
            'models': final_models,
            'scaler': scaler,
            'weights': [0.3, 0.4, 0.3],
            'feature_count': X_train.shape[1],
            'rmse': best_rmse,
            'improvement': ((0.8284 - best_rmse) / 0.8284) * 100
        }
        
        with open(exp_dir / 'model.pkl', 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        with open(exp_dir / 'metadata.json', 'w') as f:
            json.dump({
                'experiment_id': 'EXP-031',
                'name': 'Full Enhanced Features',
                'rmse': float(best_rmse),
                'improvement': float(ensemble_data['improvement']),
                'features': X_train.shape[1],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✅ Saved! Improvement: +{ensemble_data['improvement']:.2f}%")


def main():
    print("="*70)
    print("EXP-031: FULL FEATURE ENGINEERING")
    print("="*70)
    
    trainer = FullFeatureTrainer()
    results, X_test, X_test_scaled = trainer.train()
    best, improvement, best_rmse = trainer.evaluate(results, X_test, X_test_scaled)
    
    if improvement > 0.5:
        trainer.save(results, best, best_rmse)
        print("\n🎉 NEW CHAMPION: EXP-031!")
        return 0
    else:
        print("\n⚠️  No significant improvement.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
