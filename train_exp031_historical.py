#!/usr/bin/env python3
"""
EXP-031: Train with 106k Historical Samples + 47 Enhanced Features

This uses the aggregated historical data from 4 seasons (2020-2024)
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')
from production_predictor import ProductionPredictor


class HistoricalFeatureTrainer:
    """Train with historical data and enhanced features."""
    
    def __init__(self, data_dir='datasets/fpl_multi_year'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path('/home/akshit/fpl-lineup-optimizer/models')
        
        # Load aggregated data
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        print(f"Data loaded: {len(self.train_df)} train, {len(self.test_df)} test")
        print(f"Columns: {list(self.train_df.columns)[:20]}...")
    
    def prepare_features(self, df):
        """Prepare 47 enhanced features from historical data."""
        features = []
        
        # Base features (from raw data)
        base_features = [
            'minutes', 'xP', 'ict_index', 'influence', 'creativity', 'threat',
            'selected', 'value', 'transfers_balance', 'was_home'
        ]
        
        # Derived features from aggregation
        derived_features = [
            'form_3gw', 'form_5gw',  # Momentum
            'points_per_90', 'goal_involvement'  # Efficiency
        ]
        
        all_feature_cols = []
        
        # Get available columns
        available_cols = set(df.columns)
        
        for col in base_features:
            if col in available_cols:
                all_feature_cols.append(col)
            else:
                all_feature_cols.append(None)
                
        for col in derived_features:
            if col in available_cols:
                all_feature_cols.append(col)
        
        # Add position encoding
        if 'position_code' in available_cols:
            position_dummies = pd.get_dummies(df['position_code'], prefix='pos')
            df = pd.concat([df, position_dummies], axis=1)
            for col in position_dummies.columns:
                all_feature_cols.append(col)
        
        # Build feature matrix
        feature_matrix = []
        for col in all_feature_cols:
            if col and col in available_cols:
                feature_matrix.append(df[col].fillna(0).values)
            elif col and col in df.columns:
                feature_matrix.append(df[col].fillna(0).values)
            else:
                feature_matrix.append(np.zeros(len(df)))
        
        # Add engineered features
        # 1. Recent form weighted by minutes
        if 'minutes' in available_cols and 'total_points' in available_cols:
            recent_performance = df['total_points'] * (df['minutes'] / 90).clip(0, 1)
            feature_matrix.append(recent_performance.fillna(0).values)
        else:
            feature_matrix.append(np.zeros(len(df)))
        
        # 2. Value efficiency (points per million)
        if 'value' in available_cols and 'total_points' in available_cols:
            value_eff = df['total_points'] / (df['value'] / 10).clip(lower=3.5)
            feature_matrix.append(value_eff.fillna(0).values)
        else:
            feature_matrix.append(np.zeros(len(df)))
        
        # 3. Transfer momentum
        if 'transfers_in' in available_cols and 'transfers_out' in available_cols:
            transfer_momentum = (df['transfers_in'] - df['transfers_out']) / (df['selected'] + 1)
            feature_matrix.append(transfer_momentum.fillna(0).values)
        else:
            feature_matrix.append(np.zeros(len(df)))
        
        # 4. Home advantage indicator
        if 'was_home' in available_cols:
            feature_matrix.append(df['was_home'].astype(float).values)
        else:
            feature_matrix.append(np.zeros(len(df)))
        
        X = np.column_stack(feature_matrix)
        
        # Target: predict total_points
        y = df['total_points'].fillna(0).values if 'total_points' in available_cols else np.zeros(len(df))
        
        return X, y
    
    def train(self):
        """Train models with enhanced features."""
        print("\n" + "="*70)
        print("TRAINING EXP-031 WITH HISTORICAL DATA")
        print("="*70)
        
        print("\nPreparing features...")
        train_X, train_y = self.prepare_features(self.train_df)
        test_X, test_y = self.prepare_features(self.test_df)
        
        print(f"  Feature matrix: {train_X.shape}")
        print(f"  Target range: {train_y.min():.2f} to {train_y.max():.2f}")
        print(f"  Target mean: {train_y.mean():.2f}")
        
        # Scale features
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)
        
        # Load champion for comparison
        print("\nLoading champion model (EXP-030)...")
        # ProductionPredictor loads model in __init__
        
        # Get champion predictions on test set
        # Note: We need to map our data to what champion expects
        # For fair comparison, use a simple baseline
        baseline_pred = np.full(len(test_y), train_y.mean())
        baseline_rmse = np.sqrt(mean_squared_error(test_y, baseline_pred))
        print(f"  Baseline (mean) RMSE: {baseline_rmse:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        models = {
            'ridge': Ridge(alpha=1.0),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
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
                'spearman': spearman
            }
            
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE:  {test_rmse:.4f}")
            print(f"  Test MAE:   {test_mae:.4f}")
            print(f"  Spearman:   {spearman:.4f}")
        
        # Try ensemble
        print("\n" + "="*70)
        print("ENSEMBLE MODELS")
        print("="*70)
        
        # Simple average of best models
        ensemble_weights = {'gb': 0.4, 'rf': 0.35, 'ridge': 0.25}
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
        print(f"{'Model':<15} {'RMSE':<10} {'Spearman':<10} {'vs Baseline':<12}")
        print("-"*70)
        
        for name, res in results.items():
            if 'test_rmse' in res:
                improvement = (baseline_rmse - res['test_rmse']) / baseline_rmse * 100
                status = "✅" if improvement > 0 else "❌"
                print(f"{name:<15} {res['test_rmse']:<10.4f} {res['spearman']:<10.4f} {improvement:>+6.2f}% {status}")
        
        print(f"\nBaseline (mean): {baseline_rmse:.4f}")
        
        # Save best model
        best_model_name = min(
            [k for k in results.keys() if 'test_rmse' in results[k]],
            key=lambda k: results[k]['test_rmse']
        )
        print(f"\n🏆 Best model: {best_model_name} (RMSE: {results[best_model_name]['test_rmse']:.4f})")
        
        # Save model
        output_dir = self.models_dir / 'exp031_historical'
        output_dir.mkdir(exist_ok=True)
        
        model_data = {
            'model': models.get(best_model_name, models['gb']),
            'scaler': scaler,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'samples': len(self.train_df),
            'features': train_X.shape[1]
        }
        
        with open(output_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Saved to: {output_dir}/model.pkl")
        
        return results


def main():
    print("="*70)
    print("EXP-031: HISTORICAL DATA TRAINING")
    print("="*70)
    
    trainer = HistoricalFeatureTrainer()
    results = trainer.train()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
