#!/usr/bin/env python3
"""
Agent 2: EXP-033 Research - Position-Specific Models

Hypothesis: Different positions (GK/DEF/MID/FWD) have different
predictive patterns. Training separate models per position may
improve overall performance.

Target: Beat EXP-032 (Spearman 0.7666)
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


class PositionSpecificTrainer:
    """Train separate models for each position."""
    
    def __init__(self):
        self.data_dir = Path('datasets/fpl_multi_year')
        self.models_dir = Path('models/exp033_position')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading data...")
        self.train_df = pd.read_csv(self.data_dir / 'train.csv', low_memory=False)
        self.test_df = pd.read_csv(self.data_dir / 'test.csv', low_memory=False)
        
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")
    
    def prepare_features(self, df):
        """Prepare features (same as EXP-032)."""
        features_list = []
        
        base_features = {
            'form_3gw': df.get('form_3gw', pd.Series([0]*len(df))),
            'form_5gw': df.get('form_5gw', pd.Series([0]*len(df))),
            'value': df.get('value', pd.Series([50]*len(df))) / 10,
            'was_home': df.get('was_home', pd.Series([1]*len(df))),
            'selected': np.log1p(df.get('selected', pd.Series([10000]*len(df)))),
            'transfers_balance': df.get('transfers_balance', pd.Series([0]*len(df))) / 1000
        }
        
        for name, values in base_features.items():
            features_list.append(values.fillna(0).values)
        
        # Position encoding (will be filtered per model)
        if 'position_code' in df.columns:
            for pos in [1, 2, 3, 4]:
                features_list.append((df['position_code'] == pos).astype(float).values)
        
        X = np.column_stack(features_list)
        y = df['total_points'].fillna(0).values
        
        return X, y
    
    def train_position_model(self, position_code, position_name):
        """Train model for a specific position."""
        print(f"\n{'='*60}")
        print(f"Training model for {position_name} (code: {position_code})")
        print(f"{'='*60}")
        
        # Filter data for this position
        train_pos = self.train_df[self.train_df['position_code'] == position_code]
        test_pos = self.test_df[self.test_df['position_code'] == position_code]
        
        print(f"Train samples: {len(train_pos)}")
        print(f"Test samples: {len(test_pos)}")
        
        if len(train_pos) < 100 or len(test_pos) < 50:
            print(f"⚠️ Not enough data for {position_name}")
            return None
        
        # Prepare features
        X_train, y_train = self.prepare_features(train_pos)
        X_test, y_test = self.prepare_features(test_pos)
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        print("Training...")
        model.fit(X_train_s, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_s)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        spearman = spearmanr(y_test, y_pred)[0]
        
        print(f"Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Spearman: {spearman:.4f}")
        
        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'position': position_name,
            'position_code': position_code,
            'metrics': {'rmse': rmse, 'spearman': spearman},
            'samples': len(train_pos)
        }
        
        output_path = self.models_dir / f'{position_name.lower()}_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Saved to: {output_path}")
        
        return model_data
    
    def evaluate_ensemble(self, position_models):
        """Evaluate ensemble of position-specific models."""
        print(f"\n{'='*60}")
        print("Evaluating Position-Specific Ensemble")
        print(f"{'='*60}")
        
        all_predictions = []
        all_actuals = []
        
        for position_code, model_data in position_models.items():
            if model_data is None:
                continue
            
            # Get test data for this position
            test_pos = self.test_df[self.test_df['position_code'] == position_code]
            if len(test_pos) == 0:
                continue
            
            X_test, y_test = self.prepare_features(test_pos)
            
            # Predict
            scaler = model_data['scaler']
            model = model_data['model']
            
            X_test_s = scaler.transform(X_test)
            y_pred = model.predict(X_test_s)
            
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
        
        # Overall metrics
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        spearman = spearmanr(all_actuals, all_predictions)[0]
        
        print(f"\nOverall Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Spearman: {spearman:.4f}")
        print(f"  vs EXP-032 (0.7666): {spearman - 0.7666:+.4f}")
        
        if spearman > 0.7666:
            print(f"\n🎉 NEW CHAMPION! EXP-033 beats EXP-032!")
        else:
            print(f"\n⏳ No improvement over EXP-032")
        
        return {'rmse': rmse, 'spearman': spearman}


def main():
    """Agent 2 main task."""
    print("="*70)
    print("AGENT 2: EXP-033 Research - Position-Specific Models")
    print("="*70)
    print(f"Target: Beat EXP-032 (Spearman 0.7666)")
    print()
    
    trainer = PositionSpecificTrainer()
    
    # Train models for each position
    positions = {
        1: 'GK',
        2: 'DEF', 
        3: 'MID',
        4: 'FWD'
    }
    
    position_models = {}
    
    for code, name in positions.items():
        model = trainer.train_position_model(code, name)
        position_models[code] = model
    
    # Evaluate ensemble
    results = trainer.evaluate_ensemble(position_models)
    
    # Summary
    print("\n" + "="*70)
    print("AGENT 2 COMPLETE")
    print("="*70)
    
    # Save results
    output = {
        'agent': 2,
        'task': 'EXP-033 Position-Specific Models',
        'status': 'complete',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'position_models': {
            code: {
                'position': name,
                'trained': model is not None,
                'spearman': model['metrics']['spearman'] if model else None
            }
            for code, (name, model) in zip(positions.keys(), zip(positions.values(), position_models.values()))
        }
    }
    
    results_path = Path('research/agents/agent2_results')
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_path}/results.json")


if __name__ == '__main__':
    main()
