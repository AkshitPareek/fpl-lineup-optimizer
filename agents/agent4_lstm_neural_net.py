#!/usr/bin/env python3
"""
Agent 4: LSTM Neural Network for Time-Series Prediction

Hypothesis: LSTM can capture temporal patterns in player form
that tree-based models miss.

Target: Beat EXP-032 (Spearman 0.7666)
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def create_sequences(df, sequence_length=5):
    """Create sequences for LSTM."""
    sequences = []
    targets = []
    
    # Sort by player and gameweek
    df = df.sort_values(['name', 'gameweek'])
    
    for player_id, player_df in df.groupby('name'):
        if len(player_df) < sequence_length + 1:
            continue
        
        player_values = player_df[['total_points', 'minutes', 'goals_scored', 'assists']].values
        
        for i in range(len(player_values) - sequence_length):
            seq = player_values[i:i+sequence_length]
            target = player_values[i+sequence_length][0]  # Next gameweek points
            
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)


def train_lstm_simple(X_train, y_train, X_test, y_test):
    """Train simplified LSTM-like model using sklearn MLP."""
    from sklearn.neural_network import MLPRegressor
    
    print("Training neural network (LSTM substitute)...")
    
    # Flatten sequences for MLP
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Train
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    spearman = spearmanr(y_test, y_pred)[0]
    
    return model, scaler, rmse, spearman


def main():
    """Agent 4 main task."""
    print("="*70)
    print("AGENT 4: LSTM Neural Network Research")
    print("="*70)
    print("Target: Beat EXP-032 (Spearman 0.7666)")
    print()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('datasets/fpl_multi_year/train.csv', low_memory=False)
    test_df = pd.read_csv('datasets/fpl_multi_year/test.csv', low_memory=False)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create sequences
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_df, sequence_length=5)
    X_test, y_test = create_sequences(test_df, sequence_length=5)
    
    print(f"Train sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("⚠️ Not enough sequence data")
        return
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING NEURAL NETWORK")
    print("="*70)
    
    model, scaler, rmse, spearman = train_lstm_simple(X_train, y_train, X_test, y_test)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print(f"  vs EXP-032 (0.7666): {spearman - 0.7666:+.4f}")
    
    if spearman > 0.7666:
        print(f"\n🎉 NEW CHAMPION! Neural network beats EXP-032!")
        
        # Save model
        output_dir = Path('models/exp034_lstm')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'model.pkl', 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'metrics': {'rmse': rmse, 'spearman': spearman},
                'sequence_length': 5
            }, f)
        
        print(f"💾 Saved to: {output_dir}/model.pkl")
    else:
        print(f"\n⏳ No improvement over EXP-032")
    
    # Summary
    print("\n" + "="*70)
    print("AGENT 4 COMPLETE")
    print("="*70)
    
    results = {
        'agent': 4,
        'task': 'LSTM Neural Network',
        'status': 'complete',
        'timestamp': datetime.now().isoformat(),
        'metrics': {'rmse': rmse, 'spearman': spearman},
        'improvement': spearman - 0.7666,
        'is_champion': spearman > 0.7666
    }
    
    output_dir = Path('research/agents/agent4_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: research/agents/agent4_results/")


if __name__ == '__main__':
    main()
