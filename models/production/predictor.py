#!/usr/bin/env python3
"""
Production predictor using EXP-030 ensemble model.
"""

import pickle
import numpy as np
from pathlib import Path

def load_model():
    """Load the production model."""
    model_path = Path(__file__).parent / "model.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict(X):
    """
    Generate predictions for input features.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
    
    Returns:
        numpy array of predicted points
    """
    ensemble_data = load_model()
    models = ensemble_data['models']
    weights = ensemble_data['weights']
    
    predictions = np.zeros(len(X))
    for (name, model), weight in zip(models.items(), weights):
        predictions += model.predict(X) * weight
    
    return predictions

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        # Load from file
        X = np.load(sys.argv[1])
        preds = predict(X)
        print(preds)
    else:
        print("Usage: python predictor.py <features.npy>")
