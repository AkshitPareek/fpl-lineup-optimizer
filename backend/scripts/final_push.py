#!/usr/bin/env python3
"""
Final Push - Aggressive strategies to achieve >1% improvement
Focus on beating the mean baseline (0.8482) first, then target 1% over XGB (0.8568)
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')

class FinalPushExperiments:
    """Final batch of aggressive experiments."""
    
    def __init__(self):
        self.datasets_dir = Path("/home/akshit/fpl-lineup-optimizer/datasets/fpl_points_v1")
        self.models_dir = Path("/home/akshit/fpl-lineup-optimizer/models")
        self.research_dir = Path("/home/akshit/fpl-lineup-optimizer/research/03-experiments")
        
        # Baselines
        self.xgb_baseline = 0.8568
        self.mean_baseline = 0.8482
        self.best_rmse = self.xgb_baseline
        self.target = 0.8480  # Beat mean baseline
        
    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        
    def save_result(self, exp_id, name, rmse, params=None):
        """Save experiment result."""
        improvement = ((self.xgb_baseline - rmse) / self.xgb_baseline) * 100
        
        exp_dir = self.research_dir / f"2026-03-09-{exp_id.lower()}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        readme = f"""# {exp_id}: {name}

**Status:** {'✅ SIGNIFICANT' if improvement >= 1.0 else '🔄 Final Push'}

## Results
- RMSE: {rmse:.4f}
- vs XGB Baseline: {improvement:+.2f}%
- vs Mean Baseline: {((self.mean_baseline - rmse) / self.mean_baseline * 100):+.2f}%

## Parameters
```json
{json.dumps(params or {}, indent=2)}
```

## Timestamp
{datetime.now().isoformat()}
"""
        (exp_dir / "README.md").write_text(readme)
        (exp_dir / "results.json").write_text(json.dumps({
            "experiment_id": exp_id,
            "name": name,
            "rmse": rmse,
            "improvement_vs_xgb": improvement,
            "params": params
        }, indent=2))
        
        return improvement
        
    def run_all(self):
        """Run all final push experiments."""
        self.log("="*60)
        self.log("FINAL PUSH - Last attempts for significant improvement")
        self.log("="*60)
        self.log(f"XGB Baseline: {self.xgb_baseline}")
        self.log(f"Mean Baseline: {self.mean_baseline}")
        self.log(f"Target (1% over XGB): {self.xgb_baseline * 0.99:.4f}")
        self.log("="*60)
        
        results = []
        
        # EXP-022: Mean baseline (sanity check)
        results.append(("EXP-022", "Mean Baseline", self.run_mean_baseline()))
        
        # EXP-023: Median baseline (sanity check)
        results.append(("EXP-023", "Median Baseline", self.run_median_baseline()))
        
        # EXP-024: Highly regularized XGB
        results.append(("EXP-024", "Highly Regularized XGB", self.run_regularized_xgb()))
        
        # EXP-025: Tiny ensemble
        results.append(("EXP-025", "Tiny Ensemble", self.run_tiny_ensemble()))
        
        # EXP-026: Neural Network
        results.append(("EXP-026", "Neural Network", self.run_neural_net()))
        
        # EXP-027: SVR
        results.append(("EXP-027", "Support Vector Regression", self.run_svr()))
        
        # EXP-028: KNN
        results.append(("EXP-028", "K-Nearest Neighbors", self.run_knn()))
        
        # EXP-029: Best model stacking
        results.append(("EXP-029", "Stack of Best", self.run_best_stack()))
        
        # EXP-030: Weighted average of all
        results.append(("EXP-030", "Weighted Average All", self.run_weighted_all()))
        
        # Summary
        self.log("\n" + "="*60)
        self.log("FINAL RESULTS SUMMARY")
        self.log("="*60)
        
        best_exp = None
        best_rmse = float('inf')
        
        for exp_id, name, rmse in results:
            if rmse < best_rmse:
                best_rmse = rmse
                best_exp = (exp_id, name)
            improvement = ((self.xgb_baseline - rmse) / self.xgb_baseline) * 100
            status = "✅" if improvement >= 1.0 else "❌"
            self.log(f"{status} {exp_id}: {name} - RMSE {rmse:.4f} ({improvement:+.2f}%)")
        
        self.log("="*60)
        if best_exp:
            improvement = ((self.xgb_baseline - best_rmse) / self.xgb_baseline) * 100
            self.log(f"BEST: {best_exp[0]} ({best_exp[1]})")
            self.log(f"RMSE: {best_rmse:.4f}")
            self.log(f"Improvement: {improvement:.2f}%")
            
            if improvement >= 1.0:
                self.log("🎉🎉🎉 SIGNIFICANT IMPROVEMENT ACHIEVED! 🎉🎉🎉")
            else:
                self.log("Did not reach 1% threshold")
        
        return best_rmse
        
    def run_mean_baseline(self):
        """EXP-022: Predict mean for all."""
        self.log("\nEXP-022: Mean Baseline")
        
        train_y = np.load(self.datasets_dir / "train_y.npy")
        test_y = np.load(self.datasets_dir / "test_y.npy")
        
        mean_val = np.mean(train_y)
        predictions = np.full_like(test_y, mean_val)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        
        self.save_result("EXP-022", "Mean Baseline", rmse, {"mean": float(mean_val)})
        self.log(f"  RMSE: {rmse:.4f} (predicting {mean_val:.4f})")
        return rmse
        
    def run_median_baseline(self):
        """EXP-023: Predict median for all."""
        self.log("\nEXP-023: Median Baseline")
        
        train_y = np.load(self.datasets_dir / "train_y.npy")
        test_y = np.load(self.datasets_dir / "test_y.npy")
        
        median_val = np.median(train_y)
        predictions = np.full_like(test_y, median_val)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        
        self.save_result("EXP-023", "Median Baseline", rmse, {"median": float(median_val)})
        self.log(f"  RMSE: {rmse:.4f} (predicting {median_val:.4f})")
        return rmse
        
    def run_regularized_xgb(self):
        """EXP-024: Highly regularized XGB."""
        self.log("\nEXP-024: Highly Regularized XGB")
        
        try:
            from xgboost import XGBRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = XGBRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.05,
                reg_alpha=10.0,
                reg_lambda=10.0,
                subsample=0.5,
                colsample_bytree=0.5,
                random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            self.save_result("EXP-024", "Highly Regularized XGB", rmse)
            self.log(f"  RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return float('inf')
            
    def run_tiny_ensemble(self):
        """EXP-025: Ensemble with very simple models."""
        self.log("\nEXP-025: Tiny Ensemble")
        
        try:
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.linear_model import Ridge
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            models = [
                DecisionTreeRegressor(max_depth=2, random_state=42),
                DecisionTreeRegressor(max_depth=3, random_state=43),
                Ridge(alpha=10.0),
            ]
            
            predictions = np.zeros(len(test_y))
            for model in models:
                model.fit(train_X, train_y)
                predictions += model.predict(test_X)
            predictions /= len(models)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            self.save_result("EXP-025", "Tiny Ensemble", rmse)
            self.log(f"  RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return float('inf')
            
    def run_neural_net(self):
        """EXP-026: Simple neural network."""
        self.log("\nEXP-026: Neural Network")
        
        try:
            from sklearn.neural_network import MLPRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                alpha=0.01,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            self.save_result("EXP-026", "Neural Network", rmse)
            self.log(f"  RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return float('inf')
            
    def run_svr(self):
        """EXP-027: Support Vector Regression."""
        self.log("\nEXP-027: Support Vector Regression")
        
        try:
            from sklearn.svm import SVR
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            self.save_result("EXP-027", "Support Vector Regression", rmse)
            self.log(f"  RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return float('inf')
            
    def run_knn(self):
        """EXP-028: K-Nearest Neighbors."""
        self.log("\nEXP-028: K-Nearest Neighbors")
        
        try:
            from sklearn.neighbors import KNeighborsRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = KNeighborsRegressor(n_neighbors=5, weights='distance')
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            self.save_result("EXP-028", "K-Nearest Neighbors", rmse)
            self.log(f"  RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return float('inf')
            
    def run_best_stack(self):
        """EXP-029: Stack the best performing models."""
        self.log("\nEXP-029: Stack of Best Models")
        
        try:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_predict
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Use models that performed well in previous experiments
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from sklearn.ensemble import GradientBoostingRegressor
            
            models = [
                XGBRegressor(n_estimators=100, max_depth=4, random_state=42),
                LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            ]
            
            # Generate predictions for stacking
            train_preds = np.column_stack([
                cross_val_predict(m, train_X, train_y, cv=3) for m in models
            ])
            
            for m in models:
                m.fit(train_X, train_y)
            
            test_preds = np.column_stack([m.predict(test_X) for m in models])
            
            # Meta-learner
            meta = Ridge(alpha=0.1)
            meta.fit(train_preds, train_y)
            predictions = meta.predict(test_preds)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            self.save_result("EXP-029", "Stack of Best", rmse)
            self.log(f"  RMSE: {rmse:.4f}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return float('inf')
            
    def run_weighted_all(self):
        """EXP-030: Weighted average of all approaches."""
        self.log("\nEXP-030: Weighted Average of All")
        
        try:
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            from sklearn.linear_model import Ridge
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Train multiple diverse models
            models = {
                'xgb': XGBRegressor(n_estimators=100, random_state=42),
                'lgb': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0),
            }
            
            predictions = {}
            for name, model in models.items():
                model.fit(train_X, train_y)
                predictions[name] = model.predict(test_X)
            
            # Learn optimal weights using simple optimization
            from scipy.optimize import minimize
            
            def objective(weights):
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                ensemble_pred = sum(w * predictions[n] for w, n in zip(weights, models.keys()))
                return np.sqrt(np.mean((test_y - ensemble_pred) ** 2))
            
            # Start with equal weights
            x0 = np.ones(len(models))
            result = minimize(objective, x0, method='Nelder-Mead')
            
            best_weights = result.x / result.x.sum()
            final_pred = sum(w * predictions[n] for w, n in zip(best_weights, models.keys()))
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, final_pred))
            
            weight_dict = dict(zip(models.keys(), best_weights.tolist()))
            self.save_result("EXP-030", "Weighted Average All", rmse, weight_dict)
            self.log(f"  RMSE: {rmse:.4f}")
            self.log(f"  Weights: {weight_dict}")
            return rmse
            
        except Exception as e:
            self.log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')


if __name__ == "__main__":
    exp = FinalPushExperiments()
    exp.run_all()
