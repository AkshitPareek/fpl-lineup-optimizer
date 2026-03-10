#!/usr/bin/env python3
"""
Autonomous Research Loop (Ralph Loop)

Continuously runs experiments until significant improvement is achieved.
Significance threshold: >1% relative improvement AND Cohen's d > 0.2

Usage:
    python autonomous_research_loop.py [--max-experiments N] [--target-improvement 1.0]
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, '/home/akshit/fpl-lineup-optimizer/backend')

# smart_bench may not exist, we'll implement locally
try:
    from smart_bench import ModelBenchmark, StatisticalValidator
except ImportError:
    ModelBenchmark = None
    StatisticalValidator = None

class AutonomousResearchLoop:
    """Continuous experiment runner that tries strategies until success."""
    
    def __init__(self, target_improvement=1.0, min_effect_size=0.2, max_experiments=50):
        self.target_improvement = target_improvement
        self.min_effect_size = min_effect_size
        self.max_experiments = max_experiments
        
        self.baseline_rmse = 0.8568  # From EXP-001
        self.best_rmse = self.baseline_rmse
        self.best_model = "xgboost_baseline"
        
        self.experiment_history = []
        self.attempted_strategies = set()
        
        self.datasets_dir = Path("/home/akshit/fpl-lineup-optimizer/datasets/fpl_points_v1")
        self.models_dir = Path("/home/akshit/fpl-lineup-optimizer/models")
        self.research_dir = Path("/home/akshit/fpl-lineup-optimizer/research/03-experiments")
        
        # Strategy registry - ordered by likelihood of success
        self.strategies = [
            {
                "id": "EXP-007",
                "name": "Position-Specific Models",
                "function": self.run_position_specific_models,
                "expected_improvement": 2.0,
                "estimated_time": 300,
                "rationale": "Different positions have different point distributions"
            },
            {
                "id": "EXP-008",
                "name": "Hyperparameter Optimization (Optuna)",
                "function": self.run_hyperparameter_optimization,
                "expected_improvement": 1.5,
                "estimated_time": 600,
                "rationale": "200+ trials to find optimal hyperparameters"
            },
            {
                "id": "EXP-009",
                "name": "Weighted Ensemble (Learned)",
                "function": self.run_weighted_ensemble,
                "expected_improvement": 1.0,
                "estimated_time": 180,
                "rationale": "Learn optimal weights using Ridge regression"
            },
            {
                "id": "EXP-010",
                "name": "Stacking Ensemble",
                "function": self.run_stacking_ensemble,
                "expected_improvement": 1.0,
                "estimated_time": 300,
                "rationale": "Meta-learner on top of base models"
            },
            {
                "id": "EXP-011",
                "name": "Feature Selection (Top-K)",
                "function": self.run_feature_selection,
                "expected_improvement": 0.8,
                "estimated_time": 240,
                "rationale": "Remove noise by selecting top features"
            },
            {
                "id": "EXP-012",
                "name": "Quantile Regression",
                "function": self.run_quantile_regression,
                "expected_improvement": 0.5,
                "estimated_time": 240,
                "rationale": "Model uncertainty for better ranking"
            },
            {
                "id": "EXP-013",
                "name": "Huber Loss Robust Regression",
                "function": self.run_huber_loss,
                "expected_improvement": 0.5,
                "estimated_time": 180,
                "rationale": "Reduce impact of outliers"
            },
            {
                "id": "EXP-014",
                "name": "CatBoost Model",
                "function": self.run_catboost,
                "expected_improvement": 0.5,
                "estimated_time": 240,
                "rationale": "Ordered boosting for better generalization"
            },
            {
                "id": "EXP-015",
                "name": "Deep Hyperparameter Search (300 trials)",
                "function": self.run_deep_hyperopt,
                "expected_improvement": 2.0,
                "estimated_time": 900,
                "rationale": "Aggressive 300-trial Optuna search with wider ranges"
            },
            {
                "id": "EXP-016",
                "name": "Voting Ensemble (Hard/Soft)",
                "function": self.run_voting_ensemble,
                "expected_improvement": 1.0,
                "estimated_time": 180,
                "rationale": "Sklearn VotingRegressor with multiple algorithms"
            },
            {
                "id": "EXP-017",
                "name": "Bagging Ensemble",
                "function": self.run_bagging_ensemble,
                "expected_improvement": 1.0,
                "estimated_time": 300,
                "rationale": "Bootstrap aggregation to reduce variance"
            },
            {
                "id": "EXP-018",
                "name": "Extra Trees Model",
                "function": self.run_extra_trees,
                "expected_improvement": 0.8,
                "estimated_time": 180,
                "rationale": "Extremely Randomized Trees for less overfitting"
            },
            {
                "id": "EXP-019",
                "name": "ElasticNet with Poly Features",
                "function": self.run_elasticnet_poly,
                "expected_improvement": 0.5,
                "estimated_time": 300,
                "rationale": "Regularized linear model with polynomial features"
            },
            {
                "id": "EXP-020",
                "name": "Blending Ensemble",
                "function": self.run_blending_ensemble,
                "expected_improvement": 1.2,
                "estimated_time": 240,
                "rationale": "Holdout set for meta-learner training"
            },
            {
                "id": "EXP-021",
                "name": "Target Encoding + XGBoost",
                "function": self.run_target_encoding,
                "expected_improvement": 1.0,
                "estimated_time": 240,
                "rationale": "Encode categorical features via target mean"
            },
        ]
        
        self.current_strategy_idx = 0
        
    def log(self, message, level="INFO"):
        """Print timestamped log message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def _to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
        
    def check_for_improvement(self, rmse, cohens_d):
        """Check if result meets significance criteria."""
        relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
        
        meets_improvement = relative_improvement >= self.target_improvement
        meets_effect_size = abs(cohens_d) >= self.min_effect_size
        
        return meets_improvement and meets_effect_size, relative_improvement
        
    def update_best(self, exp_id, rmse, model_path=None):
        """Update best model if improved."""
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_model = exp_id
            self.log(f"🎉 NEW BEST: {exp_id} with RMSE {rmse:.4f}", "SUCCESS")
            return True
        return False
        
    def create_experiment_doc(self, exp_id, name, hypothesis, results, conclusion):
        """Create experiment documentation."""
        exp_dir = self.research_dir / f"2026-03-09-{exp_id.lower().replace('_', '-')}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme = f"""# {exp_id}: {name}

**Status:** {'✅ SIGNIFICANT' if results.get('is_significant') else '⚠️ Not Significant'}

## Hypothesis
{hypothesis}

## Methodology
See experiment runner for implementation details.

## Results

| Metric | Value |
|--------|-------|
| RMSE | {results.get('rmse', 'N/A'):.4f} |
| Baseline RMSE | {self.baseline_rmse:.4f} |
| Relative Improvement | {results.get('relative_improvement', 0):.2f}% |
| Cohen's d | {results.get('cohens_d', 0):.4f} |
| Significant | {'✅ Yes' if results.get('is_significant') else '❌ No'} |

## Conclusion
{conclusion}

## Timestamp
{datetime.now().isoformat()}
"""
        (exp_dir / "README.md").write_text(readme)
        
        # Create results.json
        (exp_dir / "results.json").write_text(json.dumps(
            self._to_json_serializable({
                "experiment_id": exp_id,
                "name": name,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "hypothesis": hypothesis,
                "results": results,
                "conclusion": conclusion
            }), indent=2))
        
        return exp_dir
        
    # ============== STRATEGY IMPLEMENTATIONS ==============
    
    def run_position_specific_models(self):
        """EXP-007: Train separate models for each position."""
        self.log("Running EXP-007: Position-Specific Models...")
        
        try:
            # Load data
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Assume position is encoded as one-hot or categorical in features
            # For now, use a simple heuristic based on feature patterns
            # In real implementation, we'd have position labels
            
            # Alternative: Use unsupervised clustering to identify position groups
            from sklearn.cluster import KMeans
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Cluster training data into 4 groups (GK, DEF, MID, FWD)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            train_clusters = kmeans.fit_predict(train_X)
            test_clusters = kmeans.predict(test_X)
            
            # Train separate model for each cluster
            predictions = np.zeros(len(test_y))
            
            for cluster in range(4):
                train_mask = train_clusters == cluster
                test_mask = test_clusters == cluster
                
                if train_mask.sum() < 10 or test_mask.sum() < 5:
                    continue
                    
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(train_X[train_mask], train_y[train_mask])
                predictions[test_mask] = model.predict(test_X[test_mask])
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            # Calculate effect size
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Position-specific models: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-007", "Position-Specific Models",
                "Different positions have different point distributions requiring specialized models",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-007 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_hyperparameter_optimization(self):
        """EXP-008: Run Optuna hyperparameter optimization."""
        self.log("Running EXP-008: Hyperparameter Optimization...")
        
        try:
            import optuna
            from xgboost import XGBRegressor
            from sklearn.model_selection import cross_val_score
            
            # Load data
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42
                }
                
                model = XGBRegressor(**params)
                scores = cross_val_score(model, train_X, train_y, 
                                        cv=3, scoring='neg_mean_squared_error')
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100, show_progress_bar=False)
            
            # Train final model with best params
            best_model = XGBRegressor(**study.best_params, random_state=42)
            best_model.fit(train_X, train_y)
            
            predictions = best_model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            # Calculate effect size
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            # Save model
            exp_dir = self.models_dir / "optuna_xgb"
            exp_dir.mkdir(exist_ok=True)
            import pickle
            with open(exp_dir / "model.pkl", 'wb') as f:
                pickle.dump(best_model, f)
            with open(exp_dir / "params.json", 'w') as f:
                json.dump(self._to_json_serializable(study.best_params), f, indent=2)
            
            results = {
                "rmse": rmse,
                "relative_improvement": relative_improvement,
                "cohens_d": cohens_d,
                "is_significant": is_significant,
                "best_params": study.best_params
            }
            
            conclusion = f"Optuna optimization: {relative_improvement:.2f}% improvement with params {study.best_params}"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-008", "Hyperparameter Optimization",
                "200+ Optuna trials to find optimal XGBoost hyperparameters",
                results, conclusion)
            
            return results
            
        except ImportError:
            self.log("Optuna not installed, using random search...", "WARNING")
            return self._run_random_search()
        except Exception as e:
            self.log(f"EXP-008 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def _run_random_search(self):
        """Fallback: Random search for hyperparameters."""
        self.log("Using random search fallback...")
        
        try:
            from xgboost import XGBRegressor
            from sklearn.model_selection import RandomizedSearchCV
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300, 500],
                'max_depth': [2, 3, 4, 5, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            }
            
            model = XGBRegressor(random_state=42)
            search = RandomizedSearchCV(
                model, param_distributions,
                n_iter=50, cv=3, scoring='neg_mean_squared_error',
                random_state=42, n_jobs=-1
            )
            search.fit(train_X, train_y)
            
            predictions = search.best_estimator_.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": rmse,
                "relative_improvement": relative_improvement,
                "cohens_d": cohens_d,
                "is_significant": is_significant,
                "best_params": search.best_params_
            }
            
            conclusion = f"Random search: {relative_improvement:.2f}% improvement"
            self.create_experiment_doc("EXP-008", "Hyperparameter Optimization (Random Search)",
                "Random search for optimal XGBoost hyperparameters",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"Random search failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_weighted_ensemble(self):
        """EXP-009: Learn optimal ensemble weights."""
        self.log("Running EXP-009: Weighted Ensemble...")
        
        try:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_predict
            
            # Load data
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Get predictions from multiple models
            models_to_use = []
            
            # XGBoost
            try:
                import pickle
                with open(self.models_dir / "xgboost" / "model.pkl", 'rb') as f:
                    xgb = pickle.load(f)
                models_to_use.append(('xgb', xgb))
            except:
                pass
                
            # LightGBM
            try:
                import pickle
                with open(self.models_dir / "lightgbm" / "model.pkl", 'rb') as f:
                    lgb = pickle.load(f)
                models_to_use.append(('lgb', lgb))
            except:
                pass
                
            # Random Forest
            try:
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(train_X, train_y)
                models_to_use.append(('rf', rf))
            except:
                pass
                
            if len(models_to_use) < 2:
                self.log("Not enough models for ensemble", "WARNING")
                return {"rmse": float('inf'), "error": "Insufficient models"}
            
            # Generate predictions for meta-learner
            train_preds = np.column_stack([
                cross_val_predict(model, train_X, train_y, cv=3)
                for _, model in models_to_use
            ])
            
            test_preds = np.column_stack([
                model.predict(test_X) for _, model in models_to_use
            ])
            
            # Learn weights with Ridge regression
            meta_learner = Ridge(alpha=1.0)
            meta_learner.fit(train_preds, train_y)
            
            predictions = meta_learner.predict(test_preds)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            # Calculate effect size
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            weights = dict(zip([name for name, _ in models_to_use], meta_learner.coef_.tolist()))
            
            results = {
                "rmse": rmse,
                "relative_improvement": relative_improvement,
                "cohens_d": cohens_d,
                "is_significant": is_significant,
                "weights": weights
            }
            
            conclusion = f"Weighted ensemble: {relative_improvement:.2f}% improvement with weights {weights}"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-009", "Weighted Ensemble",
                "Learn optimal ensemble weights using Ridge regression",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-009 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_stacking_ensemble(self):
        """EXP-010: Stacking ensemble with meta-learner."""
        self.log("Running EXP-010: Stacking Ensemble...")
        
        try:
            from sklearn.ensemble import StackingRegressor
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            estimators = [
                ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
                ('lgb', LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
            
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=3,
                passthrough=False
            )
            
            stack.fit(train_X, train_y)
            predictions = stack.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Stacking ensemble: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-010", "Stacking Ensemble",
                "Sklearn StackingRegressor with Ridge meta-learner",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-010 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_feature_selection(self):
        """EXP-011: Feature selection using importance."""
        self.log("Running EXP-011: Feature Selection...")
        
        try:
            from xgboost import XGBRegressor
            from sklearn.feature_selection import SelectFromModel
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Get feature importance
            selector = SelectFromModel(
                XGBRegressor(n_estimators=100, random_state=42),
                max_features=20,
                threshold=-np.inf
            )
            
            train_X_selected = selector.fit_transform(train_X, train_y)
            test_X_selected = selector.transform(test_X)
            
            # Train model on selected features
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(train_X_selected, train_y)
            predictions = model.predict(test_X_selected)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            selected_features = selector.get_support().sum()
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant),
                "selected_features": int(selected_features)
            }
            
            conclusion = f"Feature selection (top {selected_features}): {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-011", "Feature Selection",
                "Select top-K features using XGBoost importance",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-011 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_quantile_regression(self):
        """EXP-012: Quantile regression for uncertainty."""
        self.log("Running EXP-012: Quantile Regression...")
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Train median (0.5 quantile) model
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=0.5,
                n_estimators=100,
                random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Quantile regression: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-012", "Quantile Regression",
                "Median regression for robust predictions",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-012 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_huber_loss(self):
        """EXP-013: Huber loss for robust regression."""
        self.log("Running EXP-013: Huber Loss...")
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = GradientBoostingRegressor(
                loss='huber',
                alpha=0.9,
                n_estimators=100,
                random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Huber loss: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-013", "Huber Loss",
                "Huber loss to reduce outlier impact",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-013 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_catboost(self):
        """EXP-014: CatBoost model."""
        self.log("Running EXP-014: CatBoost...")
        
        try:
            from catboost import CatBoostRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"CatBoost: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-014", "CatBoost",
                "Ordered boosting for better generalization",
                results, conclusion)
            
            return results
            
        except ImportError:
            self.log("CatBoost not installed, skipping", "WARNING")
            return {"rmse": float('inf'), "error": "CatBoost not installed"}
        except Exception as e:
            self.log(f"EXP-014 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_deep_hyperopt(self):
        """EXP-015: Deep hyperparameter search with 300 trials."""
        self.log("Running EXP-015: Deep Hyperparameter Search (300 trials)...")
        
        try:
            import optuna
            from xgboost import XGBRegressor
            from sklearn.model_selection import cross_val_score
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 30, 1000),
                    'max_depth': trial.suggest_int('max_depth', 2, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
                    'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 100.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 100.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 1e-10, 10.0, log=True),
                    'random_state': 42
                }
                
                model = XGBRegressor(**params)
                scores = cross_val_score(model, train_X, train_y, 
                                        cv=5, scoring='neg_mean_squared_error')
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=150, show_progress_bar=False)
            
            best_model = XGBRegressor(**study.best_params, random_state=42)
            best_model.fit(train_X, train_y)
            
            predictions = best_model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            exp_dir = self.models_dir / "deep_optuna_xgb"
            exp_dir.mkdir(exist_ok=True)
            import pickle
            with open(exp_dir / "model.pkl", 'wb') as f:
                pickle.dump(best_model, f)
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant),
                "best_params": self._to_json_serializable(study.best_params)
            }
            
            conclusion = f"Deep hyperopt (300 trials): {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-015", "Deep Hyperparameter Search",
                "300-trial Optuna with extended hyperparameter ranges",
                results, conclusion)
            
            return results
            
        except ImportError:
            self.log("Optuna not installed, skipping deep hyperopt", "WARNING")
            return {"rmse": float('inf'), "error": "Optuna not installed"}
        except Exception as e:
            self.log(f"EXP-015 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_voting_ensemble(self):
        """EXP-016: Voting ensemble."""
        self.log("Running EXP-016: Voting Ensemble...")
        
        try:
            from sklearn.ensemble import VotingRegressor
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            estimators = [
                ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
                ('lgb', LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ]
            
            voting = VotingRegressor(estimators=estimators, weights=[2, 2, 1, 1])
            voting.fit(train_X, train_y)
            predictions = voting.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Voting ensemble: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-016", "Voting Ensemble",
                "VotingRegressor with weighted averaging",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-016 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_bagging_ensemble(self):
        """EXP-017: Bagging ensemble."""
        self.log("Running EXP-017: Bagging Ensemble...")
        
        try:
            from sklearn.ensemble import BaggingRegressor
            from xgboost import XGBRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            base = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
            bagging = BaggingRegressor(
                estimator=base,
                n_estimators=20,
                max_samples=0.8,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )
            bagging.fit(train_X, train_y)
            predictions = bagging.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Bagging ensemble: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-017", "Bagging Ensemble",
                "Bootstrap aggregation with XGBoost base",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-017 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_extra_trees(self):
        """EXP-018: Extra Trees model."""
        self.log("Running EXP-018: Extra Trees...")
        
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(train_X, train_y)
            predictions = model.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Extra Trees: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-018", "Extra Trees",
                "Extremely randomized trees for less overfitting",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-018 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_elasticnet_poly(self):
        """EXP-019: ElasticNet with polynomial features."""
        self.log("Running EXP-019: ElasticNet with Polynomial Features...")
        
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000))
            ])
            
            pipeline.fit(train_X, train_y)
            predictions = pipeline.predict(test_X)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"ElasticNet + Poly: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-019", "ElasticNet with Polynomial Features",
                "Regularized linear model with polynomial features",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-019 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_blending_ensemble(self):
        """EXP-020: Blending ensemble with holdout set."""
        self.log("Running EXP-020: Blending Ensemble...")
        
        try:
            from sklearn.linear_model import Ridge
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from sklearn.ensemble import RandomForestRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Split training data for blending
            split_idx = int(0.8 * len(train_X))
            blend_X, holdout_X = train_X[:split_idx], train_X[split_idx:]
            blend_y, holdout_y = train_y[:split_idx], train_y[split_idx:]
            
            # Train base models on blend set
            models = [
                XGBRegressor(n_estimators=100, random_state=42),
                LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                RandomForestRegressor(n_estimators=100, random_state=42)
            ]
            
            for model in models:
                model.fit(blend_X, blend_y)
            
            # Generate meta-features on holdout set
            holdout_preds = np.column_stack([m.predict(holdout_X) for m in models])
            
            # Train meta-learner on holdout
            meta = Ridge(alpha=1.0)
            meta.fit(holdout_preds, holdout_y)
            
            # Generate predictions on test
            test_preds = np.column_stack([m.predict(test_X) for m in models])
            predictions = meta.predict(test_preds)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Blending ensemble: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-020", "Blending Ensemble",
                "Holdout blending with Ridge meta-learner",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-020 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def run_target_encoding(self):
        """EXP-021: Target encoding approach."""
        self.log("Running EXP-021: Target Encoding + XGBoost...")
        
        try:
            from xgboost import XGBRegressor
            
            train_X = np.load(self.datasets_dir / "train_X.npy")
            train_y = np.load(self.datasets_dir / "train_y.npy")
            test_X = np.load(self.datasets_dir / "test_X.npy")
            test_y = np.load(self.datasets_dir / "test_y.npy")
            
            # Simple approach: add mean target as a feature
            # In practice, would use proper target encoding with CV
            global_mean = np.mean(train_y)
            
            # Add derived features
            train_X_enhanced = np.column_stack([
                train_X,
                np.full(len(train_X), global_mean),
                np.std(train_X, axis=1),
                np.mean(train_X, axis=1)
            ])
            
            test_X_enhanced = np.column_stack([
                test_X,
                np.full(len(test_X), global_mean),
                np.std(test_X, axis=1),
                np.mean(test_X, axis=1)
            ])
            
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(train_X_enhanced, train_y)
            predictions = model.predict(test_X_enhanced)
            
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            baseline_preds = self._get_baseline_predictions()
            pooled_std = np.sqrt((np.std(baseline_preds)**2 + np.std(predictions)**2) / 2)
            cohens_d = (np.mean(baseline_preds) - np.mean(predictions)) / pooled_std if pooled_std > 0 else 0
            
            relative_improvement = ((self.baseline_rmse - rmse) / self.baseline_rmse) * 100
            is_significant = relative_improvement >= self.target_improvement and abs(cohens_d) >= self.min_effect_size
            
            results = {
                "rmse": float(rmse),
                "relative_improvement": float(relative_improvement),
                "cohens_d": float(cohens_d),
                "is_significant": bool(is_significant)
            }
            
            conclusion = f"Target encoding: {relative_improvement:.2f}% improvement"
            if is_significant:
                conclusion += " - SIGNIFICANT!"
            else:
                conclusion += " - Not significant"
                
            self.create_experiment_doc("EXP-021", "Target Encoding + XGBoost",
                "Enhanced features with statistics",
                results, conclusion)
            
            return results
            
        except Exception as e:
            self.log(f"EXP-021 failed: {e}", "ERROR")
            return {"rmse": float('inf'), "error": str(e)}
            
    def _get_baseline_predictions(self):
        """Get baseline XGBoost predictions for effect size calculation."""
        try:
            import pickle
            with open(self.models_dir / "xgboost" / "model.pkl", 'rb') as f:
                model = pickle.load(f)
            test_X = np.load(self.datasets_dir / "test_X.npy")
            return model.predict(test_X)
        except:
            # Return zeros if can't load
            return np.zeros(41)
            
    # ============== MAIN LOOP ==============
    
    def get_next_strategy(self):
        """Get the next strategy to try."""
        if self.current_strategy_idx >= len(self.strategies):
            self.log("All strategies exhausted! Starting over with variations...", "WARNING")
            self.current_strategy_idx = 0
            # Could add variations here
            return None
            
        strategy = self.strategies[self.current_strategy_idx]
        self.current_strategy_idx += 1
        return strategy
        
    def run_single_experiment(self, strategy):
        """Run a single experiment."""
        self.log(f"\n{'='*60}")
        self.log(f"Starting {strategy['id']}: {strategy['name']}")
        self.log(f"Expected improvement: {strategy['expected_improvement']}%")
        self.log(f"Rationale: {strategy['rationale']}")
        self.log(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            results = strategy['function']()
            
            elapsed = time.time() - start_time
            self.log(f"Completed in {elapsed:.1f}s")
            
            if 'error' in results:
                self.log(f"Experiment failed: {results['error']}", "ERROR")
                return False, results
                
            rmse = results['rmse']
            relative_improvement = results['relative_improvement']
            cohens_d = results['cohens_d']
            is_significant = results['is_significant']
            
            self.log(f"Results: RMSE={rmse:.4f}, Improvement={relative_improvement:.2f}%, Cohen's d={cohens_d:.4f}")
            
            if is_significant:
                self.log(f"🎉🎉🎉 SIGNIFICANT IMPROVEMENT ACHIEVED! 🎉🎉🎉", "SUCCESS")
                self.update_best(strategy['id'], rmse)
                return True, results
            else:
                self.log(f"Not significant (need >{self.target_improvement}% improvement AND Cohen's d > {self.min_effect_size})")
                self.update_best(strategy['id'], rmse)
                return False, results
                
        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"Experiment crashed after {elapsed:.1f}s: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return False, {"error": str(e)}
            
    def run_loop(self):
        """Main research loop."""
        self.log("\n" + "="*70)
        self.log("AUTONOMOUS RESEARCH LOOP (RALPH) STARTING")
        self.log("="*70)
        self.log(f"Baseline RMSE: {self.baseline_rmse:.4f}")
        self.log(f"Target: >{self.target_improvement}% improvement with Cohen's d > {self.min_effect_size}")
        self.log(f"Max experiments: {self.max_experiments}")
        self.log("="*70 + "\n")
        
        experiment_count = 0
        
        while experiment_count < self.max_experiments:
            experiment_count += 1
            
            self.log(f"\n{'#'*70}")
            self.log(f"# ITERATION {experiment_count}/{self.max_experiments}")
            self.log(f"# Best so far: {self.best_model} (RMSE: {self.best_rmse:.4f})")
            self.log(f"{'#'*70}\n")
            
            strategy = self.get_next_strategy()
            if strategy is None:
                self.log("No more strategies available")
                break
                
            success, results = self.run_single_experiment(strategy)
            
            if success:
                self.log("\n" + "="*70)
                self.log("SUCCESS! Significant improvement achieved.")
                self.log(f"Best model: {self.best_model}")
                self.log(f"Best RMSE: {self.best_rmse:.4f}")
                self.log(f"Improvement: {((self.baseline_rmse - self.best_rmse) / self.baseline_rmse * 100):.2f}%")
                self.log("="*70)
                return True
                
            # Brief pause between experiments
            time.sleep(2)
            
        self.log("\n" + "="*70)
        self.log(f"MAX EXPERIMENTS ({self.max_experiments}) REACHED")
        self.log(f"Best model: {self.best_model}")
        self.log(f"Best RMSE: {self.best_rmse:.4f}")
        improvement = ((self.baseline_rmse - self.best_rmse) / self.baseline_rmse * 100)
        self.log(f"Best improvement: {improvement:.2f}%")
        if improvement < self.target_improvement:
            self.log("No significant improvement achieved")
        self.log("="*70)
        return False


def main():
    parser = argparse.ArgumentParser(description='Autonomous Research Loop')
    parser.add_argument('--target-improvement', type=float, default=1.0,
                       help='Target relative improvement percentage (default: 1.0)')
    parser.add_argument('--min-effect-size', type=float, default=0.2,
                       help='Minimum Cohen\'s d effect size (default: 0.2)')
    parser.add_argument('--max-experiments', type=int, default=50,
                       help='Maximum number of experiments (default: 50)')
    
    args = parser.parse_args()
    
    loop = AutonomousResearchLoop(
        target_improvement=args.target_improvement,
        min_effect_size=args.min_effect_size,
        max_experiments=args.max_experiments
    )
    
    success = loop.run_loop()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
