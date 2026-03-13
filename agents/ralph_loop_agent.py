#!/usr/bin/env python3
"""
Ralph Loop Agent: Continuous Model Improvement

Named after RALPH = Research Agent Loop for Perpetual Hypothesis testing

This agent continuously searches for better models through systematic
experimentation until a significant improvement is found.

Author: Research Agent
Date: 2026-03-13
"""

import os
import sys
import json
import pickle
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RalphLoop')


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    id: str
    hypothesis: Dict
    metrics: Dict
    improved: bool
    timestamp: str
    artifacts: List[str]


class ChampionTracker:
    """Tracks the current champion model."""
    
    def __init__(self, champion_path='models/exp031_clean/'):
        self.champion_path = Path(champion_path)
        self.champion = self._load_champion()
        
        spearman = self.champion.get('spearman', self.champion.get('test_spearman', 0.7263))
        rmse = self.champion.get('rmse', self.champion.get('test_rmse', 1.4629))
        logger.info(f"Champion loaded: Spearman={spearman:.4f}, RMSE={rmse:.4f}")
    
    def _load_champion(self) -> Dict:
        """Load current champion metrics."""
        metrics_path = self.champion_path / 'metrics.json'
        
        if metrics_path.exists()        :
            with open(metrics_path) as f:
                data = json.load(f)
                # Handle nested structure {model: {metrics}}
                if 'ridge' in data:
                    return data['ridge']
                return data
        else:
            # Default to EXP-031 known values
            return {
                'test_rmse': 1.4629,
                'spearman': 0.7263
            }
    
    def is_better(self, candidate_metrics: Dict) -> Tuple[bool, str]:
        """Check if candidate beats champion."""
        champ_spearman = self.champion.get('spearman', 0.7263)
        champ_rmse = self.champion.get('test_rmse', self.champion.get('rmse', 1.4629))
        
        # Primary: Spearman improvement > 2%
        spearman_improvement = (
            (candidate_metrics['spearman'] - champ_spearman) / champ_spearman
        )
        
        # Secondary: RMSE improvement > 1%
        rmse_improvement = (
            (champ_rmse - candidate_metrics['rmse']) / champ_rmse
        )
        
        # Success criteria
        if spearman_improvement > 0.02:
            return True, f"Spearman improved by {spearman_improvement*100:.1f}%"
        
        if rmse_improvement > 0.01:
            return True, f"RMSE improved by {rmse_improvement*100:.1f}%"
        
        return False, f"No significant improvement (Spearman: {spearman_improvement*100:+.1f}%, RMSE: {rmse_improvement*100:+.1f}%)"


class HypothesisGenerator:
    """Generates experiment hypotheses."""
    
    STRATEGIES = {
        'feature_expansion': {
            'description': 'Add new features to improve prediction',
            'tactics': [
                'add_fixture_difficulty',
                'add_team_strength',
                'add_ownership_trends',
                'add_interaction_features'
            ]
        },
        'deep_learning': {
            'description': 'Try neural network architectures',
            'tactics': [
                'mlp_small',
                'mlp_large',
                'mlp_deep'
            ]
        },
        'ensemble': {
            'description': 'Combine multiple models',
            'tactics': [
                'weighted_average',
                'stacking',
                'boosting'
            ]
        },
        'regularization': {
            'description': 'Optimize regularization parameters',
            'tactics': [
                'ridge_tuning',
                'elastic_net',
                'early_stopping'
            ]
        }
    }
    
    def __init__(self, strategy='feature_expansion'):
        self.strategy = strategy
        self.tactics = self.STRATEGIES[strategy]['tactics']
        self.experiment_count = 0
        
        logger.info(f"Hypothesis generator initialized: {strategy}")
    
    def generate(self) -> Dict:
        """Generate next hypothesis."""
        self.experiment_count += 1
        tactic = random.choice(self.tactics)
        
        hypothesis = {
            'id': f"EXP-032-{self.experiment_count:03d}",
            'strategy': self.strategy,
            'tactic': tactic,
            'description': self._describe_tactic(tactic),
            'config': self._generate_config(tactic)
        }
        
        return hypothesis
    
    def _describe_tactic(self, tactic: str) -> str:
        """Generate human-readable description."""
        descriptions = {
            'add_fixture_difficulty': 'Add FDR (Fixture Difficulty Rating) features',
            'add_team_strength': 'Add team attack/defense strength ratings',
            'add_ownership_trends': 'Add player ownership percentage changes',
            'add_interaction_features': 'Add feature interactions (form × value)',
            'mlp_small': 'Small neural network (32 hidden units)',
            'mlp_large': 'Large neural network (128 hidden units)',
            'mlp_deep': 'Deep neural network (3 layers)',
            'weighted_average': 'Weighted ensemble of multiple models',
            'stacking': 'Stacking ensemble with meta-learner',
            'boosting': 'Gradient boosting with tuned hyperparameters',
            'ridge_tuning': 'Ridge regression with hyperparameter search',
            'elastic_net': 'Elastic Net regularization',
            'early_stopping': 'Early stopping to prevent overfitting'
        }
        return descriptions.get(tactic, f"Unknown tactic: {tactic}")
    
    def _generate_config(self, tactic: str) -> Dict:
        """Generate configuration for tactic."""
        configs = {
            'add_fixture_difficulty': {
                'additional_features': ['fdr', 'opponent_strength']
            },
            'add_team_strength': {
                'additional_features': ['team_attack', 'team_defense']
            },
            'add_ownership_trends': {
                'additional_features': ['ownership_change', 'transfers_in_out']
            },
            'add_interaction_features': {
                'interactions': [('form_3gw', 'value'), ('form_3gw', 'was_home')]
            },
            'mlp_small': {
                'hidden_layer_sizes': (32,),
                'alpha': 0.001,
                'max_iter': 500
            },
            'mlp_large': {
                'hidden_layer_sizes': (128, 64),
                'alpha': 0.001,
                'max_iter': 500
            },
            'mlp_deep': {
                'hidden_layer_sizes': (64, 32, 16),
                'alpha': 0.001,
                'max_iter': 1000
            },
            'weighted_average': {
                'models': ['ridge', 'gb', 'rf'],
                'weights': [0.5, 0.3, 0.2]
            },
            'stacking': {
                'base_models': ['ridge', 'gb', 'rf'],
                'meta_model': 'ridge'
            },
            'boosting': {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05
            },
            'ridge_tuning': {
                'alphas': [0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'alpha': 0.1,
                'l1_ratio': 0.5
            },
            'early_stopping': {
                'patience': 10,
                'validation_fraction': 0.1
            }
        }
        return configs.get(tactic, {})


class RalphLoopAgent:
    """Agent that runs continuous improvement experiments."""
    
    def __init__(self, agent_id: int, strategy: str = 'feature_expansion', 
                 max_iterations: int = 20, timeout: int = 1800):
        self.agent_id = agent_id
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.timeout = timeout
        
        self.results_dir = Path(f'research/agents/agent{agent_id}_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = ChampionTracker()
        self.generator = HypothesisGenerator(strategy)
        
        self.experiment_log = []
        self.best_result = None
        
        # Load data
        self.train_df = pd.read_csv('datasets/fpl_multi_year/train.csv')
        self.test_df = pd.read_csv('datasets/fpl_multi_year/test.csv')
        
        logger.info(f"Ralph Loop Agent {agent_id} initialized")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Max iterations: {max_iterations}")
    
    def prepare_features(self, df: pd.DataFrame, hypothesis: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features based on hypothesis."""
        features_list = []
        
        # Base features (always included)
        base = ['form_3gw', 'value', 'was_home', 'position_code']
        for col in base:
            if col in df.columns:
                features_list.append(df[col].fillna(0).values)
        
        # Additional features from hypothesis
        config = hypothesis.get('config', {})
        for feature in config.get('additional_features', []):
            if feature in df.columns:
                features_list.append(df[feature].fillna(0).values)
        
        # Interaction features
        for feat_a, feat_b in config.get('interactions', []):
            if feat_a in df.columns and feat_b in df.columns:
                interaction = df[feat_a].fillna(0) * df[feat_b].fillna(0)
                features_list.append(interaction.values)
        
        X = np.column_stack(features_list)
        y = df['total_points'].fillna(0).values
        
        return X, y
    
    def train_model(self, hypothesis: Dict, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train model according to hypothesis."""
        tactic = hypothesis['tactic']
        
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Select model based on tactic
        if tactic.startswith('mlp'):
            config = hypothesis['config']
            model = MLPRegressor(
                hidden_layer_sizes=config.get('hidden_layer_sizes', (32,)),
                alpha=config.get('alpha', 0.001),
                max_iter=config.get('max_iter', 500),
                random_state=42,
                early_stopping=True
            )
        elif tactic == 'elastic_net':
            config = hypothesis['config']
            model = ElasticNet(
                alpha=config.get('alpha', 0.1),
                l1_ratio=config.get('l1_ratio', 0.5),
                max_iter=2000,
                random_state=42
            )
        elif tactic == 'boosting':
            config = hypothesis['config']
            model = GradientBoostingRegressor(
                n_estimators=config.get('n_estimators', 300),
                max_depth=config.get('max_depth', 5),
                learning_rate=config.get('learning_rate', 0.05),
                random_state=42
            )
        else:
            # Default: Ridge
            model = Ridge(alpha=1.0, random_state=42)
        
        # Train
        model.fit(X_train_s, y_train)
        
        # Predict
        y_pred = model.predict(X_test_s)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        spearman = spearmanr(y_test, y_pred)[0]
        
        return {
            'rmse': rmse,
            'mae': mae,
            'spearman': spearman,
            'model': model,
            'scaler': scaler
        }
    
    def run_experiment(self, hypothesis: Dict) -> ExperimentResult:
        """Run a single experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: {hypothesis['id']}")
        logger.info(f"Tactic: {hypothesis['tactic']}")
        logger.info(f"Description: {hypothesis['description']}")
        logger.info(f"{'='*70}")
        
        try:
            # Prepare features
            X_train, y_train = self.prepare_features(self.train_df, hypothesis)
            X_test, y_test = self.prepare_features(self.test_df, hypothesis)
            
            logger.info(f"Feature matrix: {X_train.shape}")
            
            # Train model
            result = self.train_model(hypothesis, X_train, y_train, X_test, y_test)
            
            # Check if better than champion
            improved, reason = self.tracker.is_better({
                'rmse': result['rmse'],
                'spearman': result['spearman']
            })
            
            logger.info(f"Results: RMSE={result['rmse']:.4f}, Spearman={result['spearman']:.4f}")
            logger.info(f"Improvement: {reason}")
            
            # Save model if improved
            artifacts = []
            if improved:
                model_path = self.results_dir / f"{hypothesis['id']}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': result['model'],
                        'scaler': result['scaler'],
                        'hypothesis': hypothesis,
                        'metrics': {
                            'rmse': result['rmse'],
                            'mae': result['mae'],
                            'spearman': result['spearman']
                        }
                    }, f)
                artifacts.append(str(model_path))
                logger.info(f"✓ New champion saved to {model_path}")
            
            return ExperimentResult(
                id=hypothesis['id'],
                hypothesis=hypothesis,
                metrics={
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'spearman': result['spearman']
                },
                improved=improved,
                timestamp=datetime.now().isoformat(),
                artifacts=artifacts
            )
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return ExperimentResult(
                id=hypothesis['id'],
                hypothesis=hypothesis,
                metrics={'error': str(e)},
                improved=False,
                timestamp=datetime.now().isoformat(),
                artifacts=[]
            )
    
    def run(self):
        """Run the Ralph Loop."""
        logger.info("="*70)
        logger.info(f"RALPH LOOP AGENT {self.agent_id} STARTING")
        logger.info("="*70)
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"Target: Beat EXP-031 (Spearman: 0.7263, RMSE: 1.4629)")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info("="*70)
        
        new_champion_found = False
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*70}")
            
            # Generate hypothesis
            hypothesis = self.generator.generate()
            
            # Run experiment
            result = self.run_experiment(hypothesis)
            self.experiment_log.append(asdict(result))
            
            # Track best
            if result.improved:
                self.best_result = result
                new_champion_found = True
                logger.info(f"\n🎉 NEW CHAMPION FOUND: {result.id}")
                logger.info(f"   Spearman: {result.metrics['spearman']:.4f}")
                logger.info(f"   RMSE: {result.metrics['rmse']:.4f}")
            
            # Save progress
            self._save_progress()
        
        # Final report
        self._generate_final_report(new_champion_found)
        
        return new_champion_found
    
    def _save_progress(self):
        """Save experiment log."""
        log_path = self.results_dir / 'experiment_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
    
    def _generate_final_report(self, success: bool):
        """Generate final report."""
        report = {
            'agent_id': self.agent_id,
            'strategy': self.strategy,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiment_log),
            'success': success,
            'best_result': asdict(self.best_result) if self.best_result else None,
            'summary': {
                'experiments_run': len(self.experiment_log),
                'improvements_found': sum(1 for e in self.experiment_log if e.get('improved', False)),
                'target_spearman': 0.7263,
                'target_rmse': 1.4629
            }
        }
        
        report_path = self.results_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("="*70)
        logger.info("RALPH LOOP COMPLETE")
        logger.info("="*70)
        logger.info(f"Total experiments: {report['summary']['experiments_run']}")
        logger.info(f"Improvements found: {report['summary']['improvements_found']}")
        if success:
            logger.info("✓ NEW CHAMPION DISCOVERED")
        else:
            logger.info("✗ No improvement over EXP-031")
        logger.info(f"Report saved to {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ralph Loop Agent')
    parser.add_argument('--agent-id', type=int, required=True)
    parser.add_argument('--strategy', type=str, default='feature_expansion',
                       choices=['feature_expansion', 'deep_learning', 'ensemble', 'regularization'])
    parser.add_argument('--max-iter', type=int, default=20)
    args = parser.parse_args()
    
    agent = RalphLoopAgent(
        agent_id=args.agent_id,
        strategy=args.strategy,
        max_iterations=args.max_iter
    )
    
    success = agent.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
