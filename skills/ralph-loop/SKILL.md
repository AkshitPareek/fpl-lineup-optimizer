# Ralph Loop: Continuous Model Improvement Skill

> **RALPH** = **R**esearch **A**gent **L**oop for **P**erpetual **H**ypothesis testing

## Overview

The Ralph Loop is an autonomous research system designed to continuously search for better machine learning models through systematic experimentation. Named after the concept of continuous improvement, it runs experiments in a loop until a significant improvement is found.

## Purpose

In academic research and production ML systems, finding the next breakthrough model requires:
1. Systematic hypothesis generation
2. Rapid experimentation
3. Objective evaluation
4. Knowledge accumulation

The Ralph Loop automates this process, allowing parallel agents to explore different approaches simultaneously.

## Core Principles

### 1. Hypothesis-Driven Research
Every experiment starts with a clear hypothesis:
- "Adding position-specific features will improve FPL predictions by 2%"
- "Deep learning models will outperform tree-based models with 50k+ samples"

### 2. Rapid Iteration
- Fast experiments (< 30 minutes each)
- Parallel execution
- Quick failure detection

### 3. Objective Metrics
- Primary: Spearman correlation (ranking quality)
- Secondary: RMSE (prediction accuracy)
- Tertiary: Inference speed (production readiness)

### 4. Knowledge Persistence
- All results logged
- Failed hypotheses documented
- Successful approaches built upon

## Workflow

```
┌─────────────────┐
│  Set Champion   │◄─────────┐
│  (Baseline)     │          │
└────────┬────────┘          │
         │                   │
         ▼                   │
┌─────────────────┐          │
│  Generate       │          │
│  Hypotheses     │          │
└────────┬────────┘          │
         │                   │
         ▼                   │
┌─────────────────┐          │
│  Parallel       │          │
│  Experiments    │          │
└────────┬────────┘          │
         │                   │
    ┌────┴────┐              │
    ▼         ▼              │
┌──────┐  ┌──────┐           │
│Better│  │Not   │           │
│?     │  │Better│           │
└──┬───┘  └──┬───┘           │
   │         │                │
   ▼         ▼                │
┌──────┐  ┌──────┐           │
│Deploy│  │Log & │───────────┘
│New   │  │Loop  │
│Champ │  │      │
└──────┘  └──────┘
```

## Components

### 1. Champion Tracker
```python
class ChampionTracker:
    """Tracks the current best model."""
    
    def __init__(self, champion_path='models/champion/'):
        self.champion = self.load_champion()
        
    def is_better(self, candidate_metrics):
        """Check if candidate beats champion."""
        # Primary: Spearman improvement > 1%
        spearman_improvement = (
            candidate_metrics['spearman'] - 
            self.champion['spearman']
        ) / self.champion['spearman']
        
        # Secondary: RMSE improvement > 0.5%
        rmse_improvement = (
            self.champion['rmse'] - 
            candidate_metrics['rmse']
        ) / self.champion['rmse']
        
        return spearman_improvement > 0.01 or rmse_improvement > 0.005
```

### 2. Hypothesis Generator
```python
class HypothesisGenerator:
    """Generates experiment ideas based on current knowledge."""
    
    HYPOTHESIS_TEMPLATES = [
        {
            'name': 'feature_expansion',
            'description': 'Add {n} new features: {features}',
            'params': {'n': int, 'features': list}
        },
        {
            'name': 'model_architecture',
            'description': 'Try {model_type} with {params}',
            'params': {'model_type': str, 'params': dict}
        },
        {
            'name': 'data_augmentation',
            'description': 'Add {season} season data',
            'params': {'season': str}
        },
        {
            'name': 'ensemble_strategy',
            'description': 'Combine {models} with {weights}',
            'params': {'models': list, 'weights': list}
        }
    ]
    
    def generate(self, experiment_history, max_hypotheses=5):
        """Generate new hypotheses based on what worked/failed."""
        # Analyze past experiments
        successful = [e for e in experiment_history if e['improved']]
        failed = [e for e in experiment_history if not e['improved']]
        
        # Generate new hypotheses avoiding failed approaches
        hypotheses = []
        # ... generation logic ...
        return hypotheses
```

### 3. Experiment Runner
```python
class ExperimentRunner:
    """Runs a single experiment and evaluates results."""
    
    def run(self, hypothesis, timeout=1800):
        """
        Run experiment with timeout protection.
        
        Args:
            hypothesis: Experiment configuration
            timeout: Maximum seconds to run
            
        Returns:
            Experiment results or None if failed
        """
        try:
            # Run with timeout
            result = self._run_with_timeout(hypothesis, timeout)
            
            # Evaluate against champion
            result['improved'] = self.tracker.is_better(result['metrics'])
            
            return result
            
        except TimeoutError:
            return {'status': 'timeout', 'hypothesis': hypothesis}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'hypothesis': hypothesis}
```

### 4. Results Logger
```python
class ResultsLogger:
    """Persists all experiment results."""
    
    def log(self, result):
        """Log experiment result with metadata."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.generate_id(),
            'hypothesis': result['hypothesis'],
            'metrics': result['metrics'],
            'improved': result.get('improved', False),
            'artifacts': result.get('artifacts', [])
        }
        
        # Save to JSONL for append-only logging
        with open('research/experiments.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')
```

## Configuration

```yaml
# ralph-loop-config.yaml
loop:
  max_iterations: 100
  parallel_agents: 4
  timeout_per_experiment: 1800  # 30 minutes
  
champion:
  min_spearman_improvement: 0.01  # 1%
  min_rmse_improvement: 0.005     # 0.5%
  
hypotheses:
  generation_strategy: 'evolutionary'  # or 'random', 'grid'
  max_concurrent: 4
  
logging:
  level: 'INFO'
  destination: 'research/experiments.jsonl'
```

## Usage

### Basic Usage
```python
from ralph_loop import RalphLoop

# Initialize with current champion
loop = RalphLoop(
    champion_path='models/exp031_clean/',
    research_dir='research/'
)

# Run continuous improvement loop
loop.run(
    max_iterations=50,
    parallel_agents=4,
    timeout=1800
)
```

### Advanced: Custom Hypothesis
```python
# Define custom experiment
hypothesis = {
    'name': 'EXP-032-LSTM',
    'type': 'model_architecture',
    'description': 'LSTM with 64 hidden units, 5 GW history',
    'config': {
        'model': 'LSTM',
        'hidden_units': 64,
        'sequence_length': 5,
        'features': ['form_3gw', 'value', 'was_home']
    }
}

# Run single experiment
result = loop.run_single(hypothesis)
```

### As Agent Task
```bash
# Launch Ralph Loop as background agent
python -m ralph_loop.agent \
    --champion models/exp031_clean/ \
    --output research/exp032-search/ \
    --max-iterations 100 \
    --parallel 4
```

## Success Criteria

A new champion is crowned when:

1. **Spearman Improvement > 1%**
   - Better player ranking
   - Primary metric for FPL

2. **RMSE Improvement > 0.5%**
   - More accurate predictions
   - Secondary metric

3. **Statistical Significance**
   - p < 0.05 on test set
   - Cross-validation stability

4. **Production Ready**
   - Inference < 100ms
   - Memory < 100MB
   - No data leakage

## Experiment Ideas (Hypothesis Bank)

### Feature Engineering
- [ ] Add fixture difficulty ratings (FDR)
- [ ] Add team attack/defense strength
- [ ] Add player ownership percentage trends
- [ ] Add bookmaker odds as features
- [ ] Add weather conditions

### Model Architectures
- [ ] LSTM for time-series patterns
- [ ] Transformer with attention
- [ ] TabNet for tabular data
- [ ] XGBoost with hyperopt
- [ ] CatBoost (handles categoricals well)

### Ensemble Strategies
- [ ] Position-specific ensembles
- [ ] Time-weighted averaging
- [ ] Stacking with meta-learner
- [ ] Bayesian model averaging

### Data Strategies
- [ ] Add 2024-25 season
- [ ] Weight recent seasons higher
- [ ] Player-level normalization
- [ ] Position-stratified sampling

## Integration with Agent System

The Ralph Loop integrates with the parallel agent system:

```
Agent 1: Production Testing
├── Tests EXP-031 on live FPL data
├── Compares picks with EXP-030
└── Reports ranking quality

Agent 2: Data Collection
├── Fetches 2024-25 season data
├── Updates aggregated dataset
└── Triggers retraining

Agents 3-4: Ralph Loop
├── Generate hypotheses
├── Run parallel experiments
├── Evaluate vs EXP-031
└── Report new champions
```

## Output Structure

```
research/
├── ralph-loop/
│   ├── experiments.jsonl        # All experiment logs
│   ├── champions.json           # Champion progression
│   ├── hypotheses.json          # Generated hypotheses
│   └── reports/
│       ├── weekly-summary.md
│       └── best-experiments.md
│
├── exp032-search/               # Current search space
│   ├── EXP-032-001-lstm/
│   ├── EXP-032-002-xgboost/
│   └── ...
│
└── agents/
    ├── agent-1-production/
    ├── agent-2-data-collection/
    ├── agent-3-ralph-loop/
    └── agent-4-ralph-loop/
```

## Research Goals

### EXP-032 Target
- **Spearman:** 0.75+ (from 0.7263)
- **RMSE:** 1.40 or lower
- **Approach:** Deep learning or feature expansion

### EXP-033 Target
- **Spearman:** 0.78+
- **Approach:** Ensemble of best models

### EXP-034+ Target
- **Spearman:** 0.80+
- **Approach:** Novel architecture or external data

## Best Practices

1. **Start Simple**
   - Test simple hypotheses first
   - Build complexity gradually

2. **Fail Fast**
   - 30-minute timeout per experiment
   - Kill underperforming approaches quickly

3. **Document Everything**
   - Why the hypothesis was tested
   - What was learned
   - What to try next

4. **Share Knowledge**
   - Successful approaches inform new hypotheses
   - Failed approaches prevent repetition

## References

- AutoML Research (Google)
- Hyperparameter Optimization (Optuna)
- Neural Architecture Search (NAS)
- Bayesian Optimization

---

*The Ralph Loop: Never stop searching for better.*
