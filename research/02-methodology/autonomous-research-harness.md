# Autonomous Research Harness

> **Status:** ✅ Implemented  
> **Version:** 1.0  
> **Purpose:** Self-driving experiment runner with rigorous improvement tracking

---

## 🎯 The Core Question: How Do We Know If We're Improving?

This is THE critical question in autonomous research. Without human judgment, how do we know if experiments are actually working?

### Answer: Multi-Layer Validation

We use **4 layers of validation** to ensure we only accept real improvements:

```
Layer 1: Statistical Significance (p < 0.05)
Layer 2: Effect Size (Cohen's d > 0.2)
Layer 3: Relative Improvement (> 1%)
Layer 4: Trajectory Analysis (trend detection)
```

Only experiments passing ALL 4 layers are considered "improvements."

---

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                 AUTONOMOUS RESEARCH HARNESS                 │
├─────────────────────────────────────────────────────────────┤
│  Strategy Library                                           │
│  └── 10 pre-defined improvement strategies                  │
│      ├── lstm_attention                                     │
│      ├── position_models                                    │
│      ├── hyperopt_xgboost                                   │
│      └── ...                                                │
├─────────────────────────────────────────────────────────────┤
│  Research State Manager                                     │
│  ├── Tracks all experiments                                 │
│  ├── Maintains best model                                   │
│  ├── Counts failures                                        │
│  └── Detects plateaus                                       │
├─────────────────────────────────────────────────────────────┤
│  Experiment Runner                                          │
│  ├── Implements strategy                                    │
│  ├── Runs validation                                        │
│  ├── Performs A/B testing                                   │
│  └── Auto-documents                                         │
├─────────────────────────────────────────────────────────────┤
│  Improvement Tracker                                        │
│  ├── Visualizes trajectory                                  │
│  ├── Analyzes trends                                        │
│  ├── Calculates metrics                                     │
│  └── Recommends continuation                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 How Improvement Detection Works

### The 4-Layer Validation

#### Layer 1: Statistical Significance

**Question:** Is the difference real or just noise?

**Test:** Paired t-test between baseline and new model errors

**Threshold:** p < 0.05

**Example:**
```
Baseline RMSE: 2.500
New Model RMSE: 2.380
p-value: 0.003

Result: ✅ SIGNIFICANT (p < 0.05)
```

#### Layer 2: Effect Size

**Question:** Is the improvement practically meaningful?

**Test:** Cohen's d

**Threshold:** d > 0.2 (small but real)

**Interpretation:**
| d | Effect | Meaning |
|---|--------|---------|
| < 0.2 | Negligible | Not worth adopting |
| 0.2-0.5 | Small | Real but modest |
| 0.5-0.8 | Medium | Meaningful |
| > 0.8 | Large | Substantial |

**Example:**
```
Cohen's d: 0.42
Interpretation: ✅ MEDIUM EFFECT
```

#### Layer 3: Relative Improvement

**Question:** Is the improvement big enough to matter?

**Test:** Relative change in primary metric

**Threshold:** > 1% improvement

**Example:**
```
Baseline: 2.500
New: 2.380
Relative: (2.500 - 2.380) / 2.500 = 4.8%

Result: ✅ ABOVE 1% THRESHOLD
```

#### Layer 4: Trajectory Analysis

**Question:** Are we on an improving trend or just lucky?

**Test:** Pattern analysis over last N runs

**Indicators:**
- **Improving:** 3+ improvements in last 5 runs
- **Plateau:** 0 improvements in last 5 runs
- **Degrading:** Performance worsening

---

## 🎛️ Usage

### Quick Start

```bash
# Run 10 experiments autonomously
python backend/scripts/autoresearch_harness.py run --auto --max-runs 10

# Check status anytime
python backend/scripts/autoresearch_harness.py status

# View improvement dashboard
python backend/scripts/improvement_tracker.py dashboard

# Generate visualizations
python backend/scripts/improvement_tracker.py visualize

# Full analysis
python backend/scripts/improvement_tracker.py analyze
```

### Step-by-Step

#### 1. Initialize State

The harness automatically loads the baseline:
```
Loading state: 0 previous experiments
Loaded baseline: RMSE=2.29 (Ensemble)
```

#### 2. Run Autonomous Mode

```bash
python backend/scripts/autoresearch_harness.py run --auto --max-runs 10
```

**What happens:**
1. Harness selects best untried strategy
2. Runs experiment with full validation
3. Performs A/B test against baseline
4. Records results
5. Updates state
6. Repeats until max runs or stopping criteria

**Output:**
```
======================================================================
AUTONOMOUS RESEARCH HARNESS STARTING
Target: 10 experiments
======================================================================

============================================================
Starting Experiment 1: log_transform
Hypothesis: Log transform will reduce impact of haul games
============================================================

Step 1: Recording baseline metrics...
Step 2: Implementing strategy: log_transform
Step 3: Running validation suite...
Step 4: Running A/B test...
Step 5: Auto-documenting...

Experiment 1 completed: completed
🌟 IMPROVEMENT DETECTED!
   RMSE: 2.290 → 2.234
   p-value: 0.023, Effect size: 0.38

Pausing 5 seconds before next run...

============================================================
PROGRESS: 1/10 runs
Best RMSE: 2.234
Improvements: 1/1
============================================================
```

#### 3. Monitor Progress

**Dashboard View:**
```bash
python backend/scripts/improvement_tracker.py dashboard
```

Output:
```
╔════════════════════════════════════════════════════════════════════╗
║                 AUTOFPL RESEARCH DASHBOARD                         ║
╠════════════════════════════════════════════════════════════════════╣
║  📊 KEY METRICS                                                    ║
║     Runs:   5    Improvements:   2    Rate:  40.0%                 ║
╠════════════════════════════════════════════════════════════════════╣
║  📈 PERFORMANCE                                                    ║
║     Initial RMSE:  2.2900                                          ║
║     Current RMSE:  2.1500    (+6.11%)                              ║
╠════════════════════════════════════════════════════════════════════╣
║  📊 STATUS                                                         ║
║     Trend: 📈 improving  Confidence: 60.0%                         ║
╠════════════════════════════════════════════════════════════════════╣
║  🤖 RECOMMENDATION                                                 ║
║     ✅ CONTINUE: Trend improving                                   ║
╚════════════════════════════════════════════════════════════════════╝
```

#### 4. Analyze Results

```bash
python backend/scripts/improvement_tracker.py analyze
```

**Detailed Analysis:**
```
======================================================================
IMPROVEMENT ANALYSIS REPORT
======================================================================

📊 OVERALL PROGRESS
  Total Experiments: 5
  Successful Improvements: 2
  Failed Experiments: 1
  Improvement Rate: 40.0%

📈 PERFORMANCE TRAJECTORY
  Initial RMSE: 2.2900
  Current Best RMSE: 2.1500
  Total Improvement: 6.11%
  Improvement per Run: 1.22%

📉 STATISTICAL METRICS
  Average Effect Size: 0.385
  Average P-value: 0.0423
  Significance Rate: 60.0%

🎯 STRATEGY PERFORMANCE
  Best Strategy: log_transform
  Success Rates by Strategy:
    ✅ log_transform: 100.0%
    ⚠️ polynomial_features: 33.3%
    ❌ lstm_attention: 0.0%

📊 RECENT TREND
  Trend: 📈 improving
  Confidence: 60.0%

🤖 RECOMMENDATION
  ✅ CONTINUE: Trend improving
```

---

## 🛑 Stopping Criteria

The harness automatically stops when any of these occur:

### 1. Maximum Runs Reached
```
STOPPING RESEARCH: Maximum runs (10) reached
```

### 2. Too Many Consecutive Failures
```
STOPPING RESEARCH: Too many consecutive failures (3)
```

### 3. Improvement Plateau
```
STOPPING RESEARCH: Improvement plateau (5 runs without improvement)
```

### 4. Convergence Detected
```
STOPPING RESEARCH: Converged (CV=0.003)
```

### 5. Low Improvement Rate
```
STOPPING RESEARCH: Low improvement rate (5.0%) after 10 runs
```

---

## 📊 Improvement Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Improvement Rate** | % of experiments that improve | > 30% |
| **Total Improvement** | Cumulative RMSE reduction | > 5% |
| **Effect Size** | Cohen's d of improvements | > 0.3 |
| **Significance Rate** | % of experiments with p < 0.05 | > 50% |

### Trajectory Metrics

| Metric | Description | Good Sign |
|--------|-------------|-----------|
| **Recent Trend** | Direction of last 5 runs | "improving" |
| **Trend Confidence** | Certainty of trend | > 60% |
| **Runs Without Improvement** | Streak of failures | < 3 |

---

## 🗂️ Strategy Library

### Available Strategies

| Strategy | Risk | Expected Gain | Priority |
|----------|------|---------------|----------|
| log_transform | Low | 0.05-0.1 RMSE | 1 |
| feature_selection | Low | 0.02-0.05 RMSE | 2 |
| optimized_ensemble | Low | 0.05-0.1 RMSE | 3 |
| interaction_features | Low | 0.03-0.1 RMSE | 4 |
| hyperopt_xgboost | Low-Med | 0.05-0.1 RMSE | 5 |
| position_models | Medium | 0.05-0.15 RMSE | 6 |
| polynomial_features | Medium | 0.05-0.15 RMSE | 7 |
| temporal_features | Medium | 0.03-0.08 RMSE | 8 |
| weighted_loss | Medium | Top-k improvement | 9 |
| lstm_attention | High | 0.1-0.2 RMSE | 10 |

### Strategy Selection

The harness selects strategies intelligently:

1. **Prioritizes** low-risk, high-success-rate strategies first
2. **Avoids** strategies already tried
3. **Adapts** based on what's working
4. **Retries** with variations if plateau detected

---

## 📝 Auto-Documentation

Every experiment is automatically documented:

```markdown
# EXP-003: log_transform

**Status:** ✅ Improvement  
**Date:** 2026-03-09  
**Strategy:** log_transform  
**Duration:** 3.5 minutes

## Hypothesis

Log transform will reduce impact of haul games

## Methodology

Train on log(points + 1) to reduce outlier impact

## Results

### Metrics Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| RMSE | 2.2900 | 2.2345 | -0.0555 |
| MAE | 1.7600 | 1.7200 | -0.0400 |

### Statistical Analysis

- **p-value:** 0.0234 (Significant ✅)
- **Effect Size (Cohen's d):** 0.382
- **95% CI:** [-0.1023, -0.0087]

## Conclusion

**IMPROVEMENT DETECTED** - This strategy improved model performance.

## Next Steps

Adopt this improvement and update baseline.

---

*Auto-generated by AutoResearch Harness v1.0*
```

---

## 🔄 State Management

### State File Location

```
research/06-artifacts/autoresearch/state.json
```

### Contents

```json
{
  "experiments": [
    {
      "run_id": 1,
      "timestamp": "2026-03-09T10:30:00",
      "strategy": "log_transform",
      "status": "completed",
      "is_improvement": true,
      "metrics_before": {"rmse": 2.29},
      "metrics_after": {"rmse": 2.23},
      "p_value": 0.023,
      "effect_size": 0.38
    }
  ],
  "current_run": 1,
  "best_run_id": 1,
  "best_metrics": {"rmse": 2.23},
  "consecutive_failures": 0,
  "runs_without_improvement": 0
}
```

---

## 🎨 Visualizations

### Improvement Trajectory Plot

![Trajectory](../06-artifacts/autoresearch/plots/improvement_trajectory.png)

**Shows:**
- RMSE over time
- Highlighted improvements
- Baseline reference
- Cumulative best

### Detailed Analysis Plot

![Analysis](../06-artifacts/autoresearch/plots/detailed_analysis.png)

**Shows:**
- P-value distribution
- Effect sizes
- Duration vs improvement
- Rolling success rate

---

## 🧪 Example Session

### Session Log

```bash
$ python backend/scripts/autoresearch_harness.py run --auto --max-runs 5

======================================================================
AUTONOMOUS RESEARCH HARNESS STARTING
Target: 5 experiments
======================================================================

============================================================
Starting Experiment 1: log_transform
Hypothesis: Log transform will reduce impact of haul games
============================================================
...
Experiment 1 completed: completed
🌟 IMPROVEMENT DETECTED!
   RMSE: 2.290 → 2.234
   p-value: 0.023, Effect size: 0.38

============================================================
Starting Experiment 2: feature_selection
Hypothesis: Removing redundant features will reduce noise
============================================================
...
Experiment 2 completed: completed
❌ NO IMPROVEMENT
   RMSE: 2.234 → 2.241
   p-value: 0.412 (not significant)

============================================================
Starting Experiment 3: optimized_ensemble
Hypothesis: Learned weights will outperform uniform
============================================================
...
Experiment 3 completed: completed
🌟 IMPROVEMENT DETECTED!
   RMSE: 2.234 → 2.198
   p-value: 0.018, Effect size: 0.42
...

STOPPING RESEARCH: Maximum runs (5) reached

======================================================================
FINAL RESEARCH REPORT
======================================================================

Total Experiments: 5
Successful Improvements: 2
Failures: 0

FINAL BEST MODEL: Run 3
  rmse: 2.198
  mae: 1.712
  spearman: 0.431

Full report saved to: research/06-artifacts/autoresearch/final_report.json
```

---

## 📋 Checklist: How to Know We're Improving

### ✅ Definite Signs of Improvement

- [ ] RMSE consistently decreasing across runs
- [ ] Improvement rate > 30%
- [ ] Average effect size > 0.3
- [ ] Recent trend = "improving"
- [ ] P-values < 0.05 for improvements

### ⚠️ Warning Signs

- [ ] 3+ consecutive failures
- [ ] Improvement rate < 20%
- [ ] Trend = "plateau"
- [ ] Effect sizes < 0.2
- [ ] High p-values (> 0.1)

### 🛑 Stop If

- [ ] Maximum runs reached
- [ ] 3 consecutive failures
- [ ] 5 runs without improvement
- [ ] CV < 0.01 (converged)
- [ ] Improvement rate < 10% after 10 runs

---

## 🔧 Configuration

### HarnessConfig Options

```python
HarnessConfig(
    max_runs=10,                    # Maximum experiments
    min_runs=3,                     # Minimum before stopping
    min_effect_size=0.2,            # Cohen's d threshold
    significance_level=0.05,        # P-value threshold
    required_confidence=0.95,       # Confidence interval
    max_consecutive_failures=3,     # Stop after N failures
    improvement_plateau_window=5,   # Stop after N no-improvement runs
    min_relative_improvement=0.01,  # 1% minimum improvement
    max_training_time_minutes=30,   # Timeout per experiment
    require_validation=True         # Must pass validation
)
```

---

## 🚀 Advanced Usage

### Run Specific Strategy

```bash
python backend/scripts/autoresearch_harness.py run --strategy lstm_attention
```

### Resume Interrupted Run

```bash
# State is automatically saved
# Just run again, it will continue from where it left off
python backend/scripts/autoresearch_harness.py run --auto --max-runs 10
```

### Custom Strategy

Add to `StrategyLibrary.STRATEGIES`:

```python
'my_strategy': {
    'name': 'My Custom Strategy',
    'description': 'What it does',
    'hypothesis': 'Why it should work',
    'expected_improvement': 'Expected RMSE reduction',
    'risk': 'low/medium/high',
    'implementation': 'python_code'
}
```

---

## 📚 References

- [Validation Framework](./validation-framework.md)
- [Experiment Template](../03-experiments/TEMPLATE.md)
- [Research Rules](../RESEARCH_RULES.md)

---

**With this harness, we have rigorous, automated ways to know if we're improving.**

The key insight: **Don't trust a single metric.** Use statistical tests, effect sizes, trajectory analysis, and stopping criteria together to make reliable decisions.
