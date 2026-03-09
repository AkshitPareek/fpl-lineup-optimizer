# Validation Framework

> **Status:** ✅ Implemented & Validated  
> **Date:** 2026-03-09  
> **Related Experiment:** [EXP-001](../03-experiments/2026-03-09-baseline-establishment/)  
> **Code Location:** `backend/ml/model_benchmark.py`, `backend/scripts/validation_runner.py`

---

## 1. Overview

The AutoFPL Validation Framework is a 5-layer testing system designed to ensure that model improvements are real, statistically significant, and practically meaningful.

### Why 5 Layers?

Each layer catches different types of errors:

```
Layer 1: Unit Tests ────────────────► Catch implementation bugs
Layer 2: Data Validation ───────────► Catch data quality issues
Layer 3: Model Validation ──────────► Catch model behavior issues
Layer 4: Regression Testing ────────► Catch performance degradation
Layer 5: Benchmark Testing ─────────► Catch statistical flukes
```

Only by passing all 5 layers can we confidently claim an improvement.

---

## 2. The 5 Layers

### Layer 1: Unit Tests

**Purpose:** Ensure individual components work correctly.

**Scope:**
- Data loading functions
- Feature engineering
- Model initialization
- API endpoints
- Utility functions

**Implementation:**
```python
# Example: test that feature engineering produces expected output
def test_polynomial_features():
    X = np.array([[1, 2, 3]])
    X_poly = generate_polynomial_features(X)
    assert X_poly.shape[1] > X.shape[1]  # Should add features
```

**Files:**
- `backend/tests/test_*.py`

**Command:**
```bash
pytest backend/tests/ -v
```

**When to Run:**
- After any code changes
- Before committing
- Part of CI/CD pipeline

---

### Layer 2: Data Validation

**Purpose:** Ensure training data is clean and consistent.

**Checks:**

| Check | Why It Matters | Failure Mode |
|-------|----------------|--------------|
| No NaN values | Models crash on NaN | Silent failures |
| No infinite values | Destroys training | Gradient explosion |
| Consistent shapes | Matrix dimension errors | Runtime errors |
| Targets in range | FPL points are bounded | Unrealistic predictions |
| No data leakage | Invalid results | Overly optimistic metrics |
| Similar distributions | Generalization | Train/test mismatch |

**Implementation:**
```python
# From test_data_validation.py
def test_no_nan_in_features():
    X = np.load('datasets/fpl_points_v1/train_X.npy')
    assert np.isnan(X).sum() == 0

def test_feature_target_alignment():
    X = np.load('train_X.npy')
    y = np.load('train_y.npy')
    assert X.shape[0] == y.shape[0]
```

**Command:**
```bash
pytest backend/tests/test_data_validation.py -v
```

**When to Run:**
- After data pipeline changes
- Before training new models
- After feature engineering changes

---

### Layer 3: Model Validation

**Purpose:** Ensure models behave reasonably.

**Checks:**

| Check | Purpose | Example |
|-------|---------|---------|
| Predictions finite | No crashes | `np.all(np.isfinite(preds))` |
| Predictions in range | Realistic FPL | All preds in [-5, 20] |
| Predictions vary | Model isn't trivial | `std(preds) > 0.5` |
| Rank correlation positive | Model learns something | Spearman > 0.1 |
| Handles edge cases | Robustness | Works with zeros, large values |
| Inference time acceptable | Production ready | < 100ms per prediction |

**Implementation:**
```python
# From test_model_validation.py
def test_predictions_within_reasonable_range():
    preds = model.predict(X_test)
    assert np.all(preds >= -5)
    assert np.all(preds <= 20)

def test_single_prediction_latency():
    start = time.perf_counter()
    _ = model.predict(X_test[:1])
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100
```

**Command:**
```bash
pytest backend/tests/test_model_validation.py -v
```

**When to Run:**
- After training new models
- Before deploying to production
- After architecture changes

---

### Layer 4: Regression Testing

**Purpose:** Ensure new models don't break existing functionality.

**Checks:**

| Metric | Baseline | Threshold | Test |
|--------|----------|-----------|------|
| RMSE | 2.50 | < 2.625 (+5%) | `current_rmse < baseline_rmse * 1.05` |
| MAE | 1.90 | < 1.995 (+5%) | `current_mae < baseline_mae * 1.05` |
| R² | 0.25 | > 0.20 | `current_r2 > baseline_r2 - 0.05` |
| Inference Time | 50ms | < 75ms (+50%) | `current_time < baseline_time * 1.5` |

**Implementation:**
```python
# From test_model_regression.py
def test_xgboost_performance_not_regressed():
    baseline_rmse = load_baseline()['xgboost']['rmse']
    current_rmse = evaluate_current_model()['rmse']
    assert current_rmse <= baseline_rmse * 1.05
```

**Command:**
```bash
pytest backend/tests/test_model_regression.py -v
```

**When to Run:**
- Before committing changes
- Before merging pull requests
- In CI/CD pipeline

---

### Layer 5: Benchmark Testing

**Purpose:** Statistically compare models with rigor.

**Components:**

#### A/B Testing
Compare two models with statistical significance:

```
Model A (Baseline):
  RMSE: 2.500
  MAE: 1.900

Model B (Treatment):
  RMSE: 2.380
  MAE: 1.820

Statistical Test:
  RMSE Difference: -0.120
  95% CI: [-0.198, -0.042]
  p-value: 0.003
  Significant: ✅ Yes
  Effect Size: 0.45 (medium)
```

#### Multi-Model Benchmarking
Compare multiple models across metrics:

```
Leaderboard (by RMSE):
┌─────────────┬────────┬────────┬──────────┬────────────┐
│ Model       │ RMSE   │ MAE    │ Spearman │ Top-10 Acc │
├─────────────┼────────┼────────┼──────────┼────────────┤
│ Ensemble    │ 2.290  │ 1.760  │ 0.421    │ 48.2%      │ ⭐
│ LightGBM    │ 2.380  │ 1.820  │ 0.398    │ 45.1%      │
│ XGBoost     │ 2.450  │ 1.890  │ 0.385    │ 43.8%      │
│ RandomForest│ 2.520  │ 1.950  │ 0.362    │ 41.2%      │
│ Ridge       │ 2.610  │ 2.020  │ 0.341    │ 39.5%      │
└─────────────┴────────┴────────┴──────────┴────────────┘
```

#### Statistical Significance Testing
- **Paired t-test:** Compare error distributions
- **Wilcoxon test:** Non-parametric alternative
- **Bootstrap CI:** Confidence intervals for differences
- **Effect size:** Cohen's d for practical significance

**Implementation:**
```python
# From model_benchmark.py
def ab_test_models(model_a, model_b, X_test, y_test):
    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)
    
    error_a = np.abs(y_test - pred_a)
    error_b = np.abs(y_test - pred_b)
    
    # Paired t-test
    t_stat, p_value = ttest_rel(error_a, error_b)
    
    # Effect size
    cohens_d = (np.mean(error_a) - np.mean(error_b)) / pooled_std
    
    return {
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': cohens_d,
        'recommendation': generate_recommendation(...)
    }
```

**Command:**
```bash
# Compare two models
python backend/scripts/ab_testing_cli.py compare \
    --model-a xgboost \
    --model-b lightgbm

# Full benchmark
python backend/scripts/ab_testing_cli.py benchmark \
    --models xgboost lightgbm ensemble
```

**When to Run:**
- After training candidate improvements
- Before adopting new model
- Weekly tracking of model performance

---

## 3. Evaluation Metrics

### Primary Metrics

| Metric | Description | Target | Why |
|--------|-------------|--------|-----|
| **RMSE** | Root Mean Squared Error | < 2.3 | Overall accuracy |
| **MAE** | Mean Absolute Error | < 1.8 | Interpretable error |
| **Spearman ρ** | Rank correlation | > 0.35 | Ranking quality |

### Secondary Metrics

| Metric | Description | Target | Why |
|--------|-------------|--------|-----|
| **Top-5 Acc** | % actual top-5 in predicted | > 40% | Captain picks |
| **Top-10 Acc** | % actual top-10 in predicted | > 45% | Squad selection |
| **Within 2pt** | % predictions within 2 points | > 60% | Practical accuracy |
| **Captain Acc** | Picked actual top scorer | > 20% | Key decision |

### Constraint Metrics

| Metric | Description | Limit | Why |
|--------|-------------|-------|-----|
| **Inference Time** | ms per prediction | < 100ms | Production usable |
| **Model Size** | Disk space | < 100MB | Deployable |
| **Memory** | RAM usage | < 1GB | Resource efficient |

---

## 4. Statistical Standards

### Significance Thresholds

| Test | Threshold | Interpretation |
|------|-----------|----------------|
| p-value | < 0.05 | Statistically significant |
| p-value | < 0.01 | Highly significant |
| Effect Size (d) | > 0.2 | Small but real |
| Effect Size (d) | > 0.5 | Medium, important |
| Effect Size (d) | > 0.8 | Large, substantial |

### Multiple Comparisons Correction

When testing multiple hypotheses, use Bonferroni correction:
```
α_corrected = α / n_tests
```

Example: Testing 10 improvements at α=0.05
```
α_corrected = 0.05 / 10 = 0.005
```

### Confidence Intervals

Always report 95% confidence intervals for:
- Metric differences
- Effect sizes
- Performance estimates

---

## 5. Automation

### Validation Runner

One command runs all validations:

```bash
python backend/scripts/validation_runner.py --all
```

**Output:**
```
======================================================================
VALIDATION SUMMARY
======================================================================
✓ PASS smoke_test             (0.5s)
✓ PASS data_validation        (2.1s)
✓ PASS model_validation       (5.3s)
✓ PASS regression_tests       (8.7s)
✓ PASS benchmark              (45.2s)

Total: 5/5 passed (100%)

✓ All validations passed! Ready for deployment.
```

### HTML Reports

Generate shareable reports:

```bash
python backend/scripts/validation_runner.py \
    --all \
    --html-report validation_report.html
```

**Contents:**
- Executive summary
- Detailed test results
- Performance comparisons
- Statistical analysis
- Recommendations

---

## 6. Usage Guide

### For Model Developers

**Before Training:**
```bash
# Validate data
python backend/scripts/validation_runner.py --data
```

**After Training:**
```bash
# Validate model behavior
pytest backend/tests/test_model_validation.py

# Compare to baseline
python backend/scripts/ab_testing_cli.py compare \
    --model-a baseline \
    --model-b your_model

# Full validation
python backend/scripts/validation_runner.py --all
```

**Before Committing:**
```bash
# Check for regressions
pytest backend/tests/test_model_regression.py
```

### For Researchers

**Documenting Experiments:**
1. Run full validation
2. Save results to `research/03-experiments/`
3. Fill out [experiment template](../03-experiments/TEMPLATE.md)
4. Update knowledge graph

**Tracking Progress:**
```bash
# Establish baseline
python backend/scripts/ab_testing_cli.py establish-baseline

# Track metrics over time
python backend/scripts/ab_testing_cli.py track \
    --model xgboost \
    --version v2.1
```

---

## 7. Validation Checklist

Before claiming an improvement:

- [ ] **Data Validation**
  - [ ] No NaN or infinite values
  - [ ] Consistent shapes
  - [ ] No data leakage

- [ ] **Model Validation**
  - [ ] Predictions in reasonable range
  - [ ] Handles edge cases
  - [ ] Inference time acceptable

- [ ] **Statistical Testing**
  - [ ] p-value < 0.05
  - [ ] Effect size reported
  - [ ] Confidence intervals provided

- [ ] **Regression Testing**
  - [ ] No degradation > 5%
  - [ ] All existing tests pass

- [ ] **Documentation**
  - [ ] Experiment README complete
  - [ ] Results reproducible
  - [ ] Knowledge graph updated

---

## 8. Implementation Details

### File Structure

```
backend/
├── ml/
│   ├── model_benchmark.py      # Core benchmarking
│   └── evaluation.py           # Metrics
├── tests/
│   ├── test_data_validation.py
│   ├── test_model_validation.py
│   ├── test_model_regression.py
│   └── test_*.py               # Unit tests
└── scripts/
    ├── ab_testing_cli.py       # CLI for A/B testing
    └── validation_runner.py    # Main orchestrator
```

### Key Classes

```python
# Model Benchmark
from ml.model_benchmark import ModelBenchmark

benchmark = ModelBenchmark()
result = benchmark.evaluate_model(model, X_test, y_test, "model_name")
# result.metrics: {'rmse': 2.3, 'mae': 1.8, ...}

# A/B Testing
from ml.model_benchmark import run_ab_test

result = run_ab_test(model_a, model_b, X_test, y_test)
# result['mae']['p_value']: 0.003
# result['recommendation']: "Recommend model_b"
```

---

## 9. Future Enhancements

### Planned Improvements

1. **Uncertainty Quantification**
   - Add confidence intervals to predictions
   - Monte Carlo Dropout for neural networks

2. **Drift Detection**
   - Monitor for data distribution shifts
   - Automatic retraining triggers

3. **Multi-Objective Optimization**
   - Balance accuracy vs inference time
   - Pareto frontier exploration

4. **Automated Experiment Proposals**
   - Agent suggests next experiments
   - Bayesian optimization for hyperparameters

---

## 10. References

### Internal
- [Experiment EXP-001](../03-experiments/2026-03-09-baseline-establishment/)
- [Testing Framework Docs](../../docs/TESTING_FRAMEWORK.md)
- [A/B Testing CLI](../../backend/scripts/ab_testing_cli.py)

### External
- [Papers with Code - ML Evaluation](https://paperswithcode.com/area/machine-learning/evaluation)
- [Google ML Rules of Thumb](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Statistical Significance in A/B Testing](https://towardsdatascience.com/statistical-significance-in-a-b-testing-5d5e5e6a1e)

---

**Maintained by:** AutoFPL Research Team  
**Last Updated:** 2026-03-09  
**Version:** 1.0
