# FPL ML Testing & Validation Framework

Comprehensive testing and validation framework for the FPL Lineup Optimizer ML pipeline. This framework ensures model reliability, tracks performance improvements, and prevents regressions.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Testing Components](#testing-components)
- [Running Tests](#running-tests)
- [A/B Testing Models](#ab-testing-models)
- [Benchmarking](#benchmarking)
- [Regression Testing](#regression-testing)
- [Validation Reports](#validation-reports)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The testing framework consists of five layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Benchmarking & A/B Testing                        │
│  - Compare models statistically                             │
│  - Track performance over time                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Regression Testing                                │
│  - Prevent performance degradation                          │
│  - Ensure behavioral consistency                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Model Validation                                  │
│  - Behavioral tests (predictions make sense)                │
│  - Robustness tests (handle edge cases)                     │
│  - Performance tests (latency requirements)                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Data Validation                                   │
│  - Data quality checks (no NaN, Inf)                        │
│  - Distribution checks (no drift)                           │
│  - Shape consistency                                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Unit Tests                                        │
│  - Component functionality                                  │
│  - Integration tests                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Run Everything

```bash
# Run all validations (recommended before deployment)
python backend/scripts/validation_runner.py --all --html-report report.html
```

### Run Specific Tests

```bash
# Quick smoke test (30 seconds)
python backend/scripts/validation_runner.py --smoke

# Data validation only
python backend/scripts/validation_runner.py --data

# Model validation only
python backend/scripts/validation_runner.py --models

# Regression tests (requires baseline)
python backend/scripts/validation_runner.py --regression

# Full unit test suite
python backend/scripts/validation_runner.py --unit-tests
```

### Compare Two Models

```bash
# A/B test XGBoost vs LightGBM
python backend/scripts/ab_testing_cli.py compare \
    --model-a xgboost \
    --model-b lightgbm \
    --output ab_test_results.json

# Run full benchmark on all models
python backend/scripts/ab_testing_cli.py benchmark \
    --models xgboost lightgbm ensemble \
    --report
```

## 🧪 Testing Components

### 1. Data Validation (`test_data_validation.py`)

**Purpose:** Ensure training data is clean and consistent.

**Checks:**
- ✓ All required files exist
- ✓ No NaN or infinite values
- ✓ Consistent shapes across splits
- ✓ Targets within reasonable FPL ranges
- ✓ Feature variance (not all identical)
- ✓ No data leakage between splits
- ✓ Similar distributions between train/test

**Usage:**
```bash
pytest backend/tests/test_data_validation.py -v
```

### 2. Model Validation (`test_model_validation.py`)

**Purpose:** Ensure models behave correctly and meet production requirements.

**Behavioral Tests:**
- ✓ Predictions are finite numbers
- ✓ Predictions within FPL range (-5 to 20)
- ✓ Predictions have variance (not all identical)
- ✓ High scorers ranked higher
- ✓ Model uses form features

**Performance Tests:**
- ✓ Single prediction < 100ms
- ✓ Batch of 100 < 500ms
- ✓ Model loading < 5s

**Robustness Tests:**
- ✓ Handles zero input
- ✓ Handles very large values
- ✓ Handles negative values
- ✓ Handles edge cases gracefully

**Usage:**
```bash
pytest backend/tests/test_model_validation.py -v
```

### 3. Regression Testing (`test_model_regression.py`)

**Purpose:** Prevent performance degradation when making changes.

**Requirements:**
- Baseline metrics must be established first

**Checks:**
- ✓ RMSE does not regress > 5%
- ✓ MAE does not regress > 5%
- ✓ R² does not decrease significantly
- ✓ Prediction behavior is consistent
- ✓ Inference time not degraded

**Establish Baseline:**
```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

**Run Regression Tests:**
```bash
pytest backend/tests/test_model_regression.py -v
```

### 4. Model Benchmark (`model_benchmark.py`)

**Purpose:** Comprehensive model comparison with statistical rigor.

**Features:**
- Standard metrics (RMSE, MAE, R²)
- Rank correlation (Spearman, Kendall)
- Top-k accuracy (can we find the best players?)
- Bootstrap confidence intervals
- Position-wise performance analysis
- Price bracket analysis
- FPL-specific metrics (captain accuracy, within X points)

**Statistical Tests:**
- Paired t-tests between models
- Wilcoxon signed-rank tests
- Effect size (Cohen's d)
- Confidence intervals for differences

**Usage:**
```python
from ml.model_benchmark import ModelBenchmark, quick_benchmark

# Quick comparison
comparison = quick_benchmark(
    models={'xgboost': xgb_model, 'lightgbm': lgb_model},
    X_test=X_test,
    y_test=y_test,
)

# Full benchmark with report
benchmark = ModelBenchmark()
comparison = benchmark.compare_models(models, X_test, y_test)
benchmark.generate_report(comparison, 'report.html')
```

### 5. A/B Testing CLI (`ab_testing_cli.py`)

**Purpose:** Command-line tool for rigorous model comparison.

**Commands:**

```bash
# Compare two models
python ab_testing_cli.py compare --model-a xgboost --model-b lightgbm

# Full benchmark suite
python ab_testing_cli.py benchmark --models xgboost lightgbm ensemble

# Establish baseline
python ab_testing_cli.py establish-baseline

# Track metrics over time
python ab_testing_cli.py track --model xgboost --version v2.1
```

## 📊 A/B Testing Models

When you want to compare two model versions:

### Step 1: Establish Baseline (First Time Only)

```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

This creates `benchmark_results/baseline_metrics.json` with current performance.

### Step 2: Make Your Improvements

Train your improved model and save it (e.g., as `models/xgboost/model_v2.pkl`).

### Step 3: Run A/B Test

```bash
python backend/scripts/ab_testing_cli.py compare \
    --model-a xgboost \
    --model-b lightgbm \
    --output comparison.json
```

Output includes:
- Mean Absolute Error comparison
- Root Mean Squared Error comparison
- Effect size (Cohen's d)
- Statistical significance (p-values)
- Win rates
- Recommendation

### Interpreting Results

```
📊 Mean Absolute Error (MAE):
  xgboost: 2.4567
  lightgbm: 2.3456
  Difference: +0.1111
  95% CI: [-0.0234, 0.2456]
  p-value: 0.1082 (Not Significant)

💡 Recommendation:
  No statistically significant difference. Keep xgboost (status quo).
```

## 📈 Benchmarking

Run comprehensive benchmarks to compare all models:

```bash
python backend/scripts/ab_testing_cli.py benchmark \
    --models xgboost lightgbm random_forest ensemble \
    --report
```

This generates:
1. Console output with leaderboard
2. `benchmark_results/comparison_*.json` with full results
3. `benchmark_results/benchmark_report_*.html` with visual report

### Benchmark Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| RMSE | Root Mean Squared Error | Overall prediction accuracy |
| MAE | Mean Absolute Error | Average point error |
| R² | Coefficient of Determination | Variance explained |
| Spearman | Rank correlation | Can we rank players correctly? |
| Top-5 Acc | % of actual top 5 in predicted top 5 | Captain pick quality |
| Top-10 Acc | % of actual top 10 in predicted top 10 | Squad selection |
| Within 2pt | % of predictions within 2 points | Practical accuracy |
| Captain Acc | Did we pick the actual top scorer? | Most important decision |

## 🔄 Regression Testing

Prevent performance degradation:

### Set Up Baseline

```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

### Before Committing Changes

```bash
pytest backend/tests/test_model_regression.py -v
```

Tests will fail if:
- RMSE increases by > 5%
- MAE increases by > 5%
- R² decreases significantly
- Predictions become unreasonable
- Inference time degrades

### Update Baseline (After Confirmed Improvements)

```bash
python backend/scripts/ab_testing_cli.py establish-baseline --output baseline_v2.json
```

## 📄 Validation Reports

### HTML Report

```bash
python backend/scripts/validation_runner.py --all --html-report report.html
```

Report includes:
- Executive summary (pass/fail)
- Detailed test results
- Performance metrics
- Statistical comparisons
- Recommendations

### JSON Report

```bash
python backend/scripts/validation_runner.py --all --report results.json
```

Useful for:
- CI/CD integration
- Automated alerting
- Historical tracking

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: ML Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-ml.txt
      
      - name: Run smoke test
        run: python backend/scripts/validation_runner.py --smoke
      
      - name: Run data validation
        run: python backend/scripts/validation_runner.py --data
      
      - name: Run model validation
        run: python backend/scripts/validation_runner.py --models
      
      - name: Run regression tests
        run: python backend/scripts/validation_runner.py --regression
      
      - name: Generate report
        if: always()
        run: |
          python backend/scripts/validation_runner.py \
            --all \
            --html-report validation_report.html
      
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: validation-report
          path: validation_report.html
```

### Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running ML validation..."

python backend/scripts/validation_runner.py --smoke
if [ $? -ne 0 ]; then
    echo "Smoke tests failed!"
    exit 1
fi

python backend/scripts/validation_runner.py --regression
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    echo "Run 'ab_testing_cli.py establish-baseline' if this is expected."
    exit 1
fi

echo "All validations passed!"
```

## 🔧 Troubleshooting

### "No baseline metrics found"

**Solution:** Establish baseline first:
```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

### "Missing test data"

**Solution:** Ensure datasets exist:
```bash
ls datasets/fpl_points_v1/
# Should show: train_X.npy, train_y.npy, test_X.npy, test_y.npy, etc.
```

### "Model not found"

**Solution:** Train models first:
```bash
cd backend
python ml_training.py
```

### "Tests too slow"

**Options:**
1. Run smoke test only: `--smoke`
2. Skip benchmarks: Don't use `--benchmark`
3. Reduce test data size for development

### "False positives in regression tests"

**Reason:** Baseline may be outdated.

**Solution:** Re-establish baseline after confirmed improvements:
```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

## 📚 Additional Resources

### Python API

```python
# Data validation
from tests.test_data_validation import validate_data_for_training
report = validate_data_for_training()
print(f"Data valid: {report['valid']}")

# Model validation
from tests.test_model_validation import validate_models_for_production
report = validate_models_for_production()
print(f"Models valid: {report['valid']}")

# Benchmarking
from ml.model_benchmark import ModelBenchmark
benchmark = ModelBenchmark()
result = benchmark.evaluate_model(model, X_test, y_test, "my_model")
print(f"RMSE: {result.metrics['rmse']:.4f}")

# A/B testing
from ml.model_benchmark import run_ab_test
result = run_ab_test(model_a, model_b, X_test, y_test)
print(result['recommendation'])
```

### File Locations

```
backend/
├── ml/
│   ├── model_benchmark.py          # Benchmarking framework
│   └── ...
├── tests/
│   ├── test_data_validation.py     # Data validation tests
│   ├── test_model_validation.py    # Model validation tests
│   ├── test_model_regression.py    # Regression tests
│   ├── test_evaluation.py          # Model evaluation tests
│   ├── test_ensemble.py            # Ensemble tests
│   ├── test_lstm_model.py          # LSTM tests
│   └── test_integration_ml.py      # Integration tests
└── scripts/
    ├── ab_testing_cli.py           # A/B testing CLI
    └── validation_runner.py        # Main validation runner
```

## 🤝 Best Practices

1. **Before Training:** Run data validation
2. **After Training:** Run model validation
3. **Before Committing:** Run regression tests
4. **Before Deploying:** Run full validation suite
5. **After Major Changes:** Re-establish baseline
6. **Weekly:** Run benchmarks to track performance over time

## 📞 Support

For issues or questions:
1. Check this documentation
2. Run with `--verbose` flag for detailed output
3. Check `benchmark_results/` for detailed logs
4. Review test files for specific validation logic
