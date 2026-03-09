# FPL ML Validation Framework - Quickstart Guide

This guide will help you set up and use the validation framework to confidently improve your models.

## 🎯 What You Now Have

We've built a comprehensive **5-Layer Testing Framework**:

```
✅ Layer 1: Unit Tests (existing + enhanced)
✅ Layer 2: Data Validation Tests  
✅ Layer 3: Model Validation Tests
✅ Layer 4: Regression Testing Suite
✅ Layer 5: Benchmarking & A/B Testing Framework
```

## 🚀 Getting Started (5 minutes)

### Step 1: Verify Everything Works

```bash
# Navigate to project root
cd /home/akshit/fpl-lineup-optimizer

# Activate virtual environment
source venv/bin/activate

# Run quick smoke test
python backend/scripts/validation_runner.py --smoke
```

**Expected output:**
```
======================================================================
                    FPL ML VALIDATION RUNNER
======================================================================

======================================================================
SMOKE TEST
======================================================================
ℹ Loading models...
✓ Smoke test passed (3 models tested)

======================================================================
VALIDATION SUMMARY
======================================================================
✓ PASS smoke_test             (0.5s)

Total: 1/1 passed (100%)

✓ All validations passed! Ready for deployment.
```

### Step 2: Establish Your Baseline

**IMPORTANT:** Do this before making any model improvements!

```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

This creates `benchmark_results/baseline_metrics.json` with your current model performance. This is your reference point.

### Step 3: Run Full Validation

```bash
python backend/scripts/validation_runner.py --all --html-report validation_report.html
```

Then open `validation_report.html` in your browser to see the full report.

## 🔄 Your Improvement Workflow

Now when you want to improve your models, follow this workflow:

### 1. Before Making Changes

```bash
# Verify current state
python backend/scripts/validation_runner.py --all
```

### 2. Make Your Improvements

Example: Try LSTM with attention
```python
# In your training code
from ml.lstm_model import train_lstm

model, history = train_lstm(
    X_train, y_train, X_val, y_val,
    use_attention=True,  # Enable the attention mechanism
    hidden_dim=128,      # Increase capacity
    epochs=150,
)
```

### 3. Validate Your Changes

```bash
# Run data validation (did we break anything?)
python backend/scripts/validation_runner.py --data

# Run model validation (does the model behave correctly?)
python backend/scripts/validation_runner.py --models

# Compare with baseline (did we improve?)
python backend/scripts/ab_testing_cli.py compare \
    --model-a xgboost \
    --model-b lstm  # your new model
```

### 4. Check for Regressions

```bash
# Ensure we didn't break existing models
pytest backend/tests/test_model_regression.py -v
```

### 5. Full Benchmark (if improved)

```bash
# Run comprehensive comparison
python backend/scripts/ab_testing_cli.py benchmark \
    --models xgboost lightgbm lstm ensemble \
    --report
```

### 6. Update Baseline (if significantly better)

```bash
python backend/scripts/ab_testing_cli.py establish-baseline
```

## 📊 Understanding A/B Test Results

When you run:
```bash
python backend/scripts/ab_testing_cli.py compare --model-a xgboost --model-b lstm
```

You'll see output like:

```
📊 Mean Absolute Error (MAE):
  xgboost: 2.4567
  lstm: 2.2345
  Difference: -0.2222
  95% CI: [-0.3456, -0.0987]
  p-value: 0.0004 (Significant ✓)

📈 Effect Size (Cohen's d): 0.45
  Interpretation: medium

💡 Recommendation:
  Recommend lstm over xgboost. Effect size: medium
```

### Interpreting p-values:
- **p < 0.05**: Statistically significant improvement
- **p >= 0.05**: May be due to random chance

### Interpreting Effect Size (Cohen's d):
- **< 0.2**: Negligible
- **0.2 - 0.5**: Small
- **0.5 - 0.8**: Medium  
- **> 0.8**: Large

## 🎨 Common Validation Commands

### Quick Checks (< 30 seconds)

```bash
# Smoke test
python backend/scripts/validation_runner.py --smoke

# Data only
python backend/scripts/validation_runner.py --data

# Models only
python backend/scripts/validation_runner.py --models
```

### Standard Validation (1-2 minutes)

```bash
# Data + Models
python backend/scripts/validation_runner.py --data --models

# Everything except benchmarks
python backend/scripts/validation_runner.py --all
```

### Full Validation (5-10 minutes)

```bash
# Everything including benchmarks
python backend/scripts/validation_runner.py --all --benchmark --html-report report.html
```

## 🔬 Testing Specific Improvements

### Testing Feature Engineering

```python
# 1. Add your new features
from ml.advanced_features import generate_all_features

X_train_enhanced = generate_all_features(X_train)
X_test_enhanced = generate_all_features(X_test)

# 2. Train model with new features
model.fit(X_train_enhanced, y_train)

# 3. Compare
python backend/scripts/ab_testing_cli.py compare \
    --model-a xgboost_old \
    --model-b xgboost_new_features
```

### Testing Hyperparameter Changes

```bash
# 1. Train with new hyperparameters
python backend/ml_training.py --xgb-learning-rate 0.05

# 2. Compare with baseline
python backend/scripts/ab_testing_cli.py compare \
    --model-a xgboost_baseline \
    --model-b xgboost
```

### Testing Ensemble Methods

```bash
# 1. Create new ensemble
python -c "
from ml.ensemble import EnsemblePredictor
ensemble = EnsemblePredictor(base_models=['xgboost', 'lightgbm', 'lstm'])
ensemble.save('models/ensemble_v2/ensemble.pkl')
"

# 2. Compare ensembles
python backend/scripts/ab_testing_cli.py compare \
    --model-a ensemble \
    --model-b ensemble_v2
```

## 📈 Tracking Improvements Over Time

Track your model versions:

```bash
# After each significant change
python backend/scripts/ab_testing_cli.py track \
    --model xgboost \
    --version v2.1 \
    --note "Added attention mechanism"

# View history
cat benchmark_results/metrics_history.jsonl
```

## 🚨 Handling Test Failures

### "Data validation failed"

```bash
# Check what's wrong
pytest backend/tests/test_data_validation.py -v

# Common fixes:
# - Check for NaN in your data preprocessing
# - Ensure train/test splits are correct
# - Verify feature engineering didn't introduce infinities
```

### "Regression test failed"

```bash
# Check what regressed
pytest backend/tests/test_model_regression.py -v

# If the change is expected and acceptable:
python backend/scripts/ab_testing_cli.py establish-baseline
```

### "Model validation failed"

```bash
# Check model behavior
pytest backend/tests/test_model_validation.py -v

# Common issues:
# - Predictions outside range (check your activation function)
# - Non-finite predictions (check for numerical instability)
# - Too slow (optimize inference code)
```

## 📊 Performance Benchmarks

Your models should meet these targets:

| Metric | Target | Current |
|--------|--------|---------|
| RMSE | < 2.5 | Run benchmark |
| MAE | < 2.0 | Run benchmark |
| Spearman Correlation | > 0.3 | Run benchmark |
| Top-10 Accuracy | > 40% | Run benchmark |
| Single Prediction | < 100ms | Run model validation |
| Batch (100) | < 500ms | Run model validation |

Run to check your current performance:
```bash
python backend/scripts/ab_testing_cli.py benchmark \
    --models xgboost lightgbm ensemble \
    --report
```

## 🎯 Next Steps

Now that you have the validation framework:

1. **Try your first improvement:**
   - Enable LSTM attention mechanism
   - Run A/B test
   - See if it improves!

2. **Add position-specific models:**
   - Train separate models for GK, DEF, MID, FWD
   - Compare with unified model

3. **Try ensemble weight optimization:**
   - Use the benchmark to find optimal ensemble weights

4. **Add new features:**
   - Add temporal features (rest days, momentum)
   - Validate with data validation tests
   - Compare with A/B testing

## 📚 Documentation

- **Full Testing Framework Docs:** `docs/TESTING_FRAMEWORK.md`
- **Benchmark Framework:** `backend/ml/model_benchmark.py` (docstrings)
- **A/B Testing CLI:** `backend/scripts/ab_testing_cli.py --help`
- **Validation Runner:** `backend/scripts/validation_runner.py --help`

## 💡 Pro Tips

1. **Always establish baseline first** - You can't measure improvement without a reference
2. **Run smoke tests frequently** - Catches major issues quickly
3. **Use A/B testing for small changes** - Statistical rigor matters
4. **Generate HTML reports for sharing** - Easy to share with teammates
5. **Track metrics over time** - See trends, not just snapshots
6. **Don't ignore warnings** - They often indicate future problems

## ✅ Checklist Before Deployment

Before deploying to production:

- [ ] Smoke test passes
- [ ] Data validation passes
- [ ] Model validation passes
- [ ] Regression tests pass (or baseline updated intentionally)
- [ ] A/B test shows improvement (or no significant degradation)
- [ ] Full benchmark report generated
- [ ] Performance targets met
- [ ] HTML report reviewed

---

**You're all set!** Start with a smoke test, then try your first model improvement. The framework will tell you if it's working! 🚀
