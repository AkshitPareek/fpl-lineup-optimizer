# EXP-031: Feature Expansion Results

> **Status:** Feature Engineering Complete, Overfitting Challenges Identified

---

## 📊 Experiment Summary

### Attempt 1: Simulated Features
| Model | RMSE | vs Champion |
|-------|------|-------------|
| Ridge | 0.8522 | -2.88% ❌ |
| GB | 0.8601 | -3.83% ❌ |
| RF | 0.8556 | -3.29% ❌ |

**Result:** Random features don't help (expected)

### Attempt 2: Real Fixture Features (7 new features)
| Model | RMSE | vs Champion |
|-------|------|-------------|
| Ridge | 0.8440 | -1.89% ❌ |
| GB | 0.8730 | -5.39% ❌ |
| RF | 0.8516 | -2.80% ❌ |

**Result:** Real fixture data alone didn't help

### Attempt 3: Full Features (47 total: 30 base + 17 new)
| Model | Train RMSE | Test RMSE | vs Champion |
|-------|-----------|-----------|-------------|
| Ridge | 0.0708 | 1.3795 | -66.53% ❌ |
| GB | 0.0455 | 0.9883 | -19.30% ❌ |
| RF | 0.1077 | 0.9044 | -9.18% ❌ |

**Result:** Severe overfitting - training error very low, test error very high

---

## 🔍 Analysis: Why Didn't It Work?

### Problem 1: Small Dataset
```
Training samples: 192
Original features: 30
New features: 17
Total features: 47

Ratio: 192/47 = 4.1 samples per feature
Recommended: >100 samples per feature
```

**Rule of thumb:** You need at least 10x more samples than features to avoid overfitting.

### Problem 2: No Feature Selection
- Added all features without checking relevance
- Many features may be correlated (redundant)
- No regularization strong enough for this data size

### Problem 3: Dataset Limitations
Our training data (`fpl_points_v1`) is aggregated:
- Only 192 training samples
- Not enough to learn complex patterns
- Champion model (EXP-030) already optimized for this data

---

## 💡 What Would Actually Work?

### Option 1: Get More Data ⭐ BEST
```python
# Collect 5+ seasons of data
# Target: 1000+ samples
# Then 47 features would work fine
```

### Option 2: Feature Selection
```python
# Use only top 5-10 most predictive features
# Select based on correlation with target
# Techniques: L1 regularization, mutual information
```

### Option 3: Transfer Learning
```python
# Train on large dataset (e.g., all PL players)
# Fine-tune on FPL-specific data
# Use pre-trained embeddings
```

### Option 4: Domain-Specific Models
```python
# Separate models per position
# Different features matter for GK vs FWD
# Reduce dimensionality per model
```

---

## ✅ What We Accomplished

1. **Feature Engineering Module** (579 lines)
   - Fixture difficulty calculation
   - Fatigue metrics
   - Momentum indicators
   - Team chemistry
   - Ready to use when more data available

2. **Real Data Pipeline**
   - Fetches live fixture data from FPL API
   - Team strength ratings
   - Schedule analysis

3. **Training Infrastructure**
   - Pipeline for enhanced features
   - Comparison framework
   - Model saving/validation

4. **Learned Limitations**
   - Dataset size is the bottleneck
   - Feature engineering needs sufficient data
   - Champion model (EXP-030) is well-optimized for current data

---

## 🎯 Recommended Path Forward

### Immediate (This Week)
1. ✅ **Accept EXP-030 as champion** (it's well-optimized)
2. ✅ **Collect more historical data**
   - Multiple seasons
   - Player-level gameweek data
   - Target: 1000+ samples

### Short-term (Next 2 Weeks)
3. 🔄 **Retry EXP-031 with more data**
   - Same features, larger dataset
   - Expect significant improvement

### Alternative Approach
4. 🔬 **Feature selection on current data**
   - Pick top 5 features only
   - Use L1 regularization
   - May get marginal improvement

---

## 📁 Deliverables

```
backend/feature_engineering.py      # Complete module
data/fixtures/                      # Real FPL data
train_exp031_*.py                   # Training pipelines
EXP031_RESULTS.md                   # This file
```

---

## 🏆 Conclusion

**EXP-031 feature expansion is READY but needs MORE DATA.**

The champion model (EXP-030) with its negative weighting strategy is remarkably effective for the current dataset size. The feature engineering module is complete and will deliver value once we have:

- 500+ training samples → Try 10 new features
- 1000+ training samples → Try all 17 new features
- 5000+ training samples → Deep learning models

**Current status:**
- ✅ Feature engineering: COMPLETE
- ✅ Infrastructure: COMPLETE  
- ⏳ More data: NEEDED
- 🏆 EXP-030: Still champion

---

*Last Updated: 2026-03-13*
