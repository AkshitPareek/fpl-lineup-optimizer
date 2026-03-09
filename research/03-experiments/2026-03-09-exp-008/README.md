# EXP-008: Hyperparameter Optimization

**Status:** ⚠️ Not Significant

## Hypothesis
200+ Optuna trials to find optimal XGBoost hyperparameters

## Methodology
See experiment runner for implementation details.

## Results

| Metric | Value |
|--------|-------|
| RMSE | 0.8516 |
| Baseline RMSE | 0.8568 |
| Relative Improvement | 0.61% |
| Cohen's d | 0.1015 |
| Significant | ❌ No |

## Conclusion
Optuna optimization: 0.61% improvement with params {'n_estimators': 108, 'max_depth': 8, 'learning_rate': 0.02577960402412914, 'subsample': 0.7074212753170258, 'colsample_bytree': 0.5775051138078803, 'reg_alpha': 3.6343004829739005, 'reg_lambda': 0.45463113204561917} - Not significant

## Timestamp
2026-03-09T23:54:30.606344
