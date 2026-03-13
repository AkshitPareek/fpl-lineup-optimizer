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
| Cohen's d | 0.0903 |
| Significant | ❌ No |

## Conclusion
Optuna optimization: 0.61% improvement with params {'n_estimators': 68, 'max_depth': 4, 'learning_rate': 0.02804185182317473, 'subsample': 0.9636517056958357, 'colsample_bytree': 0.5536973193241005, 'reg_alpha': 9.65250952064412, 'reg_lambda': 9.968629869823832e-05} - Not significant

## Timestamp
2026-03-13T17:01:59.449542
