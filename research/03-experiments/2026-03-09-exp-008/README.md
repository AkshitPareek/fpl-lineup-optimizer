# EXP-008: Hyperparameter Optimization

**Status:** ⚠️ Not Significant

## Hypothesis
200+ Optuna trials to find optimal XGBoost hyperparameters

## Methodology
See experiment runner for implementation details.

## Results

| Metric | Value |
|--------|-------|
| RMSE | 0.8513 |
| Baseline RMSE | 0.8568 |
| Relative Improvement | 0.64% |
| Cohen's d | 0.0795 |
| Significant | ❌ No |

## Conclusion
Optuna optimization: 0.64% improvement with params {'n_estimators': 135, 'max_depth': 8, 'learning_rate': 0.02618516118521656, 'subsample': 0.625872535725196, 'colsample_bytree': 0.5044571732361647, 'reg_alpha': 3.9057817309516327, 'reg_lambda': 0.0029255238333952965} - Not significant

## Timestamp
2026-03-09T23:58:26.681225
