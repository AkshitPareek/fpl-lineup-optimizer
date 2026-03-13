# EXP-009: Weighted Ensemble

**Status:** ⚠️ Not Significant

## Hypothesis
Learn optimal ensemble weights using Ridge regression

## Methodology
See experiment runner for implementation details.

## Results

| Metric | Value |
|--------|-------|
| RMSE | 0.8514 |
| Baseline RMSE | 0.8568 |
| Relative Improvement | 0.63% |
| Cohen's d | 0.1346 |
| Significant | ❌ No |

## Conclusion
Weighted ensemble: 0.63% improvement with weights {'xgb': -0.1809965045153463, 'lgb': 0.10719825707942215, 'rf': -0.023594357103907143} - Not significant

## Timestamp
2026-03-13T17:02:02.365314
