# Task to Test Map

Used by `backend/scripts/verify_task_completion.sh`.

| Task ID | Required Test Command |
|---------|------------------------|
| P2-T1 | `pytest backend/tests/test_baseline_models.py -v` |
| P2-T2 | `pytest backend/tests/test_advanced_features.py -v` |
| P2-T3 | `pytest backend/tests/test_gradient_boosting.py -v` |
| P2-T4 | `pytest backend/tests/test_lstm_model.py -v` |
| P2-T5 | `pytest backend/tests/test_ensemble.py -v` |
| P2-T6 | `pytest backend/tests/test_evaluation.py -v` |

## Policy
- Failing mapped tests means task stays in `In Progress`.
- Teammates may run extra tests, but mapped tests are mandatory.
