# Merge Queue (Lead)

Date: 2026-03-05
Target branch: `dev/ml-phase2`

## Preconditions
- Test gates passed for queued tasks.
- Reviewer report complete: `docs/team/reviews/REVIEW_REPORT_2026-03-05.md`
- Devil's advocate report complete: `docs/team/reviews/DEVILS_ADVOCATE_REPORT_2026-03-05.md`

## Queue Order
1. P2-T1 + P2-T2 (foundational)
2. P2-T3 (boosting)
3. P2-T5 (ensemble)
4. P2-T6 (evaluation)

## Blocked / Deferred
- P2-T4 (optional): blocked by missing `torch` in active environment.

## PR-Style Summaries

### PR-1: P2-T1 + P2-T2
- Scope:
  - Baseline model support and validations
  - Advanced feature pipeline and tests
- Gate evidence:
  - `./backend/scripts/verify_task_completion.sh P2-T1` -> pass
  - `./backend/scripts/verify_task_completion.sh P2-T2` -> pass
- Risk:
  - Feature behavior depends on dataset schema consistency.

### PR-2: P2-T3
- Scope:
  - Gradient boosting training/eval/tuning paths
  - SHAP dependency path validated in venv
- Gate evidence:
  - `./backend/scripts/verify_task_completion.sh P2-T3` -> pass
- Risk:
  - External package availability (`shap`) required.

### PR-3: P2-T5
- Scope:
  - Ensemble module (`backend/ml/ensemble.py`)
  - Stacking, blending, uncertainty, persistence
- Gate evidence:
  - `./backend/scripts/verify_task_completion.sh P2-T5` -> pass
- Risk:
  - Current calibration approach reads local validation files implicitly.

### PR-4: P2-T6
- Scope:
  - Evaluation module (`backend/ml/evaluation.py`)
  - Backtest, CV, report, interval utilities
- Gate evidence:
  - `./backend/scripts/verify_task_completion.sh P2-T6` -> pass
- Risk:
  - Utility implementations are harness-driven and should be hardened for production.

## Lead Merge Checklist
- [x] Re-run all queued task gates in one pass
- [x] Confirm no accidental scope creep in diff
- [x] Merge in queue order
- [x] Update `backend/TASK_TRACKING.md` statuses to `Done`
- [x] Record merge SHAs in this file

## Merge SHAs
- PR-1 (P2-T1 + P2-T2): `3381afe`
- PR-2 (P2-T3): `32d4b54`
- PR-3/PR-4 (P2-T5 + P2-T6): `79566fc`
