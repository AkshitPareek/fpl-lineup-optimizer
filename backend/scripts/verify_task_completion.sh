#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <TASK_ID>"
  echo "Example: $0 P2-T1"
  exit 2
fi

TASK_ID="$1"

if [[ -x "./venv/bin/pytest" ]]; then
  PYTEST_BIN="./venv/bin/pytest"
elif command -v pytest >/dev/null 2>&1; then
  PYTEST_BIN="pytest"
else
  echo "pytest not found in PATH and ./venv/bin/pytest is missing"
  exit 2
fi

run_test_cmd() {
  local cmd="$1"
  echo "[gate] Running: ${cmd}"
  eval "${cmd}"
}

case "$TASK_ID" in
  P2-T1)
    run_test_cmd "${PYTEST_BIN} backend/tests/test_baseline_models.py -v"
    ;;
  P2-T2)
    run_test_cmd "${PYTEST_BIN} backend/tests/test_advanced_features.py -v"
    ;;
  P2-T3)
    run_test_cmd "${PYTEST_BIN} backend/tests/test_gradient_boosting.py -v"
    ;;
  P2-T4)
    run_test_cmd "${PYTEST_BIN} backend/tests/test_lstm_model.py -v"
    ;;
  P2-T5)
    run_test_cmd "${PYTEST_BIN} backend/tests/test_ensemble.py -v"
    ;;
  P2-T6)
    run_test_cmd "${PYTEST_BIN} backend/tests/test_evaluation.py -v"
    ;;
  *)
    echo "Unknown task id: ${TASK_ID}"
    echo "See docs/team/TASK_TEST_MAP.md"
    exit 2
    ;;
esac

echo "[gate] ${TASK_ID} passed mapped tests. Task may proceed to review." 
