#!/bin/bash
# Multi-agent workspace launcher (lead-managed, test-gated)

set -euo pipefail

SESSION="fpl-ml-team"
ROOT="/home/akshit/fpl-lineup-optimizer"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Attaching..."
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n lead

for win in architect teammate-a teammate-b teammate-c test-harness reviewer devil dashboard; do
  tmux new-window -t "$SESSION" -n "$win"
done

setup_window() {
  local window="$1"
  local title="$2"
  local lines="$3"
  tmux send-keys -t "$SESSION:$window" "cd $ROOT && clear" C-m
  tmux send-keys -t "$SESSION:$window" "echo '=== $title ==='" C-m
  tmux send-keys -t "$SESSION:$window" "echo ''" C-m
  IFS='|' read -ra chunks <<< "$lines"
  for line in "${chunks[@]}"; do
    tmux send-keys -t "$SESSION:$window" "echo '$line'" C-m
  done
}

setup_window "lead" "LEAD AGENT" \
"Branch: dev/ml-phase2|Rule: no task is done until mapped tests pass|Use docs/team/LEAD_AGENT_SYSTEM.md"
setup_window "architect" "ARCHITECT" \
"Maintain optimized execution plan|Approve teammate plan before coding|Use docs/team/EXECUTION_PLAN.md"
setup_window "teammate-a" "TEAMMATE A" \
"Primary tasks: P2-T1, P2-T2|Worktree + feature branch required|Run gate script before PR"
setup_window "teammate-b" "TEAMMATE B" \
"Primary tasks: P2-T3, P2-T5|Worktree + feature branch required|Run gate script before PR"
setup_window "teammate-c" "TEAMMATE C" \
"Primary tasks: P2-T4 (optional), P2-T6|Worktree + feature branch required|Run gate script before PR"
setup_window "test-harness" "TEST HARNESS TEAMMATE" \
"Prepare/maintain mapped tests per task|Parallel to teammate planning|Block completion on failing tests"
setup_window "reviewer" "REVIEW AGENT" \
"Review only after mapped tests pass|Check regressions and scope|Approve before merge"
setup_window "devil" "DEVIL'S ADVOCATE" \
"Challenge complexity and hidden risk|Push simplification|Final gate before sign-off"
setup_window "dashboard" "TEAM DASHBOARD" \
"cat backend/TASK_TRACKING.md|cat docs/team/TASK_TEST_MAP.md|./backend/scripts/verify_task_completion.sh P2-T1"

tmux select-window -t "$SESSION:lead"

# Optional visual settings
tmux set-option -t "$SESSION" status on
tmux set-option -t "$SESSION" status-interval 5
tmux set-option -t "$SESSION" status-left "#[fg=green]#S"
tmux set-option -t "$SESSION" status-right "%Y-%m-%d %H:%M"

echo "Team workspace created: $SESSION"
echo "Attach with: tmux attach -t $SESSION"

tmux attach -t "$SESSION"
