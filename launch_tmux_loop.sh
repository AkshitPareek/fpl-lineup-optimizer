#!/bin/bash
# Launch Autonomous Research Loop in a detached tmux session

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION_NAME="autofpl-research"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists!"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    echo "Or kill it with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new detached session
echo "Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" -n "research-loop"

# Send commands to the session
tmux send-keys -t "$SESSION_NAME" "cd '$SCRIPT_DIR'" C-m
tmux send-keys -t "$SESSION_NAME" "./run_autonomous_loop.sh" C-m

echo ""
echo "✅ Autonomous Research Loop started in tmux session '$SESSION_NAME'"
echo ""
echo "Commands:"
echo "  Attach:    tmux attach -t $SESSION_NAME"
echo "  Detach:    Ctrl+B then D"
echo "  View logs: tail -f research/loop_logs/autonomous_loop_*.log"
echo "  Kill:      tmux kill-session -t $SESSION_NAME"
echo ""
echo "The loop will run continuously until significant improvement is achieved."
echo ""

# Optionally attach immediately
if [ "$1" == "--attach" ]; then
    tmux attach -t "$SESSION_NAME"
fi
