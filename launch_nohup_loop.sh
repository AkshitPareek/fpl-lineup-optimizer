#!/bin/bash
# Launch Autonomous Research Loop with nohup (alternative to tmux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/research/loop_logs"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
NOHUP_LOG="$SCRIPT_DIR/research/loop_logs/nohup_${TIMESTAMP}.log"

echo "Starting Autonomous Research Loop with nohup..."
echo "Log file: $NOHUP_LOG"

nohup "$SCRIPT_DIR/run_autonomous_loop.sh" > "$NOHUP_LOG" 2>&1 &

PID=$!
echo $PID > "$SCRIPT_DIR/research/loop_logs/autonomous_loop.pid"

echo ""
echo "✅ Autonomous Research Loop started (PID: $PID)"
echo ""
echo "Commands:"
echo "  View logs:  tail -f $NOHUP_LOG"
echo "  Check PID:  ps aux | grep $PID"
echo "  Stop:       kill $PID"
echo "  Status:     cat research/loop_logs/autonomous_loop.pid"
echo ""
echo "The loop will run continuously until significant improvement is achieved."
echo ""
