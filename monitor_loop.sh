#!/bin/bash
# Monitor the Autonomous Research Loop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  AutoFPL Research Loop Monitor"
echo "=========================================="
echo ""

# Check if running via tmux
if tmux has-session -t "autofpl-research" 2>/dev/null; then
    echo "✅ Tmux session 'autofpl-research' is RUNNING"
    echo ""
    echo "Recent activity:"
    tmux capture-pane -t "autofpl-research" -p | tail -20
else
    echo "❌ Tmux session not found"
fi

echo ""
echo "--- Latest Log Files ---"
if [ -d "$SCRIPT_DIR/research/loop_logs" ]; then
    ls -lt "$SCRIPT_DIR/research/loop_logs"/*.log 2>/dev/null | head -5 || echo "No log files found"
else
    echo "No log directory found"
fi

echo ""
echo "--- Recent Experiments ---"
if [ -d "$SCRIPT_DIR/research/03-experiments" ]; then
    ls -lt "$SCRIPT_DIR/research/03-experiments" | head -10
else
    echo "No experiments directory found"
fi

echo ""
echo "--- Best Results ---"
if [ -f "$SCRIPT_DIR/research/loop_logs"/*.log ]; then
    LATEST_LOG=$(ls -t "$SCRIPT_DIR/research/loop_logs"/*.log | head -1)
    echo "From $LATEST_LOG:"
    grep -E "(SIGNIFICANT|Best RMSE|Improvement)" "$LATEST_LOG" | tail -10 || echo "No results yet"
fi

echo ""
echo "=========================================="
echo "Commands:"
echo "  Attach:    tmux attach -t autofpl-research"
echo "  View logs: tail -f research/loop_logs/*.log"
echo "=========================================="
