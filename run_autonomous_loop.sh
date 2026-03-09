#!/bin/bash
# Autonomous Research Loop Runner
# Runs experiments continuously until significant improvement is achieved

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Activate virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    log "Virtual environment activated"
else
    error "Virtual environment not found at $SCRIPT_DIR/venv"
    exit 1
fi

# Install missing dependencies if needed
log "Checking dependencies..."
pip install -q optuna catboost 2>/dev/null || warning "Some optional dependencies may not be available"

# Create run log directory
mkdir -p "$SCRIPT_DIR/research/loop_logs"

RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="$SCRIPT_DIR/research/loop_logs/autonomous_loop_${RUN_TIMESTAMP}.log"

log "Starting Autonomous Research Loop (Ralph)"
log "Log file: $LOG_FILE"
log "Target: >1% improvement with Cohen's d > 0.2"
echo ""

# Run the autonomous loop with restart capability
MAX_RESTARTS=5
RESTART_COUNT=0

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    log "=== Loop iteration $((RESTART_COUNT + 1))/$MAX_RESTARTS ==="
    
    if python "$SCRIPT_DIR/backend/scripts/autonomous_research_loop.py" \
        --target-improvement 1.0 \
        --min-effect-size 0.2 \
        --max-experiments 50 2>&1 | tee -a "$LOG_FILE"; then
        
        success "SIGNIFICANT IMPROVEMENT ACHIEVED!"
        log "Check $LOG_FILE for details"
        
        # Send notification (if available)
        if command -v notify-send &> /dev/null; then
            notify-send "AutoFPL Research" "Significant improvement achieved! Check logs."
        fi
        
        exit 0
    else
        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -eq 1 ]; then
            warning "No significant improvement in this batch"
            RESTART_COUNT=$((RESTART_COUNT + 1))
            
            if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
                log "Restarting with modified strategies..."
                sleep 5
            fi
        else
            error "Loop crashed with exit code $EXIT_CODE"
            RESTART_COUNT=$((RESTART_COUNT + 1))
            
            if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
                log "Restarting after crash..."
                sleep 10
            fi
        fi
    fi
done

error "Max restarts reached without achieving significant improvement"
log "Check $LOG_FILE for details"
exit 1
