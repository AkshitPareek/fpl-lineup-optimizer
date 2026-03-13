#!/bin/bash
# Run Data Collection Agents in Parallel
# 
# Usage: ./run_agents_parallel.sh
# This launches 4-6 agents simultaneously to collect historical data

set -e

cd "$(dirname "$0")"

echo "================================================================================"
echo "  PARALLEL AGENT DEPLOYMENT - Historical Data Collection"
echo "================================================================================"
echo ""

# Create directories
mkdir -p data/historical/raw
mkdir -p data/historical/agents
mkdir -p logs/agents

# Check prerequisites
echo "🔍 Pre-flight checks..."

# Check disk space
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 5 ]; then
    echo "❌ Insufficient disk space: ${AVAILABLE_GB}GB (need 5GB+)"
    exit 1
fi
echo "  ✅ Disk space: ${AVAILABLE_GB}GB available"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
echo "  ✅ Python available"

# Check virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "  ✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

echo ""
echo "🚀 Launching agents..."
echo ""

# Agent launch function
launch_agent() {
    local AGENT_NUM=$1
    local SEASON=$2
    local LOG_FILE="logs/agents/agent_${AGENT_NUM}_${SEASON}.log"
    
    echo "  🚀 Agent $AGENT_NUM: Starting collection for $SEASON"
    
    # Run agent in background
    python3 agents/fetch_historical_data.py \
        --season "$SEASON" \
        --output data/historical \
        --source github \
        > "$LOG_FILE" 2>&1 &
    
    echo $!  # Return PID
}

# Launch all agents in parallel
PIDS=()

# Agent 1: 2020-21 season
PID1=$(launch_agent 1 "2020-21")
PIDS+=($PID1)
echo "     PID: $PID1"

# Agent 2: 2021-22 season
PID2=$(launch_agent 2 "2021-22")
PIDS+=($PID2)
echo "     PID: $PID2"

# Agent 3: 2022-23 season
PID3=$(launch_agent 3 "2022-23")
PIDS+=($PID3)
echo "     PID: $PID3"

# Agent 4: 2023-24 season
PID4=$(launch_agent 4 "2023-24")
PIDS+=($PID4)
echo "     PID: $PID4"

echo ""
echo "📊 All 4 agents launched in parallel"
echo ""
echo "Monitoring..."
echo ""

# Wait for all agents with progress display
COMPLETED=0
FAILED=0
TOTAL=4

for PID in "${PIDS[@]}"; do
    if wait $PID; then
        ((COMPLETED++))
        echo "  ✅ Agent completed (PID: $PID)"
    else
        ((FAILED++))
        echo "  ❌ Agent failed (PID: $PID)"
    fi
    echo "     Progress: $COMPLETED/$TOTAL completed, $FAILED failed"
done

echo ""
echo "================================================================================"
echo "  AGENT COLLECTION COMPLETE"
echo "================================================================================"
echo ""
echo "Results:"
echo "  ✅ Successful: $COMPLETED"
echo "  ❌ Failed: $FAILED"
echo ""
echo "Output files:"
ls -lh data/historical/*.csv 2>/dev/null || echo "  No CSV files found"
echo ""
echo "Logs:"
ls -lh logs/agents/*.log 2>/dev/null || echo "  No log files found"
echo ""
echo "Next steps:"
echo "  1. Check individual agent logs: cat logs/agents/agent_*.log"
echo "  2. Aggregate data: python3 aggregate_historical_data.py"
echo "  3. Retry failed agents if needed"
echo ""

# Summary
if [ $COMPLETED -eq $TOTAL ]; then
    echo "🎉 All agents completed successfully!"
    exit 0
elif [ $COMPLETED -gt 0 ]; then
    echo "⚠️  Some agents completed ($COMPLETED/$TOTAL)"
    exit 0
else
    echo "❌ All agents failed"
    exit 1
fi
