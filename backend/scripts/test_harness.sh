#!/bin/bash
# Test script for Autonomous Research Harness
# Runs a quick demonstration of the harness functionality

echo "========================================================================"
echo "AutoFPL Research Harness - Demo Test"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "research/README.md" ]; then
    echo "Error: Must run from project root"
    echo "Usage: cd /home/akshit/fpl-lineup-optimizer && bash backend/scripts/test_harness.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "========================================================================"
echo "Step 1: Check Current Status"
echo "========================================================================"
python backend/scripts/autoresearch_harness.py status

echo ""
echo "========================================================================"
echo "Step 2: Show Improvement Dashboard"
echo "========================================================================"
python backend/scripts/improvement_tracker.py dashboard

echo ""
echo "========================================================================"
echo "Step 3: List Available Strategies"
echo "========================================================================"
python -c "
import sys
sys.path.insert(0, 'backend')
from scripts.autoresearch_harness import StrategyLibrary
print('Available Strategies:')
for i, strategy in enumerate(StrategyLibrary.list_strategies(), 1):
    info = StrategyLibrary.get_strategy(strategy)
    print(f'{i:2d}. {strategy:20s} - {info.get(\"name\", \"\")}')
"

echo ""
echo "========================================================================"
echo "Step 4: Run Single Test Experiment"
echo "========================================================================"
echo "Running: log_transform strategy (low risk, fast)"
echo ""
python backend/scripts/autoresearch_harness.py run --strategy log_transform

echo ""
echo "========================================================================"
echo "Step 5: Check Updated Status"
echo "========================================================================"
python backend/scripts/autoresearch_harness.py status

echo ""
echo "========================================================================"
echo "Step 6: Generate Analysis Report"
echo "========================================================================"
python backend/scripts/improvement_tracker.py analyze

echo ""
echo "========================================================================"
echo "Demo Complete!"
echo "========================================================================"
echo ""
echo "To run full autonomous research (10 experiments):"
echo "  python backend/scripts/autoresearch_harness.py run --auto --max-runs 10"
echo ""
echo "To monitor in real-time:"
echo "  python backend/scripts/improvement_tracker.py dashboard"
echo ""
