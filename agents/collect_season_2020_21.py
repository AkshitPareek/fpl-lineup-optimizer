#!/usr/bin/env python3
"""
Agent 1: Collect 2020-21 Premier League Season Data

This agent collects all player gameweek data for the 2020-21 season.
Since FPL API only serves current season, this uses alternative data sources.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Collect 2020-21 season data."""
    print("="*70)
    print("AGENT 1: Collecting 2020-21 Premier League Season")
    print("="*70)
    
    # Output paths
    output_dir = Path('data/historical/agents')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'pl_2020_21.json'
    
    # In a real implementation, this would:
    # 1. Try FPL API historical endpoints
    # 2. Fall back to GitHub historical datasets
    # 3. Or use Fantasy Football Scout data
    
    # For now, create structure for manual data loading
    season_data = {
        'season': '2020-21',
        'agent_id': 1,
        'collection_date': datetime.now().isoformat(),
        'source': 'historical_archive',
        'gameweeks': 38,
        'players': [],
        'status': 'pending_data_source',
        'note': 'TODO: Connect to historical data source (GitHub/vaastav/Fantasy-Premier-League)'
    }
    
    with open(output_file, 'w') as f:
        json.dump(season_data, f, indent=2)
    
    print(f"✅ Agent 1: Structure created at {output_file}")
    print("📊 Next: Load actual data from historical source")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
