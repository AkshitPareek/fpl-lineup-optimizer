#!/usr/bin/env python3
"""
Fetch Historical FPL Data from Multiple Sources

Tries multiple sources for historical FPL data:
1. GitHub: vaastav/Fantasy-Premier-League (most popular)
2. FPL API historical endpoints
3. Fantasy Football Scout (if accessible)
4. Local cache

Usage:
    python fetch_historical_data.py --season 2020-21 --output data/historical/
"""

import os
import sys
import json
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


def fetch_from_github(season: str, output_dir: Path) -> bool:
    """
    Fetch historical data from GitHub repository.
    Uses: https://github.com/vaastav/Fantasy-Premier-League
    """
    print(f"📡 Trying GitHub for {season}...")
    
    # The GitHub repo has data in format: data/2020-21/
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    
    # Map season format
    season_path = season  # Already in format 2020-21
    
    # Files to fetch
    files = ['players_raw.csv', 'gws/merged_gw.csv', 'teams.csv']
    
    season_data = {
        'season': season,
        'source': 'github/vaastav/Fantasy-Premier-League',
        'collection_date': datetime.now().isoformat(),
        'files': {}
    }
    
    success = False
    
    for file in files:
        url = f"{base_url}/{season_path}/{file}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                # Save
                file_name = file.replace('/', '_')
                output_file = output_dir / f"{season_path.replace('-', '_')}_{file_name}"
                df.to_csv(output_file, index=False)
                
                season_data['files'][file] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'path': str(output_file)
                }
                
                print(f"  ✅ {file}: {len(df)} rows")
                success = True
            else:
                print(f"  ⚠️  {file}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ {file}: {e}")
    
    # Save metadata
    if success:
        meta_file = output_dir / f"{season_path.replace('-', '_')}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(season_data, f, indent=2)
        print(f"  💾 Metadata saved")
    
    return success


def aggregate_season_data(season: str, input_dir: Path, output_dir: Path) -> int:
    """
    Aggregate raw season data into ML-ready format.
    Returns number of samples created.
    """
    print(f"🔧 Aggregating {season} data...")
    
    season_path = season.replace('-', '_')
    
    # Load files
    players_file = input_dir / f"{season_path}_players_raw.csv"
    gws_file = input_dir / f"{season_path}_merged_gw.csv"
    
    if not players_file.exists() or not gws_file.exists():
        print(f"  ❌ Missing files for {season}")
        return 0
    
    try:
        players = pd.read_csv(players_file)
        gws = pd.read_csv(gws_file)
        
        print(f"  📊 Players: {len(players)}, GW records: {len(gws)}")
        
        # Create gameweek-player features
        # This would merge player attributes with per-GW performance
        
        # For now, count potential samples
        # Each GW record is one sample
        samples = len(gws)
        
        # Save aggregated
        output_file = output_dir / f"{season_path}_aggregated.csv"
        
        # Select relevant columns
        feature_cols = [
            'name', 'position', 'team', 'GW', 'season',
            'minutes', 'total_points', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'bonus', 'bps',
            'influence', 'creativity', 'threat', 'ict_index',
            'value', 'was_home', 'opponent_team'
        ]
        
        available_cols = [c for c in feature_cols if c in gws.columns]
        aggregated = gws[available_cols].copy()
        aggregated['season'] = season
        
        aggregated.to_csv(output_file, index=False)
        print(f"  💾 Saved: {output_file} ({len(aggregated)} samples)")
        
        return len(aggregated)
        
    except Exception as e:
        print(f"  ❌ Aggregation failed: {e}")
        return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', required=True, help='Season to fetch (e.g., 2020-21)')
    parser.add_argument('--output', default='data/historical', help='Output directory')
    parser.add_argument('--source', choices=['github', 'fpl', 'all'], default='all')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"FETCHING HISTORICAL DATA: {args.season}")
    print("="*70)
    print()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Raw data directory
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)
    
    success = False
    
    # Try GitHub
    if args.source in ['github', 'all']:
        success = fetch_from_github(args.season, raw_dir)
    
    if success:
        # Aggregate
        samples = aggregate_season_data(args.season, raw_dir, output_dir)
        
        print()
        print("="*70)
        print("✅ SUCCESS")
        print("="*70)
        print(f"Season: {args.season}")
        print(f"Samples collected: {samples}")
        print(f"Output: {output_dir}")
        
        return 0
    else:
        print()
        print("="*70)
        print("❌ FAILED")
        print("="*70)
        print("Could not fetch data from any source")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
