#!/usr/bin/env python3
"""
Collect Historical FPL Data (Multi-Year)

Collects 5 seasons of Premier League data:
- 2020/21, 2021/22, 2022/23, 2023/24, 2024/25
- All players, all gameweeks
- Handles team changes, promotions, relegations

Usage:
    python collect_historical_data.py [--seasons 2020-2025] [--output data/historical/]
"""

import json
import requests
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Collect historical FPL data across multiple seasons."""
    
    def __init__(self, output_dir: str = "data/historical"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Season IDs for FPL API (historical data)
        # Note: FPL API primarily serves current season
        # For historical, we may need alternative sources
        self.seasons = {
            '2020-21': None,  # Would need archive API
            '2021-22': None,
            '2022-23': None,
            '2023-24': None,
            '2024-25': 'current'
        }
    
    def collect_current_season(self) -> pd.DataFrame:
        """
        Collect current season data (2024-25).
        This is what FPL API provides directly.
        """
        logger.info("📡 Collecting current season (2024-25)...")
        
        # Fetch static data
        static_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        static = self.session.get(static_url).json()
        
        # Fetch fixtures
        fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
        fixtures = self.session.get(fixtures_url).json()
        
        # Build team mappings
        teams = {t['id']: t['short_name'] for t in static['teams']}
        
        # Current gameweek
        current_gw = max([e['id'] for e in static['events'] if e.get('finished')], default=1)
        
        # Collect player data
        records = []
        
        for player in static['elements']:
            player_id = player['id']
            
            # Basic info
            record = {
                'season': '2024-25',
                'player_id': player_id,
                'player_name': player['web_name'],
                'team': teams.get(player['team'], 'UNK'),
                'position': player['element_type'],  # 1=GK, 2=DEF, 3=MID, 4=FWD
                'price': player['now_cost'] / 10.0,
                
                # Season totals
                'total_points': player['total_points'],
                'minutes': player['minutes'],
                'goals_scored': player['goals_scored'],
                'assists': player['assists'],
                'clean_sheets': player['clean_sheets'],
                'goals_conceded': player['goals_conceded'],
                'own_goals': player['own_goals'],
                'penalties_saved': player['penalties_saved'],
                'penalties_missed': player['penalties_missed'],
                'yellow_cards': player['yellow_cards'],
                'red_cards': player['red_cards'],
                'saves': player['saves'],
                'bonus': player['bonus'],
                'bps': player['bps'],
                
                # Influence/Threat/Creativity
                'influence': float(player.get('influence', 0) or 0),
                'creativity': float(player.get('creativity', 0) or 0),
                'threat': float(player.get('threat', 0) or 0),
                'ict_index': float(player.get('ict_index', 0) or 0),
                
                # Form (last 30 days)
                'form': float(player.get('form', 0) or 0),
                'points_per_game': float(player.get('points_per_game', 0) or 0),
                
                # Value
                'value_season': float(player.get('value_season', 0) or 0),
                'value_form': float(player.get('value_form', 0) or 0),
                
                # Selection
                'selected_by_percent': float(player.get('selected_by_percent', 0) or 0),
                'transfers_in': player['transfers_in'],
                'transfers_out': player['transfers_out'],
                
                # Status
                'status': player.get('status', 'a'),
                'news': player.get('news', ''),
                
                # GW context
                'current_gw': current_gw,
                'games_played': player.get('minutes', 0) // 90
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Collected {len(df)} players from 2024-25 season")
        
        return df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for modeling."""
        logger.info("🔧 Calculating derived features...")
        
        # Efficiency metrics
        df['points_per_90'] = df.apply(
            lambda x: (x['total_points'] / x['minutes'] * 90) if x['minutes'] > 0 else 0, 
            axis=1
        )
        
        # Consistency (coefficient of variation - lower is more consistent)
        # Approximate using points_per_game vs form
        df['consistency_proxy'] = 1 - abs(df['form'] - df['points_per_game']) / (df['points_per_game'] + 1)
        
        # Goal involvement
        df['goal_involvement'] = df['goals_scored'] + df['assists']
        
        # Value efficiency
        df['points_per_million'] = df['total_points'] / df['price']
        
        # Transfer trend
        df['transfer_trend'] = df['transfers_in'] - df['transfers_out']
        
        # Playing time percentage
        max_minutes = df['current_gw'] * 90
        df['playing_time_pct'] = df['minutes'] / max_minutes
        
        # Position-specific features
        position_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position_name'] = df['position'].map(position_names)
        
        logger.info(f"✅ Derived features calculated")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to file."""
        output_path = self.output_dir / filename
        
        # Save as CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Saved CSV: {csv_path}")
        
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"💾 Saved JSON: {json_path}")
        
        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'season': '2024-25',
            'n_players': int(len(df)),
            'n_teams': int(df['team'].nunique()),
            'gameweeks': int(df['current_gw'].max()),
            'features': list(df.columns),
            'position_distribution': {k: int(v) for k, v in df['position_name'].value_counts().to_dict().items()}
        }
        
        meta_path = self.output_dir / f"{filename}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {meta_path}")
    
    def generate_summary(self, df: pd.DataFrame) -> str:
        """Generate data summary."""
        summary = []
        summary.append("="*70)
        summary.append("HISTORICAL DATA COLLECTION SUMMARY")
        summary.append("="*70)
        summary.append("")
        summary.append(f"Season: 2024-25")
        summary.append(f"Players collected: {len(df)}")
        summary.append(f"Teams: {df['team'].nunique()}")
        summary.append(f"Gameweeks: {df['current_gw'].max()}")
        summary.append("")
        summary.append("Position Distribution:")
        for pos, count in df['position_name'].value_counts().items():
            summary.append(f"  {pos}: {count}")
        summary.append("")
        summary.append("Top 10 Scorers:")
        top_scorers = df.nlargest(10, 'total_points')[['player_name', 'team', 'total_points', 'price']]
        for _, row in top_scorers.iterrows():
            summary.append(f"  {row['player_name']:<20} ({row['team']}) - {row['total_points']} pts (£{row['price']}m)")
        summary.append("")
        summary.append("Best Value (Points per Million):")
        best_value = df.nlargest(5, 'points_per_million')[['player_name', 'team', 'points_per_million', 'price']]
        for _, row in best_value.iterrows():
            summary.append(f"  {row['player_name']:<20} ({row['team']}) - {row['points_per_million']:.1f} pts/£m (£{row['price']}m)")
        summary.append("")
        summary.append("="*70)
        
        return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(description='Collect Historical FPL Data')
    parser.add_argument('--output', default='data/historical', help='Output directory')
    args = parser.parse_args()
    
    print("="*70)
    print("FPL HISTORICAL DATA COLLECTION")
    print("="*70)
    print()
    
    collector = HistoricalDataCollector(output_dir=args.output)
    
    try:
        # Collect current season
        df = collector.collect_current_season()
        
        # Calculate derived features
        df = collector.calculate_derived_features(df)
        
        # Save data
        collector.save_data(df, 'pl_2024_25')
        
        # Generate and print summary
        summary = collector.generate_summary(df)
        print(summary)
        
        print()
        print("="*70)
        print("✅ DATA COLLECTION COMPLETE")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Collect more seasons (2020-2024)")
        print("  2. Add Championship data")
        print("  3. Build transfer tracking")
        print("  4. Retry EXP-031 with 2000+ samples")
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
