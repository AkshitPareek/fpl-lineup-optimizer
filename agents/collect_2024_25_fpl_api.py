#!/usr/bin/env python3
"""
Agent 2 (Enhanced): Collect 2024-25 Season Data via FPL API

Fetches current season data directly from Fantasy Premier League API.
This is needed because the GitHub archive (vaastav/FPL) only has completed seasons.

Author: Research Agent
Date: 2026-03-13
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FPL-API-Collector')


@dataclass
class FPLPlayerGameweek:
    """Single player gameweek record."""
    player_id: int
    player_name: str
    team: str
    position: str
    gameweek: int
    
    # Performance metrics
    minutes: int
    total_points: int
    goals_scored: int
    assists: int
    clean_sheets: int
    bonus: int
    bps: int
    
    # Underlying stats
    influence: float
    creativity: float
    threat: float
    ict_index: float
    
    # Expected goals
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    
    # Fixture context
    was_home: bool
    opponent_team: str
    fixture_difficulty: int  # FDR rating
    
    # Ownership/Value
    value: int  # In 0.1m (e.g., 105 = £10.5m)
    transfers_balance: int
    selected: int
    transfers_in: int
    transfers_out: int


class FPLAPIClient:
    """Client for FPL API."""
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit  # Seconds between requests
        self._last_request = 0
        self._cache = {}
        
    def _get(self, endpoint: str, use_cache: bool = True) -> Dict:
        """Make rate-limited GET request."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Check cache
        if use_cache and url in self._cache:
            return self._cache[url]
        
        # Rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            self._last_request = time.time()
            self._cache[url] = data
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            raise
    
    def get_bootstrap_static(self) -> Dict:
        """Get static data (players, teams, fixtures)."""
        logger.info("Fetching bootstrap-static data...")
        return self._get("bootstrap-static/")
    
    def get_player_summary(self, player_id: int) -> Dict:
        """Get detailed player history."""
        logger.debug(f"Fetching player summary for ID {player_id}...")
        return self._get(f"element-summary/{player_id}/")
    
    def get_fixtures(self) -> List[Dict]:
        """Get all fixtures."""
        logger.info("Fetching fixtures...")
        return self._get("fixtures/")
    
    def get_gameweek_live(self, gameweek: int) -> Dict:
        """Get live data for a specific gameweek."""
        logger.info(f"Fetching live data for GW{gameweek}...")
        return self._get(f"event/{gameweek}/live/")


class CurrentSeasonCollector:
    """Collect 2024-25 season data via FPL API."""
    
    def __init__(self, season: str = '2024-25', output_dir: str = 'data/current_season'):
        self.season = season
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api = FPLAPIClient(rate_limit=0.5)
        
        # Metadata tracking
        self.metadata = {
            'season': season,
            'collection_date': datetime.now().isoformat(),
            'source': 'Fantasy Premier League API',
            'api_endpoint': self.api.BASE_URL,
            'players_collected': 0,
            'gameweeks_collected': 0,
            'total_records': 0
        }
        
        # Data storage
        self.static_data = None
        self.fixtures_data = None
        self.player_histories = []
        
        logger.info(f"FPL API Collector initialized for season {season}")
    
    def collect_static_data(self) -> Tuple[Dict, Dict]:
        """Collect static data (players, teams)."""
        logger.info("="*70)
        logger.info("STEP 1: Collecting Static Data")
        logger.info("="*70)
        
        # Fetch bootstrap static
        bootstrap = self.api.get_bootstrap_static()
        
        # Extract players
        players = bootstrap.get('elements', [])
        logger.info(f"Found {len(players)} players")
        
        # Extract teams
        teams = bootstrap.get('teams', [])
        logger.info(f"Found {len(teams)} teams")
        
        # Create team ID mapping
        self.team_map = {t['id']: t['name'] for t in teams}
        self.team_short_map = {t['id']: t['short_name'] for t in teams}
        
        # Create player ID mapping
        self.player_map = {}
        for p in players:
            self.player_map[p['id']] = {
                'name': f"{p['first_name']} {p['second_name']}",
                'team': self.team_map.get(p['team'], 'Unknown'),
                'position': self._get_position_name(p['element_type']),
                'now_cost': p['now_cost'],
                'selected_by_percent': p['selected_by_percent']
            }
        
        self.static_data = bootstrap
        logger.info("✓ Static data collected")
        
        return players, teams
    
    def _get_position_name(self, position_id: int) -> str:
        """Convert position ID to name."""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_id, 'Unknown')
    
    def collect_fixtures(self) -> List[Dict]:
        """Collect fixture data with FDR ratings."""
        logger.info("="*70)
        logger.info("STEP 2: Collecting Fixtures")
        logger.info("="*70)
        
        fixtures = self.api.get_fixtures()
        logger.info(f"Found {len(fixtures)} fixtures")
        
        # Create fixture lookup for gameweek context
        self.fixture_lookup = {}
        for f in fixtures:
            gw = f.get('event')
            if gw:
                if gw not in self.fixture_lookup:
                    self.fixture_lookup[gw] = []
                self.fixture_lookup[gw].append(f)
        
        self.fixtures_data = fixtures
        logger.info(f"✓ Fixtures organized by gameweek (GW1-{max(self.fixture_lookup.keys())})")
        
        return fixtures
    
    def collect_player_histories(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Collect detailed gameweek history for all players."""
        logger.info("="*70)
        logger.info("STEP 3: Collecting Player Histories")
        logger.info("="*70)
        
        if not self.static_data:
            self.collect_static_data()
        
        players = self.static_data.get('elements', [])
        if limit:
            players = players[:limit]
            logger.info(f"Limited to {limit} players for testing")
        
        all_records = []
        total_players = len(players)
        
        for i, player in enumerate(players):
            player_id = player['id']
            player_name = f"{player['first_name']} {player['second_name']}"
            
            logger.info(f"[{i+1}/{total_players}] Collecting history for {player_name} (ID: {player_id})...")
            
            try:
                summary = self.api.get_player_summary(player_id)
                history = summary.get('history', [])
                
                for gw_record in history:
                    record = self._process_gameweek_record(player, gw_record)
                    if record:
                        all_records.append(record)
                
                self.metadata['players_collected'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to collect for {player_name}: {e}")
                continue
        
        logger.info(f"✓ Collected {len(all_records)} gameweek records from {self.metadata['players_collected']} players")
        
        # Create DataFrame
        df = pd.DataFrame(all_records)
        self.player_histories = df
        self.metadata['total_records'] = len(df)
        
        return df
    
    def _process_gameweek_record(self, player: Dict, gw_record: Dict) -> Optional[Dict]:
        """Process a single gameweek record."""
        try:
            # Get fixture details
            fixture_id = gw_record.get('fixture')
            gw = gw_record.get('round')
            
            # Get opponent team
            opponent_team_id = gw_record.get('opponent_team')
            opponent_team = self.team_map.get(opponent_team_id, 'Unknown')
            
            # Get FDR from fixture if available
            fdr = 3  # Default medium difficulty
            if gw and gw in self.fixture_lookup:
                for fixture in self.fixture_lookup[gw]:
                    if fixture['id'] == fixture_id:
                        # Determine if player was home or away
                        was_home = gw_record.get('was_home', True)
                        if was_home:
                            fdr = fixture.get('team_h_difficulty', 3)
                        else:
                            fdr = fixture.get('team_a_difficulty', 3)
                        break
            
            return {
                'player_id': player['id'],
                'player_name': f"{player['first_name']} {player['second_name']}",
                'team': self.team_map.get(player['team'], 'Unknown'),
                'position': self._get_position_name(player['element_type']),
                'position_code': player['element_type'],
                'gameweek': gw,
                'season': self.season,
                
                # Performance
                'minutes': gw_record.get('minutes', 0),
                'total_points': gw_record.get('total_points', 0),
                'goals_scored': gw_record.get('goals_scored', 0),
                'assists': gw_record.get('assists', 0),
                'clean_sheets': gw_record.get('clean_sheets', 0),
                'bonus': gw_record.get('bonus', 0),
                'bps': gw_record.get('bps', 0),
                
                # Underlying stats
                'influence': float(gw_record.get('influence', 0)),
                'creativity': float(gw_record.get('creativity', 0)),
                'threat': float(gw_record.get('threat', 0)),
                'ict_index': float(gw_record.get('ict_index', 0)),
                
                # xG
                'expected_goals': float(gw_record.get('expected_goals', 0)),
                'expected_assists': float(gw_record.get('expected_assists', 0)),
                'expected_goal_involvements': float(gw_record.get('expected_goal_involvements', 0)),
                'expected_goals_conceded': float(gw_record.get('expected_goals_conceded', 0)),
                
                # Context
                'was_home': gw_record.get('was_home', True),
                'opponent_team': opponent_team,
                'fixture_difficulty': fdr,
                
                # Value/Ownership
                'value': gw_record.get('value', player['now_cost']),
                'transfers_balance': gw_record.get('transfers_balance', 0),
                'selected': gw_record.get('selected', 0),
                'transfers_in': gw_record.get('transfers_in', 0),
                'transfers_out': gw_record.get('transfers_out', 0),
                
                # Timestamp
                'collection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Error processing record: {e}")
            return None
    
    def calculate_derived_features(self, df: pd.DataForme) -> pd.DataFrame:
        """Calculate derived features (form, etc.)."""
        logger.info("="*70)
        logger.info("STEP 4: Calculating Derived Features")
        logger.info("="*70)
        
        # Sort by player and gameweek
        df = df.sort_values(['player_id', 'gameweek'])
        
        # Calculate rolling form
        df['form_3gw'] = df.groupby('player_id')['total_points'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df['form_5gw'] = df.groupby('player_id')['total_points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Points per 90 minutes
        df['points_per_90'] = (
            df['total_points'] / (df['minutes'] / 90)
        ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Goal involvement
        df['goal_involvement'] = df['goals_scored'] + df['assists']
        
        logger.info("✓ Derived features calculated")
        logger.info(f"  - Form 3GW: {df['form_3gw'].notna().sum()} records")
        logger.info(f"  - Form 5GW: {df['form_5gw'].notna().sum()} records")
        
        return df
    
    def save_data(self, df: pd.DataFrame):
        """Save collected data."""
        logger.info("="*70)
        logger.info("STEP 5: Saving Data")
        logger.info("="*70)
        
        # Save raw data
        csv_path = self.output_dir / f'{self.season}_fpl_api_data.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved CSV: {csv_path}")
        
        # Save Parquet (more efficient)
        parquet_path = self.output_dir / f'{self.season}_fpl_api_data.parquet'
        df.to_parquet(parquet_path, compression='gzip')
        logger.info(f"✓ Saved Parquet: {parquet_path}")
        
        # Save metadata
        metadata_path = self.output_dir / f'{self.season}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_path}")
        
        return csv_path, parquet_path
    
    def update_master_dataset(self, df: pd.DataFrame):
        """Add to master training dataset."""
        logger.info("="*70)
        logger.info("STEP 6: Updating Master Dataset")
        logger.info("="*70)
        
        master_path = Path('datasets/fpl_multi_year/fpl_historical_unified.csv')
        
        if master_path.exists():
            master = pd.read_csv(master_path)
            logger.info(f"Loaded existing master: {len(master)} records")
            
            # Combine
            combined = pd.concat([master, df], ignore_index=True)
            
            # Remove duplicates
            before = len(combined)
            combined = combined.drop_duplicates(
                subset=['player_name', 'gameweek', 'season'],
                keep='last'
            )
            after = len(combined)
            
            logger.info(f"Combined: {before} → {after} records (removed {before - after} duplicates)")
            
            samples_added = after - len(master)
            self.metadata['samples_added_to_master'] = samples_added
            logger.info(f"✓ New samples added: {samples_added}")
            
        else:
            combined = df
            logger.info(f"Created new master dataset: {len(combined)} records")
            self.metadata['samples_added_to_master'] = len(df)
        
        # Save updated master
        combined.to_csv(master_path, index=False)
        combined.to_parquet(master_path.with_suffix('.parquet'), compression='gzip')
        
        return combined
    
    def generate_report(self) -> Dict:
        """Generate collection report."""
        report = {
            'experiment': '2024-25 FPL API Data Collection',
            'timestamp': datetime.now().isoformat(),
            'season': self.season,
            'metadata': self.metadata,
            'status': 'complete',
            'summary': {
                'players_collected': self.metadata['players_collected'],
                'total_records': self.metadata['total_records'],
                'samples_added_to_master': self.metadata.get('samples_added_to_master', 0)
            },
            'next_steps': [
                'Regenerate train/test splits: python aggregate_historical_data.py',
                'Retrain EXP-031 with new data',
                'Re-run Ralph Loop with updated dataset',
                'Update dashboard with new statistics'
            ]
        }
        
        # Save report
        report_path = self.output_dir / f'collection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Report saved: {report_path}")
        return report
    
    def run(self, player_limit: Optional[int] = None):
        """Run full collection pipeline."""
        logger.info("="*70)
        logger.info(f"FPL API DATA COLLECTION - Season {self.season}")
        logger.info("="*70)
        
        try:
            # Step 1: Static data
            self.collect_static_data()
            
            # Step 2: Fixtures
            self.collect_fixtures()
            
            # Step 3: Player histories
            df = self.collect_player_histories(limit=player_limit)
            
            if len(df) == 0:
                logger.error("No data collected!")
                return None
            
            # Step 4: Derived features
            df = self.calculate_derived_features(df)
            
            # Step 5: Save
            self.save_data(df)
            
            # Step 6: Update master
            self.update_master_dataset(df)
            
            # Step 7: Report
            report = self.generate_report()
            
            logger.info("="*70)
            logger.info("COLLECTION COMPLETE")
            logger.info("="*70)
            logger.info(f"Players: {report['summary']['players_collected']}")
            logger.info(f"Records: {report['summary']['total_records']}")
            logger.info(f"Added to master: {report['summary']['samples_added_to_master']}")
            
            return report
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Collect 2024-25 data via FPL API')
    parser.add_argument('--season', type=str, default='2024-25')
    parser.add_argument('--limit', type=int, default=None, help='Limit players for testing')
    parser.add_argument('--output', type=str, default='data/current_season')
    args = parser.parse_args()
    
    collector = CurrentSeasonCollector(
        season=args.season,
        output_dir=args.output
    )
    
    report = collector.run(player_limit=args.limit)
    
    if report:
        print("\n" + "="*70)
        print("FPL API DATA COLLECTION COMPLETE")
        print("="*70)
        print(f"Season: {report['season']}")
        print(f"Players: {report['summary']['players_collected']}")
        print(f"Records: {report['summary']['total_records']}")
        print(f"New samples: {report['summary']['samples_added_to_master']}")
        print()
        print("Next steps:")
        for step in report['next_steps']:
            print(f"  • {step}")
    else:
        print("\n✗ Collection failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
