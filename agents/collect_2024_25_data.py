#!/usr/bin/env python3
"""
Agent 2: 2024-25 Data Collection

Fetches the current season's FPL data to add to our training dataset.
Target: 15,000+ new samples

Author: Research Agent
Date: 2026-03-13
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataCollection-2024-25')


class CurrentSeasonCollector:
    """Collect current season FPL data."""
    
    def __init__(self, season='2024-25'):
        self.season = season
        self.base_url = 'https://raw.githubusercontent.com/vaastav/FPL/master/data/2024-25/'
        self.output_dir = Path('data/historical/raw')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            'season': season,
            'collection_date': datetime.now().isoformat(),
            'source': 'vaastav/FPL GitHub',
            'files': {}
        }
        
        logger.info(f"Initialized collector for season {season}")
    
    def fetch_file(self, filename: str, subdir: str = '') -> Optional[pd.DataFrame]:
        """Fetch a single file from GitHub."""
        url = f"{self.base_url}{subdir}{filename}"
        
        try:
            logger.info(f"Fetching {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save raw content
            output_path = self.output_dir / f"2024_25_{subdir.replace('/', '_')}{filename}"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Parse as DataFrame
            df = pd.read_csv(output_path)
            logger.info(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            return None
    
    def collect_gameweek_data(self) -> Optional[pd.DataFrame]:
        """Collect merged gameweek data."""
        logger.info("Collecting gameweek data...")
        
        df = self.fetch_file('merged_gw.csv', 'gws/')
        
        if df is None:
            logger.warning("2024-25 data not yet available on GitHub.")
            logger.info("This is expected for the current season.")
            logger.info("Consider collecting via FPL API instead.")
            return None
        
        self.metadata['files']['gws_merged_gw.csv'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'players': df['name'].nunique() if 'name' in df.columns else None
        }
        
        return df
    
    def collect_player_data(self) -> Optional[pd.DataFrame]:
        """Collect player raw data."""
        logger.info("Collecting player data...")
        
        df = self.fetch_file('players_raw.csv')
        
        if df is not None:
            self.metadata['files']['players_raw.csv'] = {
                'rows': len(df),
                'columns': len(df.columns)
            }
        
        return df
    
    def collect_team_data(self) -> Optional[pd.DataFrame]:
        """Collect team data."""
        logger.info("Collecting team data...")
        
        df = self.fetch_file('teams.csv')
        
        if df is not None:
            self.metadata['files']['teams.csv'] = {
                'rows': len(df),
                'columns': len(df.columns)
            }
        
        return df
    
    def aggregate_data(self, gws_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate and process the data."""
        logger.info("Aggregating data...")
        
        # Add metadata columns
        gws_df['season'] = self.season
        gws_df['collection_timestamp'] = datetime.now().isoformat()
        
        # Calculate derived features (same as historical aggregation)
        if 'total_points' in gws_df.columns and 'minutes' in gws_df.columns:
            gws_df['points_per_90'] = (
                gws_df['total_points'] / (gws_df['minutes'] / 90)
            ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Position encoding
        if 'position' in gws_df.columns:
            position_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            gws_df['position_code'] = gws_df['position'].map(position_map)
        
        # Form calculations (rolling averages)
        if all(col in gws_df.columns for col in ['name', 'GW', 'total_points']):
            gws_df = gws_df.sort_values(['name', 'GW'])
            gws_df['form_3gw'] = gws_df.groupby('name')['total_points'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            gws_df['form_5gw'] = gws_df.groupby('name')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
        
        logger.info(f"  ✓ Aggregated {len(gws_df)} rows")
        return gws_df
    
    def update_master_dataset(self, new_data: pd.DataFrame):
        """Update the master training dataset."""
        logger.info("Updating master dataset...")
        
        master_path = Path('datasets/fpl_multi_year/fpl_historical_unified.csv')
        
        if master_path.exists():
            # Load existing
            master = pd.read_csv(master_path)
            logger.info(f"  Loaded existing: {len(master)} rows")
            
            # Append new
            combined = pd.concat([master, new_data], ignore_index=True)
            
            # Remove duplicates
            before_dedup = len(combined)
            combined = combined.drop_duplicates(subset=['name', 'GW', 'season'], keep='last')
            after_dedup = len(combined)
            
            logger.info(f"  Combined: {before_dedup} rows")
            logger.info(f"  After dedup: {after_dedup} rows")
            logger.info(f"  New samples added: {after_dedup - len(master)}")
            
            self.metadata['samples_added'] = after_dedup - len(master)
            self.metadata['total_samples'] = after_dedup
        else:
            combined = new_data
            logger.info(f"  Created new dataset: {len(combined)} rows")
            self.metadata['samples_added'] = len(combined)
            self.metadata['total_samples'] = len(combined)
        
        # Save updated master
        combined.to_csv(master_path, index=False)
        
        # Also save Parquet for efficiency
        combined.to_parquet(
            master_path.with_suffix('.parquet'),
            compression='gzip'
        )
        
        logger.info(f"  ✓ Saved to {master_path}")
        return combined
    
    def generate_report(self) -> Dict:
        """Generate collection report."""
        report = {
            'experiment': '2024-25 Data Collection',
            'timestamp': datetime.now().isoformat(),
            'season': self.season,
            'metadata': self.metadata,
            'status': 'complete',
            'next_steps': [
                'Run aggregate_historical_data.py to regenerate train/test splits',
                'Retrain EXP-031 with updated dataset',
                'Evaluate if more data improves model performance'
            ]
        }
        
        # Save report
        report_path = Path('research/agents/agent2_results') / f'collection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run(self):
        """Run data collection."""
        logger.info("="*70)
        logger.info(f"2024-25 DATA COLLECTION - Agent 2")
        logger.info("="*70)
        
        # Collect data
        gws_df = self.collect_gameweek_data()
        players_df = self.collect_player_data()
        teams_df = self.collect_team_data()
        
        if gws_df is None:
            logger.warning("2024-25 season data not yet available.")
            logger.info("This is normal - the season is still in progress.")
            logger.info("Falling back to using existing dataset for research.")
            
            # Generate report with status
            report = {
                'experiment': '2024-25 Data Collection',
                'timestamp': datetime.now().isoformat(),
                'season': self.season,
                'status': 'skipped',
                'reason': 'Data not yet available on GitHub (season in progress)',
                'alternative': 'Use existing 52k sample dataset',
                'next_steps': [
                    'Try again in a few weeks when season is archived',
                    'Use FPL API for real-time data collection',
                    'Continue with existing dataset (52k samples is sufficient)'
                ]
            }
            
            # Save report
            report_path = Path('research/agents/agent2_results') / f'collection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
        
        # Aggregate
        aggregated = self.aggregate_data(gws_df)
        
        # Update master dataset
        updated = self.update_master_dataset(aggregated)
        
        # Generate report
        report = self.generate_report()
        
        logger.info("="*70)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Samples collected: {self.metadata.get('samples_added', 0)}")
        logger.info(f"Total dataset: {self.metadata.get('total_samples', 0)}")
        
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Collect 2024-25 season data')
    parser.add_argument('--season', type=str, default='2024-25')
    args = parser.parse_args()
    
    collector = CurrentSeasonCollector(season=args.season)
    report = collector.run()
    
    if report:
        print("\n" + "="*70)
        print("2024-25 DATA COLLECTION COMPLETE")
        print("="*70)
        
        if 'metadata' in report:
            print(f"Samples added: {report['metadata'].get('samples_added', 0)}")
            print(f"Total samples: {report['metadata'].get('total_samples', 0)}")
        else:
            print(f"Status: {report.get('status', 'unknown')}")
            print(f"Reason: {report.get('reason', 'N/A')}")
        
        print()
        print("Next steps:")
        for step in report.get('next_steps', []):
            print(f"  • {step}")
    else:
        print("\n✗ Data collection failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
