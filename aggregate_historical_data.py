#!/usr/bin/env python3
"""
Aggregate Historical Data from All Agents

Combines data collected by parallel agents into unified training dataset.

Usage:
    python aggregate_historical_data.py [--input data/historical/] [--output datasets/fpl_multi_year/]
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_season_data(season: str, data_dir: Path) -> pd.DataFrame:
    """Load aggregated data for a season."""
    # Try multiple locations
    file_paths = [
        data_dir / f"{season.replace('-', '_')}_aggregated.csv",
        data_dir / 'raw' / f"{season.replace('-', '_')}_gws_merged_gw.csv",
        data_dir / f"{season.replace('-', '_')}_gws_merged_gw.csv",
    ]
    
    for file_path in file_paths:
        if file_path.exists():
            df = pd.read_csv(file_path)
            logger.info(f"  ✅ {season}: {len(df)} samples from {file_path.name}")
            return df
    
    logger.warning(f"  ⚠️  No data found for {season}")
    return pd.DataFrame()


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features across seasons."""
    # Handle different column names across seasons
    column_mapping = {
        'name': 'player_name',
        'element': 'player_id',
        'GW': 'gameweek',
        'value': 'price',
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Ensure all required columns exist
    required_cols = [
        'player_name', 'position', 'team', 'gameweek', 'season',
        'minutes', 'total_points', 'goals_scored', 'assists',
        'clean_sheets', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional features for modeling."""
    logger.info("🔧 Calculating derived features...")
    
    # Points per 90 minutes
    df['points_per_90'] = np.where(
        df['minutes'] > 0,
        df['total_points'] / df['minutes'] * 90,
        0
    )
    
    # Goal involvement
    df['goal_involvement'] = df['goals_scored'] + df['assists']
    
    # Rolling averages (if we have gameweek data)
    if 'gameweek' in df.columns:
        df = df.sort_values(['player_name', 'season', 'gameweek'])
        
        # 3-gameweek rolling form
        df['form_3gw'] = df.groupby(['player_name', 'season'])['total_points'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        
        # 5-gameweek rolling form
        df['form_5gw'] = df.groupby(['player_name', 'season'])['total_points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    
    # Position encoding
    position_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4, 'GKP': 1}
    if df['position'].dtype == object:
        df['position_code'] = df['position'].map(position_map).fillna(3).astype(int)
    else:
        df['position_code'] = df['position']
    
    logger.info(f"  ✅ Features calculated")
    return df


def aggregate_all_seasons(seasons: List[str], input_dir: Path, output_dir: Path):
    """Aggregate all season data into unified dataset."""
    logger.info("="*70)
    logger.info("AGGREGATING HISTORICAL DATA")
    logger.info("="*70)
    
    all_data = []
    
    for season in seasons:
        df = load_season_data(season, input_dir)
        if not df.empty:
            df = normalize_features(df)
            all_data.append(df)
    
    if not all_data:
        logger.error("❌ No data found from any season!")
        return None
    
    # Combine all seasons
    logger.info(f"\n📊 Combining {len(all_data)} seasons...")
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"  Total samples: {len(combined)}")
    
    # Calculate derived features
    combined = calculate_derived_features(combined)
    
    # Remove duplicates
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['player_name', 'season', 'gameweek'], keep='first')
    after_dedup = len(combined)
    if before_dedup != after_dedup:
        logger.info(f"  Removed {before_dedup - after_dedup} duplicates")
    
    # Save unified dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / 'fpl_historical_unified.csv'
    combined.to_csv(csv_path, index=False)
    logger.info(f"\n💾 Saved unified CSV: {csv_path}")
    
    # Save as Parquet (compressed)
    parquet_path = output_dir / 'fpl_historical_unified.parquet'
    combined.to_parquet(parquet_path, compression='gzip')
    logger.info(f"💾 Saved Parquet: {parquet_path}")
    
    # Save metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'seasons': seasons,
        'total_samples': len(combined),
        'players': int(combined['player_name'].nunique()),
        'gameweeks': int(combined['gameweek'].nunique()) if 'gameweek' in combined.columns else 0,
        'columns': list(combined.columns),
        'file_sizes': {
            'csv_mb': round(csv_path.stat().st_size / (1024*1024), 2),
            'parquet_mb': round(parquet_path.stat().st_size / (1024*1024), 2)
        }
    }
    
    meta_path = output_dir / 'unified_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"💾 Saved metadata: {meta_path}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("AGGREGATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nTotal samples: {len(combined):,}")
    logger.info(f"Unique players: {combined['player_name'].nunique():,}")
    logger.info(f"Seasons: {', '.join(seasons)}")
    
    # Position distribution
    if 'position_code' in combined.columns:
        pos_dist = combined.groupby('position_code')['player_name'].count()
        logger.info(f"\nPosition distribution:")
        pos_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        for code, count in pos_dist.items():
            logger.info(f"  {pos_names.get(code, code)}: {count:,}")
    
    # Season distribution
    season_dist = combined.groupby('season').size()
    logger.info(f"\nSeason distribution:")
    for season, count in season_dist.items():
        logger.info(f"  {season}: {count:,} samples")
    
    return combined


def create_train_test_split(df: pd.DataFrame, output_dir: Path, test_size: float = 0.2):
    """Create train/test split for ML."""
    logger.info("\n🔀 Creating train/test split...")
    
    # Sort by season and gameweek to ensure temporal split
    if 'season' in df.columns and 'gameweek' in df.columns:
        df = df.sort_values(['season', 'gameweek'])
    
    # Split: Use earlier seasons for train, later for test
    # Or random split if no temporal info
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Save
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"  Train: {len(train_df):,} samples ({train_path})")
    logger.info(f"  Test:  {len(test_df):,} samples ({test_path})")
    
    return train_df, test_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/historical', help='Input directory')
    parser.add_argument('--output', default='datasets/fpl_multi_year', help='Output directory')
    parser.add_argument('--seasons', nargs='+', 
                       default=['2020-21', '2021-22', '2022-23', '2023-24', '2024-25'],
                       help='Seasons to aggregate')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Aggregate
    combined = aggregate_all_seasons(args.seasons, input_dir, output_dir)
    
    if combined is not None and len(combined) > 100:
        # Create train/test split
        create_train_test_split(combined, output_dir)
        
        print("\n" + "="*70)
        print("✅ DATA READY FOR ML TRAINING")
        print("="*70)
        print(f"\nUse: datasets/fpl_multi_year/")
        print(f"  - train.csv: Training data")
        print(f"  - test.csv: Test data")
        print(f"\nRun: python train_exp031_full_features.py --data datasets/fpl_multi_year/")
        
        return 0
    else:
        print("\n❌ Insufficient data for ML training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
