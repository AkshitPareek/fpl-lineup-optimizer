#!/usr/bin/env python3
"""
Create dataset numpy arrays from the ml_training pipeline.

This script:
1. Loads data using FPLService/HistoricalDataService (as in ml_training.py)
2. Extracts features using extract_features_for_player
3. Splits temporally (train/val/test)
4. Scales features with StandardScaler
5. Saves to datasets/<name>/ train_X.npy, train_y.npy, etc.

Usage:
    python create_dataset_from_ml_pipeline.py --season 2025-26 --name fpl_2025_v1
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from ml_training import (
    load_data,
    extract_features_for_player,
    prepare_training_data,
    TEST_GW as DEFAULT_TEST_GW,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Create dataset from ml_training pipeline")
    parser.add_argument("--season", default="2025-26", help="Season to process")
    parser.add_argument("--test-gw", type=int, default=38, help="Test cutoff gameweek")
    parser.add_argument("--name", default="fpl_ml_latest", help="Dataset directory name (under datasets/)")
    parser.add_argument("--output-dir", default="datasets", help="Parent output directory")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    output_dir = project_root / args.output_dir / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating dataset '{args.name}' for season {args.season}, test_gw={args.test_gw}")

    # 1. Load data
    features_df, context_data = load_data([args.season])
    logger.info(f"Loaded {len(features_df)} total samples")

    # 2. Define base feature columns (same as ml_training.py)
    base_features = [
        "position",
        "now_cost",
        "team",
        "form_points",
        "form_minutes",
        "form_xg",
        "form_xa",
        "form_ict",
        "goals_per_90",
        "assists_per_90",
        "fixture_score",
        "double_gw_flag",
        "blank_gw_flag",
        "recent_4_minutes_avg",
        "start_probability",
        "ict_per_90",
        "understat_matched",
        "understat_xG_per_90",
        "understat_xA_per_90",
        "injury_status",
        "availability_factor",
    ]

    # 3. Temporal split (same as ml_training)
    TEST_GW = args.test_gw
    train_mask = features_df["target_gw"] < (TEST_GW - 3)
    val_mask = (features_df["target_gw"] >= (TEST_GW - 3)) & (features_df["target_gw"] < TEST_GW)
    test_mask = features_df["target_gw"] >= TEST_GW

    train_df = features_df[train_mask]
    val_df = features_df[val_mask]
    test_df = features_df[test_mask]

    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 4. Prepare feature matrices (returns arrays and column list)
    X_train, y_train, transformed_columns = prepare_training_data(train_df, base_features)
    X_val, y_val, _ = prepare_training_data(val_df, base_features, transformed_columns)
    X_test, y_test, _ = prepare_training_data(test_df, base_features, transformed_columns)

    logger.info(f"Feature matrix shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # 5. Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 6. Save numpy arrays
    np.save(output_dir / "train_X.npy", X_train_scaled)
    np.save(output_dir / "train_y.npy", y_train)
    np.save(output_dir / "validation_X.npy", X_val_scaled)
    np.save(output_dir / "validation_y.npy", y_val)
    np.save(output_dir / "test_X.npy", X_test_scaled)
    np.save(output_dir / "test_y.npy", y_test)

    logger.info(f"Saved scaled arrays to {output_dir}")

    # 7. Also save raw CSVs for inspection (optional)
    train_df.to_csv(output_dir / "train_raw.csv", index=False)
    val_df.to_csv(output_dir / "validation_raw.csv", index=False)
    test_df.to_csv(output_dir / "test_raw.csv", index=False)

    # 8. Create metadata.json with feature names and scaler stats
    metadata = {
        "dataset_name": args.name,
        "created_at": datetime.now().isoformat(),
        "n_splits": 3,
        "feature_metadata": {
            "feature_names": transformed_columns,
            "n_features": len(transformed_columns),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "target_column": "target_points",
        "preprocessing": "StandardScaler",
        "season": args.season,
        "test_gw": args.test_gw,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset creation complete!")

if __name__ == "__main__":
    main()
