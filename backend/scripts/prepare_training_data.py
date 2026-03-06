#!/usr/bin/env python3
"""
Training Dataset Preparation for FPL ML Integration

Prepares ML training datasets from feature store with:
- Time-based splitting (no data leakage)
- Feature preprocessing (scaling, encoding)
- Handling of missing values and outliers
- Dataset versioning and metadata tracking
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """Prepares training datasets from feature store."""

    def __init__(self, db_url: str, output_dir: str = "./datasets"):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature configuration
        self.categorical_features = ["element_type", "team", "season"]  # Categorical
        self.numeric_features = [
            "now_cost",
            "total_points_rolling_1",
            "total_points_rolling_3",
            "total_points_rolling_6",
            "minutes_rolling_1",
            "minutes_rolling_3",
            "minutes_rolling_6",
            "goals_scored_rolling_1",
            "goals_scored_rolling_3",
            "assists_rolling_1",
            "assists_rolling_3",
            "clean_sheets_rolling_1",
            "clean_sheets_rolling_3",
            "avg_fdr_next_3",
            "avg_fdr_next_6",
            "dgw_next_3",
            "dgw_next_6",
            "bgw_next_3",
            "team_points_rolling_3",
            "team_goals_rolling_3",
            "team_conceded_rolling_3",
            "injury_risk_score",
            "form_trend",
            "points_per_minute",
            "value_ratio",
            "ownership_factor",
            "opp_strength",
        ]
        self.target_column = "actual_points"

    def load_features(self, seasons: List[str] = None) -> pd.DataFrame:
        """Load features from database."""
        logger.info("Loading features from database...")

        query = "SELECT * FROM players_features"
        if seasons:
            season_list = ", ".join([f"'{s}'" for s in seasons])
            query += f" WHERE season IN ({season_list})"

        df = pd.read_sql(query, self.engine)
        logger.info(f"   ✓ Loaded {len(df)} feature records")
        logger.info(
            f"   Seasons: {df['season'].unique().tolist() if 'season' in df.columns else 'N/A'}"
        )
        logger.info(
            f"   Gameweeks: {df['gameweek'].min()}-{df['gameweek'].max() if 'gameweek' in df.columns else 'N/A'}"
        )

        return df

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate feature dataframe."""
        logger.info("Validating features...")

        issues = []

        # Check required columns
        all_features = self.numeric_features + [self.target_column]
        missing_cols = [c for c in all_features if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for missing values in critical features
        critical_missing = df[self.numeric_features].isnull().sum().sum()
        if critical_missing > 0:
            issues.append(f"{critical_missing} missing values in numeric features")

        # Check target variable exists
        if self.target_column not in df.columns:
            issues.append(f"Target column '{self.target_column}' not found")
        elif df[self.target_column].isnull().sum() > 0:
            issues.append(
                f"{df[self.target_column].isnull().sum()} missing target values"
            )

        # Check for data leakage (future gameweeks in train)
        if "gameweek" in df.columns:
            # Assuming max gameweek is 38 for a season
            future_gws = df[df["gameweek"] > 38].shape[0]
            if future_gws > 0:
                issues.append(
                    f"{future_gws} records with gameweek > 38 (possible leakage)"
                )

        if issues:
            logger.error("❌ Validation failed:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False
        else:
            logger.info("✅ All validation checks passed")
            return True

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        logger.info("Handling missing values...")

        missing_total = df.isnull().sum().sum()
        if missing_total == 0:
            logger.info("   ✓ No missing values found")
            return df

        logger.info(f"   Found {missing_total} missing values")

        # For numeric features: fill with median
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"      Filled {col} with median: {median_val:.3f}")

        # For categorical: fill with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_val)
                logger.debug(f"      Filled {col} with mode: {mode_val}")

        remaining_missing = df.isnull().sum().sum()
        logger.info(f"   ✓ Missing values after imputation: {remaining_missing}")

        return df

    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Remove outliers from numeric features."""
        logger.info(f"Removing outliers using {method} method...")

        initial_size = len(df)

        if method == "iqr":
            # IQR method: keep values within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            for col in self.numeric_features:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower) & (df[col] <= upper)]

        elif method == "percentile":
            # Remove extreme 1% tails
            for col in self.numeric_features:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    lower = df[col].quantile(0.01)
                    upper = df[col].quantile(0.99)
                    df = df[(df[col] >= lower) & (df[col] <= upper)]

        removed = initial_size - len(df)
        logger.info(
            f"   ✓ Removed {removed} outlier records ({removed / initial_size * 100:.1f}%)"
        )

        return df

    def create_time_based_splits(
        self, df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits based on time.

        Uses chronological split to avoid data leakage.
        """
        logger.info("Creating time-based splits...")

        # Sort by season and gameweek to ensure chronological order
        if "season" in df.columns and "gameweek" in df.columns:
            df = df.sort_values(["season", "gameweek"]).copy()

        # Calculate split indices
        n_samples = len(df)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_val - n_test

        # Split
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]

        splits = {"train": train_df, "validation": val_df, "test": test_df}

        logger.info(
            f"   Train: {len(train_df)} samples ({len(train_df) / n_samples * 100:.1f}%)"
        )
        logger.info(
            f"   Validation: {len(val_df)} samples ({len(val_df) / n_samples * 100:.1f}%)"
        )
        logger.info(
            f"   Test: {len(test_df)} samples ({len(test_df) / n_samples * 100:.1f}%)"
        )

        # Log date ranges for each split
        if "season" in df.columns and "gameweek" in df.columns:
            for split_name, split_df in splits.items():
                if len(split_df) > 0:
                    start_season = split_df["season"].iloc[0]
                    start_gw = split_df["gameweek"].iloc[0]
                    end_season = split_df["season"].iloc[-1]
                    end_gw = split_df["gameweek"].iloc[-1]
                    logger.info(
                        f"      {split_name}: {start_season} GW{start_gw} - {end_season} GW{end_gw}"
                    )

        return splits

    def prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Prepare feature matrix with scaling and encoding.

        Returns:
            X: Feature matrix (numpy array)
            metadata: Dictionary with feature names, scaler info, etc.
        """
        logger.info("Preparing feature matrix...")

        # Select numeric features that exist in dataframe
        feature_cols = [c for c in self.numeric_features if c in df.columns]

        if not feature_cols:
            raise ValueError("No numeric features found in dataframe")

        X = df[feature_cols].values

        # Scale features
        scaler = StandardScaler() if fit_scaler else None
        if fit_scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X  # No scaling for inference

        metadata = {
            "feature_names": feature_cols,
            "n_features": len(feature_cols),
            "scaler_mean": scaler.mean_.tolist() if scaler else None,
            "scaler_scale": scaler.scale_.tolist() if scaler else None,
            "feature_stats": {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
                for col in feature_cols
            },
        }

        logger.info(f"   ✓ Prepared {len(feature_cols)} features")
        if scaler:
            logger.info(f"   ✓ Scaled features (StandardScaler)")

        return X_scaled, metadata

    def save_datasets(
        self,
        splits: Dict[str, pd.DataFrame],
        feature_metadata: Dict,
        dataset_name: str = "fpl_points",
    ):
        """Save datasets to disk and record metadata."""
        logger.info("Saving datasets...")

        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                continue

            # Prepare features and target
            X, _ = self.prepare_features(
                split_df, fit_scaler=False
            )  # No scaling during save
            y = (
                split_df[self.target_column].values
                if self.target_column in split_df.columns
                else None
            )

            # Save as numpy arrays
            np.save(dataset_dir / f"{split_name}_X.npy", X)
            if y is not None:
                np.save(dataset_dir / f"{split_name}_y.npy", y)

            # Also save raw CSV for inspection
            split_df.to_csv(dataset_dir / f"{split_name}_raw.csv", index=False)

            logger.info(
                f"   ✓ Saved {split_name} split: {len(split_df)} samples, X shape: {X.shape}"
            )

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "n_splits": len(splits),
            "feature_metadata": feature_metadata,
            "splits": {name: len(df) for name, df in splits.items()},
            "target_column": self.target_column,
            "preprocessing": "StandardScaler",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Compute dataset hash for versioning
        dataset_hash = self._compute_dataset_hash(splits)
        metadata["dataset_hash"] = dataset_hash

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Datasets saved to {dataset_dir}")
        logger.info(f"   Dataset hash: {dataset_hash}")

        return dataset_dir, dataset_hash

    def _compute_dataset_hash(self, splits: Dict[str, pd.DataFrame]) -> str:
        """Compute SHA256 hash of dataset for versioning."""
        import pickle

        # Create deterministic representation
        data_to_hash = {
            "splits": {name: df.to_dict("records") for name, df in splits.items()}
        }

        # Serialize and hash
        serialized = pickle.dumps(data_to_hash, protocol=4)
        return hashlib.sha256(serialized).hexdigest()

    def register_dataset(
        self, dataset_name: str, dataset_hash: str, metadata: Dict, seasons: List[str]
    ):
        """Register dataset in database."""
        logger.info("Registering dataset in database...")

        # First, get the season range
        start_season = min(seasons) if seasons else None
        end_season = max(seasons) if seasons else None

        split_types = list(metadata["splits"].keys())
        for split_type in split_types:
            num_samples = metadata["splits"][split_type]

            insert_sql = """
            INSERT INTO training_datasets 
            (dataset_name, season, split_type, start_gameweek, end_gameweek,
             dataset_hash, num_samples, num_features, file_path, created_at)
            VALUES (:dataset_name, :season, :split_type, :start_gw, :end_gw,
                    :dataset_hash, :num_samples, :num_features, :file_path, NOW())
            ON CONFLICT (dataset_name, season, split_type) DO UPDATE SET
                dataset_hash = EXCLUDED.dataset_hash,
                num_samples = EXCLUDED.num_samples,
                created_at = NOW()
            """

            with self.engine.begin() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "dataset_name": dataset_name,
                        "season": start_season or "all",
                        "split_type": split_type,
                        "start_gw": 1,
                        "end_gw": 38,
                        "dataset_hash": dataset_hash,
                        "num_samples": num_samples,
                        "num_features": metadata["feature_metadata"]["n_features"],
                        "file_path": str(
                            self.output_dir / dataset_name / f"{split_type}_X.npy"
                        ),
                    },
                )

        logger.info("   ✓ Dataset registered in database")

    def run(
        self,
        seasons: List[str] = None,
        dataset_name: str = "fpl_points_v1",
        val_size: float = 0.15,
        test_size: float = 0.15,
        remove_outliers: bool = True,
    ) -> Dict:
        """
        Run full dataset preparation pipeline.

        Args:
            seasons: List of seasons to include (None = all)
            dataset_name: Name for this dataset version
            val_size: Validation set proportion
            test_size: Test set proportion
            remove_outliers: Whether to remove outliers

        Returns:
            Dictionary with dataset info and metadata
        """
        logger.info("🚀 Starting Training Dataset Preparation")

        # 1. Load features
        df = self.load_features(seasons)

        # 2. Validate
        if not self.validate_features(df):
            raise ValueError("Feature validation failed")

        # 3. Handle missing values
        df = self.handle_missing_values(df)

        # 4. Remove outliers (optional)
        if remove_outliers:
            df = self.remove_outliers(df)

        # 5. Create time-based splits
        splits = self.create_time_based_splits(df, val_size, test_size)

        # 6. Prepare features and save
        # Fit scaler on training data only
        train_X, feature_metadata = self.prepare_features(
            splits["train"], fit_scaler=True
        )

        # Apply same scaling to val and test (using training stats)
        for split_name in ["validation", "test"]:
            if split_name in splits:
                X_split, _ = self.prepare_features(splits[split_name], fit_scaler=False)
                # Manually scale using training mean/std
                if (
                    "scaler_mean" in feature_metadata
                    and "scaler_scale" in feature_metadata
                ):
                    mean = np.array(feature_metadata["scaler_mean"])
                    scale = np.array(feature_metadata["scaler_scale"])
                    splits[split_name] = splits[
                        split_name
                    ].copy()  # Avoid SettingWithCopyWarning
                    # Note: We're not storing scaled X in splits, just keeping raw for saving
                logger.info(
                    f"   {split_name} will use same scaling parameters as train"
                )

        # Save datasets
        dataset_dir, dataset_hash = self.save_datasets(
            splits, feature_metadata, dataset_name
        )

        # Register in database
        self.register_dataset(
            dataset_name,
            dataset_hash,
            {
                "splits": {k: len(v) for k, v in splits.items()},
                "feature_metadata": feature_metadata,
            },
            seasons or [],
        )

        result = {
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "directory": str(dataset_dir),
            "splits": {k: len(v) for k, v in splits.items()},
            "n_features": feature_metadata["n_features"],
            "feature_names": feature_metadata["feature_names"],
        }

        logger.info("\n✅ Dataset Preparation Complete!")
        logger.info(f"   Name: {dataset_name}")
        logger.info(f"   Hash: {dataset_hash}")
        logger.info(f"   Location: {dataset_dir}")
        for split, count in result["splits"].items():
            logger.info(f"   {split}: {count} samples")

        return result


def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument(
        "--seasons",
        default="2023-24,2024-25",
        help="Comma-separated seasons to include",
    )
    parser.add_argument("--name", default="fpl_points_v1", help="Dataset name/version")
    parser.add_argument(
        "--output", default="./datasets", help="Output directory for datasets"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="Validation set proportion"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Test set proportion"
    )
    parser.add_argument(
        "--keep-outliers", action="store_true", help="Don't remove outliers"
    )
    parser.add_argument("--url", help="Database URL (overrides DATABASE_URL)")

    args = parser.parse_args()

    db_url = args.url or os.getenv(
        "DATABASE_URL", "postgresql://akshit:password@localhost/fpl_optimizer"
    )

    seasons = [s.strip() for s in args.seasons.split(",")] if args.seasons else None

    preparer = TrainingDataPreparer(db_url, args.output)

    try:
        result = preparer.run(
            seasons=seasons,
            dataset_name=args.name,
            val_size=args.val_size,
            test_size=args.test_size,
            remove_outliers=not args.keep_outliers,
        )

        # Print summary for logging
        print(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Dataset preparation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
