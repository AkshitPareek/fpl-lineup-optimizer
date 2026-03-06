#!/usr/bin/env python3
"""
Data Exploration & Profiling for FPL ML Integration

Performs comprehensive EDA on historical FPL data to:
- Understand data distributions
- Identify predictive signals
- Detect data quality issues
- Generate feature importance insights
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FPLDataProfiler:
    """Profiles FPL data for ML feature engineering."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.report = {
            "generated_at": datetime.now().isoformat(),
            "database": db_url.split("/")[-1],
            "sections": {},
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load players, teams, and features from database."""
        logger.info("Loading data from database...")

        players = pd.read_sql("SELECT * FROM players", self.engine)
        teams = pd.read_sql("SELECT * FROM teams", self.engine)
        features = pd.read_sql("SELECT * FROM players_features", self.engine)

        logger.info(f"   ✓ Players: {len(players)} rows")
        logger.info(f"   ✓ Teams: {len(teams)} rows")
        logger.info(f"   ✓ Features: {len(features)} rows")

        return players, teams, features

    def basic_statistics(self, players: pd.DataFrame, features: pd.DataFrame):
        """Compute basic statistics."""
        logger.info("\n📊 Basic Statistics")

        stats = {
            "total_players": len(players),
            "total_teams": players["team"].nunique(),
            "seasons": features["season"].unique().tolist()
            if "season" in features.columns
            else [],
            "gameweeks": features["gameweek"].unique().tolist()
            if "gameweek" in features.columns
            else [],
            "avg_points_per_player": float(players["total_points"].mean()),
            "avg_minutes": float(players["minutes"].mean()),
            "avg_cost": float(players["now_cost"].mean() / 10.0),
        }

        self.report["sections"]["basic_statistics"] = stats

        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        return stats

    def position_analysis(self, players: pd.DataFrame) -> Dict:
        """Analyze player distributions by position."""
        logger.info("\n🏃 Position Analysis")

        position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        players["position_name"] = players["element_type"].map(position_map)

        analysis = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_players = players[players["position_name"] == pos]
            if len(pos_players) > 0:
                analysis[pos] = {
                    "count": len(pos_players),
                    "avg_points": float(pos_players["total_points"].mean()),
                    "avg_minutes": float(pos_players["minutes"].mean()),
                    "avg_cost": float(pos_players["now_cost"].mean() / 10.0),
                    "avg_goals": float(pos_players["goals_scored"].mean()),
                    "avg_assists": float(pos_players["assists"].mean()),
                }

        self.report["sections"]["position_analysis"] = analysis

        for pos, metrics in analysis.items():
            logger.info(
                f"   {pos}: {metrics['count']} players, "
                f"avg points={metrics['avg_points']:.1f}, "
                f"avg cost={metrics['avg_cost']:.1f}m"
            )

        return analysis

    def correlation_analysis(self, features: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlations between features and target (actual_points)."""
        logger.info("\n🔗 Correlation Analysis")

        # Select numeric features only
        numeric_features = features.select_dtypes(include=[np.number]).copy()

        # Remove non-predictive columns
        cols_to_drop = [
            "id",
            "player_id",
            "gameweek",
            "season",
            "feature_version",
            "actual_points",
        ]
        feature_cols = [c for c in numeric_features.columns if c not in cols_to_drop]

        if "actual_points" in numeric_features.columns:
            # Calculate correlations with target
            correlations = numeric_features[feature_cols + ["actual_points"]].corr()

            target_corr = (
                correlations["actual_points"]
                .drop("actual_points")
                .sort_values(ascending=False)
            )

            logger.info("   Top 10 features correlated with points:")
            for feature, corr in target_corr.head(10).items():
                logger.info(f"      {feature}: {corr:.3f}")

            self.report["sections"]["correlation_analysis"] = {
                "top_positive": target_corr.head(10).to_dict(),
                "top_negative": target_corr.tail(10).to_dict(),
            }

            return correlations
        else:
            logger.warning("   'actual_points' column not found in features")
            return pd.DataFrame()

    def data_quality_checks(self, players: pd.DataFrame, features: pd.DataFrame):
        """Run data quality validation."""
        logger.info("\n✅ Data Quality Checks")

        checks = {}

        # 1. Missing values
        missing_players = players.isnull().sum().sum()
        missing_features = features.isnull().sum().sum()
        checks["missing_values"] = {
            "players": int(missing_players),
            "features": int(missing_features),
        }

        # 2. Duplicates
        duplicate_players = players["id"].duplicated().sum()
        duplicate_features = features.duplicated(
            subset=["player_id", "gameweek", "season"]
        ).sum()
        checks["duplicates"] = {
            "players": int(duplicate_players),
            "features": int(duplicate_features),
        }

        # 3. Invalid costs (should be 30-150 in 10ths)
        invalid_costs = players[
            (players["now_cost"] < 30) | (players["now_cost"] > 150)
        ].shape[0]
        checks["invalid_costs"] = int(invalid_costs)

        # 4. Future leakage: features should not contain data from future gameweeks
        if "gameweek" in features.columns:
            future_data = features[features["gameweek"] > 38].shape[
                0
            ]  # Assuming max 38 GWs
            checks["future_leakage"] = int(future_data)

        # 5. Check for unreasonable values
        unreasonable_minutes = players[players["minutes"] > 3420].shape[0]  # > 38*90
        checks["unreasonable_minutes"] = int(unreasonable_minutes)

        self.report["sections"]["data_quality"] = checks

        # Log results
        all_passed = True
        for check, value in checks.items():
            if isinstance(value, dict):
                for subcheck, count in value.items():
                    if count > 0:
                        logger.warning(f"   ⚠️  {check}.{subcheck}: {count} issues")
                        all_passed = False
            elif value > 0:
                logger.warning(f"   ⚠️  {check}: {value} issues")
                all_passed = False

        if all_passed:
            logger.info("   ✅ All data quality checks passed!")

        return checks

    def feature_importance_analysis(self, features: pd.DataFrame) -> Dict:
        """
        Simple feature importance using correlation and variance.
        """
        logger.info("\n🎯 Feature Importance (Based on Correlation + Variance)")

        numeric_features = features.select_dtypes(include=[np.number]).copy()

        # Remove identifier columns
        identifier_cols = ["id", "player_id", "gameweek", "season", "feature_version"]
        feature_cols = [
            c
            for c in numeric_features.columns
            if c not in identifier_cols + ["actual_points"]
        ]

        importance_scores = {}

        if "actual_points" in numeric_features.columns:
            # Correlation-based importance
            correlations = (
                numeric_features[feature_cols + ["actual_points"]]
                .corr()["actual_points"]
                .drop("actual_points")
            )

            # Variance-based importance (higher variance = more information)
            variances = numeric_features[feature_cols].var()

            # Combined score: |correlation| * variance
            for feature in feature_cols:
                if feature in correlations.index and feature in variances.index:
                    corr = abs(correlations[feature])
                    var = variances[feature]
                    importance_scores[feature] = {
                        "correlation": float(corr),
                        "variance": float(var),
                        "score": float(corr * var),
                    }

            # Sort by combined score
            sorted_features = sorted(
                importance_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )

            logger.info("   Top 15 most important features:")
            for i, (feature, scores) in enumerate(sorted_features[:15], 1):
                logger.info(
                    f"      {i:2}. {feature:40} corr={scores['correlation']:.3f}, var={scores['variance']:.3f}"
                )

            self.report["sections"]["feature_importance"] = {
                "top_features": dict(sorted_features[:20])
            }

        return importance_scores

    def generate_visualizations(
        self, players: pd.DataFrame, features: pd.DataFrame, output_dir: Path
    ):
        """Generate EDA visualizations."""
        logger.info("\n📈 Generating Visualizations")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Points distribution by position
        if all(col in players.columns for col in ["total_points", "element_type"]):
            position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            players["position"] = players["element_type"].map(position_map)

            plt.figure(figsize=(10, 6))
            for pos in ["GK", "DEF", "MID", "FWD"]:
                subset = players[players["position"] == pos]
                if len(subset) > 0:
                    plt.hist(subset["total_points"], alpha=0.5, label=pos, bins=50)
            plt.xlabel("Total Points")
            plt.ylabel("Count")
            plt.title("Points Distribution by Position")
            plt.legend()
            plt.savefig(
                output_dir / "points_by_position.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            logger.info(f"   ✓ Created: points_by_position.png")

        # 2. Correlation heatmap (top 20 features)
        if "actual_points" in features.columns:
            numeric_features = features.select_dtypes(include=[np.number])
            cols_to_drop = ["id", "player_id", "gameweek", "season", "feature_version"]
            feature_cols = [
                c
                for c in numeric_features.columns
                if c not in cols_to_drop + ["actual_points"]
            ]

            if len(feature_cols) > 5:
                corr_matrix = numeric_features[
                    feature_cols[:20] + ["actual_points"]
                ].corr()

                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
                plt.title("Feature Correlation Matrix (Top 20 + Target)")
                plt.tight_layout()
                plt.savefig(
                    output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight"
                )
                plt.close()
                logger.info(f"   ✓ Created: correlation_heatmap.png")

        # 3. Cost vs Points scatter
        if all(col in players.columns for col in ["now_cost", "total_points"]):
            plt.figure(figsize=(10, 6))
            position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            players["position"] = players["element_type"].map(position_map)

            for pos in ["GK", "DEF", "MID", "FWD"]:
                subset = players[players["position"] == pos]
                if len(subset) > 0:
                    plt.scatter(
                        subset["now_cost"] / 10.0,
                        subset["total_points"],
                        alpha=0.5,
                        label=pos,
                        s=20,
                    )
            plt.xlabel("Cost (£m)")
            plt.ylabel("Total Points")
            plt.title("Player Cost vs Points by Position")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / "cost_vs_points.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"   ✓ Created: cost_vs_points.png")

        logger.info(f"✅ Visualizations saved to {output_dir}")

    def generate_report(self, output_file: Path):
        """Generate EDA report as JSON."""
        import json

        with open(output_file, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        logger.info(f"✅ EDA report saved to {output_file}")

    def run(self, output_dir: Path = Path("./eda_output")):
        """Run full EDA pipeline."""
        logger.info("🚀 Starting Data Profiling")

        # Load data
        players, teams, features = self.load_data()

        # Run analyses
        self.basic_statistics(players, features)
        self.position_analysis(players)
        self.correlation_analysis(features)
        self.data_quality_checks(players, features)
        self.feature_importance_analysis(features)

        # Generate visualizations
        output_dir.mkdir(parents=True, exist_ok=True)
        self.generate_visualizations(players, features, output_dir)

        # Save report
        report_file = output_dir / "eda_report.json"
        self.generate_report(report_file)

        logger.info("\n✅ EDA Complete!")
        logger.info(f"   Report: {report_file}")
        logger.info(f"   Visualizations: {output_dir}/")

        return self.report


def main():
    parser = argparse.ArgumentParser(description="Data profiling and EDA")
    parser.add_argument("--output", default="./eda_output", help="Output directory")
    parser.add_argument("--url", help="Database URL (overrides DATABASE_URL)")

    args = parser.parse_args()

    db_url = args.url or os.getenv(
        "DATABASE_URL", "postgresql://akshit:password@localhost/fpl_optimizer"
    )

    profiler = FPLDataProfiler(db_url)

    try:
        report = profiler.run(Path(args.output))

        # Exit with warnings if any data quality issues
        dq = report.get("sections", {}).get("data_quality", {})
        if any(
            v > 0 if isinstance(v, int) else any(subv > 0 for subv in v.values())
            for v in dq.values()
        ):
            logger.warning("Data quality issues detected - review required")
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"❌ EDA failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
