#!/usr/bin/env python3
"""
Feature Engineering Pipeline for FPL ML Integration

Computes ML features from raw FPL data and stores them in the feature store.

Features include:
- Rolling statistics (1, 3, 6 gameweek windows)
- Fixture difficulty (FDR)
- Team form metrics
- Player availability indicators
- Interaction features
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes ML features from raw FPL data."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def load_raw_data(self, season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw players and fixtures data for a season."""
        logger.info(f"Loading raw data for {season}...")

        # Load players from database
        players_query = """
        SELECT * FROM players 
        WHERE created_at >= :season_start
        """
        # For simplicity, just load all players
        players_df = pd.read_sql(
            "SELECT * FROM players WHERE team IS NOT NULL", self.engine
        )

        # Load fixtures
        fixtures_query = "SELECT * FROM fixtures WHERE season = :season"
        # We'll need to load fixtures from a file or table
        # For now, create a placeholder
        fixtures_df = pd.DataFrame(
            columns=["event", "team_h", "team_a", "kickoff_time"]
        )

        logger.info(
            f"   ✓ Loaded {len(players_df)} players, {len(fixtures_df)} fixtures"
        )
        return players_df, fixtures_df

    def compute_rolling_stats(
        self, df: pd.DataFrame, player_id: int, metric: str, window: int = 3
    ) -> float:
        """
        Compute rolling statistic for a player.
        This is a simplified version - in production would use actual gameweek history.
        """
        # Placeholder: in real implementation, would query gameweek history
        # For now, return a simple value
        player_row = df[df["id"] == player_id]
        if player_row.empty:
            return 0.0

        base_value = float(player_row.iloc[0].get(metric, 0))
        return base_value / max(1, window)  # Simulate rolling average

    def compute_fixture_difficulty(
        self,
        player_id: int,
        team: int,
        fixtures_df: pd.DataFrame,
        num_fixtures: int = 3,
    ) -> float:
        """
        Compute average fixture difficulty for a player's team over next N fixtures.
        """
        # Placeholder: would look at upcoming fixtures and team FDR
        # For now, return random-ish value between 2-5
        return 3.0 + np.random.random() * 2

    def compute_team_form(
        self, team: int, fixtures_df: pd.DataFrame, window: int = 3
    ) -> Dict[str, float]:
        """
        Compute team form metrics (points, goals, conceded) over last N fixtures.
        """
        # Placeholder
        return {
            "team_points_rolling_3": np.random.randint(0, 10),
            "team_goals_rolling_3": np.random.randint(0, 10),
            "team_conceded_rolling_3": np.random.randint(0, 10),
        }

    def create_features_for_player(
        self,
        player_row: pd.Series,
        all_players: pd.DataFrame,
        fixtures_df: pd.DataFrame,
    ) -> Dict:
        """
        Create feature vector for a single player for the upcoming gameweek.
        """
        features = {
            "player_id": int(player_row["id"]),
            "now_cost": int(player_row["now_cost"]),
            "element_type": int(player_row["element_type"]),
            "team": int(player_row["team"]),
            "season": "2024-25",
            "gameweek": 1,  # TODO: Get actual next gameweek
        }

        # Rolling statistics
        for metric in [
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
        ]:
            for window in [1, 3, 6]:
                col_name = f"{metric}_rolling_{window}"
                features[col_name] = self.compute_rolling_stats(
                    all_players, player_row["id"], metric, window
                )

        # Fixture difficulty
        features["avg_fdr_next_3"] = self.compute_fixture_difficulty(
            player_row["id"], player_row["team"], fixtures_df, 3
        )
        features["avg_fdr_next_6"] = self.compute_fixture_difficulty(
            player_row["id"], player_row["team"], fixtures_df, 6
        )

        # Double/blank gameweeks
        features["dgw_next_3"] = 0  # Placeholder
        features["dgw_next_6"] = 0
        features["bgw_next_3"] = 0

        # Team form
        team_form = self.compute_team_form(player_row["team"], fixtures_df)
        features.update(team_form)

        # Player availability
        features["is_injured"] = player_row.get("status") in ["i", "u"]
        features["is_suspended"] = False  # TODO: Determine from cards data
        features["injury_risk_score"] = 0.0  # TODO: Compute from injury history

        # Form trend (simple linear regression on recent points)
        features["form_trend"] = np.random.normal(0, 1)  # Placeholder

        # Efficiency metrics
        minutes = float(player_row.get("minutes", 0))
        total_points = float(player_row.get("total_points", 0))
        cost = float(player_row["now_cost"]) / 10.0

        features["points_per_minute"] = total_points / max(1, minutes)
        features["value_ratio"] = total_points / max(1, cost)

        # Ownership factor
        selected_pct = float(player_row.get("selected_by_percent", 0))
        features["ownership_factor"] = selected_pct

        # Interaction features
        features["home_advantage"] = True  # Placeholder - need fixture data
        features["opp_strength"] = 3.0  # Placeholder - opponent defensive strength

        # Target variable (actual points for this GW - will be filled later)
        features["actual_points"] = None

        return features

    def generate_feature_store(self, season: str, gameweek: int = None):
        """
        Generate features for all players and store in database.

        Args:
            season: Season identifier (e.g., '2024-25')
            gameweek: Specific gameweek to compute features for (None = all)
        """
        logger.info(f"Generating feature store for {season}, GW{gameweek or 'all'}")

        # Load raw data
        players_df, fixtures_df = self.load_raw_data(season)

        # Generate features for each player
        all_features = []

        for idx, player_row in players_df.iterrows():
            try:
                features = self.create_features_for_player(
                    player_row, players_df, fixtures_df
                )
                all_features.append(features)
            except Exception as e:
                logger.warning(
                    f"Failed to compute features for player {player_row['id']}: {e}"
                )
                continue

        features_df = pd.DataFrame(all_features)

        # Store in database
        logger.info(f"Storing {len(features_df)} feature records...")

        # Drop existing features for this season/gameweek if updating
        with self.engine.begin() as conn:
            if gameweek:
                conn.execute(
                    text("""
                    DELETE FROM players_features 
                    WHERE season = :season AND gameweek = :gameweek
                """),
                    {"season": season, "gameweek": gameweek},
                )
            else:
                conn.execute(
                    text("""
                    DELETE FROM players_features WHERE season = :season
                """),
                    {"season": season},
                )

        # Insert new features
        features_df.to_sql(
            "players_features", self.engine, if_exists="append", index=False
        )

        logger.info("✅ Feature store generation complete!")

        # Log feature statistics
        self.log_feature_stats(features_df)

        return features_df

    def log_feature_stats(self, features_df: pd.DataFrame):
        """Log statistics about generated features."""
        logger.info("\nFeature Statistics:")
        logger.info(f"   Total players: {len(features_df)}")

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        logger.info(f"   Numeric features: {len(numeric_cols)}")

        # Check for missing values
        missing = features_df[numeric_cols].isnull().sum().sum()
        if missing > 0:
            logger.warning(f"   ⚠️  Found {missing} missing values in numeric features")
        else:
            logger.info("   ✅ No missing values in numeric features")

    def run(self, seasons: List[str], gameweek: int = None):
        """Run feature engineering for multiple seasons."""
        logger.info("🚀 Starting Feature Engineering Pipeline")
        logger.info(f"Seasons: {seasons}")

        results = {}
        for season in seasons:
            try:
                features_df = self.generate_feature_store(season, gameweek)
                results[season] = {
                    "status": "success",
                    "num_players": len(features_df),
                    "num_features": len(features_df.columns),
                }
            except Exception as e:
                logger.error(f"❌ Failed for {season}: {e}")
                results[season] = {"status": "failed", "error": str(e)}

        return results


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--seasons", default="2024-25", help="Comma-separated seasons")
    parser.add_argument("--gameweek", type=int, help="Specific gameweek (default: all)")
    parser.add_argument("--url", help="Database URL (overrides DATABASE_URL)")

    args = parser.parse_args()

    db_url = args.url or os.getenv(
        "DATABASE_URL", "postgresql://akshit:password@localhost/fpl_optimizer"
    )

    seasons = [s.strip() for s in args.seasons.split(",")]

    engineer = FeatureEngineer(db_url)
    results = engineer.run(seasons, args.gameweek)

    # Exit code based on success
    if all(r["status"] == "success" for r in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
