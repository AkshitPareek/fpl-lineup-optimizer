#!/usr/bin/env python3
"""
Historical FPL Data Loader

Downloads and loads historical FPL data from vaastav/Fantasy-Premier-League repository:
https://github.com/vaastav/Fantasy-Premier-League/tree/master/data

Processes data for multiple seasons and loads into PostgreSQL database.

Usage:
    python load_historical_data.py --season 2024-25
    python load_historical_data.py --all-seasons
    python load_historical_data.py --seasons 2023-24,2024-25
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import zipfile
import tempfile
from datetime import datetime

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Vaastav FPL data repository structure
VASTAAV_REPO = "https://github.com/vaastav/Fantasy-Premier-League/raw/master/data"
SEASON_MAPPING = {
    "2019-20": "2019-20",
    "2020-21": "2020-21",
    "2021-22": "2021-22",
    "2022-23": "2022-23",
    "2023-24": "2023-24",
    "2024-25": "2024-25",
}


class FPLDataLoader:
    """Loads historical FPL data into database."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def download_season_data(self, season: str, temp_dir: Path) -> Path:
        """
        Download season data from vaastav repository.

        Returns path to extracted data directory.
        """
        season_folder = SEASON_MAPPING.get(season)
        if not season_folder:
            raise ValueError(
                f"Invalid season: {season}. Valid: {list(SEASON_MAPPING.keys())}"
            )

        logger.info(f"Downloading {season} data...")

        # Download players_<season>.csv
        players_url = f"{VASTAAV_REPO}/{season_folder}/players_raw.csv"
        players_path = temp_dir / f"players_{season}.csv"

        response = requests.get(players_url, stream=True)
        response.raise_for_status()

        with open(players_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"   ✓ Downloaded players data: {players_path}")

        # Download fixtures
        fixtures_url = f"{VASTAAV_REPO}/{season_folder}/fixtures.csv"
        fixtures_path = temp_dir / f"fixtures_{season}.csv"

        response = requests.get(fixtures_url, stream=True)
        response.raise_for_status()

        with open(fixtures_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"   ✓ Downloaded fixtures: {fixtures_path}")

        # Download teams
        teams_url = f"{VASTAAV_REPO}/{season_folder}/teams.csv"
        teams_path = temp_dir / f"teams_{season}.csv"

        response = requests.get(teams_url, stream=True)
        response.raise_for_status()

        with open(teams_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"   ✓ Downloaded teams: {teams_path}")

        return temp_dir

    def load_teams(self, teams_path: Path, season: str):
        """Load teams data into database."""
        logger.info(f"Loading teams from {teams_path}")
        df = pd.read_csv(teams_path)

        # Map columns to our schema
        df_renamed = df.rename(
            columns={
                "id": "id",
                "name": "name",
                "short_name": "short_name",
                "code": "code",
                "strength": "strength",
                "strength_overall_home": "strength_overall_home",
                "strength_overall_away": "strength_overall_away",
                "strength_attack_home": "strength_attack_home",
                "strength_attack_away": "strength_attack_away",
                "strength_defence_home": "strength_defence_home",
                "strength_defence_away": "strength_defence_away",
                "pulse_id": "pulse_id",
            }
        )

        # Keep only columns we have in our schema
        columns_to_keep = [
            "id",
            "name",
            "short_name",
            "code",
            "strength",
            "strength_overall_home",
            "strength_overall_away",
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
            "pulse_id",
        ]

        df_clean = df_renamed[
            [c for c in columns_to_keep if c in df_renamed.columns]
        ].copy()
        df_clean["created_at"] = datetime.now()
        df_clean["updated_at"] = datetime.now()

        # Insert using ON CONFLICT DO NOTHING for idempotency
        with self.engine.begin() as conn:
            for _, row in df_clean.iterrows():
                # Build dynamic upsert
                cols = list(df_clean.columns)
                vals = [row[col] for col in cols]
                placeholders = ", ".join([":{}".format(c) for c in cols])

                insert_sql = f"""
                INSERT INTO teams ({", ".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at
                """
                conn.execute(text(insert_sql), dict(zip(cols, vals)))

        logger.info(f"   ✓ Loaded {len(df_clean)} teams")

    def load_players(self, players_path: Path, season: str):
        """Load players data into database."""
        logger.info(f"Loading players from {players_path}")
        df = pd.read_csv(players_path)

        # Map vaastav columns to our schema
        column_mapping = {
            "id": "id",
            "first_name": "first_name",
            "second_name": "second_name",
            "web_name": "web_name",
            "team": "team",
            "element_type": "element_type",
            "now_cost": "now_cost",
            "total_points": "total_points",
            "minutes": "minutes",
            "goals_scored": "goals_scored",
            "assists": "assists",
            "clean_sheets": "clean_sheets",
            "goals_conceded": "goals_conceded",
            "own_goals": "own_goals",
            "penalties_saved": "penalties_saved",
            "penalties_missed": "penalties_missed",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
            "saves": "saves",
            "bonus": "bonus",
            "bps": "bps",
            "selected_by_percent": "selected_by_percent",
            "form": "form",
            "ep_next": "ep_next",
            "ep_this": "ep_this",
            "status": "status",
        }

        # Rename columns that exist
        df_renamed = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Add missing columns with defaults
        for col in [
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "selected_by_percent",
            "form",
            "ep_next",
            "ep_this",
        ]:
            if col not in df_renamed.columns:
                df_renamed[col] = 0

        # Set status default
        if "status" not in df_renamed.columns:
            df_renamed["status"] = "a"

        # Add timestamps
        df_renamed["created_at"] = datetime.now()
        df_renamed["updated_at"] = datetime.now()

        # Ensure proper types
        numeric_cols = [
            "now_cost",
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "selected_by_percent",
            "form",
            "ep_next",
            "ep_this",
        ]

        for col in numeric_cols:
            if col in df_renamed.columns:
                df_renamed[col] = pd.to_numeric(
                    df_renamed[col], errors="coerce"
                ).fillna(0)

        # Select only columns in our table schema
        table_columns = [
            "id",
            "first_name",
            "second_name",
            "web_name",
            "team",
            "element_type",
            "now_cost",
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "selected_by_percent",
            "form",
            "ep_next",
            "ep_this",
            "status",
            "created_at",
            "updated_at",
        ]

        df_final = df_renamed[
            [c for c in table_columns if c in df_renamed.columns]
        ].copy()

        # Insert/update players
        with self.engine.begin() as conn:
            for _, row in df_final.iterrows():
                cols = list(df_final.columns)
                vals = [row[col] for col in cols]
                placeholders = ", ".join([":{}".format(c) for c in cols])

                insert_sql = f"""
                INSERT INTO players ({", ".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at
                """
                conn.execute(text(insert_sql), dict(zip(cols, vals)))

        logger.info(f"   ✓ Loaded {len(df_final)} players for {season}")

    def validate_data_quality(self, season: str) -> Dict:
        """
        Run data quality checks on loaded data.
        Returns dict with validation results.
        """
        logger.info(f"Running data quality validation for {season}...")

        validation = {
            "season": season,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "passed": True,
            "warnings": [],
        }

        with self.engine.connect() as conn:
            # Check team count
            result = conn.execute(text("SELECT COUNT(*) FROM teams")).scalar()
            validation["checks"]["teams_count"] = result
            if result < 20 or result > 30:  # PL has 20 teams
                validation["warnings"].append(f"Unexpected team count: {result}")

            # Check player count
            result = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
            validation["checks"]["players_count"] = result
            if result < 500 or result > 800:
                validation["warnings"].append(f"Unexpected player count: {result}")

            # Check for nulls in critical columns
            result = conn.execute(
                text("""
                SELECT COUNT(*) FROM players 
                WHERE web_name IS NULL OR team IS NULL OR element_type IS NULL
            """)
            ).scalar()
            validation["checks"]["null_critical_fields"] = result
            if result > 0:
                validation["passed"] = False
                validation["warnings"].append(
                    f"Found {result} players with null critical fields"
                )

            # Check that costs are reasonable (in 10ths, 30-130 range)
            result = conn.execute(
                text("""
                SELECT COUNT(*) FROM players 
                WHERE now_cost < 30 OR now_cost > 150
            """)
            ).scalar()
            validation["checks"]["unusual_costs"] = result
            if result > 0:
                validation["warnings"].append(
                    f"Found {result} players with unusual costs"
                )

            # Check for duplicate IDs
            result = conn.execute(
                text("""
                SELECT COUNT(*) FROM (
                    SELECT id FROM players GROUP BY id HAVING COUNT(*) > 1
                ) duplicates
            """)
            ).scalar()
            validation["checks"]["duplicate_players"] = result
            if result > 0:
                validation["passed"] = False
                validation["warnings"].append(f"Found {result} duplicate player IDs")

        # Log results
        if validation["passed"] and not validation["warnings"]:
            logger.info("   ✅ All quality checks passed")
        elif validation["warnings"]:
            logger.warning(f"   ⚠️  {len(validation['warnings'])} warnings:")
            for w in validation["warnings"]:
                logger.warning(f"      - {w}")
        else:
            logger.error("   ❌ Quality checks failed")

        return validation

    def process_season(self, season: str):
        """Process a single season: download, load, validate."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing season: {season}")
        logger.info(f"{'=' * 60}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download data
            data_dir = self.download_season_data(season, temp_path)

            # Load teams
            teams_path = data_dir / f"teams_{season}.csv"
            if teams_path.exists():
                self.load_teams(teams_path, season)

            # Load players
            players_path = data_dir / f"players_{season}.csv"
            if players_path.exists():
                self.load_players(players_path, season)

            # Run validation
            validation = self.validate_data_quality(season)

            return validation

    def run(self, seasons: List[str]):
        """Run loader for multiple seasons."""
        logger.info("🚀 Starting FPL historical data loader")
        logger.info(f"Database: {self.db_url}")
        logger.info(f"Seasons to process: {seasons}")

        results = []
        for season in seasons:
            try:
                result = self.process_season(season)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to process {season}: {e}")
                results.append({"season": season, "passed": False, "error": str(e)})

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        for result in results:
            status = "✅" if result.get("passed", False) else "❌"
            logger.info(f"{status} {result['season']}")

        # Count successes
        passed = sum(1 for r in results if r.get("passed", False))
        logger.info(f"\nSuccessfully processed {passed}/{len(seasons)} seasons")

        return results


def main():
    parser = argparse.ArgumentParser(description="Load historical FPL data")
    parser.add_argument(
        "--seasons",
        default="2024-25",
        help="Comma-separated seasons (e.g., '2023-24,2024-25')",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Load all available seasons (2019-20 to 2024-25)",
    )
    parser.add_argument("--url", help="Database URL (overrides DATABASE_URL)")

    args = parser.parse_args()

    # Get database URL
    db_url = args.url or os.getenv(
        "DATABASE_URL", "postgresql://akshit:password@localhost/fpl_optimizer"
    )

    # Determine seasons
    if args.all_seasons:
        seasons = list(SEASON_MAPPING.keys())
    else:
        seasons = [s.strip() for s in args.seasons.split(",")]

    # Validate seasons
    invalid = [s for s in seasons if s not in SEASON_MAPPING]
    if invalid:
        print(f"Invalid seasons: {invalid}")
        print(f"Valid seasons: {list(SEASON_MAPPING.keys())}")
        sys.exit(1)

    # Run loader
    loader = FPLDataLoader(db_url)
    results = loader.run(seasons)

    # Exit with appropriate code
    if all(r.get("passed", False) for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
