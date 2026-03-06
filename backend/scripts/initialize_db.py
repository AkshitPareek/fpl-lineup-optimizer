#!/usr/bin/env python3
"""
Database initialization and migration script for FPL ML Integration.

Usage:
    python initialize_db.py --url postgresql://user:pass@localhost/dbname
    python initialize_db.py --local  # Uses local fpl_optimizer DB
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports (do this after sys import)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


# Check system dependencies first
def check_system_dependencies():
    """Check if required system tools are installed."""
    try:
        import shutil

        missing = []
        for tool in ["psql", "pg_dump"]:
            if shutil.which(tool) is None:
                missing.append(tool)

        if missing:
            print("❌ Missing required system tools:")
            for tool in missing:
                print(f"   - {tool}")
            print("\n📦 Install PostgreSQL client tools:")
            print("   Ubuntu/Debian: sudo apt-get install postgresql-client")
            print("   macOS: brew install libpq && brew link libpq --force")
            print(
                "   Windows: Download from https://www.postgresql.org/download/windows/"
            )
            print("\n💡 After installation, ensure tools are in PATH and retry.")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not check system dependencies: {e}")
        return True  # Continue anyway, will fail later if missing


# Check dependencies first
if not check_system_dependencies():
    sys.exit(1)


def check_postgresql_installed():
    """Check if PostgreSQL client tools are available."""
    try:
        subprocess.run(["psql", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PostgreSQL client tools (psql) not found. Please install PostgreSQL.")
        return False


def get_database_url(env_var: str = "DATABASE_URL", default_local: str = None) -> str:
    """Get database URL from environment or use default."""
    db_url = os.getenv(env_var)
    if db_url:
        return db_url
    if default_local:
        return default_local
    raise ValueError(
        f"Database URL not found. Set {env_var} environment variable "
        "or pass --url argument."
    )


def run_migration(db_url: str, migration_file: Path):
    """Run a SQL migration file against the database."""
    print(f"📄 Running migration: {migration_file.name}")

    # Extract database name from URL for creation
    db_name = db_url.split("/")[-1].split("?")[0]

    # Try to connect to target database
    result = subprocess.run(
        ["psql", db_url, "-f", str(migration_file)], capture_output=True, text=True
    )

    if result.returncode != 0:
        # If database doesn't exist, try creating it first
        if "does not exist" in result.stderr:
            print(f"   ℹ️  Database '{db_name}' doesn't exist. Creating...")
            base_url = db_url.rsplit("/", 1)[0] + "/postgres"
            subprocess.run(
                ["psql", base_url, "-c", f"CREATE DATABASE {db_name}"],
                capture_output=True,
            )
            # Retry migration
            result = subprocess.run(
                ["psql", db_url, "-f", str(migration_file)],
                capture_output=True,
                text=True,
            )

    if result.returncode == 0:
        print(f"   ✅ Migration successful")
        return True
    else:
        print(f"   ❌ Migration failed:\n{result.stderr}")
        return False


def verify_schema(db_url: str) -> bool:
    """Verify that key tables exist after migration."""
    print("\n🔍 Verifying schema...")

    tables_to_check = [
        "teams",
        "players",
        "players_features",
        "model_registry",
        "model_predictions",
        "training_metadata",
        "prediction_logs",
    ]

    for table in tables_to_check:
        query = f"SELECT to_regclass('public.{table}');"
        result = subprocess.run(
            ["psql", db_url, "-c", query, "-t"], capture_output=True, text=True
        )
        if result.stdout.strip() == table:
            print(f"   ✅ Table '{table}' exists")
        else:
            print(f"   ❌ Table '{table}' missing")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Initialize FPL ML database")
    parser.add_argument("--url", help="Database URL (overrides DATABASE_URL)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local PostgreSQL database 'fpl_optimizer'",
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip running migrations, just verify",
    )

    args = parser.parse_args()

    # Determine database URL
    if args.local:
        db_url = "postgresql://akshit:password@localhost/fpl_optimizer"
    elif args.url:
        db_url = args.url
    else:
        db_url = get_database_url(
            default_local="postgresql://akshit:password@localhost/fpl_optimizer"
        )

    print(f"🗄️  Using database: {db_url.split('@')[-1]}")

    # Check PostgreSQL
    if not check_postgresql_installed():
        sys.exit(1)

    # Find migrations directory
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        print(f"❌ Migrations directory not found: {migrations_dir}")
        sys.exit(1)

    # Get all migration files sorted by name
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        print("❌ No migration files found")
        sys.exit(1)

    # Run migrations
    if not args.skip_migrations:
        print("\n🚀 Running migrations...")
        success = True
        for migration in migration_files:
            if not run_migration(db_url, migration):
                success = False
                break

        if not success:
            print("\n❌ Migration failed")
            sys.exit(1)

    # Verify schema
    if not verify_schema(db_url):
        print("\n❌ Schema verification failed")
        sys.exit(1)

    print("\n✅ Database initialization complete!")
    print("\nNext steps:")
    print("  1. Load historical FPL data: python scripts/load_historical_data.py")
    print("  2. Compute features: python scripts/compute_features.py")
    print("  3. Train ML models: python ml_training.py")


if __name__ == "__main__":
    main()
