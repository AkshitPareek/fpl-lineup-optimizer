#!/usr/bin/env python3
"""
Database Backup & Recovery Script for FPL ML Integration

Provides automated backup, restore, and point-in-time recovery capabilities.

Usage:
    python backup_database.py backup --output /backup/path
    python backup_database.py restore --input /backup/path/backup_20250304.sql
    python backup_database.py list-backups
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import tarfile
import tempfile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseBackupManager:
    """Manages database backups and recovery."""

    def __init__(self, db_url: str, backup_dir: str = "/backup/fpl_optimizer"):
        self.db_url = db_url
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Parse database connection info
        parts = db_url.split("/")
        self.db_name = parts[-1]
        self.pg_host = "localhost"
        self.pg_port = "5432"

    def create_backup(self, compress: bool = True) -> Path:
        """
        Create a full database backup.

        Returns path to backup file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"fpl_optimizer_{timestamp}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sql_backup = temp_path / f"{backup_name}.sql"

            logger.info(f"Creating backup to {sql_backup}...")

            # Extract connection details
            # Format: postgresql://user:pass@host:port/dbname
            if self.db_url.startswith("postgresql://"):
                creds_host = self.db_url.split("@")[0].split("://")[1]
                user_pass = creds_host.split(":")[0]
                host_port = (
                    creds_host.split(":", 1)[1]
                    if ":" in creds_host
                    else "localhost:5432"
                )

                user = user_pass.split(":")[0]
                password = user_pass.split(":")[1] if ":" in user_pass else ""

                host = host_port.split(":")[0]
                port = host_port.split(":")[1] if ":" in host_port else "5432"

                # Set password via environment
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password

                # Run pg_dump
                cmd = [
                    "pg_dump",
                    "-h",
                    host,
                    "-p",
                    port,
                    "-U",
                    user,
                    "-d",
                    self.db_name,
                    "-f",
                    str(sql_backup),
                    "--no-owner",
                    "--no-acl",
                    "--verbose",
                ]

                result = subprocess.run(cmd, env=env, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"pg_dump failed: {result.stderr}")
                    raise RuntimeError(f"Backup failed: {result.stderr}")

                logger.info(
                    f"   ✓ SQL dump created: {sql_backup} ({sql_backup.stat().st_size / 1024 / 1024:.1f} MB)"
                )

            else:
                raise ValueError(f"Unsupported database URL format: {self.db_url}")

            # Compress if requested
            if compress:
                backup_file = self.backup_dir / f"{backup_name}.tar.gz"
                logger.info(f"Compressing backup to {backup_file}...")

                with tarfile.open(backup_file, "w:gz") as tar:
                    tar.add(sql_backup, arcname=f"{backup_name}.sql")

                # Clean up uncompressed SQL
                sql_backup.unlink()

                logger.info(
                    f"   ✓ Compressed backup created: {backup_file} ({backup_file.stat().st_size / 1024 / 1024:.1f} MB)"
                )
            else:
                backup_file = self.backup_dir / sql_backup.name
                sql_backup.rename(backup_file)

            # Create metadata file
            metadata = {
                "backup_name": backup_name,
                "database": self.db_name,
                "timestamp": timestamp,
                "compressed": compress,
                "size_bytes": backup_file.stat().st_size,
                "pg_dump_version": subprocess.run(
                    ["pg_dump", "--version"], capture_output=True, text=True
                ).stdout.strip(),
            }

            metadata_file = backup_file.with_suffix(".meta.json")
            import json

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Backup complete: {backup_file}")
            logger.info(f"   Metadata: {metadata_file}")

            return backup_file

    def restore_backup(self, backup_file: Path, target_db: str = None):
        """
        Restore database from backup file.

        Args:
            backup_file: Path to .tar.gz or .sql backup file
            target_db: Target database name (defaults to original)
        """
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        logger.info(f"Restoring from {backup_file}...")

        # Extract if compressed
        if backup_file.suffix == ".gz":
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                with tarfile.open(backup_file, "r:gz") as tar:
                    tar.extractall(temp_path)

                # Find SQL file
                sql_files = list(temp_path.glob("*.sql"))
                if not sql_files:
                    raise ValueError("No SQL file found in backup archive")

                sql_file = sql_files[0]
                target_db = target_db or self.db_name

                self._execute_sql_file(sql_file, target_db)
        else:
            target_db = target_db or self.db_name
            self._execute_sql_file(backup_file, target_db)

        logger.info(f"✅ Restore complete to database: {target_db}")

    def _execute_sql_file(self, sql_file: Path, target_db: str):
        """Execute SQL file to restore database."""
        # Extract connection details (same as backup)
        if self.db_url.startswith("postgresql://"):
            creds_host = self.db_url.split("@")[0].split("://")[1]
            user_pass = creds_host.split(":")[0]
            host_port = (
                creds_host.split(":", 1)[1] if ":" in creds_host else "localhost:5432"
            )

            user = user_pass.split(":")[0]
            password = user_pass.split(":")[1] if ":" in user_pass else ""
            host = host_port.split(":")[0]
            port = host_port.split(":")[1] if ":" in host_port else "5432"

            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password

            # Drop and recreate database
            logger.info(f"Dropping and recreating database '{target_db}'...")
            cmd_drop = [
                "psql",
                "-h",
                host,
                "-p",
                port,
                "-U",
                user,
                "-d",
                "postgres",
                "-c",
                f"DROP DATABASE IF EXISTS {target_db}",
            ]
            subprocess.run(cmd_drop, env=env, capture_output=True)

            cmd_create = [
                "psql",
                "-h",
                host,
                "-p",
                port,
                "-U",
                user,
                "-d",
                "postgres",
                "-c",
                f"CREATE DATABASE {target_db}",
            ]
            subprocess.run(cmd_create, env=env, capture_output=True)

            # Restore
            logger.info(f"Restoring SQL to '{target_db}'...")
            cmd_restore = [
                "psql",
                "-h",
                host,
                "-p",
                port,
                "-U",
                user,
                "-d",
                target_db,
                "-f",
                str(sql_file),
            ]

            result = subprocess.run(
                cmd_restore, env=env, capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"psql restore failed: {result.stderr}")
                raise RuntimeError(f"Restore failed: {result.stderr}")

            logger.info("   ✓ Database restored")

    def list_backups(self) -> list:
        """List all available backups."""
        backups = []

        for ext in [".tar.gz", ".sql"]:
            for backup_file in self.backup_dir.glob(f"fpl_optimizer_*{ext}"):
                meta_file = backup_file.with_suffix(".meta.json")

                backup_info = {
                    "file": backup_file,
                    "size_mb": backup_file.stat().st_size / 1024 / 1024,
                    "created": datetime.fromtimestamp(backup_file.stat().st_ctime),
                }

                if meta_file.exists():
                    import json

                    with open(meta_file) as f:
                        meta = json.load(f)
                        backup_info.update(meta)

                backups.append(backup_info)

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)

        return backups

    def cleanup_old_backups(self, keep_last: int = 10):
        """
        Delete old backups to save space.
        Keep the N most recent backups.
        """
        backups = self.list_backups()

        if len(backups) <= keep_last:
            logger.info(f"Only {len(backups)} backups exist, no cleanup needed")
            return

        to_delete = backups[keep_last:]
        logger.info(
            f"Cleaning up {len(to_delete)} old backups (keeping last {keep_last})..."
        )

        for backup in to_delete:
            try:
                backup["file"].unlink()
                meta_file = backup["file"].with_suffix(".meta.json")
                if meta_file.exists():
                    meta_file.unlink()
                logger.info(f"   ✓ Deleted {backup['file'].name}")
            except Exception as e:
                logger.error(f"   ❌ Failed to delete {backup['file']}: {e}")

        logger.info(
            f"✅ Cleanup complete, {len(self.list_backups())} backups remaining"
        )


def main():
    parser = argparse.ArgumentParser(description="Database backup and recovery")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument(
        "--output", help="Output directory (default: /backup/fpl_optimizer)"
    )
    backup_parser.add_argument(
        "--no-compress", action="store_true", help="Skip compression"
    )

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument(
        "--input", required=True, help="Backup file path (.sql or .tar.gz)"
    )
    restore_parser.add_argument(
        "--target-db", help="Target database name (default: original)"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    cleanup_parser.add_argument(
        "--keep", type=int, default=10, help="Number of backups to keep"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    db_url = os.getenv(
        "DATABASE_URL", "postgresql://akshit:password@localhost/fpl_optimizer"
    )
    backup_dir = args.output or os.getenv("BACKUP_DIR", "/backup/fpl_optimizer")

    manager = DatabaseBackupManager(db_url, backup_dir)

    try:
        if args.command == "backup":
            backup_file = manager.create_backup(compress=not args.no_compress)
            print(f"Backup created: {backup_file}")

        elif args.command == "restore":
            backup_path = Path(args.input)
            manager.restore_backup(backup_path, args.target_db)

        elif args.command == "list":
            backups = manager.list_backups()
            print(f"\nFound {len(backups)} backups:")
            print("-" * 80)
            for b in backups:
                meta_str = (
                    f" (v{b.get('model_version', 'N/A')})"
                    if "model_version" in b
                    else ""
                )
                print(
                    f"{b['created'].strftime('%Y-%m-%d %H:%M')} | "
                    f"{b['size_mb']:7.1f} MB | "
                    f"{b['file'].name}{meta_str}"
                )

        elif args.command == "cleanup":
            manager.cleanup_old_backups(args.keep)

    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
