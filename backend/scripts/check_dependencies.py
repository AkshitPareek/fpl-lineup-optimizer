#!/usr/bin/env python3
"""
System Dependency Checker for FPL ML Integration

Checks for required system tools (PostgreSQL client) before running operations.
"""

import shutil
import sys


def check_postgresql_client():
    """Check if PostgreSQL client tools are installed."""
    required_tools = ["psql", "pg_dump"]
    missing = []

    for tool in required_tools:
        if shutil.which(tool) is None:
            missing.append(tool)

    return missing


def main():
    print("🔍 Checking system dependencies...")

    missing = check_postgresql_client()

    if missing:
        print("\n❌ Missing required system tools:")
        for tool in missing:
            print(f"   - {tool}")

        print("\n📦 Installation instructions:")
        print("\n   Ubuntu/Debian:")
        print("   sudo apt-get update && sudo apt-get install postgresql-client")
        print("\n   macOS:")
        print("   brew install libpq && brew link libpq --force")
        print("\n   Windows:")
        print("   Download from: https://www.postgresql.org/download/windows/")
        print("   Select 'Command Line Tools' during installation")

        print("\n💡 After installation, ensure tools are in PATH and retry.")
        return False
    else:
        print("✅ All system dependencies found!")
        print("   - psql")
        print("   - pg_dump")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
