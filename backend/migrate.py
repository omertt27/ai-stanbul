#!/usr/bin/env python3
"""
Database Migration Utility
Provides easy commands for database migration management.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def init_database():
    """Initialize the database with current schema."""
    print("🚀 Initializing Database Migration System")
    
    # Get the Python executable path
    python_path = sys.executable
    
    # Mark database as current (no migrations to run)
    success = run_command(
        f"cd {backend_path} && {python_path} -m alembic stamp head",
        "Marking database as current"
    )
    
    if success:
        print("✅ Database migration system initialized!")
        print("📊 Current migration status:")
        run_command(
            f"cd {backend_path} && {python_path} -m alembic current",
            "Checking current migration status"
        )
    
    return success

def create_migration(message="Auto-generated migration"):
    """Create a new migration."""
    python_path = sys.executable
    
    success = run_command(
        f"cd {backend_path} && {python_path} -m alembic revision --autogenerate -m \"{message}\"",
        f"Creating migration: {message}"
    )
    
    if success:
        print(f"✅ Migration '{message}' created successfully!")
    
    return success

def upgrade_database():
    """Apply all pending migrations."""
    python_path = sys.executable
    
    success = run_command(
        f"cd {backend_path} && {python_path} -m alembic upgrade head",
        "Applying database migrations"
    )
    
    if success:
        print("✅ Database upgraded successfully!")
        show_status()
    
    return success

def downgrade_database(revision="base"):
    """Downgrade database to a specific revision."""
    python_path = sys.executable
    
    success = run_command(
        f"cd {backend_path} && {python_path} -m alembic downgrade {revision}",
        f"Downgrading database to {revision}"
    )
    
    if success:
        print(f"✅ Database downgraded to {revision} successfully!")
        show_status()
    
    return success

def show_status():
    """Show current migration status."""
    python_path = sys.executable
    
    print("📊 Current Migration Status:")
    run_command(
        f"cd {backend_path} && {python_path} -m alembic current",
        "Getting current revision"
    )
    
    print("\n📝 Migration History:")
    run_command(
        f"cd {backend_path} && {python_path} -m alembic history",
        "Getting migration history"
    )

def show_help():
    """Show available commands."""
    print("""
🗄️  Database Migration Utility

Available commands:
  init      - Initialize migration system (mark current DB as baseline)
  create    - Create a new migration (with optional message)
  upgrade   - Apply all pending migrations
  downgrade - Downgrade to previous revision (or specify revision)
  status    - Show current migration status
  help      - Show this help message

Examples:
  python migrate.py init
  python migrate.py create "Add user preferences table"
  python migrate.py upgrade
  python migrate.py downgrade
  python migrate.py status
""")

def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "init":
        init_database()
    elif command == "create":
        message = sys.argv[2] if len(sys.argv) > 2 else "Auto-generated migration"
        create_migration(message)
    elif command == "upgrade":
        upgrade_database()
    elif command == "downgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "-1"
        downgrade_database(revision)
    elif command == "status":
        show_status()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()
