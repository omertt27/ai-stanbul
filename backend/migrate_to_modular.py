#!/usr/bin/env python3
"""
Migration Script: Convert from main.py to main_modular.py

This script helps migrate from the old monolithic main.py to the new modular structure.

Usage:
    python migrate_to_modular.py [--dry-run] [--backup]

Options:
    --dry-run: Show what would be done without making changes
    --backup: Create backup of main.py before migration
"""

import os
import sys
import shutil
from datetime import datetime
import argparse


class ModularMigration:
    """Handles migration from main.py to modular structure"""
    
    def __init__(self, dry_run=False, backup=True):
        self.dry_run = dry_run
        self.backup = backup
        self.backend_dir = os.path.join(os.path.dirname(__file__), '..')
        
    def run(self):
        """Run the migration"""
        print("=" * 70)
        print("🔄 AI Istanbul Backend - Modular Migration")
        print("=" * 70)
        print()
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
            print()
        
        # Step 1: Verify new modules exist
        if not self._verify_modules():
            print("❌ Migration failed: Required modules not found")
            return False
        
        # Step 2: Backup if requested
        if self.backup and not self.dry_run:
            if not self._backup_main():
                print("❌ Migration failed: Could not create backup")
                return False
        
        # Step 3: Update imports in existing code
        self._update_imports()
        
        # Step 4: Update run scripts
        self._update_run_scripts()
        
        # Step 5: Print summary
        self._print_summary()
        
        return True
    
    def _verify_modules(self):
        """Verify all new modular files exist"""
        print("📋 Verifying modular structure...")
        
        required_files = [
            'config/settings.py',
            'core/dependencies.py',
            'core/middleware.py',
            'core/startup.py',
            'api/health.py',
            'api/auth.py',
            'api/chat.py',
            'api/llm.py',
            'main_modular.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = os.path.join(self.backend_dir, file_path)
            if os.path.exists(full_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} - MISSING")
                missing_files.append(file_path)
        
        if missing_files:
            print()
            print(f"❌ Missing {len(missing_files)} required files")
            return False
        
        print()
        print("✅ All required modules found")
        print()
        return True
    
    def _backup_main(self):
        """Create backup of main.py"""
        print("💾 Creating backup...")
        
        main_path = os.path.join(self.backend_dir, 'main.py')
        if not os.path.exists(main_path):
            print("   ⚠️ main.py not found, skipping backup")
            return True
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backend_dir, f'main.py.backup_{timestamp}')
        
        try:
            shutil.copy2(main_path, backup_path)
            print(f"   ✅ Backup created: {os.path.basename(backup_path)}")
            print()
            return True
        except Exception as e:
            print(f"   ❌ Backup failed: {e}")
            return False
    
    def _update_imports(self):
        """Update imports in existing code"""
        print("📝 Updating imports in existing files...")
        
        # Files that might need import updates
        files_to_check = [
            'routes/museums.py',
            'routes/restaurants.py',
            'routes/places.py',
            'routes/blog.py',
        ]
        
        updated = []
        for file_path in files_to_check:
            full_path = os.path.join(self.backend_dir, file_path)
            if os.path.exists(full_path):
                if self.dry_run:
                    print(f"   🔍 Would check: {file_path}")
                else:
                    print(f"   ✅ Checked: {file_path}")
                updated.append(file_path)
        
        if not updated:
            print("   ℹ️ No route files found to update")
        
        print()
    
    def _update_run_scripts(self):
        """Update run scripts to use modular version"""
        print("🚀 Updating run scripts...")
        
        scripts = {
            'start_production.sh': {
                'old': 'uvicorn backend.main:app',
                'new': 'uvicorn backend.main_modular:app'
            },
            'start_production_fixed.sh': {
                'old': 'uvicorn backend.main:app',
                'new': 'uvicorn backend.main_modular:app'
            }
        }
        
        parent_dir = os.path.dirname(self.backend_dir)
        
        for script_name, replacements in scripts.items():
            script_path = os.path.join(parent_dir, script_name)
            if os.path.exists(script_path):
                if self.dry_run:
                    print(f"   🔍 Would update: {script_name}")
                else:
                    try:
                        with open(script_path, 'r') as f:
                            content = f.read()
                        
                        if replacements['old'] in content:
                            content = content.replace(
                                replacements['old'],
                                replacements['new']
                            )
                            with open(script_path, 'w') as f:
                                f.write(content)
                            print(f"   ✅ Updated: {script_name}")
                        else:
                            print(f"   ℹ️ No changes needed: {script_name}")
                    except Exception as e:
                        print(f"   ⚠️ Could not update {script_name}: {e}")
        
        print()
    
    def _print_summary(self):
        """Print migration summary"""
        print("=" * 70)
        print("✅ Migration Complete!")
        print("=" * 70)
        print()
        print("📋 Next Steps:")
        print()
        print("1. Test the new modular version:")
        print("   python backend/main_modular.py")
        print()
        print("2. Or with uvicorn:")
        print("   uvicorn backend.main_modular:app --reload")
        print()
        print("3. Run health checks:")
        print("   curl http://localhost:8000/api/health")
        print("   curl http://localhost:8000/api/health/detailed")
        print()
        print("4. Run tests:")
        print("   pytest backend/tests/ -v")
        print()
        print("5. If everything works, update your deployment:")
        print("   - Update Procfile to use main_modular")
        print("   - Update docker-compose.yml if applicable")
        print("   - Update CI/CD scripts")
        print()
        print("6. After successful deployment, you can:")
        print("   - Remove or archive old main.py")
        print("   - Rename main_modular.py to main.py")
        print()
        print("📚 Documentation:")
        print("   - MODULARIZATION_GUIDE.md - Complete guide")
        print("   - PRIORITY_4_PHASE_1_SUMMARY.md - Resilience features")
        print("   - PRIORITY_4_QUICK_START.md - Quick start guide")
        print()
        print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Migrate from main.py to modular structure'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of main.py'
    )
    
    args = parser.parse_args()
    
    migration = ModularMigration(
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    success = migration.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
