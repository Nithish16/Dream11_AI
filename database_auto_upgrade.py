#!/usr/bin/env python3
"""
Automatic Database Upgrade System
Handles automatic detection and upgrade of database schemas on startup
"""

import logging
import sys
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_logic.database_schema_manager import DatabaseSchemaManager, get_schema_manager
from core_logic.database_migrations import DatabaseMigrator, get_database_migrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseAutoUpgrader:
    """
    Automatic database upgrade system that runs on application startup
    """
    
    def __init__(self, base_path: str = None, backup_enabled: bool = True):
        self.base_path = Path(base_path) if base_path else Path('.')
        self.backup_enabled = backup_enabled
        
        self.schema_manager = get_schema_manager()
        self.migrator = get_database_migrator()
        
        # Configuration
        self.auto_upgrade_enabled = True
        self.backup_before_upgrade = True
        self.max_backup_files = 10
        
        self.upgrade_results = {}
    
    def check_and_upgrade_all(self, force_upgrade: bool = False) -> Dict[str, Dict]:
        """
        Check all databases and perform automatic upgrades if needed
        
        Args:
            force_upgrade: Force upgrade even if auto-upgrade is disabled
            
        Returns:
            Dict with upgrade results for each database
        """
        logger.info("🔍 Checking all databases for upgrades...")
        
        if not self.auto_upgrade_enabled and not force_upgrade:
            logger.info("ℹ️ Auto-upgrade is disabled, skipping")
            return {'status': 'skipped', 'reason': 'auto_upgrade_disabled'}
        
        # Validate all databases first
        validation_results = self.schema_manager.validate_all_databases()
        
        upgrade_results = {
            'timestamp': str(Path.cwd()),
            'databases': {},
            'overall_success': True,
            'backups_created': []
        }
        
        for db_name, status in validation_results.items():
            db_path = self.base_path / db_name
            
            try:
                result = self._process_single_database(str(db_path), db_name, status)
                upgrade_results['databases'][db_name] = result
                
                if not result['success']:
                    upgrade_results['overall_success'] = False
                    
            except Exception as e:
                logger.error(f"❌ Error processing {db_name}: {e}")
                upgrade_results['databases'][db_name] = {
                    'success': False,
                    'error': str(e),
                    'action': 'error'
                }
                upgrade_results['overall_success'] = False
        
        self._log_upgrade_summary(upgrade_results)
        self.upgrade_results = upgrade_results
        
        return upgrade_results
    
    def _process_single_database(self, db_path: str, db_name: str, status: Dict) -> Dict:
        """Process a single database for upgrades"""
        
        result = {
            'success': True,
            'action': 'none',
            'from_version': status.get('current_version'),
            'to_version': status.get('expected_version'),
            'backup_created': None,
            'error': None
        }
        
        try:
            # Database doesn't exist - create it
            if not status['exists']:
                logger.info(f"🆕 Creating new database: {db_name}")
                
                if self.schema_manager.initialize_database(db_path):
                    result['action'] = 'created'
                    logger.info(f"✅ Created {db_name} successfully")
                else:
                    result['success'] = False
                    result['error'] = 'Failed to create database'
                    logger.error(f"❌ Failed to create {db_name}")
                
                return result
            
            # Database needs initialization (exists but no schema)
            if status['needs_initialization']:
                logger.info(f"🔧 Initializing schema for: {db_name}")
                
                # Backup existing database before initialization
                if self.backup_enabled:
                    backup_path = self.migrator.backup_database_before_migration(db_path)
                    result['backup_created'] = backup_path
                
                if self.schema_manager.initialize_database(db_path):
                    result['action'] = 'initialized'
                    logger.info(f"✅ Initialized {db_name} successfully")
                else:
                    result['success'] = False
                    result['error'] = 'Failed to initialize database'
                    logger.error(f"❌ Failed to initialize {db_name}")
                
                return result
            
            # Database needs upgrade
            if status.get('needs_upgrade', False):
                logger.info(f"⬆️ Upgrading database: {db_name}")
                logger.info(f"   From: {status['current_version']} → To: {status['expected_version']}")
                
                # Create backup before upgrade
                if self.backup_enabled:
                    backup_path = self.migrator.backup_database_before_migration(db_path)
                    result['backup_created'] = backup_path
                
                if self.migrator.migrate_database(db_path, status['expected_version']):
                    result['action'] = 'upgraded'
                    logger.info(f"✅ Upgraded {db_name} successfully")
                else:
                    result['success'] = False
                    result['error'] = 'Migration failed'
                    logger.error(f"❌ Failed to upgrade {db_name}")
                
                return result
            
            # Database is up to date
            logger.info(f"✅ {db_name} is up to date (v{status['current_version']})")
            result['action'] = 'up_to_date'
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {db_name}: {e}")
            result['success'] = False
            result['error'] = str(e)
            result['action'] = 'error'
            return result
    
    def _log_upgrade_summary(self, results: Dict):
        """Log summary of upgrade operations"""
        logger.info("=" * 60)
        logger.info("📊 DATABASE UPGRADE SUMMARY")
        logger.info("=" * 60)
        
        total_dbs = len(results['databases'])
        successful = sum(1 for r in results['databases'].values() if r['success'])
        failed = total_dbs - successful
        
        logger.info(f"Total databases: {total_dbs}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("")
        
        # Action breakdown
        actions = {}
        for db_name, result in results['databases'].items():
            action = result['action']
            if action not in actions:
                actions[action] = []
            actions[action].append(db_name)
        
        for action, db_list in actions.items():
            logger.info(f"{action.upper()}: {len(db_list)} database(s)")
            for db_name in db_list:
                result = results['databases'][db_name]
                if result['success']:
                    logger.info(f"  ✅ {db_name}")
                else:
                    logger.info(f"  ❌ {db_name} - {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)
        
        if results['overall_success']:
            logger.info("🎉 ALL DATABASES SUCCESSFULLY UPGRADED!")
        else:
            logger.warning("⚠️ SOME DATABASES FAILED TO UPGRADE")
        
        logger.info("=" * 60)
    
    def get_database_status(self) -> Dict[str, Dict]:
        """Get current status of all databases"""
        return self.schema_manager.validate_all_databases()
    
    def cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            backup_dir = self.base_path / 'backups'
            if not backup_dir.exists():
                return
            
            # Get all backup files sorted by modification time
            backup_files = sorted(
                backup_dir.glob('*.db'),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the most recent ones
            if len(backup_files) > self.max_backup_files:
                files_to_remove = backup_files[self.max_backup_files:]
                
                for backup_file in files_to_remove:
                    try:
                        backup_file.unlink()
                        logger.info(f"🗑️ Removed old backup: {backup_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to remove backup {backup_file.name}: {e}")
                        
                logger.info(f"🧹 Cleaned up {len(files_to_remove)} old backup files")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up backups: {e}")
    
    def validate_upgrade_results(self) -> bool:
        """Validate that all upgrades were successful"""
        if not self.upgrade_results:
            logger.warning("⚠️ No upgrade results available")
            return False
        
        return self.upgrade_results.get('overall_success', False)
    
    def get_upgrade_report(self) -> Dict:
        """Get detailed upgrade report"""
        if not self.upgrade_results:
            return {'status': 'no_upgrades_performed'}
        
        return self.upgrade_results
    
    def emergency_reset_database(self, db_name: str) -> bool:
        """Emergency reset of a single database (use with caution)"""
        logger.warning(f"🚨 EMERGENCY RESET requested for {db_name}")
        logger.warning("This will DELETE all data in the database!")
        
        try:
            db_path = self.base_path / db_name
            
            # Create emergency backup
            if db_path.exists():
                backup_path = self.migrator.backup_database_before_migration(str(db_path))
                logger.info(f"💾 Emergency backup created: {backup_path}")
                
                # Remove existing database
                db_path.unlink()
                logger.info(f"🗑️ Removed existing database: {db_name}")
            
            # Initialize fresh database
            if self.schema_manager.initialize_database(str(db_path)):
                logger.info(f"✅ Emergency reset completed for {db_name}")
                return True
            else:
                logger.error(f"❌ Failed to initialize fresh database: {db_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Emergency reset failed for {db_name}: {e}")
            return False

# Global auto upgrader instance
auto_upgrader = DatabaseAutoUpgrader()

def get_auto_upgrader() -> DatabaseAutoUpgrader:
    """Get the global auto upgrader instance"""
    return auto_upgrader

def auto_upgrade_databases(force: bool = False) -> bool:
    """Quick function to auto-upgrade all databases"""
    upgrader = DatabaseAutoUpgrader()
    results = upgrader.check_and_upgrade_all(force_upgrade=force)
    return results['overall_success']

def startup_database_check():
    """
    Perform database check and upgrade on application startup
    This should be called by main application files
    """
    logger.info("🚀 Starting application database check...")
    
    try:
        upgrader = DatabaseAutoUpgrader()
        results = upgrader.check_and_upgrade_all()
        
        # Clean up old backups
        upgrader.cleanup_old_backups()
        
        if results['overall_success']:
            logger.info("✅ Database startup check completed successfully")
            return True
        else:
            logger.error("❌ Database startup check failed")
            logger.error("🛑 Application may not function correctly")
            return False
            
    except Exception as e:
        logger.error(f"❌ Critical error during database startup check: {e}")
        return False

if __name__ == "__main__":
    # Command line usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Auto Upgrade System')
    parser.add_argument('--force', action='store_true', help='Force upgrade even if auto-upgrade is disabled')
    parser.add_argument('--status', action='store_true', help='Show database status only')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old backups')
    parser.add_argument('--emergency-reset', type=str, help='Emergency reset specified database')
    
    args = parser.parse_args()
    
    upgrader = DatabaseAutoUpgrader()
    
    if args.status:
        status = upgrader.get_database_status()
        print("Database Status:")
        for db_name, db_status in status.items():
            print(f"  {db_name}: {db_status}")
    
    elif args.cleanup:
        upgrader.cleanup_old_backups()
    
    elif args.emergency_reset:
        upgrader.emergency_reset_database(args.emergency_reset)
    
    else:
        results = upgrader.check_and_upgrade_all(force_upgrade=args.force)
        exit(0 if results['overall_success'] else 1)