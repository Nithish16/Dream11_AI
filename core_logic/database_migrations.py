#!/usr/bin/env python3
"""
Database Migration System
Handles automatic database schema upgrades and version management
"""

import sqlite3
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .database_schema_manager import DatabaseSchemaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """
    Handles database migrations and version upgrades
    """
    
    def __init__(self, schema_manager: DatabaseSchemaManager = None):
        self.schema_manager = schema_manager or DatabaseSchemaManager()
        self.migrations_dir = self.schema_manager.migrations_dir
        self.base_path = self.schema_manager.base_path
        
        # Predefined migration paths for known version upgrades
        self.migration_definitions = {
            'universal_cricket_intelligence.db': {
                '1.0.0_to_1.1.0': {
                    'description': 'Add player performance indices and venue analysis',
                    'sql_commands': [
                        '''
                        CREATE INDEX IF NOT EXISTS idx_player_intelligence_format 
                        ON player_intelligence(format_type)
                        ''',
                        '''
                        CREATE INDEX IF NOT EXISTS idx_player_intelligence_updated 
                        ON player_intelligence(updated_date)
                        ''',
                        '''
                        ALTER TABLE player_intelligence 
                        ADD COLUMN venue_performance TEXT DEFAULT '{}'
                        ''',
                        '''
                        ALTER TABLE format_patterns 
                        ADD COLUMN validation_matches INTEGER DEFAULT 0
                        '''
                    ]
                }
            },
            'ai_learning_database.db': {
                '1.0.0_to_1.1.0': {
                    'description': 'Add performance tracking and advanced analytics',
                    'sql_commands': [
                        '''
                        CREATE TABLE IF NOT EXISTS prediction_accuracy (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            prediction_id INTEGER NOT NULL,
                            accuracy_score REAL NOT NULL,
                            error_margin REAL,
                            confidence_validated BOOLEAN DEFAULT FALSE,
                            calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
                        )
                        ''',
                        '''
                        CREATE INDEX IF NOT EXISTS idx_predictions_match_format 
                        ON predictions(match_id, match_format)
                        ''',
                        '''
                        CREATE INDEX IF NOT EXISTS idx_results_performance_gap 
                        ON results(performance_gap)
                        '''
                    ]
                }
            }
        }
    
    def get_available_migrations(self, db_name: str, current_version: str) -> List[Dict]:
        """Get available migrations for a database from current version"""
        migrations = []
        
        # Check predefined migrations
        if db_name in self.migration_definitions:
            for migration_key, migration_data in self.migration_definitions[db_name].items():
                from_version, to_version = migration_key.split('_to_')
                if from_version == current_version:
                    migrations.append({
                        'from_version': from_version,
                        'to_version': to_version,
                        'source': 'predefined',
                        'data': migration_data
                    })
        
        # Check migration files
        migration_pattern = f"{db_name}_{current_version}_to_*.json"
        for migration_file in self.migrations_dir.glob(migration_pattern):
            try:
                with open(migration_file, 'r') as f:
                    migration_data = json.load(f)
                
                migrations.append({
                    'from_version': migration_data['from_version'],
                    'to_version': migration_data['to_version'],
                    'source': 'file',
                    'file_path': str(migration_file),
                    'data': migration_data
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to load migration file {migration_file}: {e}")
        
        return migrations
    
    def apply_migration(self, db_path: str, migration: Dict) -> bool:
        """Apply a single migration to a database"""
        try:
            db_name = os.path.basename(db_path)
            migration_data = migration['data']
            
            logger.info(f"🔄 Applying migration {migration['from_version']} → {migration['to_version']} to {db_name}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Start transaction
            cursor.execute('BEGIN TRANSACTION')
            
            try:
                # Apply each SQL command
                for sql_command in migration_data['sql_commands']:
                    cursor.execute(sql_command)
                
                # Update schema version
                cursor.execute('''
                    INSERT OR REPLACE INTO schema_version (version, description)
                    VALUES (?, ?)
                ''', (migration['to_version'], migration_data.get('description', 'Migration applied')))
                
                # Commit transaction
                conn.commit()
                
                logger.info(f"✅ Migration {migration['from_version']} → {migration['to_version']} applied successfully")
                return True
                
            except Exception as e:
                # Rollback on error
                conn.rollback()
                logger.error(f"❌ Error applying migration: {e}")
                return False
            
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ Error applying migration to {db_path}: {e}")
            return False
    
    def migrate_database(self, db_path: str, target_version: Optional[str] = None) -> bool:
        """Migrate a database to target version (or latest if not specified)"""
        try:
            db_name = os.path.basename(db_path)
            current_version = self.schema_manager.get_database_version(db_path)
            
            if not current_version:
                logger.error(f"❌ Cannot migrate {db_name}: no current version found")
                return False
            
            # Get target version (latest if not specified)
            if not target_version:
                expected_version = self.schema_manager.database_schemas.get(db_name, {}).get('current_version')
                if not expected_version:
                    logger.error(f"❌ No expected version defined for {db_name}")
                    return False
                target_version = expected_version
            
            if current_version == target_version:
                logger.info(f"✅ {db_name} is already at version {current_version}")
                return True
            
            logger.info(f"🔄 Migrating {db_name} from {current_version} to {target_version}")
            
            # Build migration path
            migration_path = self._build_migration_path(db_name, current_version, target_version)
            
            if not migration_path:
                logger.error(f"❌ No migration path found from {current_version} to {target_version}")
                return False
            
            # Apply migrations in sequence
            success = True
            for migration in migration_path:
                if not self.apply_migration(db_path, migration):
                    success = False
                    break
            
            if success:
                logger.info(f"✅ Successfully migrated {db_name} to version {target_version}")
            else:
                logger.error(f"❌ Failed to migrate {db_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error migrating database {db_path}: {e}")
            return False
    
    def _build_migration_path(self, db_name: str, from_version: str, to_version: str) -> List[Dict]:
        """Build a sequence of migrations from source to target version"""
        # For now, implement simple direct migration
        # In future versions, this could handle complex migration chains
        
        available_migrations = self.get_available_migrations(db_name, from_version)
        
        # Find direct migration
        for migration in available_migrations:
            if migration['to_version'] == to_version:
                return [migration]
        
        # TODO: Implement complex migration path finding for future versions
        logger.warning(f"⚠️ No direct migration path found from {from_version} to {to_version}")
        return []
    
    def migrate_all_databases(self) -> bool:
        """Migrate all databases to their latest versions"""
        logger.info("🔄 Starting migration of all databases...")
        
        validation_results = self.schema_manager.validate_all_databases()
        
        success = True
        for db_name, status in validation_results.items():
            db_path = self.base_path / db_name
            
            if status['needs_initialization']:
                logger.info(f"🆕 Initializing {db_name}...")
                if not self.schema_manager.initialize_database(str(db_path)):
                    success = False
            
            elif status.get('needs_upgrade', False):
                logger.info(f"⬆️ Upgrading {db_name}...")
                if not self.migrate_database(str(db_path)):
                    success = False
            
            else:
                logger.info(f"✅ {db_name} is up to date")
        
        if success:
            logger.info("✅ All databases migrated successfully")
        else:
            logger.error("❌ Some database migrations failed")
        
        return success
    
    def create_migration_from_schema_changes(self, db_name: str, from_version: str, 
                                           to_version: str, description: str = "") -> bool:
        """Create migration file based on schema differences"""
        try:
            # This is a simplified version - in production you'd compare actual schemas
            logger.info(f"🔧 Creating migration for {db_name}: {from_version} → {to_version}")
            
            # Generate migration commands based on schema differences
            # For now, use predefined migrations
            if db_name in self.migration_definitions:
                migration_key = f"{from_version}_to_{to_version}"
                if migration_key in self.migration_definitions[db_name]:
                    migration_data = self.migration_definitions[db_name][migration_key]
                    
                    return self.schema_manager.create_migration(
                        db_name, from_version, to_version,
                        migration_data['sql_commands'],
                        description or migration_data['description']
                    )
            
            logger.warning(f"⚠️ No predefined migration found for {db_name} {from_version} → {to_version}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error creating migration: {e}")
            return False
    
    def rollback_migration(self, db_path: str, to_version: str) -> bool:
        """Rollback database to a previous version"""
        # This is a complex operation that would require careful implementation
        # For now, we'll log that it's not implemented
        logger.warning("⚠️ Migration rollback not yet implemented")
        logger.info("💡 To rollback, restore from backup or reinitialize database")
        return False
    
    def backup_database_before_migration(self, db_path: str) -> Optional[str]:
        """Create backup before applying migrations"""
        try:
            backup_dir = self.base_path / 'backups'
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            db_name = os.path.basename(db_path)
            backup_name = f"{db_name.replace('.db', '')}_{timestamp}.db"
            backup_path = backup_dir / backup_name
            
            # Copy database file
            import shutil
            shutil.copy2(db_path, backup_path)
            
            logger.info(f"✅ Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"❌ Error creating backup: {e}")
            return None

# Global migrator instance
migrator = DatabaseMigrator()

def get_database_migrator() -> DatabaseMigrator:
    """Get the global database migrator instance"""
    return migrator

def migrate_all_databases():
    """Quick function to migrate all databases"""
    migrator = DatabaseMigrator()
    return migrator.migrate_all_databases()