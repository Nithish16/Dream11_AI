#!/usr/bin/env python3
"""
Database Schema Management System
Handles database schema versioning, migrations, and automatic upgrades
"""

import sqlite3
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSchemaManager:
    """
    Manages database schema versions and migrations
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path('.')
        self.migrations_dir = self.base_path / 'migrations'
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Database schema definitions
        self.database_schemas = {
            'universal_cricket_intelligence.db': {
                'current_version': '1.0.0',
                'tables': self._get_universal_schema()
            },
            'ai_learning_database.db': {
                'current_version': '1.0.0', 
                'tables': self._get_learning_schema()
            },
            'format_specific_learning.db': {
                'current_version': '1.0.0',
                'tables': self._get_format_schema()
            },
            'smart_local_predictions.db': {
                'current_version': '1.0.0',
                'tables': self._get_predictions_schema()
            },
            'optimized_predictions.db': {
                'current_version': '1.0.0',
                'tables': self._get_optimized_schema()
            },
            'api_usage_tracking.db': {
                'current_version': '1.0.0',
                'tables': self._get_api_tracking_schema()
            },
            'dream11_unified.db': {
                'current_version': '1.0.0',
                'tables': self._get_unified_schema()
            }
        }
    
    def _get_universal_schema(self) -> Dict[str, str]:
        """Get universal cricket intelligence database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'system_config': '''
                CREATE TABLE IF NOT EXISTS system_config (
                    config_key TEXT PRIMARY KEY,
                    config_value TEXT NOT NULL,
                    config_type TEXT DEFAULT 'string',
                    description TEXT,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'player_intelligence': '''
                CREATE TABLE IF NOT EXISTS player_intelligence (
                    player_id INTEGER PRIMARY KEY,
                    player_name TEXT NOT NULL,
                    format_type TEXT NOT NULL,
                    performance_data TEXT,
                    consistency_score REAL DEFAULT 0.0,
                    form_momentum REAL DEFAULT 0.0,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, format_type)
                )
            ''',
            'format_patterns': '''
                CREATE TABLE IF NOT EXISTS format_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    format_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'winning_patterns': '''
                CREATE TABLE IF NOT EXISTS winning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    match_format TEXT NOT NULL,
                    captain_pattern TEXT,
                    vice_captain_pattern TEXT,
                    team_composition TEXT,
                    success_instances INTEGER DEFAULT 0,
                    validation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
    
    def _get_learning_schema(self) -> Dict[str, str]:
        """Get AI learning database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'predictions': '''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    teams_data TEXT NOT NULL,
                    ai_strategies TEXT NOT NULL,
                    match_format TEXT,
                    venue TEXT,
                    teams_playing TEXT
                )
            ''',
            'idx_predictions_match_id': '''
                CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id)
            ''',
            'results': '''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    result_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    winning_team TEXT NOT NULL,
                    winning_score INTEGER NOT NULL,
                    ai_best_score INTEGER,
                    performance_gap INTEGER,
                    analysis_data TEXT NOT NULL,
                    key_learnings TEXT
                )
            ''',
            'idx_results_match_id': '''
                CREATE INDEX IF NOT EXISTS idx_results_match_id ON results(match_id)
            ''',
            'learning_insights': '''
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact_level TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    implementation_date TIMESTAMP
                )
            ''',
            'algorithm_improvements': '''
                CREATE TABLE IF NOT EXISTS algorithm_improvements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    improvement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    component TEXT NOT NULL,
                    old_logic TEXT,
                    new_logic TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    performance_impact TEXT,
                    matches_tested TEXT
                )
            '''
        }
    
    def _get_format_schema(self) -> Dict[str, str]:
        """Get format-specific learning database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'format_learnings': '''
                CREATE TABLE IF NOT EXISTS format_learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    format_name TEXT NOT NULL,
                    learning_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0,
                    sample_size INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'idx_format_learnings_format_name': '''
                CREATE INDEX IF NOT EXISTS idx_format_learnings_format_name ON format_learnings(format_name)
            ''',
            'performance_metrics': '''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    format_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    baseline_value REAL,
                    improvement_percentage REAL,
                    calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
    
    def _get_predictions_schema(self) -> Dict[str, str]:
        """Get predictions database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'local_predictions': '''
                CREATE TABLE IF NOT EXISTS local_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    team_data TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0,
                    strategy_type TEXT
                )
            ''',
            'idx_local_predictions_match_id': '''
                CREATE INDEX IF NOT EXISTS idx_local_predictions_match_id ON local_predictions(match_id)
            ''',
            'prediction_performance': '''
                CREATE TABLE IF NOT EXISTS prediction_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    prediction_id INTEGER,
                    actual_score INTEGER,
                    predicted_score INTEGER,
                    accuracy_percentage REAL,
                    performance_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(prediction_id) REFERENCES local_predictions(id)
                )
            '''
        }
    
    def _get_optimized_schema(self) -> Dict[str, str]:
        """Get optimized predictions database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'optimized_teams': '''
                CREATE TABLE IF NOT EXISTS optimized_teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    optimization_method TEXT NOT NULL,
                    team_composition TEXT NOT NULL,
                    expected_score REAL DEFAULT 0.0,
                    risk_level TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'idx_optimized_teams_match_id': '''
                CREATE INDEX IF NOT EXISTS idx_optimized_teams_match_id ON optimized_teams(match_id)
            ''',
            'optimization_metrics': '''
                CREATE TABLE IF NOT EXISTS optimization_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    calculation_method TEXT,
                    FOREIGN KEY(team_id) REFERENCES optimized_teams(id)
                )
            '''
        }
    
    def _get_api_tracking_schema(self) -> Dict[str, str]:
        """Get API tracking database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'api_calls': '''
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_code INTEGER,
                    response_time_ms INTEGER,
                    data_size_bytes INTEGER
                )
            ''',
            'idx_api_calls_endpoint': '''
                CREATE INDEX IF NOT EXISTS idx_api_calls_endpoint ON api_calls(endpoint)
            ''',
            'idx_api_calls_timestamp': '''
                CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp)
            ''',
            'api_quotas': '''
                CREATE TABLE IF NOT EXISTS api_quotas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    quota_type TEXT NOT NULL,
                    quota_limit INTEGER NOT NULL,
                    quota_used INTEGER DEFAULT 0,
                    reset_date TIMESTAMP,
                    UNIQUE(service_name, quota_type)
                )
            '''
        }
    
    def _get_unified_schema(self) -> Dict[str, str]:
        """Get unified database schema"""
        return {
            'schema_version': '''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''',
            'unified_data': '''
                CREATE TABLE IF NOT EXISTS unified_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    data_key TEXT NOT NULL,
                    data_value TEXT NOT NULL,
                    metadata TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(data_type, data_key)
                )
            '''
        }
    
    def get_database_version(self, db_path: str) -> Optional[str]:
        """Get current version of a database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if schema_version table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            ''')
            
            if not cursor.fetchone():
                conn.close()
                return None
            
            # Get latest version
            cursor.execute('''
                SELECT version FROM schema_version 
                ORDER BY applied_date DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting database version for {db_path}: {e}")
            return None
    
    def initialize_database(self, db_path: str) -> bool:
        """Initialize database with proper schema"""
        try:
            db_name = os.path.basename(db_path)
            if db_name not in self.database_schemas:
                logger.warning(f"No schema defined for {db_name}")
                return False
            
            schema_info = self.database_schemas[db_name]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create all tables
            for table_name, table_sql in schema_info['tables'].items():
                cursor.execute(table_sql)
            
            # Set initial version
            cursor.execute('''
                INSERT OR REPLACE INTO schema_version (version, description)
                VALUES (?, ?)
            ''', (schema_info['current_version'], f'Initial schema for {db_name}'))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Initialized database {db_name} with schema version {schema_info['current_version']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing database {db_path}: {e}")
            return False
    
    def validate_all_databases(self) -> Dict[str, Dict]:
        """Validate all databases and return status"""
        validation_results = {}
        
        for db_name in self.database_schemas.keys():
            db_path = self.base_path / db_name
            
            if not db_path.exists():
                validation_results[db_name] = {
                    'exists': False,
                    'needs_initialization': True,
                    'current_version': None,
                    'expected_version': self.database_schemas[db_name]['current_version']
                }
            else:
                current_version = self.get_database_version(str(db_path))
                expected_version = self.database_schemas[db_name]['current_version']
                
                validation_results[db_name] = {
                    'exists': True,
                    'needs_initialization': current_version is None,
                    'needs_upgrade': current_version != expected_version if current_version else False,
                    'current_version': current_version,
                    'expected_version': expected_version
                }
        
        return validation_results
    
    def initialize_all_databases(self) -> bool:
        """Initialize all databases with proper schemas"""
        logger.info("🔧 Initializing all databases with proper schemas...")
        
        success = True
        for db_name in self.database_schemas.keys():
            db_path = self.base_path / db_name
            
            if not self.initialize_database(str(db_path)):
                success = False
        
        if success:
            logger.info("✅ All databases initialized successfully")
        else:
            logger.error("❌ Some databases failed to initialize")
        
        return success
    
    def create_migration(self, db_name: str, from_version: str, to_version: str, 
                        migration_sql: List[str], description: str = "") -> bool:
        """Create a migration file"""
        try:
            migration_filename = f"{db_name}_{from_version}_to_{to_version}.json"
            migration_path = self.migrations_dir / migration_filename
            
            migration_data = {
                'database': db_name,
                'from_version': from_version,
                'to_version': to_version,
                'description': description,
                'created_date': datetime.now().isoformat(),
                'sql_commands': migration_sql
            }
            
            with open(migration_path, 'w') as f:
                json.dump(migration_data, f, indent=2)
            
            logger.info(f"✅ Created migration file: {migration_filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating migration: {e}")
            return False

# Global schema manager instance
schema_manager = DatabaseSchemaManager()

def get_schema_manager() -> DatabaseSchemaManager:
    """Get the global schema manager instance"""
    return schema_manager

def validate_and_initialize_databases():
    """Quick function to validate and initialize all databases"""
    manager = DatabaseSchemaManager()
    validation_results = manager.validate_all_databases()
    
    needs_init = []
    for db_name, status in validation_results.items():
        if status['needs_initialization']:
            needs_init.append(db_name)
    
    if needs_init:
        logger.info(f"Databases need initialization: {needs_init}")
        return manager.initialize_all_databases()
    
    logger.info("✅ All databases are properly initialized")
    return True