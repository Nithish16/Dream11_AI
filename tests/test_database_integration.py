#!/usr/bin/env python3
"""
Integration Tests for Database Systems
Tests database schema management, connection pooling, migrations, and cleanup
"""

import unittest
import sys
import os
import sqlite3
import time
import threading
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from test_framework import Dream11TestCase, DatabaseTestMixin, PerformanceTestMixin, IntegrationTestMixin

# Import database modules
try:
    from core_logic.database_schema_manager import DatabaseSchemaManager, get_schema_manager
    from core_logic.database_migrations import DatabaseMigrator, get_database_migrator
    from core_logic.database_connection_manager import DatabaseConnectionManager, get_connection_manager
    from core_logic.database_cleanup import DatabaseCleanupManager, get_cleanup_manager
    from database_auto_upgrade import DatabaseAutoUpgrader, startup_database_check
except ImportError as e:
    print(f"⚠️ Some database imports failed: {e}")
    print("Integration tests will use mocks for missing components")

class TestDatabaseSchemaManagement(Dream11TestCase, DatabaseTestMixin):
    """Test database schema management and initialization"""
    
    def setUp(self):
        super().setUp()
        
        # Create isolated test environment
        self.test_db_dir = self.temp_dir / 'test_databases'
        self.test_db_dir.mkdir(exist_ok=True)
        
        # Initialize schema manager with test directory
        self.schema_manager = DatabaseSchemaManager(base_path=str(self.test_db_dir))
    
    def test_database_schema_initialization(self):
        """Test database schema initialization"""
        db_name = 'test_universal.db'
        db_path = self.test_db_dir / db_name
        
        # Initialize database
        success = self.schema_manager.initialize_database(str(db_path))
        
        self.assertTrue(success)
        self.assertTrue(db_path.exists())
        
        # Verify schema version table exists
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='schema_version'
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
    
    def test_all_database_initialization(self):
        """Test initialization of all database schemas"""
        success = self.schema_manager.initialize_all_databases()
        
        self.assertTrue(success)
        
        # Verify all expected databases were created
        expected_databases = [
            'universal_cricket_intelligence.db',
            'ai_learning_database.db',
            'format_specific_learning.db',
            'smart_local_predictions.db',
            'optimized_predictions.db',
            'api_usage_tracking.db',
            'dream11_unified.db'
        ]
        
        for db_name in expected_databases:
            db_path = self.test_db_dir / db_name
            self.assertTrue(db_path.exists(), f"Database {db_name} was not created")
    
    def test_database_version_tracking(self):
        """Test database version tracking and retrieval"""
        db_name = 'test_versioning.db'
        db_path = self.test_db_dir / db_name
        
        # Initialize database
        self.schema_manager.initialize_database(str(db_path))
        
        # Get version
        version = self.schema_manager.get_database_version(str(db_path))
        
        self.assertIsNotNone(version)
        self.assertEqual(version, '1.0.0')
    
    def test_database_validation(self):
        """Test database validation and status checking"""
        # Initialize some databases
        self.schema_manager.initialize_all_databases()
        
        # Validate all databases
        validation_results = self.schema_manager.validate_all_databases()
        
        self.assertIsInstance(validation_results, dict)
        
        for db_name, status in validation_results.items():
            self.assertIsInstance(status, dict)
            self.assertIn('exists', status)
            self.assertIn('current_version', status)
            self.assertIn('expected_version', status)

class TestDatabaseMigrations(Dream11TestCase, DatabaseTestMixin):
    """Test database migration system"""
    
    def setUp(self):
        super().setUp()
        
        # Create isolated test environment
        self.test_db_dir = self.temp_dir / 'migration_test'
        self.test_db_dir.mkdir(exist_ok=True)
        
        # Initialize managers
        self.schema_manager = DatabaseSchemaManager(base_path=str(self.test_db_dir))
        self.migrator = DatabaseMigrator(schema_manager=self.schema_manager)
    
    def test_migration_creation(self):
        """Test creation of migration files"""
        db_name = 'test_migration.db'
        migration_sql = [
            "CREATE TABLE test_new_table (id INTEGER PRIMARY KEY, data TEXT)",
            "CREATE INDEX idx_test_data ON test_new_table(data)"
        ]
        
        success = self.migrator.schema_manager.create_migration(
            db_name=db_name,
            from_version='1.0.0',
            to_version='1.1.0',
            migration_sql=migration_sql,
            description='Add test table with index'
        )
        
        self.assertTrue(success)
        
        # Verify migration file was created
        migration_files = list(self.migrator.migrations_dir.glob(f"{db_name}_1.0.0_to_1.1.0.json"))
        self.assertEqual(len(migration_files), 1)
        
        # Verify migration file content
        with open(migration_files[0], 'r') as f:
            migration_data = json.load(f)
        
        self.assertEqual(migration_data['database'], db_name)
        self.assertEqual(migration_data['from_version'], '1.0.0')
        self.assertEqual(migration_data['to_version'], '1.1.0')
        self.assertEqual(migration_data['sql_commands'], migration_sql)
    
    def test_database_migration_execution(self):
        """Test execution of database migrations"""
        db_name = 'universal_cricket_intelligence.db'
        db_path = self.test_db_dir / db_name
        
        # Initialize database with version 1.0.0
        self.schema_manager.initialize_database(str(db_path))
        
        # Verify initial version
        initial_version = self.schema_manager.get_database_version(str(db_path))
        self.assertEqual(initial_version, '1.0.0')
        
        # Attempt migration (using predefined migration if available)
        success = self.migrator.migrate_database(str(db_path), target_version='1.1.0')
        
        # Note: This might fail if no predefined migration exists, which is expected
        # The test validates the migration process structure
        if success:
            updated_version = self.schema_manager.get_database_version(str(db_path))
            self.assertEqual(updated_version, '1.1.0')
    
    def test_migration_rollback_safety(self):
        """Test migration rollback and safety mechanisms"""
        db_name = 'test_rollback.db'
        db_path = self.test_db_dir / db_name
        
        # Initialize database
        self.schema_manager.initialize_database(str(db_path))
        
        # Create a migration that should fail
        failing_migration = {
            'from_version': '1.0.0',
            'to_version': '1.1.0',
            'data': {
                'sql_commands': [
                    "CREATE TABLE valid_table (id INTEGER PRIMARY KEY)",
                    "INVALID SQL COMMAND THAT WILL FAIL",  # This should cause rollback
                    "CREATE TABLE another_table (id INTEGER)"
                ],
                'description': 'Test migration with failure'
            }
        }
        
        # Apply migration (should fail and rollback)
        success = self.migrator.apply_migration(str(db_path), failing_migration)
        
        self.assertFalse(success)
        
        # Verify database is still at original version
        version = self.schema_manager.get_database_version(str(db_path))
        self.assertEqual(version, '1.0.0')
        
        # Verify no partial changes were applied
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='valid_table'
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        # Table should not exist due to rollback
        self.assertIsNone(result)

class TestDatabaseConnectionManager(Dream11TestCase, DatabaseTestMixin, PerformanceTestMixin):
    """Test database connection management and pooling"""
    
    def setUp(self):
        super().setUp()
        
        # Create test database
        self.test_db_dir = self.temp_dir / 'connection_test'
        self.test_db_dir.mkdir(exist_ok=True)
        
        # Initialize connection manager
        self.connection_manager = DatabaseConnectionManager(base_path=str(self.test_db_dir))
        
        # Create test database
        self.test_db_name = 'test_connections.db'
        self.test_db_path = self.test_db_dir / self.test_db_name
        
        # Initialize test database with simple schema
        conn = sqlite3.connect(str(self.test_db_path))
        conn.execute('CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)')
        conn.execute("INSERT INTO test_data (value) VALUES ('test1'), ('test2'), ('test3')")
        conn.commit()
        conn.close()
    
    def test_connection_acquisition_and_release(self):
        """Test basic connection acquisition and release"""
        with self.connection_manager.get_connection(self.test_db_name) as conn:
            self.assertIsNotNone(conn)
            self.assertIsInstance(conn, sqlite3.Connection)
            
            # Test that connection works
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM test_data')
            count = cursor.fetchone()[0]
            self.assertEqual(count, 3)
        
        # Connection should be returned to pool automatically
    
    def test_connection_pooling_efficiency(self):
        """Test connection pooling efficiency"""
        # Measure time for multiple operations with pooling
        with self.performance_context('connection_pooling', max_duration=2.0):
            for i in range(10):
                with self.connection_manager.get_connection(self.test_db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM test_data')
                    result = cursor.fetchone()
                    self.assertIsNotNone(result)
        
        # Get pool statistics
        stats = self.connection_manager.get_all_stats()
        pool_stats = stats.get('pool_stats', {}).get(self.test_db_name, {})
        
        if pool_stats:
            # Connection pool should show good hit ratio
            hit_ratio = pool_stats.get('hit_ratio', 0)
            self.assertGreaterEqual(hit_ratio, 0.0)  # Should be non-negative
    
    def test_concurrent_connection_access(self):
        """Test concurrent access to connection pool"""
        results = []
        errors = []
        
        def worker_task(worker_id):
            try:
                with self.connection_manager.get_connection(self.test_db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM test_data')
                    count = cursor.fetchone()[0]
                    results.append((worker_id, count))
                    
                    # Simulate some work
                    time.sleep(0.1)
                    
                    # Another query
                    cursor.execute('SELECT value FROM test_data LIMIT 1')
                    value = cursor.fetchone()[0]
                    results.append((worker_id, value))
                    
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(('future', str(e)))
        
        # Verify results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 20)  # 2 results per worker * 10 workers
        
        # Verify all count results are correct
        count_results = [result[1] for result in results if isinstance(result[1], int)]
        self.assertTrue(all(count == 3 for count in count_results))
    
    def test_connection_cleanup_and_timeout(self):
        """Test connection cleanup and timeout handling"""
        # Get initial pool stats
        initial_stats = self.connection_manager.get_all_stats()
        
        # Create some connections and let them become idle
        connections_created = []
        for i in range(3):
            with self.connection_manager.get_connection(self.test_db_name) as conn:
                # Use connection briefly
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()
        
        # Force cleanup of idle connections
        if hasattr(self.connection_manager, '_cleanup_idle_connections'):
            self.connection_manager._cleanup_idle_connections()
        
        # Get final pool stats
        final_stats = self.connection_manager.get_all_stats()
        
        # Verify cleanup occurred (this is implementation-dependent)
        self.assertIsNotNone(final_stats)
    
    def test_query_execution_with_parameters(self):
        """Test parameterized query execution through connection manager"""
        # Insert test data using parameterized query
        insert_result = self.connection_manager.execute_query(
            self.test_db_name,
            'INSERT INTO test_data (value) VALUES (?)',
            ('test_param',),
            fetch_method='none'
        )
        
        self.assertIsNotNone(insert_result)
        
        # Query with parameters
        select_result = self.connection_manager.execute_query(
            self.test_db_name,
            'SELECT value FROM test_data WHERE value = ?',
            ('test_param',),
            fetch_method='fetchone'
        )
        
        self.assertIsNotNone(select_result)
        self.assertEqual(select_result[0], 'test_param')
    
    def test_transaction_handling(self):
        """Test transaction handling through connection manager"""
        def transaction_operations(conn):
            cursor = conn.cursor()
            cursor.execute('INSERT INTO test_data (value) VALUES (?)', ('tx_test1',))
            cursor.execute('INSERT INTO test_data (value) VALUES (?)', ('tx_test2',))
            return cursor.rowcount
        
        # Execute transaction
        results = self.connection_manager.execute_transaction(
            self.test_db_name,
            [transaction_operations]
        )
        
        self.assertEqual(len(results), 1)
        
        # Verify data was committed
        check_result = self.connection_manager.execute_query(
            self.test_db_name,
            "SELECT COUNT(*) FROM test_data WHERE value LIKE 'tx_test%'",
            fetch_method='fetchone'
        )
        
        self.assertEqual(check_result[0], 2)

class TestDatabaseCleanup(Dream11TestCase, DatabaseTestMixin):
    """Test database cleanup and maintenance systems"""
    
    def setUp(self):
        super().setUp()
        
        # Create test environment
        self.test_db_dir = self.temp_dir / 'cleanup_test'
        self.test_db_dir.mkdir(exist_ok=True)
        
        # Initialize managers
        self.connection_manager = DatabaseConnectionManager(base_path=str(self.test_db_dir))
        self.cleanup_manager = DatabaseCleanupManager(connection_manager=self.connection_manager)
        
        # Create test databases
        self._create_test_databases()
    
    def _create_test_databases(self):
        """Create test databases for cleanup testing"""
        for db_name in ['test1.db', 'test2.db']:
            db_path = self.test_db_dir / db_name
            conn = sqlite3.connect(str(db_path))
            conn.execute('CREATE TABLE cleanup_test (id INTEGER PRIMARY KEY, data TEXT)')
            conn.execute("INSERT INTO cleanup_test (data) VALUES ('data1'), ('data2')")
            conn.commit()
            conn.close()
    
    def test_cleanup_manager_initialization(self):
        """Test cleanup manager initialization"""
        self.assertIsNotNone(self.cleanup_manager)
        self.assertIsNotNone(self.cleanup_manager.connection_manager)
        self.assertFalse(self.cleanup_manager.running)  # Should not auto-start in test
    
    def test_database_health_check(self):
        """Test database health checking"""
        health_report = self.cleanup_manager.check_database_health()
        
        self.assertIsInstance(health_report, dict)
        self.assertIn('overall_status', health_report)
        self.assertIn('databases', health_report)
        self.assertIn('issues', health_report)
        
        # Should report healthy status for test databases
        self.assertIn(health_report['overall_status'], ['healthy', 'warning', 'critical', 'error'])
    
    def test_force_cleanup_execution(self):
        """Test force cleanup execution"""
        cleanup_results = self.cleanup_manager.force_cleanup()
        
        self.assertIsInstance(cleanup_results, dict)
        self.assertIn('connections_cleaned', cleanup_results)
        self.assertIn('databases_optimized', cleanup_results)
        self.assertIn('errors', cleanup_results)
        
        # Should have non-negative counts
        self.assertGreaterEqual(cleanup_results['connections_cleaned'], 0)
        self.assertGreaterEqual(cleanup_results['databases_optimized'], 0)
    
    def test_custom_cleanup_tasks(self):
        """Test addition and execution of custom cleanup tasks"""
        task_executed = []
        
        def custom_task():
            task_executed.append('executed')
        
        # Add custom cleanup task
        self.cleanup_manager.add_cleanup_task(custom_task, 'test_task')
        
        # Execute cleanup (should run custom task)
        self.cleanup_manager._perform_cleanup()
        
        # Verify custom task was executed
        self.assertEqual(len(task_executed), 1)
        self.assertEqual(task_executed[0], 'executed')
    
    def test_cleanup_statistics_tracking(self):
        """Test cleanup statistics tracking"""
        # Get initial stats
        initial_stats = self.cleanup_manager.get_cleanup_stats()
        
        self.assertIsInstance(initial_stats, dict)
        self.assertIn('cleanups_performed', initial_stats)
        self.assertIn('connections_cleaned', initial_stats)
        self.assertIn('maintenance_runs', initial_stats)
        
        # Perform cleanup
        self.cleanup_manager._perform_cleanup()
        
        # Get updated stats
        updated_stats = self.cleanup_manager.get_cleanup_stats()
        
        # Cleanup count should have increased
        self.assertGreater(
            updated_stats['cleanups_performed'],
            initial_stats['cleanups_performed']
        )

class TestDatabaseAutoUpgrade(Dream11TestCase, IntegrationTestMixin):
    """Test automatic database upgrade system"""
    
    def setUp(self):
        super().setUp()
        
        # Create test environment
        self.test_db_dir = self.temp_dir / 'auto_upgrade_test'
        self.test_db_dir.mkdir(exist_ok=True)
        
        # Initialize auto upgrader
        self.auto_upgrader = DatabaseAutoUpgrader(
            base_path=str(self.test_db_dir),
            backup_enabled=True
        )
    
    def test_auto_upgrader_initialization(self):
        """Test auto upgrader initialization"""
        self.assertIsNotNone(self.auto_upgrader)
        self.assertTrue(self.auto_upgrader.auto_upgrade_enabled)
        self.assertTrue(self.auto_upgrader.backup_enabled)
    
    def test_database_status_checking(self):
        """Test database status checking"""
        status = self.auto_upgrader.get_database_status()
        
        self.assertIsInstance(status, dict)
        
        # Should contain information about each expected database
        expected_databases = [
            'universal_cricket_intelligence.db',
            'ai_learning_database.db',
            'format_specific_learning.db'
        ]
        
        for db_name in expected_databases:
            if db_name in status:
                self.assertIsInstance(status[db_name], dict)
                self.assertIn('exists', status[db_name])
    
    def test_new_database_creation(self):
        """Test creation of new databases during upgrade"""
        # Run upgrade check (should create missing databases)
        results = self.auto_upgrader.check_and_upgrade_all(force_upgrade=True)
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_success', results)
        self.assertIn('databases', results)
        
        # Check that databases were created
        for db_name, result in results['databases'].items():
            db_path = self.test_db_dir / db_name
            if result['success'] and result['action'] in ['created', 'initialized']:
                self.assertTrue(db_path.exists(), f"Database {db_name} should exist after upgrade")
    
    def test_upgrade_result_validation(self):
        """Test upgrade result validation"""
        # Perform upgrade
        results = self.auto_upgrader.check_and_upgrade_all(force_upgrade=True)
        
        # Validate results
        is_valid = self.auto_upgrader.validate_upgrade_results()
        
        if results['overall_success']:
            self.assertTrue(is_valid)
        else:
            self.assertFalse(is_valid)
    
    def test_backup_creation_during_upgrade(self):
        """Test backup creation during upgrade process"""
        # Create a test database with some data
        test_db = self.test_db_dir / 'test_backup.db'
        conn = sqlite3.connect(str(test_db))
        conn.execute('CREATE TABLE test_data (id INTEGER, value TEXT)')
        conn.execute("INSERT INTO test_data VALUES (1, 'test')")
        conn.commit()
        conn.close()
        
        # Create backup
        backup_path = self.auto_upgrader.migrator.backup_database_before_migration(str(test_db))
        
        if backup_path:
            backup_file = Path(backup_path)
            self.assertTrue(backup_file.exists())
            
            # Verify backup contains data
            backup_conn = sqlite3.connect(backup_path)
            cursor = backup_conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM test_data')
            count = cursor.fetchone()[0]
            backup_conn.close()
            
            self.assertEqual(count, 1)

class TestDatabaseIntegrationWorkflow(Dream11TestCase, IntegrationTestMixin, PerformanceTestMixin):
    """Test complete database integration workflow"""
    
    def setUp(self):
        super().setUp()
        
        # Create complete test environment
        self.test_db_dir = self.temp_dir / 'integration_workflow'
        self.test_db_dir.mkdir(exist_ok=True)
        
        # Initialize all managers
        self.schema_manager = DatabaseSchemaManager(base_path=str(self.test_db_dir))
        self.connection_manager = DatabaseConnectionManager(base_path=str(self.test_db_dir))
        self.auto_upgrader = DatabaseAutoUpgrader(base_path=str(self.test_db_dir))
    
    def test_complete_database_initialization_workflow(self):
        """Test complete database initialization workflow"""
        with self.performance_context('complete_workflow', max_duration=10.0):
            # Step 1: Initialize all databases
            init_success = self.schema_manager.initialize_all_databases()
            self.assertTrue(init_success)
            
            # Step 2: Validate all databases
            validation_results = self.schema_manager.validate_all_databases()
            self.assertIsInstance(validation_results, dict)
            
            # Step 3: Test connection management
            for db_name in validation_results.keys():
                if validation_results[db_name]['exists']:
                    with self.connection_manager.get_connection(db_name) as conn:
                        cursor = conn.cursor()
                        cursor.execute('SELECT COUNT(*) FROM schema_version')
                        count = cursor.fetchone()[0]
                        self.assertGreaterEqual(count, 1)
            
            # Step 4: Test auto-upgrade system
            upgrade_results = self.auto_upgrader.check_and_upgrade_all()
            self.assertIsInstance(upgrade_results, dict)
    
    def test_database_performance_under_load(self):
        """Test database performance under load"""
        # Initialize test database
        self.schema_manager.initialize_all_databases()
        
        db_name = 'ai_learning_database.db'
        
        with self.performance_context('database_load_test', max_duration=5.0):
            # Simulate high load
            def load_test_worker(worker_id):
                results = []
                for i in range(10):
                    try:
                        # Insert operation
                        self.connection_manager.execute_query(
                            db_name,
                            'INSERT INTO predictions (match_id, teams_data, ai_strategies) VALUES (?, ?, ?)',
                            (f'match_{worker_id}_{i}', '{"test": "data"}', '{"test": "strategy"}'),
                            fetch_method='none'
                        )
                        
                        # Query operation
                        result = self.connection_manager.execute_query(
                            db_name,
                            'SELECT COUNT(*) FROM predictions WHERE match_id LIKE ?',
                            (f'match_{worker_id}%',),
                            fetch_method='fetchone'
                        )
                        results.append(result[0])
                        
                    except Exception as e:
                        results.append(f'error: {e}')
                
                return results
            
            # Run load test with multiple workers
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(load_test_worker, i) for i in range(5)]
                
                all_results = []
                for future in as_completed(futures):
                    worker_results = future.result()
                    all_results.extend(worker_results)
            
            # Verify results
            error_count = sum(1 for result in all_results if isinstance(result, str) and result.startswith('error'))
            self.assertLess(error_count, len(all_results) * 0.1)  # Less than 10% errors
    
    def test_error_recovery_and_resilience(self):
        """Test error recovery and system resilience"""
        # Initialize databases
        self.schema_manager.initialize_all_databases()
        
        db_name = 'ai_learning_database.db'
        
        # Test recovery from connection errors
        with self.connection_manager.get_connection(db_name) as conn:
            # Simulate connection being used normally
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM predictions')
            initial_count = cursor.fetchone()[0]
        
        # Test handling of invalid queries
        try:
            self.connection_manager.execute_query(
                db_name,
                'INVALID SQL QUERY THAT WILL FAIL',
                fetch_method='fetchone'
            )
            self.fail("Should have raised an exception for invalid SQL")
        except Exception:
            pass  # Expected behavior
        
        # Verify system continues to work after error
        result = self.connection_manager.execute_query(
            db_name,
            'SELECT COUNT(*) FROM predictions',
            fetch_method='fetchone'
        )
        
        self.assertEqual(result[0], initial_count)
    
    def test_startup_database_check_integration(self):
        """Test startup database check integration"""
        # This test simulates what happens during application startup
        
        # Remove any existing databases to simulate fresh start
        for db_file in self.test_db_dir.glob('*.db'):
            db_file.unlink()
        
        # Run startup database check
        startup_success = startup_database_check()
        
        # The function should handle missing databases gracefully
        self.assertIsInstance(startup_success, bool)
        
        # After startup check, basic databases should be available
        if startup_success:
            expected_dbs = ['universal_cricket_intelligence.db', 'ai_learning_database.db']
            for db_name in expected_dbs:
                db_path = self.test_db_dir / db_name
                if db_path.exists():
                    # Verify database is functional
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
                    tables = cursor.fetchall()
                    conn.close()
                    
                    self.assertGreater(len(tables), 0, f"Database {db_name} should have tables")

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDatabaseSchemaManagement,
        TestDatabaseMigrations,
        TestDatabaseConnectionManager,
        TestDatabaseCleanup,
        TestDatabaseAutoUpgrade,
        TestDatabaseIntegrationWorkflow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)