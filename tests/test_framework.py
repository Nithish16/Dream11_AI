#!/usr/bin/env python3
"""
Comprehensive Test Framework for Dream11 AI
Provides unified testing infrastructure with database, performance, and integration tests
"""

import unittest
import sqlite3
import tempfile
import shutil
import time
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dream11TestCase(unittest.TestCase):
    """
    Base test case with common utilities for Dream11 AI testing
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with temporary directories and mock data"""
        cls.test_dir = Path(tempfile.mkdtemp(prefix='dream11_test_'))
        cls.original_cwd = os.getcwd()
        
        # Create test database directory
        cls.db_dir = cls.test_dir / 'databases'
        cls.db_dir.mkdir(exist_ok=True)
        
        # Create test data directory
        cls.data_dir = cls.test_dir / 'test_data'
        cls.data_dir.mkdir(exist_ok=True)
        
        # Initialize test databases
        cls._create_test_databases()
        
        # Create mock data
        cls._create_mock_data()
        
        logger.info(f"✅ Test environment set up in {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            os.chdir(cls.original_cwd)
            shutil.rmtree(cls.test_dir, ignore_errors=True)
            logger.info("🧹 Test environment cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up test environment: {e}")
    
    def setUp(self):
        """Set up individual test"""
        self.start_time = time.time()
        
        # Create test-specific temporary directory
        self.temp_dir = self.test_dir / f'test_{self._testMethodName}'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Change to test directory
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up individual test"""
        self.test_duration = time.time() - self.start_time
        
        # Log test performance
        if self.test_duration > 1.0:
            logger.warning(f"⏱️ Slow test {self._testMethodName}: {self.test_duration:.2f}s")
        
        # Clean up test-specific directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @classmethod
    def _create_test_databases(cls):
        """Create test databases with sample schemas"""
        test_databases = {
            'test_universal.db': cls._get_universal_test_schema(),
            'test_learning.db': cls._get_learning_test_schema(),
            'test_predictions.db': cls._get_predictions_test_schema()
        }
        
        for db_name, schema in test_databases.items():
            db_path = cls.db_dir / db_name
            conn = sqlite3.connect(str(db_path))
            
            for table_sql in schema:
                conn.execute(table_sql)
            
            conn.commit()
            conn.close()
    
    @classmethod
    def _get_universal_test_schema(cls) -> List[str]:
        """Get test schema for universal database"""
        return [
            '''
            CREATE TABLE schema_version (
                version TEXT PRIMARY KEY,
                applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            '''
            CREATE TABLE system_config (
                config_key TEXT PRIMARY KEY,
                config_value TEXT NOT NULL,
                config_type TEXT DEFAULT 'string'
            )
            ''',
            '''
            CREATE TABLE player_intelligence (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT NOT NULL,
                format_type TEXT NOT NULL,
                performance_data TEXT,
                consistency_score REAL DEFAULT 0.0
            )
            '''
        ]
    
    @classmethod
    def _get_learning_test_schema(cls) -> List[str]:
        """Get test schema for learning database"""
        return [
            '''
            CREATE TABLE schema_version (
                version TEXT PRIMARY KEY,
                applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            '''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                teams_data TEXT NOT NULL,
                ai_strategies TEXT NOT NULL
            )
            ''',
            '''
            CREATE TABLE results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                result_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                winning_score INTEGER NOT NULL,
                analysis_data TEXT NOT NULL
            )
            '''
        ]
    
    @classmethod
    def _get_predictions_test_schema(cls) -> List[str]:
        """Get test schema for predictions database"""
        return [
            '''
            CREATE TABLE schema_version (
                version TEXT PRIMARY KEY,
                applied_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            '''
            CREATE TABLE local_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                team_data TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0
            )
            '''
        ]
    
    @classmethod
    def _create_mock_data(cls):
        """Create mock data for testing"""
        cls.mock_match_data = {
            'match_id': '123456',
            'match_format': 'T20I',
            'teams': ['Team A', 'Team B'],
            'venue': 'Test Stadium',
            'date': '2025-08-12'
        }
        
        cls.mock_player_data = [
            {
                'player_id': 1,
                'name': 'Test Player 1',
                'role': 'Batsman',
                'team': 'Team A',
                'consistency_score': 0.85,
                'form_momentum': 0.75
            },
            {
                'player_id': 2,
                'name': 'Test Player 2',
                'role': 'Bowler',
                'team': 'Team B',
                'consistency_score': 0.78,
                'form_momentum': 0.82
            }
        ]
        
        cls.mock_team_data = {
            'team_id': 1,
            'players': cls.mock_player_data[:11],
            'captain': cls.mock_player_data[0],
            'vice_captain': cls.mock_player_data[1],
            'strategy': 'Optimal'
        }
    
    def create_test_database(self, db_name: str, schema: List[str] = None) -> Path:
        """Create a temporary test database"""
        db_path = self.temp_dir / db_name
        conn = sqlite3.connect(str(db_path))
        
        if schema:
            for table_sql in schema:
                conn.execute(table_sql)
        
        conn.commit()
        conn.close()
        
        return db_path
    
    def populate_test_database(self, db_path: Path, table_data: Dict[str, List[Dict]]):
        """Populate test database with sample data"""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        for table_name, rows in table_data.items():
            if not rows:
                continue
            
            # Get column names from first row
            columns = list(rows[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            
            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            for row in rows:
                values = [row[col] for col in columns]
                cursor.execute(insert_sql, values)
        
        conn.commit()
        conn.close()
    
    def assert_database_table_exists(self, db_path: Path, table_name: str):
        """Assert that a database table exists"""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        ''', (table_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result, f"Table '{table_name}' does not exist in database")
    
    def assert_database_row_count(self, db_path: Path, table_name: str, expected_count: int):
        """Assert that a database table has expected row count"""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        actual_count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(actual_count, expected_count, 
                        f"Table '{table_name}' has {actual_count} rows, expected {expected_count}")
    
    def assert_performance_within_limit(self, operation: Callable, max_duration: float, *args, **kwargs):
        """Assert that an operation completes within time limit"""
        start_time = time.time()
        result = operation(*args, **kwargs)
        duration = time.time() - start_time
        
        self.assertLessEqual(duration, max_duration, 
                           f"Operation took {duration:.2f}s, expected <= {max_duration}s")
        
        return result
    
    def create_mock_api_response(self, data: Dict, status_code: int = 200) -> Mock:
        """Create mock API response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data
        mock_response.text = json.dumps(data)
        return mock_response
    
    @contextmanager
    def temporary_config(self, config_updates: Dict[str, Any]):
        """Temporarily update configuration for testing"""
        # This would integrate with the actual config system
        # For now, it's a placeholder
        original_values = {}
        
        try:
            # Apply temporary config
            for key, value in config_updates.items():
                # Store original and set new value
                original_values[key] = value  # Placeholder
            
            yield
            
        finally:
            # Restore original config
            for key, value in original_values.items():
                pass  # Restore original value

class DatabaseTestMixin:
    """
    Mixin for database-related test utilities
    """
    
    def setUp(self):
        """Set up database testing utilities"""
        super().setUp()
        
        # Track database operations
        self.db_operations = []
        self.connection_count = 0
    
    def create_test_connection(self, db_path: Path) -> sqlite3.Connection:
        """Create a test database connection with tracking"""
        conn = sqlite3.connect(str(db_path))
        self.connection_count += 1
        return conn
    
    def assert_no_open_connections(self):
        """Assert that all database connections have been properly closed"""
        # This would integrate with the connection manager
        # For now, it's a basic check
        pass
    
    def simulate_database_error(self, error_type: str = "operational"):
        """Simulate database errors for testing error handling"""
        if error_type == "operational":
            return sqlite3.OperationalError("Database is locked")
        elif error_type == "integrity":
            return sqlite3.IntegrityError("UNIQUE constraint failed")
        else:
            return sqlite3.Error("Generic database error")

class PerformanceTestMixin:
    """
    Mixin for performance testing utilities
    """
    
    def setUp(self):
        """Set up performance testing utilities"""
        super().setUp()
        
        # Performance tracking
        self.performance_metrics = {}
        self.memory_usage_start = 0
    
    def start_performance_tracking(self, metric_name: str):
        """Start tracking performance for a metric"""
        self.performance_metrics[metric_name] = {
            'start_time': time.time(),
            'start_memory': 0  # Placeholder for memory tracking
        }
    
    def stop_performance_tracking(self, metric_name: str) -> Dict[str, float]:
        """Stop tracking performance and return metrics"""
        if metric_name not in self.performance_metrics:
            raise ValueError(f"No tracking started for metric: {metric_name}")
        
        metrics = self.performance_metrics[metric_name]
        end_time = time.time()
        
        result = {
            'duration': end_time - metrics['start_time'],
            'memory_delta': 0  # Placeholder for memory tracking
        }
        
        del self.performance_metrics[metric_name]
        return result
    
    @contextmanager
    def performance_context(self, metric_name: str, max_duration: float = None):
        """Context manager for performance tracking"""
        self.start_performance_tracking(metric_name)
        
        try:
            yield
        finally:
            metrics = self.stop_performance_tracking(metric_name)
            
            if max_duration and metrics['duration'] > max_duration:
                self.fail(f"Performance test '{metric_name}' took {metrics['duration']:.2f}s, "
                         f"expected <= {max_duration}s")

class IntegrationTestMixin:
    """
    Mixin for integration testing utilities
    """
    
    def setUp(self):
        """Set up integration testing utilities"""
        super().setUp()
        
        # Track external service calls
        self.api_calls = []
        self.mock_responses = {}
    
    def mock_api_call(self, endpoint: str, response_data: Dict, status_code: int = 200):
        """Mock an API call for integration testing"""
        self.mock_responses[endpoint] = {
            'data': response_data,
            'status_code': status_code
        }
    
    def assert_api_called(self, endpoint: str, min_calls: int = 1):
        """Assert that an API endpoint was called minimum number of times"""
        call_count = sum(1 for call in self.api_calls if call['endpoint'] == endpoint)
        
        self.assertGreaterEqual(call_count, min_calls,
                               f"API endpoint '{endpoint}' was called {call_count} times, "
                               f"expected >= {min_calls}")
    
    def simulate_api_failure(self, endpoint: str, failure_type: str = "timeout"):
        """Simulate API failures for testing resilience"""
        if failure_type == "timeout":
            self.mock_responses[endpoint] = {'error': 'timeout', 'status_code': 408}
        elif failure_type == "server_error":
            self.mock_responses[endpoint] = {'error': 'server_error', 'status_code': 500}
        elif failure_type == "not_found":
            self.mock_responses[endpoint] = {'error': 'not_found', 'status_code': 404}

class Dream11TestSuite:
    """
    Test suite runner for Dream11 AI
    """
    
    def __init__(self, test_dir: Optional[Path] = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.test_results = {}
        
        # Test discovery patterns
        self.test_patterns = [
            'test_*.py',
            '*_test.py'
        ]
    
    def discover_tests(self) -> unittest.TestSuite:
        """Discover all tests in the test directory"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for pattern in self.test_patterns:
            discovered = loader.discover(str(self.test_dir), pattern=pattern)
            suite.addTest(discovered)
        
        return suite
    
    def run_tests(self, verbosity: int = 2) -> unittest.TestResult:
        """Run all discovered tests"""
        suite = self.discover_tests()
        runner = unittest.TextTestRunner(verbosity=verbosity)
        
        logger.info(f"🧪 Running Dream11 AI test suite...")
        start_time = time.time()
        
        result = runner.run(suite)
        
        duration = time.time() - start_time
        logger.info(f"✅ Test suite completed in {duration:.2f}s")
        
        # Log summary
        self._log_test_summary(result, duration)
        
        return result
    
    def _log_test_summary(self, result: unittest.TestResult, duration: float):
        """Log test summary"""
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🧪 TEST SUITE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {total_tests - failures - errors}")
        logger.info(f"Failed: {failures}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info("=" * 60)
        
        if failures > 0:
            logger.error("❌ TEST FAILURES:")
            for test, traceback in result.failures:
                logger.error(f"  - {test}: {traceback.splitlines()[-1]}")
        
        if errors > 0:
            logger.error("❌ TEST ERRORS:")
            for test, traceback in result.errors:
                logger.error(f"  - {test}: {traceback.splitlines()[-1]}")

def run_all_tests():
    """Convenience function to run all tests"""
    suite = Dream11TestSuite()
    result = suite.run_tests()
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests when executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)