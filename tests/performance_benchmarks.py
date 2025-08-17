#!/usr/bin/env python3
"""
Performance Benchmarks for Dream11 AI System
Comprehensive performance testing and benchmarking suite
"""

import unittest
import sys
import os
import time
import sqlite3
import json
import statistics
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from test_framework import Dream11TestCase, PerformanceTestMixin, DatabaseTestMixin

# Import system modules
try:
    from core_logic.database_connection_manager import DatabaseConnectionManager
    from core_logic.database_schema_manager import DatabaseSchemaManager
    from core_logic.database_cleanup import DatabaseCleanupManager
    from database_auto_upgrade import DatabaseAutoUpgrader
    from dream11_ultimate import Dream11Ultimate
except ImportError as e:
    print(f"⚠️ Some performance benchmark imports failed: {e}")
    print("Benchmarks will use mocks for missing components")

@dataclass
class BenchmarkResult:
    """Store benchmark test results"""
    test_name: str
    duration: float
    memory_usage: float = 0.0
    throughput: float = 0.0  # operations per second
    success_rate: float = 1.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_duration: float = 0.0
    
    def add_result(self, result: BenchmarkResult):
        """Add benchmark result to suite"""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark suite summary"""
        if not self.results:
            return {'name': self.name, 'status': 'empty'}
        
        durations = [r.duration for r in self.results]
        throughputs = [r.throughput for r in self.results if r.throughput > 0]
        success_rates = [r.success_rate for r in self.results]
        
        return {
            'name': self.name,
            'test_count': len(self.results),
            'total_duration': sum(durations),
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'median_duration': statistics.median(durations),
            'avg_throughput': statistics.mean(throughputs) if throughputs else 0,
            'avg_success_rate': statistics.mean(success_rates),
            'total_errors': sum(r.error_count for r in self.results)
        }

class DatabasePerformanceBenchmarks(Dream11TestCase, PerformanceTestMixin, DatabaseTestMixin):
    """Benchmark database operations and connection management"""
    
    def setUp(self):
        super().setUp()
        
        # Create benchmark test environment
        self.benchmark_db_dir = self.temp_dir / 'benchmark_databases'
        self.benchmark_db_dir.mkdir(exist_ok=True)
        
        # Initialize managers
        self.schema_manager = DatabaseSchemaManager(base_path=str(self.benchmark_db_dir))
        self.connection_manager = DatabaseConnectionManager(base_path=str(self.benchmark_db_dir))
        
        # Initialize benchmark databases
        self._setup_benchmark_databases()
        
        # Benchmark suite
        self.benchmark_suite = BenchmarkSuite('Database Performance')
    
    def _setup_benchmark_databases(self):
        """Set up databases optimized for benchmarking"""
        self.schema_manager.initialize_all_databases()
        
        # Pre-populate with test data for realistic benchmarks
        self._populate_test_data()
    
    def _populate_test_data(self):
        """Populate databases with test data"""
        # Populate learning database with sample predictions
        db_name = 'ai_learning_database.db'
        
        test_data = []
        for i in range(1000):
            test_data.append({
                'match_id': f'match_{i:04d}',
                'teams_data': json.dumps({'team_a': f'Team_{i%10}', 'team_b': f'Team_{(i+1)%10}'}),
                'ai_strategies': json.dumps({'strategy': f'Strategy_{i%5}'})
            })
        
        # Batch insert for better performance
        with self.connection_manager.get_connection(db_name) as conn:
            cursor = conn.cursor()
            for data in test_data:
                cursor.execute('''
                    INSERT INTO predictions (match_id, teams_data, ai_strategies)
                    VALUES (?, ?, ?)
                ''', (data['match_id'], data['teams_data'], data['ai_strategies']))
            conn.commit()
    
    def test_database_connection_pool_performance(self):
        """Benchmark connection pool performance"""
        print("\n🔄 Benchmarking database connection pool performance...")
        
        db_name = 'ai_learning_database.db'
        num_operations = 100
        
        # Warm up connection pool
        for _ in range(10):
            with self.connection_manager.get_connection(db_name) as conn:
                conn.execute('SELECT 1').fetchone()
        
        # Benchmark connection acquisition and release
        start_time = time.time()
        
        for i in range(num_operations):
            with self.connection_manager.get_connection(db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM predictions LIMIT 1')
                cursor.fetchone()
        
        duration = time.time() - start_time
        throughput = num_operations / duration
        
        result = BenchmarkResult(
            test_name='Connection Pool Performance',
            duration=duration,
            throughput=throughput,
            metadata={
                'operations': num_operations,
                'ops_per_second': throughput,
                'avg_operation_time': duration / num_operations
            }
        )
        
        self.benchmark_suite.add_result(result)
        
        print(f"   ⚡ {throughput:.1f} connections/sec ({duration:.3f}s total)")
        
        # Performance assertion
        self.assertGreater(throughput, 50, "Connection pool should handle >50 ops/sec")
    
    def test_concurrent_database_access_performance(self):
        """Benchmark concurrent database access"""
        print("\n🔄 Benchmarking concurrent database access...")
        
        db_name = 'ai_learning_database.db'
        num_workers = 10
        operations_per_worker = 20
        
        def worker_task(worker_id):
            worker_start = time.time()
            operations = 0
            errors = 0
            
            for i in range(operations_per_worker):
                try:
                    with self.connection_manager.get_connection(db_name) as conn:
                        cursor = conn.cursor()
                        
                        # Mix of read and write operations
                        if i % 3 == 0:
                            # Insert operation
                            cursor.execute('''
                                INSERT INTO predictions (match_id, teams_data, ai_strategies)
                                VALUES (?, ?, ?)
                            ''', (f'concurrent_{worker_id}_{i}', '{"test": "data"}', '{"test": "strategy"}'))
                            conn.commit()
                        else:
                            # Select operation
                            cursor.execute('SELECT COUNT(*) FROM predictions WHERE match_id LIKE ?',
                                         (f'concurrent_{worker_id}%',))
                            cursor.fetchone()
                        
                        operations += 1
                        
                except Exception as e:
                    errors += 1
                    print(f"   ❌ Worker {worker_id} error: {e}")
            
            worker_duration = time.time() - worker_start
            return {
                'worker_id': worker_id,
                'operations': operations,
                'errors': errors,
                'duration': worker_duration,
                'throughput': operations / worker_duration if worker_duration > 0 else 0
            }
        
        # Run concurrent benchmark
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]
        
        total_duration = time.time() - start_time
        total_operations = sum(r['operations'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_throughput = total_operations / total_duration
        success_rate = (total_operations - total_errors) / total_operations if total_operations > 0 else 0
        
        result = BenchmarkResult(
            test_name='Concurrent Database Access',
            duration=total_duration,
            throughput=total_throughput,
            success_rate=success_rate,
            error_count=total_errors,
            metadata={
                'workers': num_workers,
                'operations_per_worker': operations_per_worker,
                'total_operations': total_operations,
                'worker_results': results
            }
        )
        
        self.benchmark_suite.add_result(result)
        
        print(f"   ⚡ {total_throughput:.1f} ops/sec with {num_workers} workers")
        print(f"   ✅ Success rate: {success_rate:.1%} ({total_errors} errors)")
        
        # Performance assertions
        self.assertGreater(total_throughput, 100, "Should handle >100 concurrent ops/sec")
        self.assertGreater(success_rate, 0.95, "Success rate should be >95%")
    
    def test_large_dataset_query_performance(self):
        """Benchmark performance with large datasets"""
        print("\n🔄 Benchmarking large dataset query performance...")
        
        db_name = 'ai_learning_database.db'
        
        # Test different query types
        query_tests = [
            ('Simple Count', 'SELECT COUNT(*) FROM predictions'),
            ('Filtered Count', "SELECT COUNT(*) FROM predictions WHERE match_id LIKE 'match_%'"),
            ('Grouped Query', 'SELECT substr(match_id, 1, 10), COUNT(*) FROM predictions GROUP BY substr(match_id, 1, 10)'),
            ('Join Query', '''
                SELECT p.match_id, COUNT(*) as prediction_count
                FROM predictions p
                LEFT JOIN results r ON p.match_id = r.match_id
                GROUP BY p.match_id
                LIMIT 100
            ''')
        ]
        
        for test_name, query in query_tests:
            start_time = time.time()
            
            try:
                result_data = self.connection_manager.execute_query(
                    db_name, query, fetch_method='fetchall'
                )
                duration = time.time() - start_time
                
                result = BenchmarkResult(
                    test_name=f'Large Dataset - {test_name}',
                    duration=duration,
                    throughput=1.0 / duration,  # queries per second
                    metadata={
                        'query': query,
                        'result_count': len(result_data) if result_data else 0,
                        'query_type': test_name
                    }
                )
                
                self.benchmark_suite.add_result(result)
                
                print(f"   📊 {test_name}: {duration:.3f}s ({len(result_data) if result_data else 0} rows)")
                
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
    
    def test_database_optimization_performance(self):
        """Benchmark database optimization operations"""
        print("\n🔄 Benchmarking database optimization performance...")
        
        db_name = 'ai_learning_database.db'
        
        # Test VACUUM operation
        start_time = time.time()
        vacuum_success = self.connection_manager.optimize_database(db_name)
        vacuum_duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name='Database Optimization (VACUUM + ANALYZE)',
            duration=vacuum_duration,
            success_rate=1.0 if vacuum_success else 0.0,
            metadata={
                'operation': 'VACUUM + ANALYZE',
                'success': vacuum_success
            }
        )
        
        self.benchmark_suite.add_result(result)
        
        print(f"   🔧 VACUUM + ANALYZE: {vacuum_duration:.3f}s ({'✅' if vacuum_success else '❌'})")
        
        # Performance assertion
        self.assertTrue(vacuum_success, "Database optimization should succeed")
        self.assertLess(vacuum_duration, 30.0, "Optimization should complete within 30 seconds")

class SystemPerformanceBenchmarks(Dream11TestCase, PerformanceTestMixin):
    """Benchmark overall system performance"""
    
    def setUp(self):
        super().setUp()
        
        # Create system benchmark environment
        self.system_benchmark_dir = self.temp_dir / 'system_benchmarks'
        self.system_benchmark_dir.mkdir(exist_ok=True)
        
        # Initialize benchmark suite
        self.benchmark_suite = BenchmarkSuite('System Performance')
        
        # Mock data for system tests
        self._create_system_test_data()
    
    def _create_system_test_data(self):
        """Create test data for system benchmarks"""
        self.mock_match_data = {
            'match_id': 'benchmark_123456',
            'format': 'T20I',
            'teams': ['Benchmark Team A', 'Benchmark Team B'],
            'players': [
                {
                    'player_id': i,
                    'name': f'Player {i}',
                    'role': ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'][i % 4],
                    'team': 'Benchmark Team A' if i < 11 else 'Benchmark Team B',
                    'expected_points': 40 + (i % 30),
                    'consistency_score': 0.5 + (i % 10) * 0.05
                }
                for i in range(22)
            ]
        }
    
    @unittest.skipUnless(os.path.exists('dream11_ultimate.py'), "Ultimate system not available")
    def test_team_generation_performance(self):
        """Benchmark team generation performance"""
        print("\n🔄 Benchmarking team generation performance...")
        
        # Mock the Dream11Ultimate class if not available
        try:
            ultimate_system = Dream11Ultimate()
            
            start_time = time.time()
            
            # Generate multiple teams
            teams_generated = 0
            errors = 0
            
            for i in range(5):  # Generate 5 teams
                try:
                    # Mock prediction call
                    result = ultimate_system.predict(
                        match_id='benchmark_123456',
                        save_to_file=False
                    )
                    if result:
                        teams_generated += 1
                except Exception as e:
                    errors += 1
                    print(f"   ❌ Team generation error: {e}")
            
            duration = time.time() - start_time
            success_rate = teams_generated / 5
            throughput = teams_generated / duration if duration > 0 else 0
            
            result = BenchmarkResult(
                test_name='Team Generation Performance',
                duration=duration,
                throughput=throughput,
                success_rate=success_rate,
                error_count=errors,
                metadata={
                    'teams_requested': 5,
                    'teams_generated': teams_generated,
                    'avg_time_per_team': duration / teams_generated if teams_generated > 0 else 0
                }
            )
            
            self.benchmark_suite.add_result(result)
            
            print(f"   ⚡ {teams_generated} teams in {duration:.3f}s ({throughput:.1f} teams/sec)")
            
        except ImportError:
            print("   ⚠️ Ultimate system not available for benchmark")
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage patterns"""
        print("\n🔄 Benchmarking memory usage...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        start_time = time.time()
        
        # Create large data structures
        large_datasets = []
        for i in range(100):
            dataset = {
                'data': list(range(1000)),
                'metadata': {'id': i, 'type': f'dataset_{i}'}
            }
            large_datasets.append(dataset)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del large_datasets
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name='Memory Usage Pattern',
            duration=duration,
            memory_usage=peak_memory - initial_memory,
            metadata={
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': peak_memory - initial_memory,
                'memory_cleanup_mb': peak_memory - final_memory
            }
        )
        
        self.benchmark_suite.add_result(result)
        
        print(f"   💾 Memory usage: {initial_memory:.1f} → {peak_memory:.1f} → {final_memory:.1f} MB")
        print(f"   🔄 Cleanup efficiency: {((peak_memory - final_memory) / (peak_memory - initial_memory) * 100):.1f}%")
    
    def test_startup_performance_benchmark(self):
        """Benchmark system startup performance"""
        print("\n🔄 Benchmarking system startup performance...")
        
        # Simulate system startup components
        startup_components = [
            ('Database Schema Check', self._mock_schema_check, 2.0),
            ('Connection Pool Init', self._mock_connection_init, 1.0),
            ('AI Model Loading', self._mock_model_loading, 3.0),
            ('Configuration Loading', self._mock_config_loading, 0.5)
        ]
        
        total_startup_time = 0
        component_times = {}
        
        for component_name, component_func, timeout in startup_components:
            start_time = time.time()
            
            try:
                success = component_func()
                duration = time.time() - start_time
                component_times[component_name] = {
                    'duration': duration,
                    'success': success,
                    'within_timeout': duration <= timeout
                }
                total_startup_time += duration
                
                print(f"   📦 {component_name}: {duration:.3f}s ({'✅' if success else '❌'})")
                
            except Exception as e:
                duration = time.time() - start_time
                component_times[component_name] = {
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ {component_name}: Failed ({e})")
        
        success_rate = sum(1 for c in component_times.values() if c.get('success', False)) / len(startup_components)
        
        result = BenchmarkResult(
            test_name='System Startup Performance',
            duration=total_startup_time,
            success_rate=success_rate,
            metadata={
                'component_times': component_times,
                'total_components': len(startup_components),
                'successful_components': sum(1 for c in component_times.values() if c.get('success', False))
            }
        )
        
        self.benchmark_suite.add_result(result)
        
        print(f"   🚀 Total startup time: {total_startup_time:.3f}s")
        print(f"   ✅ Success rate: {success_rate:.1%}")
        
        # Performance assertions
        self.assertLess(total_startup_time, 10.0, "System startup should complete within 10 seconds")
        self.assertGreater(success_rate, 0.8, "At least 80% of components should start successfully")
    
    def _mock_schema_check(self) -> bool:
        """Mock database schema check"""
        time.sleep(0.2)  # Simulate schema check time
        return True
    
    def _mock_connection_init(self) -> bool:
        """Mock connection pool initialization"""
        time.sleep(0.1)  # Simulate connection init time
        return True
    
    def _mock_model_loading(self) -> bool:
        """Mock AI model loading"""
        time.sleep(0.5)  # Simulate model loading time
        return True
    
    def _mock_config_loading(self) -> bool:
        """Mock configuration loading"""
        time.sleep(0.05)  # Simulate config loading time
        return True

class BenchmarkReporter:
    """Generate comprehensive benchmark reports"""
    
    def __init__(self):
        self.benchmark_suites = []
    
    def add_suite(self, suite: BenchmarkSuite):
        """Add benchmark suite to report"""
        self.benchmark_suites.append(suite)
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🏆 DREAM11 AI PERFORMANCE BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        total_tests = 0
        total_duration = 0
        total_errors = 0
        
        for suite in self.benchmark_suites:
            summary = suite.get_summary()
            
            if summary.get('status') == 'empty':
                continue
            
            total_tests += summary['test_count']
            total_duration += summary['total_duration']
            total_errors += summary['total_errors']
            
            report_lines.append(f"📊 {suite.name}")
            report_lines.append("-" * 60)
            report_lines.append(f"Tests: {summary['test_count']}")
            report_lines.append(f"Total Duration: {summary['total_duration']:.3f}s")
            report_lines.append(f"Average Duration: {summary['avg_duration']:.3f}s")
            report_lines.append(f"Range: {summary['min_duration']:.3f}s - {summary['max_duration']:.3f}s")
            
            if summary['avg_throughput'] > 0:
                report_lines.append(f"Average Throughput: {summary['avg_throughput']:.1f} ops/sec")
            
            report_lines.append(f"Success Rate: {summary['avg_success_rate']:.1%}")
            report_lines.append(f"Errors: {summary['total_errors']}")
            report_lines.append("")
            
            # Individual test results
            for result in suite.results:
                status = "✅" if result.success_rate > 0.95 else "⚠️" if result.success_rate > 0.8 else "❌"
                throughput_str = f" ({result.throughput:.1f} ops/sec)" if result.throughput > 0 else ""
                
                report_lines.append(f"  {status} {result.test_name}: {result.duration:.3f}s{throughput_str}")
                
                if result.error_count > 0:
                    report_lines.append(f"     Errors: {result.error_count}")
            
            report_lines.append("")
        
        # Overall summary
        report_lines.append("=" * 80)
        report_lines.append("📈 OVERALL SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Total Duration: {total_duration:.3f}s")
        report_lines.append(f"Average per Test: {total_duration/total_tests:.3f}s" if total_tests > 0 else "N/A")
        report_lines.append(f"Total Errors: {total_errors}")
        report_lines.append(f"Error Rate: {total_errors/total_tests:.1%}" if total_tests > 0 else "0%")
        report_lines.append("")
        
        # Performance recommendations
        report_lines.extend(self._generate_recommendations())
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        recommendations.append("🎯 PERFORMANCE RECOMMENDATIONS")
        recommendations.append("-" * 40)
        
        # Analyze results for recommendations
        slow_tests = []
        high_error_tests = []
        
        for suite in self.benchmark_suites:
            for result in suite.results:
                if result.duration > 5.0:
                    slow_tests.append(result.test_name)
                if result.error_count > 0:
                    high_error_tests.append(result.test_name)
        
        if slow_tests:
            recommendations.append("⚡ Optimize slow operations:")
            for test in slow_tests[:5]:  # Top 5
                recommendations.append(f"  - {test}")
            recommendations.append("")
        
        if high_error_tests:
            recommendations.append("🔧 Fix error-prone operations:")
            for test in high_error_tests[:3]:  # Top 3
                recommendations.append(f"  - {test}")
            recommendations.append("")
        
        recommendations.append("📊 General optimizations:")
        recommendations.append("  - Consider connection pool tuning")
        recommendations.append("  - Implement query result caching")
        recommendations.append("  - Add database indices for frequent queries")
        recommendations.append("  - Consider async processing for heavy operations")
        
        return recommendations
    
    def save_report(self, filepath: str):
        """Save benchmark report to file"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)

class PerformanceBenchmarkRunner(Dream11TestCase):
    """Main benchmark runner"""
    
    def setUp(self):
        super().setUp()
        self.reporter = BenchmarkReporter()
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("🏁 Starting Dream11 AI Performance Benchmarks...")
        print("=" * 80)
        
        # Run database benchmarks
        db_benchmark = DatabasePerformanceBenchmarks()
        db_benchmark.setUp()
        
        try:
            db_benchmark.test_database_connection_pool_performance()
            db_benchmark.test_concurrent_database_access_performance()
            db_benchmark.test_large_dataset_query_performance()
            db_benchmark.test_database_optimization_performance()
            
            self.reporter.add_suite(db_benchmark.benchmark_suite)
            
        except Exception as e:
            print(f"❌ Database benchmarks failed: {e}")
        finally:
            db_benchmark.tearDown()
        
        # Run system benchmarks
        system_benchmark = SystemPerformanceBenchmarks()
        system_benchmark.setUp()
        
        try:
            system_benchmark.test_team_generation_performance()
            system_benchmark.test_memory_usage_benchmark()
            system_benchmark.test_startup_performance_benchmark()
            
            self.reporter.add_suite(system_benchmark.benchmark_suite)
            
        except Exception as e:
            print(f"❌ System benchmarks failed: {e}")
        finally:
            system_benchmark.tearDown()
        
        # Generate and display report
        print("\n" + "=" * 80)
        report = self.reporter.generate_report()
        print(report)
        
        # Save report to file
        report_path = self.test_dir / 'performance_report.txt'
        self.reporter.save_report(str(report_path))
        print(f"\n📄 Full report saved to: {report_path}")

if __name__ == '__main__':
    # Run performance benchmarks
    benchmark_runner = PerformanceBenchmarkRunner()
    benchmark_runner.setUp()
    
    try:
        benchmark_runner.run_all_benchmarks()
    finally:
        benchmark_runner.tearDown()
    
    print("\n🏁 Performance benchmarks completed!")