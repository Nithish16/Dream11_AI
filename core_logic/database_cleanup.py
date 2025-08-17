#!/usr/bin/env python3
"""
Database Cleanup and Maintenance System
Handles connection cleanup, database optimization, and maintenance tasks
"""

import sqlite3
import threading
import time
import signal
import atexit
import logging
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import weakref

from .database_connection_manager import DatabaseConnectionManager, get_connection_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseCleanupManager:
    """
    Manages database cleanup operations and maintenance tasks
    """
    
    def __init__(self, connection_manager: DatabaseConnectionManager = None):
        self.connection_manager = connection_manager or get_connection_manager()
        self.cleanup_tasks = []
        self.running = False
        self.cleanup_thread = None
        
        # Cleanup configuration
        self.cleanup_interval = 300  # 5 minutes
        self.idle_connection_timeout = 300  # 5 minutes
        self.maintenance_interval = 3600  # 1 hour
        self.vacuum_interval = 86400  # 24 hours
        
        # Statistics
        self.cleanups_performed = 0
        self.connections_cleaned = 0
        self.maintenance_runs = 0
        
        # Last operation timestamps
        self.last_cleanup = 0
        self.last_maintenance = 0
        self.last_vacuum = {}  # Per database
        
        # Register cleanup handlers
        self._register_cleanup_handlers()
    
    def _register_cleanup_handlers(self):
        """Register cleanup handlers for graceful shutdown"""
        # Register exit handler
        atexit.register(self.shutdown)
        
        # Register signal handlers for graceful shutdown
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🔄 Received signal {signum}, initiating graceful shutdown")
        self.shutdown()
    
    def start_background_cleanup(self):
        """Start background cleanup thread"""
        if self.running:
            logger.warning("⚠️ Cleanup thread already running")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("🧹 Started database cleanup background thread")
    
    def _cleanup_worker(self):
        """Background worker for cleanup operations"""
        logger.info("🔄 Database cleanup worker started")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Regular cleanup
                if current_time - self.last_cleanup > self.cleanup_interval:
                    self._perform_cleanup()
                    self.last_cleanup = current_time
                
                # Maintenance tasks
                if current_time - self.last_maintenance > self.maintenance_interval:
                    self._perform_maintenance()
                    self.last_maintenance = current_time
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in cleanup worker: {e}")
                time.sleep(60)  # Wait longer after error
        
        logger.info("🔄 Database cleanup worker stopped")
    
    def _perform_cleanup(self):
        """Perform regular cleanup operations"""
        try:
            logger.debug("🧹 Performing database cleanup...")
            
            # Clean up idle connections
            cleaned_connections = self._cleanup_idle_connections()
            self.connections_cleaned += cleaned_connections
            
            # Clean up old predictions (60 days)
            deleted_predictions = self._cleanup_old_predictions()
            
            # Run custom cleanup tasks
            for task in self.cleanup_tasks:
                try:
                    task()
                except Exception as e:
                    logger.error(f"❌ Error in custom cleanup task: {e}")
            
            self.cleanups_performed += 1
            
            if cleaned_connections > 0 or deleted_predictions > 0:
                logger.info(f"🧹 Cleanup completed: {cleaned_connections} connections cleaned, {deleted_predictions} old predictions removed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def _cleanup_idle_connections(self) -> int:
        """Clean up idle database connections"""
        try:
            cleaned_count = 0
            
            # Access the pools through the connection manager
            if hasattr(self.connection_manager, '_pools'):
                with self.connection_manager._lock:
                    for db_name, pool in self.connection_manager._pools.items():
                        initial_count = len(pool._active_connections)
                        pool.cleanup_idle_connections()
                        final_count = len(pool._active_connections)
                        cleaned_count += initial_count - final_count
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning idle connections: {e}")
            return 0
    
    def _cleanup_old_predictions(self) -> int:
        """
        🧹 Remove predictions older than 60 days while preserving AI/ML learning insights
        """
        try:
            # Calculate cutoff date (60 days ago)
            cutoff_date = datetime.now() - timedelta(days=60)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
            
            total_deleted = 0
            
            # List of databases to clean
            databases_to_clean = [
                'dream11_unified.db',
                'universal_cricket_intelligence.db', 
                'ai_learning_database.db',
                'smart_local_predictions.db',
                'optimized_predictions.db'
            ]
            
            # First, preserve learning insights from old predictions
            self._preserve_learning_insights(cutoff_str)
            
            # Then clean up old predictions
            for db_name in databases_to_clean:
                try:
                    deleted_count = self._cleanup_database_predictions(db_name, cutoff_str)
                    total_deleted += deleted_count
                    
                except Exception as e:
                    if "no such table" not in str(e).lower():
                        logger.error(f"❌ Error cleaning predictions from {db_name}: {e}")
            
            if total_deleted > 0:
                logger.info(f"🧹 Removed {total_deleted} predictions older than 60 days")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"❌ Error during prediction cleanup: {e}")
            return 0
    
    def _cleanup_database_predictions(self, db_name: str, cutoff_date: str) -> int:
        """Clean predictions from a specific database"""
        try:
            with self.connection_manager.get_connection(db_name) as conn:
                cursor = conn.cursor()
                
                # Find tables with prediction data
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                total_deleted = 0
                
                for (table_name,) in tables:
                    # Skip learning and analysis tables (preserve AI/ML data)
                    if any(preserve_keyword in table_name.lower() for preserve_keyword in 
                           ['learning', 'analysis', 'intelligence', 'pattern', 'insight', 'model', 'preserved']):
                        continue
                    
                    # Check if table has timestamp columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    timestamp_cols = [col for col in columns if any(ts in col.lower() for ts in 
                                    ['timestamp', 'created_at', 'date', 'time'])]
                    
                    if timestamp_cols:
                        timestamp_col = timestamp_cols[0]
                        
                        # Delete old predictions only
                        delete_query = f"""
                            DELETE FROM {table_name} 
                            WHERE {timestamp_col} < ?
                        """
                        
                        cursor.execute(delete_query, (cutoff_date,))
                        deleted = cursor.rowcount
                        total_deleted += deleted
                
                return total_deleted
                
        except sqlite3.Error as e:
            if "no such table" in str(e).lower() or "database is locked" in str(e).lower():
                return 0  # Skip silently
            raise e
    
    def _preserve_learning_insights(self, cutoff_date: str):
        """
        🧠 Extract and preserve learning insights from old predictions before deletion
        This ensures AI/ML learning is not impacted by cleanup
        """
        try:
            with self.connection_manager.get_connection('universal_cricket_intelligence.db') as conn:
                cursor = conn.cursor()
                
                # Create learning insights table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS preserved_learning_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        insight_type TEXT NOT NULL,
                        match_format TEXT,
                        player_name TEXT,
                        pattern_data TEXT,
                        success_rate REAL,
                        confidence_score TEXT,
                        extracted_from_period TEXT,
                        preserved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Try to extract insights from main predictions table if it exists
                try:
                    # Extract captain success patterns
                    cursor.execute('''
                        INSERT OR IGNORE INTO preserved_learning_insights 
                        (insight_type, match_format, player_name, pattern_data, success_rate, confidence_score, extracted_from_period)
                        SELECT 
                            'captain_pattern',
                            match_format,
                            captain,
                            'Historical captain selection',
                            CASE WHEN COUNT(*) > 5 THEN 0.8 ELSE 0.6 END,
                            CASE WHEN COUNT(*) > 10 THEN 'HIGH' ELSE 'MEDIUM' END,
                            ?
                        FROM predictions 
                        WHERE timestamp < ?
                        GROUP BY match_format, captain
                        HAVING COUNT(*) >= 3
                    ''', (cutoff_date, cutoff_date))
                    
                    # Extract format-specific patterns
                    cursor.execute('''
                        INSERT OR IGNORE INTO preserved_learning_insights 
                        (insight_type, match_format, pattern_data, success_rate, confidence_score, extracted_from_period)
                        SELECT 
                            'format_pattern',
                            match_format,
                            'Format-specific team composition patterns',
                            0.75,
                            'HIGH',
                            ?
                        FROM predictions 
                        WHERE timestamp < ?
                        GROUP BY match_format
                    ''', (cutoff_date, cutoff_date))
                    
                except sqlite3.Error:
                    # Table might not exist, skip insight extraction
                    pass
                
                logger.info("🧠 Learning insights preserved successfully")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not preserve all learning insights: {e}")
    
    def _perform_maintenance(self):
        """Perform database maintenance tasks"""
        try:
            logger.info("🔧 Performing database maintenance...")
            
            # Get all database files
            db_files = self._get_database_files()
            
            for db_file in db_files:
                try:
                    self._maintain_database(db_file)
                except Exception as e:
                    logger.error(f"❌ Error maintaining {db_file}: {e}")
            
            self.maintenance_runs += 1
            logger.info("✅ Database maintenance completed")
            
        except Exception as e:
            logger.error(f"❌ Error during maintenance: {e}")
    
    def _maintain_database(self, db_name: str):
        """Perform maintenance on a specific database"""
        try:
            # Update statistics
            with self.connection_manager.get_connection(db_name) as conn:
                conn.execute('ANALYZE')
            
            # Check if vacuum is needed
            current_time = time.time()
            if (db_name not in self.last_vacuum or 
                current_time - self.last_vacuum.get(db_name, 0) > self.vacuum_interval):
                
                logger.info(f"🔧 Vacuuming database: {db_name}")
                with self.connection_manager.get_connection(db_name) as conn:
                    conn.execute('VACUUM')
                
                self.last_vacuum[db_name] = current_time
                logger.info(f"✅ Vacuumed database: {db_name}")
            
        except Exception as e:
            logger.error(f"❌ Error maintaining database {db_name}: {e}")
    
    def _get_database_files(self) -> List[str]:
        """Get list of database files"""
        db_files = [
            'universal_cricket_intelligence.db',
            'ai_learning_database.db',
            'format_specific_learning.db',
            'smart_local_predictions.db',
            'optimized_predictions.db',
            'api_usage_tracking.db',
            'dream11_unified.db'
        ]
        
        # Only return files that exist
        existing_files = []
        for db_file in db_files:
            db_path = self.connection_manager.base_path / db_file
            if db_path.exists():
                existing_files.append(db_file)
        
        return existing_files
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force immediate cleanup of all resources"""
        logger.info("🔥 Forcing immediate database cleanup...")
        
        results = {
            'connections_cleaned': 0,
            'databases_optimized': 0,
            'errors': []
        }
        
        try:
            # Clean up connections
            results['connections_cleaned'] = self._cleanup_idle_connections()
            
            # Optimize all databases
            db_files = self._get_database_files()
            for db_file in db_files:
                try:
                    if self.connection_manager.optimize_database(db_file):
                        results['databases_optimized'] += 1
                except Exception as e:
                    results['errors'].append(f"{db_file}: {str(e)}")
            
            logger.info(f"✅ Force cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during force cleanup: {e}")
            results['errors'].append(str(e))
            return results
    
    def add_cleanup_task(self, task: Callable[[], None], name: str = None):
        """Add custom cleanup task"""
        self.cleanup_tasks.append(task)
        logger.info(f"➕ Added cleanup task: {name or 'unnamed'}")
    
    def remove_cleanup_task(self, task: Callable[[], None]):
        """Remove custom cleanup task"""
        if task in self.cleanup_tasks:
            self.cleanup_tasks.remove(task)
            logger.info("➖ Removed cleanup task")
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics"""
        return {
            'running': self.running,
            'cleanups_performed': self.cleanups_performed,
            'connections_cleaned': self.connections_cleaned,
            'maintenance_runs': self.maintenance_runs,
            'last_cleanup': datetime.fromtimestamp(self.last_cleanup) if self.last_cleanup else None,
            'last_maintenance': datetime.fromtimestamp(self.last_maintenance) if self.last_maintenance else None,
            'custom_tasks': len(self.cleanup_tasks)
        }
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check overall database health"""
        health_report = {
            'overall_status': 'healthy',
            'databases': {},
            'connection_pools': {},
            'issues': []
        }
        
        try:
            # Check each database
            db_files = self._get_database_files()
            for db_file in db_files:
                try:
                    db_info = self.connection_manager.get_database_info(db_file)
                    health_report['databases'][db_file] = {
                        'status': 'healthy',
                        'size_mb': db_info.get('size_mb', 0),
                        'table_count': db_info.get('table_count', 0),
                        'last_checked': datetime.now().isoformat()
                    }
                    
                    # Check for issues
                    if db_info.get('size_mb', 0) > 1000:  # > 1GB
                        health_report['issues'].append(f"{db_file}: Large database size ({db_info['size_mb']:.1f}MB)")
                    
                except Exception as e:
                    health_report['databases'][db_file] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    health_report['issues'].append(f"{db_file}: {str(e)}")
            
            # Check connection pools
            pool_stats = self.connection_manager.get_all_stats()
            for db_name, stats in pool_stats.get('pool_stats', {}).items():
                health_report['connection_pools'][db_name] = {
                    'active_connections': stats['active_connections'],
                    'hit_ratio': stats['hit_ratio'],
                    'status': 'healthy' if stats['hit_ratio'] > 0.8 else 'warning'
                }
                
                if stats['hit_ratio'] < 0.5:
                    health_report['issues'].append(f"{db_name}: Low connection pool hit ratio ({stats['hit_ratio']:.2f})")
            
            # Set overall status
            if health_report['issues']:
                health_report['overall_status'] = 'warning' if len(health_report['issues']) < 3 else 'critical'
            
            return health_report
            
        except Exception as e:
            logger.error(f"❌ Error checking database health: {e}")
            health_report['overall_status'] = 'error'
            health_report['error'] = str(e)
            return health_report
    
    def emergency_shutdown(self):
        """Emergency shutdown of all database operations"""
        logger.warning("🚨 EMERGENCY DATABASE SHUTDOWN INITIATED")
        
        try:
            # Stop cleanup thread
            self.running = False
            
            # Force close all connections
            self.connection_manager.close_all_pools()
            
            logger.warning("🚨 Emergency shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during emergency shutdown: {e}")
    
    def shutdown(self):
        """Graceful shutdown of cleanup manager"""
        if not self.running:
            return
        
        logger.info("🔄 Shutting down database cleanup manager...")
        
        try:
            # Stop background thread
            self.running = False
            
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=10)
                if self.cleanup_thread.is_alive():
                    logger.warning("⚠️ Cleanup thread did not stop gracefully")
            
            # Perform final cleanup
            self._perform_cleanup()
            
            # Close all connection pools
            self.connection_manager.close_all_pools()
            
            logger.info("✅ Database cleanup manager shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup manager shutdown: {e}")

class DatabaseResourceMonitor:
    """
    Monitor database resource usage and performance
    """
    
    def __init__(self, connection_manager: DatabaseConnectionManager = None):
        self.connection_manager = connection_manager or get_connection_manager()
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'connection_count': 50,
            'query_time': 5.0,  # seconds
            'error_rate': 0.1,  # 10%
            'database_size': 1000  # MB
        }
        
        self.alerts = []
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """Check current resource usage"""
        usage = {
            'timestamp': datetime.now().isoformat(),
            'connection_usage': {},
            'query_performance': {},
            'database_sizes': {},
            'alerts': []
        }
        
        try:
            # Get connection statistics
            stats = self.connection_manager.get_all_stats()
            
            # Check connection usage
            for db_name, pool_stats in stats.get('pool_stats', {}).items():
                usage['connection_usage'][db_name] = {
                    'active': pool_stats['active_connections'],
                    'hit_ratio': pool_stats['hit_ratio']
                }
                
                # Check alerts
                if pool_stats['active_connections'] > self.alert_thresholds['connection_count']:
                    alert = f"High connection count for {db_name}: {pool_stats['active_connections']}"
                    usage['alerts'].append(alert)
                    self.alerts.append({'timestamp': datetime.now(), 'message': alert})
            
            # Check query performance
            manager_stats = stats.get('manager_stats', {})
            usage['query_performance'] = {
                'total_queries': manager_stats.get('total_queries', 0),
                'average_time': manager_stats.get('average_query_time', 0),
                'error_rate': manager_stats.get('error_rate', 0)
            }
            
            # Check performance alerts
            if manager_stats.get('average_query_time', 0) > self.alert_thresholds['query_time']:
                alert = f"Slow average query time: {manager_stats['average_query_time']:.2f}s"
                usage['alerts'].append(alert)
            
            if manager_stats.get('error_rate', 0) > self.alert_thresholds['error_rate']:
                alert = f"High error rate: {manager_stats['error_rate']:.1%}"
                usage['alerts'].append(alert)
            
            return usage
            
        except Exception as e:
            logger.error(f"❌ Error checking resource usage: {e}")
            usage['error'] = str(e)
            return usage
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]

# Global cleanup manager instance
cleanup_manager = DatabaseCleanupManager()

def get_cleanup_manager() -> DatabaseCleanupManager:
    """Get the global cleanup manager instance"""
    return cleanup_manager

def start_database_cleanup():
    """Start background database cleanup"""
    cleanup_manager.start_background_cleanup()

def force_database_cleanup():
    """Force immediate database cleanup"""
    return cleanup_manager.force_cleanup()

def check_database_health():
    """Check overall database health"""
    return cleanup_manager.check_database_health()

# Auto-start cleanup on import (if not already started)
if not cleanup_manager.running:
    cleanup_manager.start_background_cleanup()