#!/usr/bin/env python3
"""
Unified Database Connection Manager
Handles all database connections with pooling, caching, and proper cleanup
"""

import sqlite3
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import weakref
from collections import defaultdict
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    """Track connection usage metrics"""
    created_at: float
    last_used: float
    query_count: int = 0
    total_execution_time: float = 0.0
    error_count: int = 0

@dataclass
class PoolConfiguration:
    """Connection pool configuration"""
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    max_retries: int = 3
    retry_delay: float = 1.0

class DatabaseConnectionPool:
    """
    Connection pool for a single database
    """
    
    def __init__(self, db_path: str, config: PoolConfiguration):
        self.db_path = db_path
        self.config = config
        
        # Thread-safe connection pool
        self._pool: queue.Queue = queue.Queue(maxsize=config.max_connections)
        self._active_connections: Dict[int, sqlite3.Connection] = {}
        self._connection_metrics: Dict[int, ConnectionMetrics] = {}
        self._lock = threading.RLock()
        self._connection_id_counter = 0
        
        # Statistics
        self.total_connections_created = 0
        self.total_connections_closed = 0
        self.pool_hits = 0
        self.pool_misses = 0
        
        # Initialize minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections"""
        with self._lock:
            for _ in range(self.config.min_connections):
                try:
                    conn = self._create_connection()
                    self._pool.put(conn, block=False)
                except Exception as e:
                    logger.error(f"❌ Failed to initialize connection pool for {self.db_path}: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.config.connection_timeout,
                check_same_thread=False
            )
            
            # Configure connection
            conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
            conn.execute('PRAGMA synchronous=NORMAL')  # Balance safety/performance
            conn.execute('PRAGMA cache_size=10000')  # 10MB cache
            conn.execute('PRAGMA temp_store=memory')  # Use memory for temp tables
            conn.execute('PRAGMA mmap_size=268435456')  # 256MB memory mapped I/O
            
            # Enable foreign keys
            conn.execute('PRAGMA foreign_keys=ON')
            
            conn_id = self._get_next_connection_id()
            
            # Store weak reference to avoid circular references
            conn._pool_id = conn_id
            conn._pool_ref = weakref.ref(self)
            
            with self._lock:
                self._active_connections[conn_id] = conn
                self._connection_metrics[conn_id] = ConnectionMetrics(
                    created_at=time.time(),
                    last_used=time.time()
                )
                self.total_connections_created += 1
            
            logger.debug(f"🔗 Created new connection {conn_id} for {self.db_path}")
            return conn
            
        except Exception as e:
            logger.error(f"❌ Failed to create connection to {self.db_path}: {e}")
            raise
    
    def _get_next_connection_id(self) -> int:
        """Get next connection ID"""
        with self._lock:
            self._connection_id_counter += 1
            return self._connection_id_counter
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        try:
            # Try to get from pool first
            try:
                conn = self._pool.get(block=False)
                self.pool_hits += 1
                
                # Update last used time
                if hasattr(conn, '_pool_id') and conn._pool_id in self._connection_metrics:
                    self._connection_metrics[conn._pool_id].last_used = time.time()
                
                logger.debug(f"♻️ Reused connection from pool for {self.db_path}")
                return conn
                
            except queue.Empty:
                # Pool is empty, create new connection if under limit
                with self._lock:
                    if len(self._active_connections) < self.config.max_connections:
                        self.pool_misses += 1
                        return self._create_connection()
                    else:
                        # Wait for a connection to become available
                        logger.warning(f"⏳ Connection pool full for {self.db_path}, waiting...")
                        conn = self._pool.get(timeout=self.config.connection_timeout)
                        self.pool_hits += 1
                        return conn
                        
        except Exception as e:
            logger.error(f"❌ Failed to get connection for {self.db_path}: {e}")
            raise
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        try:
            if not hasattr(conn, '_pool_id'):
                logger.warning("⚠️ Attempting to return connection without pool ID")
                self._close_connection(conn)
                return
            
            conn_id = conn._pool_id
            
            # Check if connection is still valid
            if not self._is_connection_valid(conn):
                logger.warning(f"⚠️ Connection {conn_id} is invalid, closing")
                self._close_connection(conn)
                return
            
            # Check if connection has been idle too long
            with self._lock:
                if conn_id in self._connection_metrics:
                    metrics = self._connection_metrics[conn_id]
                    if time.time() - metrics.last_used > self.config.idle_timeout:
                        logger.debug(f"⏰ Connection {conn_id} idle timeout, closing")
                        self._close_connection(conn)
                        return
            
            # Rollback any uncommitted transactions
            try:
                conn.rollback()
            except Exception:
                pass
            
            # Return to pool
            try:
                self._pool.put(conn, block=False)
                logger.debug(f"↩️ Returned connection {conn_id} to pool")
            except queue.Full:
                # Pool is full, close the connection
                logger.debug(f"🗑️ Pool full, closing connection {conn_id}")
                self._close_connection(conn)
                
        except Exception as e:
            logger.error(f"❌ Error returning connection: {e}")
            self._close_connection(conn)
    
    def _is_connection_valid(self, conn: sqlite3.Connection) -> bool:
        """Check if connection is still valid"""
        try:
            conn.execute('SELECT 1').fetchone()
            return True
        except Exception:
            return False
    
    def _close_connection(self, conn: sqlite3.Connection):
        """Close a connection and clean up"""
        try:
            if hasattr(conn, '_pool_id'):
                conn_id = conn._pool_id
                with self._lock:
                    if conn_id in self._active_connections:
                        del self._active_connections[conn_id]
                    if conn_id in self._connection_metrics:
                        del self._connection_metrics[conn_id]
                    self.total_connections_closed += 1
                
                logger.debug(f"🔌 Closed connection {conn_id}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error closing connection: {e}")
    
    def cleanup_idle_connections(self):
        """Clean up idle connections"""
        current_time = time.time()
        idle_connections = []
        
        with self._lock:
            for conn_id, metrics in self._connection_metrics.items():
                if current_time - metrics.last_used > self.config.idle_timeout:
                    if conn_id in self._active_connections:
                        idle_connections.append(self._active_connections[conn_id])
        
        for conn in idle_connections:
            try:
                self._close_connection(conn)
            except Exception as e:
                logger.error(f"❌ Error cleaning up idle connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._lock:
            return {
                'db_path': self.db_path,
                'active_connections': len(self._active_connections),
                'pool_size': self._pool.qsize(),
                'total_created': self.total_connections_created,
                'total_closed': self.total_connections_closed,
                'pool_hits': self.pool_hits,
                'pool_misses': self.pool_misses,
                'hit_ratio': self.pool_hits / (self.pool_hits + self.pool_misses) if (self.pool_hits + self.pool_misses) > 0 else 0
            }
    
    def close_all(self):
        """Close all connections in the pool"""
        logger.info(f"🔒 Closing all connections for {self.db_path}")
        
        # Close pooled connections
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                self._close_connection(conn)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"❌ Error closing pooled connection: {e}")
        
        # Close active connections
        with self._lock:
            active_conns = list(self._active_connections.values())
            
        for conn in active_conns:
            try:
                self._close_connection(conn)
            except Exception as e:
                logger.error(f"❌ Error closing active connection: {e}")

class DatabaseConnectionManager:
    """
    Unified database connection manager with pooling
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path('.')
        
        # Connection pools for each database
        self._pools: Dict[str, DatabaseConnectionPool] = {}
        self._lock = threading.RLock()
        
        # Default configuration
        self.default_config = PoolConfiguration()
        
        # Database-specific configurations
        self.db_configs = {
            'universal_cricket_intelligence.db': PoolConfiguration(max_connections=15, min_connections=3),
            'ai_learning_database.db': PoolConfiguration(max_connections=10, min_connections=2),
            'format_specific_learning.db': PoolConfiguration(max_connections=8, min_connections=2),
            'smart_local_predictions.db': PoolConfiguration(max_connections=8, min_connections=2),
            'optimized_predictions.db': PoolConfiguration(max_connections=6, min_connections=1),
            'api_usage_tracking.db': PoolConfiguration(max_connections=5, min_connections=1),
            'dream11_unified.db': PoolConfiguration(max_connections=8, min_connections=2)
        }
        
        # Statistics
        self.query_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _get_pool(self, db_name: str) -> DatabaseConnectionPool:
        """Get or create connection pool for database"""
        with self._lock:
            if db_name not in self._pools:
                db_path = self.base_path / db_name
                config = self.db_configs.get(db_name, self.default_config)
                
                self._pools[db_name] = DatabaseConnectionPool(str(db_path), config)
                logger.info(f"🏊 Created connection pool for {db_name}")
            
            return self._pools[db_name]
    
    @contextmanager
    def get_connection(self, db_name: str) -> ContextManager[sqlite3.Connection]:
        """Get a managed database connection with automatic cleanup"""
        pool = self._get_pool(db_name)
        conn = None
        
        try:
            conn = pool.get_connection()
            yield conn
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Database error for {db_name}: {e}")
            raise
            
        finally:
            if conn:
                pool.return_connection(conn)
    
    def execute_query(self, db_name: str, query: str, params: tuple = None, 
                     fetch_method: str = 'fetchall') -> Any:
        """Execute a query with connection management"""
        start_time = time.time()
        
        try:
            with self.get_connection(db_name) as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Handle different fetch methods
                if fetch_method == 'fetchone':
                    result = cursor.fetchone()
                elif fetch_method == 'fetchall':
                    result = cursor.fetchall()
                elif fetch_method == 'fetchmany':
                    result = cursor.fetchmany(100)  # Default size
                elif fetch_method == 'none':
                    result = cursor.rowcount
                    conn.commit()
                else:
                    result = cursor.fetchall()
                
                self.query_count += 1
                execution_time = time.time() - start_time
                self.total_execution_time += execution_time
                
                logger.debug(f"📊 Query executed in {execution_time:.3f}s: {query[:100]}")
                return result
                
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            logger.error(f"❌ Query failed in {execution_time:.3f}s: {query[:100]} - {e}")
            raise
    
    def execute_transaction(self, db_name: str, operations: List[Callable[[sqlite3.Connection], Any]]) -> List[Any]:
        """Execute multiple operations in a transaction"""
        with self.get_connection(db_name) as conn:
            try:
                conn.execute('BEGIN TRANSACTION')
                results = []
                
                for operation in operations:
                    result = operation(conn)
                    results.append(result)
                
                conn.commit()
                return results
                
            except Exception as e:
                conn.rollback()
                logger.error(f"❌ Transaction failed for {db_name}: {e}")
                raise
    
    def get_database_info(self, db_name: str) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            with self.get_connection(db_name) as conn:
                cursor = conn.cursor()
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # Get table count
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                # Get index count
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
                index_count = cursor.fetchone()[0]
                
                return {
                    'database': db_name,
                    'size_bytes': db_size,
                    'size_mb': db_size / (1024 * 1024),
                    'table_count': table_count,
                    'index_count': index_count
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting database info for {db_name}: {e}")
            return {'database': db_name, 'error': str(e)}
    
    def optimize_database(self, db_name: str) -> bool:
        """Optimize database by running VACUUM and ANALYZE"""
        try:
            logger.info(f"🔧 Optimizing database: {db_name}")
            
            with self.get_connection(db_name) as conn:
                # Analyze tables for query optimizer
                conn.execute('ANALYZE')
                
                # Vacuum to reclaim space and defragment
                conn.execute('VACUUM')
                
                logger.info(f"✅ Optimized database: {db_name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error optimizing database {db_name}: {e}")
            return False
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all connection pools"""
        stats = {
            'manager_stats': {
                'total_queries': self.query_count,
                'total_errors': self.error_count,
                'total_execution_time': self.total_execution_time,
                'average_query_time': self.total_execution_time / self.query_count if self.query_count > 0 else 0,
                'error_rate': self.error_count / self.query_count if self.query_count > 0 else 0
            },
            'pool_stats': {}
        }
        
        with self._lock:
            for db_name, pool in self._pools.items():
                stats['pool_stats'][db_name] = pool.get_stats()
        
        return stats
    
    def _start_cleanup_thread(self):
        """Start background thread for cleanup tasks"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_idle_connections()
                except Exception as e:
                    logger.error(f"❌ Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Started database cleanup thread")
    
    def _cleanup_idle_connections(self):
        """Clean up idle connections across all pools"""
        with self._lock:
            for pool in self._pools.values():
                pool.cleanup_idle_connections()
    
    def close_all_pools(self):
        """Close all connection pools"""
        logger.info("🔒 Closing all database connection pools")
        
        with self._lock:
            for pool in self._pools.values():
                pool.close_all()
            self._pools.clear()
        
        logger.info("✅ All connection pools closed")

# Global connection manager instance
connection_manager = DatabaseConnectionManager()

def get_connection_manager() -> DatabaseConnectionManager:
    """Get the global connection manager instance"""
    return connection_manager

def get_database_connection(db_name: str):
    """Quick function to get a managed database connection"""
    return connection_manager.get_connection(db_name)

def execute_query(db_name: str, query: str, params: tuple = None, fetch_method: str = 'fetchall'):
    """Quick function to execute a query"""
    return connection_manager.execute_query(db_name, query, params, fetch_method)