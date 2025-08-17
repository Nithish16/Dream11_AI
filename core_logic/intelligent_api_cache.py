#!/usr/bin/env python3
"""
Intelligent API Caching System
Advanced caching with smart expiration, request optimization, and cost tracking
"""

import sqlite3
import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import requests
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached API response"""
    cache_key: str
    endpoint: str
    response_data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    cost_saved: float = 0.0
    data_freshness: str = "fresh"  # fresh, stale, expired

@dataclass
class APICallMetrics:
    """Track API call performance and costs"""
    endpoint: str
    total_calls: int = 0
    cached_calls: int = 0
    api_calls: int = 0
    total_cost: float = 0.0
    cost_saved: float = 0.0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0

class IntelligentAPICache:
    """
    Advanced API caching system with smart expiration and cost optimization
    """
    
    def __init__(self, cache_db_path: str = "api_cache.db", base_path: str = None):
        self.cache_db_path = Path(base_path) / cache_db_path if base_path else Path(cache_db_path)
        self.base_path = Path(base_path) if base_path else Path('.')
        
        # Cache configuration
        self.default_cache_duration = 3600  # 1 hour
        self.max_cache_size = 10000  # Maximum cached entries
        self.cleanup_interval = 1800  # 30 minutes
        
        # Cost tracking (adjust per your API costs)
        self.api_costs = {
            'match_center': 0.01,      # $0.01 per call
            'upcoming_matches': 0.005,  # $0.005 per call
            'player_stats': 0.02,      # $0.02 per call
            'live_scores': 0.015,      # $0.015 per call
            'default': 0.01           # Default cost
        }
        
        # Cache strategies per endpoint
        self.cache_strategies = {
            'match_center': {
                'duration': 1800,      # 30 minutes
                'strategy': 'time_based',
                'priority': 'high'
            },
            'upcoming_matches': {
                'duration': 3600,      # 1 hour  
                'strategy': 'time_based',
                'priority': 'medium'
            },
            'player_stats': {
                'duration': 7200,      # 2 hours
                'strategy': 'smart_refresh',
                'priority': 'high'
            },
            'live_scores': {
                'duration': 60,        # 1 minute
                'strategy': 'time_based', 
                'priority': 'low'
            }
        }
        
        # Statistics
        self.metrics = {}
        self.total_cost_saved = 0.0
        self.total_api_calls = 0
        self.total_cached_calls = 0
        
        # Thread safety
        self._cache_lock = threading.RLock()
        
        # Initialize database
        self._init_cache_database()
        
        # Start background cleanup
        self._start_cleanup_thread()
    
    def _init_cache_database(self):
        """Initialize cache database with optimized schema"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Cache entries table with indices for performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                response_data TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cost_saved REAL DEFAULT 0.0,
                data_size INTEGER DEFAULT 0,
                freshness_score REAL DEFAULT 1.0
            )
        ''')
        
        # Index for efficient queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_endpoint ON cache_entries(endpoint)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_accessed ON cache_entries(last_accessed)')
        
        # API metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_metrics (
                endpoint TEXT PRIMARY KEY,
                total_calls INTEGER DEFAULT 0,
                cached_calls INTEGER DEFAULT 0,
                api_calls INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                cost_saved REAL DEFAULT 0.0,
                avg_response_time REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Cost tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_tracking (
                date TEXT PRIMARY KEY,
                total_api_calls INTEGER DEFAULT 0,
                total_cached_calls INTEGER DEFAULT 0,
                daily_cost REAL DEFAULT 0.0,
                daily_saved REAL DEFAULT 0.0,
                efficiency_ratio REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ API cache database initialized: {self.cache_db_path}")
    
    def _generate_cache_key(self, endpoint: str, params: Dict = None, headers: Dict = None) -> str:
        """Generate unique cache key for request"""
        # Create deterministic hash from endpoint and parameters
        cache_data = {
            'endpoint': endpoint,
            'params': params or {},
            'headers': {k: v for k, v in (headers or {}).items() if k.lower() not in ['authorization', 'user-agent']}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _get_cache_strategy(self, endpoint: str) -> Dict[str, Any]:
        """Get caching strategy for endpoint"""
        # Extract endpoint name from URL
        endpoint_name = self._extract_endpoint_name(endpoint)
        return self.cache_strategies.get(endpoint_name, {
            'duration': self.default_cache_duration,
            'strategy': 'time_based',
            'priority': 'medium'
        })
    
    def _extract_endpoint_name(self, endpoint: str) -> str:
        """Extract meaningful endpoint name from URL"""
        if 'match-center' in endpoint.lower():
            return 'match_center'
        elif 'upcoming' in endpoint.lower():
            return 'upcoming_matches'  
        elif 'player' in endpoint.lower():
            return 'player_stats'
        elif 'live' in endpoint.lower() or 'score' in endpoint.lower():
            return 'live_scores'
        else:
            return 'default'
    
    def get_cached_response(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Optional[Dict[str, Any]]:
        """Get cached response if available and fresh"""
        cache_key = self._generate_cache_key(endpoint, params, headers)
        
        with self._cache_lock:
            try:
                conn = sqlite3.connect(self.cache_db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT response_data, expires_at, access_count, cost_saved
                    FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > ?
                ''', (cache_key, datetime.now()))
                
                result = cursor.fetchone()
                
                if result:
                    response_data, expires_at, access_count, cost_saved = result
                    
                    # Update access statistics
                    cursor.execute('''
                        UPDATE cache_entries 
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE cache_key = ?
                    ''', (datetime.now(), cache_key))
                    
                    conn.commit()
                    conn.close()
                    
                    # Update metrics
                    self._update_cache_metrics(endpoint, cached=True, cost_saved=cost_saved)
                    
                    logger.info(f"🎯 Cache HIT: {self._extract_endpoint_name(endpoint)} (${cost_saved:.4f} saved)")
                    
                    return json.loads(response_data)
                
                conn.close()
                return None
                
            except Exception as e:
                logger.error(f"❌ Error retrieving from cache: {e}")
                return None
    
    def cache_response(self, endpoint: str, response_data: Dict[str, Any], 
                      params: Dict = None, headers: Dict = None) -> bool:
        """Cache API response with smart expiration"""
        cache_key = self._generate_cache_key(endpoint, params, headers)
        strategy = self._get_cache_strategy(endpoint)
        
        expires_at = datetime.now() + timedelta(seconds=strategy['duration'])
        data_size = len(json.dumps(response_data))
        
        # Calculate cost saved
        endpoint_name = self._extract_endpoint_name(endpoint)
        cost_saved = self.api_costs.get(endpoint_name, self.api_costs['default'])
        
        with self._cache_lock:
            try:
                conn = sqlite3.connect(self.cache_db_path)
                cursor = conn.cursor()
                
                # Check cache size limit
                cursor.execute('SELECT COUNT(*) FROM cache_entries')
                cache_size = cursor.fetchone()[0]
                
                if cache_size >= self.max_cache_size:
                    # Remove oldest entries
                    cursor.execute('''
                        DELETE FROM cache_entries 
                        WHERE cache_key IN (
                            SELECT cache_key FROM cache_entries 
                            ORDER BY last_accessed ASC 
                            LIMIT 100
                        )
                    ''')
                
                # Insert/update cache entry
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, endpoint, response_data, created_at, expires_at, 
                     cost_saved, data_size, freshness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (cache_key, endpoint, json.dumps(response_data), 
                      datetime.now(), expires_at, cost_saved, data_size, 1.0))
                
                conn.commit()
                conn.close()
                
                logger.info(f"💾 Cached: {endpoint_name} (expires in {strategy['duration']}s)")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error caching response: {e}")
                return False
    
    def make_cached_request(self, url: str, method: str = 'GET', params: Dict = None, 
                          headers: Dict = None, json_data: Dict = None, timeout: int = 30) -> Dict[str, Any]:
        """Make API request with intelligent caching"""
        
        # Check cache first
        cached_response = self.get_cached_response(url, params, headers)
        if cached_response:
            return cached_response
        
        # Make actual API request
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, params=params, headers=headers, json=json_data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            response_time = time.time() - start_time
            
            # Parse response
            if response.headers.get('content-type', '').startswith('application/json'):
                response_data = response.json()
            else:
                response_data = {'text': response.text, 'status_code': response.status_code}
            
            # Cache the response
            self.cache_response(url, response_data, params, headers)
            
            # Update metrics
            endpoint_name = self._extract_endpoint_name(url)
            cost = self.api_costs.get(endpoint_name, self.api_costs['default'])
            self._update_api_metrics(url, response_time, cost)
            
            logger.info(f"🌐 API CALL: {endpoint_name} ({response_time:.2f}s, ${cost:.4f})")
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            
            # Try to return stale cache if available
            return self._get_stale_cache(url, params, headers) or {'error': str(e)}
    
    def _get_stale_cache(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Optional[Dict[str, Any]]:
        """Get stale cache as fallback"""
        cache_key = self._generate_cache_key(endpoint, params, headers)
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT response_data FROM cache_entries 
                WHERE cache_key = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (cache_key,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                logger.warning(f"⚠️ Using STALE cache for {self._extract_endpoint_name(endpoint)}")
                return json.loads(result[0])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving stale cache: {e}")
            return None
    
    def _update_cache_metrics(self, endpoint: str, cached: bool, cost_saved: float = 0.0):
        """Update cache hit metrics"""
        endpoint_name = self._extract_endpoint_name(endpoint)
        
        if endpoint_name not in self.metrics:
            self.metrics[endpoint_name] = APICallMetrics(endpoint=endpoint_name)
        
        metrics = self.metrics[endpoint_name]
        metrics.total_calls += 1
        
        if cached:
            metrics.cached_calls += 1
            metrics.cost_saved += cost_saved
            self.total_cached_calls += 1
        else:
            metrics.api_calls += 1
            self.total_api_calls += 1
        
        metrics.cache_hit_rate = metrics.cached_calls / metrics.total_calls if metrics.total_calls > 0 else 0
    
    def _update_api_metrics(self, endpoint: str, response_time: float, cost: float):
        """Update API call metrics"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            endpoint_name = self._extract_endpoint_name(endpoint)
            
            cursor.execute('''
                INSERT OR REPLACE INTO api_metrics 
                (endpoint, total_calls, api_calls, total_cost, avg_response_time, last_updated)
                VALUES (?, 
                    COALESCE((SELECT total_calls FROM api_metrics WHERE endpoint = ?), 0) + 1,
                    COALESCE((SELECT api_calls FROM api_metrics WHERE endpoint = ?), 0) + 1,
                    COALESCE((SELECT total_cost FROM api_metrics WHERE endpoint = ?), 0) + ?,
                    ?,
                    ?
                )
            ''', (endpoint_name, endpoint_name, endpoint_name, endpoint_name, cost, response_time, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error updating API metrics: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Cache statistics
            cursor.execute('SELECT COUNT(*) FROM cache_entries')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM cache_entries WHERE expires_at > ?', (datetime.now(),))
            fresh_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(cost_saved) FROM cache_entries')
            total_cost_saved = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(data_size) FROM cache_entries')
            total_cache_size = cursor.fetchone()[0] or 0
            
            # API metrics
            cursor.execute('''
                SELECT endpoint, total_calls, cached_calls, api_calls, total_cost, cost_saved
                FROM api_metrics
            ''')
            api_metrics = cursor.fetchall()
            
            conn.close()
            
            # Calculate overall metrics
            total_calls = sum(row[1] for row in api_metrics)
            total_cached = sum(row[2] for row in api_metrics)
            total_api_calls = sum(row[3] for row in api_metrics)
            total_api_cost = sum(row[4] for row in api_metrics)
            
            cache_hit_rate = (total_cached / total_calls * 100) if total_calls > 0 else 0
            
            return {
                'cache_statistics': {
                    'total_entries': total_entries,
                    'fresh_entries': fresh_entries,
                    'expired_entries': total_entries - fresh_entries,
                    'cache_utilization': f"{(total_entries / self.max_cache_size * 100):.1f}%",
                    'total_size_mb': total_cache_size / 1024 / 1024
                },
                'performance_metrics': {
                    'total_requests': total_calls,
                    'cache_hits': total_cached,
                    'api_calls': total_api_calls,
                    'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                    'total_api_cost': f"${total_api_cost:.4f}",
                    'total_cost_saved': f"${total_cost_saved:.4f}",
                    'efficiency_ratio': f"{(total_cost_saved / total_api_cost * 100):.1f}%" if total_api_cost > 0 else "N/A"
                },
                'endpoint_breakdown': [
                    {
                        'endpoint': row[0],
                        'total_calls': row[1],
                        'cache_hit_rate': f"{(row[2] / row[1] * 100):.1f}%" if row[1] > 0 else "0%",
                        'api_cost': f"${row[4]:.4f}",
                        'cost_saved': f"${row[5]:.4f}"
                    }
                    for row in api_metrics
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cache statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM cache_entries WHERE expires_at <= ?', (datetime.now(),))
            expired_count = cursor.fetchone()[0]
            
            cursor.execute('DELETE FROM cache_entries WHERE expires_at <= ?', (datetime.now(),))
            
            conn.commit()
            conn.close()
            
            if expired_count > 0:
                logger.info(f"🧹 Cleaned up {expired_count} expired cache entries")
            
            return expired_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {e}")
            return 0
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_expired_cache()
                except Exception as e:
                    logger.error(f"❌ Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Started cache cleanup background thread")
    
    def preload_common_data(self, endpoints: List[str]):
        """Preload commonly used endpoints to warm cache"""
        logger.info(f"🔥 Preloading cache for {len(endpoints)} endpoints...")
        
        for endpoint in endpoints:
            try:
                self.make_cached_request(endpoint)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"❌ Error preloading {endpoint}: {e}")
        
        logger.info("✅ Cache preloading completed")

# Global cache instance
api_cache = IntelligentAPICache()

def get_api_cache() -> IntelligentAPICache:
    """Get global API cache instance"""
    return api_cache

def cached_api_call(url: str, method: str = 'GET', **kwargs) -> Dict[str, Any]:
    """Make cached API call using global cache"""
    return api_cache.make_cached_request(url, method, **kwargs)

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return api_cache.get_cache_statistics()