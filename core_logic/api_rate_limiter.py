#!/usr/bin/env python3
"""
Advanced API Rate Limiting and Request Optimization
Smart rate limiting, request batching, and API quota management
"""

import time
import threading
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration for an API endpoint"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = None  # Maximum burst requests
    backoff_multiplier: float = 1.5
    max_backoff: float = 300.0  # Maximum backoff time in seconds

@dataclass
class APIQuota:
    """API quota tracking"""
    service_name: str
    quota_type: str  # daily, monthly, per_request
    quota_limit: int
    quota_used: int = 0
    reset_time: datetime = None
    cost_per_request: float = 0.0
    priority: int = 1  # 1=highest, 5=lowest

@dataclass
class RequestMetrics:
    """Track request performance metrics"""
    endpoint: str
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_success: datetime = None
    last_error: datetime = None
    consecutive_errors: int = 0
    backoff_until: datetime = None

class SmartRateLimiter:
    """
    Intelligent rate limiter with adaptive backoff and quota management
    """
    
    def __init__(self, db_path: str = "rate_limiter.db"):
        self.db_path = db_path
        self.request_windows = defaultdict(deque)  # Track request timestamps
        self.metrics = defaultdict(RequestMetrics)
        self.quotas = {}
        
        # Default rate limits for common APIs
        self.rate_limits = {
            'cricbuzz': RateLimit(60, 1000, 10000, burst_limit=10),
            'espncricinfo': RateLimit(30, 500, 5000, burst_limit=5),
            'default': RateLimit(30, 300, 1000, burst_limit=5)
        }
        
        # Optimization settings
        self.batch_size = 5  # Maximum requests to batch
        self.batch_delay = 0.1  # Delay between batched requests
        self.adaptive_backoff = True
        self.priority_queue_enabled = True
        
        # Thread safety
        self._lock = threading.RLock()
        self._quota_lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Load quotas from database
        self._load_quotas()
    
    def _init_database(self):
        """Initialize rate limiter database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Request tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS request_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                response_time REAL,
                status_code INTEGER,
                error_message TEXT,
                quota_used INTEGER DEFAULT 0
            )
        ''')
        
        # API quotas table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_quotas (
                service_name TEXT NOT NULL,
                quota_type TEXT NOT NULL,
                quota_limit INTEGER NOT NULL,
                quota_used INTEGER DEFAULT 0,
                reset_time TIMESTAMP,
                cost_per_request REAL DEFAULT 0.0,
                priority INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (service_name, quota_type)
            )
        ''')
        
        # Rate limit violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limit_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                backoff_time REAL,
                quota_exceeded BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Daily cost tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_costs (
                date TEXT PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                quota_utilization REAL DEFAULT 0.0,
                efficiency_score REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Rate limiter database initialized: {self.db_path}")
    
    def _load_quotas(self):
        """Load API quotas from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM api_quotas')
            quotas = cursor.fetchall()
            
            for quota in quotas:
                service_name, quota_type, quota_limit, quota_used, reset_time, cost_per_request, priority, _ = quota
                
                quota_key = f"{service_name}_{quota_type}"
                self.quotas[quota_key] = APIQuota(
                    service_name=service_name,
                    quota_type=quota_type,
                    quota_limit=quota_limit,
                    quota_used=quota_used,
                    reset_time=datetime.fromisoformat(reset_time) if reset_time else None,
                    cost_per_request=cost_per_request,
                    priority=priority
                )
            
            conn.close()
            logger.info(f"📊 Loaded {len(self.quotas)} API quotas")
            
        except Exception as e:
            logger.error(f"❌ Error loading quotas: {e}")
    
    def add_quota(self, service_name: str, quota_type: str, quota_limit: int, 
                  cost_per_request: float = 0.0, priority: int = 1):
        """Add or update API quota"""
        quota_key = f"{service_name}_{quota_type}"
        
        with self._quota_lock:
            # Reset time based on quota type
            if quota_type == 'daily':
                reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            elif quota_type == 'monthly':
                next_month = datetime.now().replace(day=1) + timedelta(days=32)
                reset_time = next_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                reset_time = None
            
            self.quotas[quota_key] = APIQuota(
                service_name=service_name,
                quota_type=quota_type,
                quota_limit=quota_limit,
                quota_used=0,
                reset_time=reset_time,
                cost_per_request=cost_per_request,
                priority=priority
            )
            
            # Save to database
            self._save_quota(quota_key)
            
            logger.info(f"📊 Added quota: {service_name} ({quota_type}) - {quota_limit} requests")
    
    def _save_quota(self, quota_key: str):
        """Save quota to database"""
        try:
            quota = self.quotas[quota_key]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO api_quotas 
                (service_name, quota_type, quota_limit, quota_used, reset_time, 
                 cost_per_request, priority, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (quota.service_name, quota.quota_type, quota.quota_limit, quota.quota_used,
                  quota.reset_time.isoformat() if quota.reset_time else None,
                  quota.cost_per_request, quota.priority, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving quota: {e}")
    
    def _get_service_name(self, endpoint: str) -> str:
        """Extract service name from endpoint"""
        if 'cricbuzz' in endpoint.lower():
            return 'cricbuzz'
        elif 'espncricinfo' in endpoint.lower() or 'espn' in endpoint.lower():
            return 'espncricinfo'
        else:
            return 'default'
    
    def _get_rate_limit(self, service_name: str) -> RateLimit:
        """Get rate limit configuration for service"""
        return self.rate_limits.get(service_name, self.rate_limits['default'])
    
    def can_make_request(self, endpoint: str, check_quota: bool = True) -> Tuple[bool, str, float]:
        """
        Check if request can be made considering rate limits and quotas
        
        Returns:
            (can_make_request, reason, wait_time_seconds)
        """
        service_name = self._get_service_name(endpoint)
        rate_limit = self._get_rate_limit(service_name)
        
        with self._lock:
            # Check if in backoff period
            if endpoint in self.metrics:
                metrics = self.metrics[endpoint]
                if metrics.backoff_until and datetime.now() < metrics.backoff_until:
                    wait_time = (metrics.backoff_until - datetime.now()).total_seconds()
                    return False, f"Backoff period active", wait_time
            
            # Check rate limits
            now = datetime.now()
            window = self.request_windows[service_name]
            
            # Clean old requests from window
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            # Remove old requests
            while window and window[0] < day_ago:
                window.popleft()
            
            # Count requests in different windows
            minute_count = sum(1 for req_time in window if req_time >= minute_ago)
            hour_count = sum(1 for req_time in window if req_time >= hour_ago)
            day_count = len(window)
            
            # Check rate limits
            if minute_count >= rate_limit.requests_per_minute:
                return False, "Per-minute rate limit exceeded", 60 - (now - minute_ago).total_seconds()
            
            if hour_count >= rate_limit.requests_per_hour:
                return False, "Per-hour rate limit exceeded", 3600 - (now - hour_ago).total_seconds()
            
            if day_count >= rate_limit.requests_per_day:
                return False, "Per-day rate limit exceeded", 86400 - (now - day_ago).total_seconds()
            
            # Check burst limit
            if rate_limit.burst_limit:
                last_5_seconds = now - timedelta(seconds=5)
                burst_count = sum(1 for req_time in window if req_time >= last_5_seconds)
                if burst_count >= rate_limit.burst_limit:
                    return False, "Burst limit exceeded", 5
            
            # Check quotas
            if check_quota:
                for quota_key, quota in self.quotas.items():
                    if service_name in quota.service_name:
                        # Check if quota needs reset
                        if quota.reset_time and now >= quota.reset_time:
                            quota.quota_used = 0
                            if quota.quota_type == 'daily':
                                quota.reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                            self._save_quota(quota_key)
                        
                        if quota.quota_used >= quota.quota_limit:
                            wait_time = (quota.reset_time - now).total_seconds() if quota.reset_time else 3600
                            return False, f"Quota exceeded for {quota.quota_type}", wait_time
            
            return True, "OK", 0
    
    def record_request(self, endpoint: str, response_time: float, status_code: int, 
                      error_message: str = None):
        """Record request for rate limiting and metrics"""
        service_name = self._get_service_name(endpoint)
        now = datetime.now()
        
        with self._lock:
            # Add to request window
            self.request_windows[service_name].append(now)
            
            # Update metrics
            if endpoint not in self.metrics:
                self.metrics[endpoint] = RequestMetrics(endpoint=endpoint)
            
            metrics = self.metrics[endpoint]
            
            if status_code < 400:
                metrics.success_count += 1
                metrics.last_success = now
                metrics.consecutive_errors = 0
                metrics.backoff_until = None
                
                # Update average response time
                total_requests = metrics.success_count + metrics.error_count
                metrics.avg_response_time = ((metrics.avg_response_time * (total_requests - 1)) + response_time) / total_requests
                
            else:
                metrics.error_count += 1
                metrics.last_error = now
                metrics.consecutive_errors += 1
                
                # Apply adaptive backoff
                if self.adaptive_backoff and metrics.consecutive_errors >= 3:
                    rate_limit = self._get_rate_limit(service_name)
                    backoff_time = min(
                        rate_limit.backoff_multiplier ** metrics.consecutive_errors,
                        rate_limit.max_backoff
                    )
                    metrics.backoff_until = now + timedelta(seconds=backoff_time)
                    
                    logger.warning(f"⏰ Applied backoff to {endpoint}: {backoff_time:.1f}s")
                    
                    # Record violation
                    self._record_violation(endpoint, "consecutive_errors", backoff_time)
            
            # Update quota usage
            self._update_quota_usage(service_name)
            
            # Record in database
            self._record_request_db(endpoint, now, response_time, status_code, error_message)
    
    def _update_quota_usage(self, service_name: str):
        """Update quota usage for service"""
        with self._quota_lock:
            for quota_key, quota in self.quotas.items():
                if service_name in quota.service_name:
                    quota.quota_used += 1
                    self._save_quota(quota_key)
    
    def _record_request_db(self, endpoint: str, timestamp: datetime, response_time: float,
                          status_code: int, error_message: str = None):
        """Record request in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO request_tracking 
                (endpoint, timestamp, response_time, status_code, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (endpoint, timestamp, response_time, status_code, error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error recording request: {e}")
    
    def _record_violation(self, endpoint: str, violation_type: str, backoff_time: float = None):
        """Record rate limit violation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rate_limit_violations 
                (endpoint, violation_type, timestamp, backoff_time)
                VALUES (?, ?, ?, ?)
            ''', (endpoint, violation_type, datetime.now(), backoff_time))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error recording violation: {e}")
    
    def get_request_delay(self, endpoint: str) -> float:
        """Get recommended delay before next request"""
        service_name = self._get_service_name(endpoint)
        rate_limit = self._get_rate_limit(service_name)
        
        with self._lock:
            window = self.request_windows[service_name]
            if not window:
                return 0.1  # Minimum delay
            
            # Calculate optimal delay based on rate limits
            now = datetime.now()
            recent_requests = [req_time for req_time in window if req_time >= now - timedelta(minutes=1)]
            
            if len(recent_requests) >= rate_limit.requests_per_minute * 0.8:  # 80% of limit
                # Slow down significantly
                return 60.0 / rate_limit.requests_per_minute * 2
            elif len(recent_requests) >= rate_limit.requests_per_minute * 0.6:  # 60% of limit
                # Moderate slowdown
                return 60.0 / rate_limit.requests_per_minute * 1.5
            else:
                # Normal rate
                return 60.0 / rate_limit.requests_per_minute
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Request statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time) as avg_response_time,
                    COUNT(CASE WHEN status_code < 400 THEN 1 END) as successful_requests,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as failed_requests
                FROM request_tracking 
                WHERE timestamp >= datetime('now', '-24 hours')
            ''')
            
            request_stats = cursor.fetchone()
            
            # Quota utilization
            quota_stats = []
            for quota_key, quota in self.quotas.items():
                utilization = (quota.quota_used / quota.quota_limit * 100) if quota.quota_limit > 0 else 0
                quota_stats.append({
                    'service': quota.service_name,
                    'type': quota.quota_type,
                    'used': quota.quota_used,
                    'limit': quota.quota_limit,
                    'utilization': f"{utilization:.1f}%",
                    'cost': f"${quota.quota_used * quota.cost_per_request:.4f}"
                })
            
            # Recent violations
            cursor.execute('''
                SELECT endpoint, violation_type, COUNT(*) as count
                FROM rate_limit_violations 
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY endpoint, violation_type
            ''')
            violations = cursor.fetchall()
            
            conn.close()
            
            total_requests, avg_response_time, successful_requests, failed_requests = request_stats
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'request_statistics': {
                    'total_requests_24h': total_requests or 0,
                    'successful_requests': successful_requests or 0,
                    'failed_requests': failed_requests or 0,
                    'success_rate': f"{success_rate:.1f}%",
                    'avg_response_time': f"{avg_response_time or 0:.3f}s"
                },
                'quota_utilization': quota_stats,
                'rate_limit_violations': [
                    {'endpoint': v[0], 'type': v[1], 'count': v[2]} 
                    for v in violations
                ],
                'current_metrics': {
                    endpoint: {
                        'success_count': metrics.success_count,
                        'error_count': metrics.error_count,
                        'consecutive_errors': metrics.consecutive_errors,
                        'avg_response_time': f"{metrics.avg_response_time:.3f}s",
                        'in_backoff': metrics.backoff_until > datetime.now() if metrics.backoff_until else False
                    }
                    for endpoint, metrics in self.metrics.items()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {'error': str(e)}
    
    def optimize_request_timing(self, endpoints: List[str]) -> List[Tuple[str, float]]:
        """Optimize timing for multiple requests"""
        optimized_schedule = []
        current_time = 0.0
        
        # Sort endpoints by priority (based on quotas and error rates)
        def get_priority_score(endpoint: str) -> float:
            service_name = self._get_service_name(endpoint)
            
            # Check quotas for priority
            priority_score = 1.0
            for quota in self.quotas.values():
                if service_name in quota.service_name:
                    priority_score = quota.priority
                    break
            
            # Adjust for error rates
            if endpoint in self.metrics:
                metrics = self.metrics[endpoint]
                total_requests = metrics.success_count + metrics.error_count
                if total_requests > 0:
                    error_rate = metrics.error_count / total_requests
                    priority_score += error_rate * 2  # Higher score = lower priority
            
            return priority_score
        
        sorted_endpoints = sorted(endpoints, key=get_priority_score)
        
        for endpoint in sorted_endpoints:
            can_make, reason, wait_time = self.can_make_request(endpoint, check_quota=True)
            
            if not can_make:
                current_time += wait_time
            
            delay = self.get_request_delay(endpoint)
            optimized_schedule.append((endpoint, current_time))
            current_time += delay
        
        return optimized_schedule
    
    def reset_backoff(self, endpoint: str):
        """Manually reset backoff for endpoint"""
        if endpoint in self.metrics:
            self.metrics[endpoint].backoff_until = None
            self.metrics[endpoint].consecutive_errors = 0
            logger.info(f"🔄 Reset backoff for {endpoint}")

# Global rate limiter instance
rate_limiter = SmartRateLimiter()

def get_rate_limiter() -> SmartRateLimiter:
    """Get global rate limiter instance"""
    return rate_limiter

def can_make_api_request(endpoint: str) -> Tuple[bool, str, float]:
    """Check if API request can be made"""
    return rate_limiter.can_make_request(endpoint)

def record_api_request(endpoint: str, response_time: float, status_code: int, error: str = None):
    """Record API request for rate limiting"""
    rate_limiter.record_request(endpoint, response_time, status_code, error)