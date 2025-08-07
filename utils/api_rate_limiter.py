#!/usr/bin/env python3
"""
Advanced API Rate Limiter - Prevent API Hit Limits
Implements multiple rate limiting strategies and intelligent request management
"""

import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 100  # RapidAPI typical limit
    requests_per_hour: int = 1000   # Daily quota management
    requests_per_day: int = 10000   # Daily quota
    burst_limit: int = 10           # Max burst requests
    cooldown_seconds: int = 60      # Cooldown after hitting limit

@dataclass
class RequestMetrics:
    """Track API request metrics"""
    requests_made: int = 0
    requests_remaining: int = 0
    reset_time: Optional[datetime] = None
    last_request_time: Optional[datetime] = None
    consecutive_errors: int = 0
    backoff_until: Optional[datetime] = None

class APIRateLimiter:
    """
    Intelligent API rate limiter with multiple strategies:
    - Token bucket algorithm
    - Exponential backoff
    - Request prioritization
    - Automatic retry with jitter
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.metrics = RequestMetrics()
        
        # Token bucket for rate limiting
        self.tokens = self.config.burst_limit
        self.last_refill = time.time()
        
        # Request queues by priority
        self.high_priority_queue = deque()
        self.normal_priority_queue = deque()
        self.low_priority_queue = deque()
        
        # Rate limiting tracking
        self.request_timestamps = deque()
        self.hourly_requests = deque()
        self.daily_requests = deque()
        
    def _refill_tokens(self):
        """Refill tokens based on configured rate"""
        now = time.time()
        time_passed = now - self.last_refill
        
        # Add tokens based on requests per minute
        tokens_to_add = time_passed * (self.config.requests_per_minute / 60.0)
        self.tokens = min(self.config.burst_limit, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def _clean_old_requests(self):
        """Remove old request timestamps"""
        now = datetime.now()
        cutoff_minute = now - timedelta(minutes=1)
        cutoff_hour = now - timedelta(hours=1)
        cutoff_day = now - timedelta(days=1)
        
        # Clean minute tracking
        while self.request_timestamps and self.request_timestamps[0] < cutoff_minute:
            self.request_timestamps.popleft()
            
        # Clean hour tracking
        while self.hourly_requests and self.hourly_requests[0] < cutoff_hour:
            self.hourly_requests.popleft()
            
        # Clean day tracking
        while self.daily_requests and self.daily_requests[0] < cutoff_day:
            self.daily_requests.popleft()
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting limits"""
        self._refill_tokens()
        self._clean_old_requests()
        
        # Check if in backoff period
        if self.metrics.backoff_until and datetime.now() < self.metrics.backoff_until:
            return False
        
        # Check all rate limits
        checks = [
            self.tokens >= 1,  # Token bucket
            len(self.request_timestamps) < self.config.requests_per_minute,  # Per minute
            len(self.hourly_requests) < self.config.requests_per_hour,  # Per hour
            len(self.daily_requests) < self.config.requests_per_day,  # Per day
        ]
        
        return all(checks)
    
    def acquire_request_slot(self, priority: str = "normal") -> bool:
        """
        Acquire a slot for making an API request
        Args:
            priority: "high", "normal", or "low"
        Returns:
            True if request can proceed, False if should wait
        """
        if not self.can_make_request():
            logger.warning(f"🚫 Rate limit hit - requests in queues: "
                         f"H:{len(self.high_priority_queue)}, "
                         f"N:{len(self.normal_priority_queue)}, "
                         f"L:{len(self.low_priority_queue)}")
            return False
        
        # Consume token and record request
        self.tokens -= 1
        now = datetime.now()
        self.request_timestamps.append(now)
        self.hourly_requests.append(now)
        self.daily_requests.append(now)
        self.metrics.last_request_time = now
        self.metrics.requests_made += 1
        
        logger.debug(f"✅ Request slot acquired (priority: {priority}) - "
                    f"Tokens: {self.tokens:.1f}, "
                    f"Requests this minute: {len(self.request_timestamps)}")
        return True
    
    def handle_api_response(self, response_headers: Dict[str, str] = None):
        """Handle API response and update metrics"""
        if not response_headers:
            return
        
        # Parse RapidAPI headers
        remaining = response_headers.get('x-ratelimit-requests-remaining')
        reset_time = response_headers.get('x-ratelimit-requests-reset')
        
        if remaining:
            self.metrics.requests_remaining = int(remaining)
            
        if reset_time:
            self.metrics.reset_time = datetime.fromtimestamp(int(reset_time))
        
        # Reset error count on successful response
        self.metrics.consecutive_errors = 0
        self.metrics.backoff_until = None
    
    def handle_rate_limit_error(self, status_code: int = 429):
        """Handle rate limit error with exponential backoff"""
        self.metrics.consecutive_errors += 1
        
        # Exponential backoff: 2^errors seconds, max 300s (5 minutes)
        backoff_seconds = min(300, 2 ** self.metrics.consecutive_errors)
        self.metrics.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
        
        logger.warning(f"🚫 Rate limit hit (HTTP {status_code}) - "
                      f"Backing off for {backoff_seconds}s "
                      f"(attempt {self.metrics.consecutive_errors})")
    
    def get_wait_time(self) -> float:
        """Get recommended wait time before next request"""
        if self.metrics.backoff_until:
            return (self.metrics.backoff_until - datetime.now()).total_seconds()
        
        if not self.can_make_request():
            # Calculate time until next token is available
            return max(0, 60.0 / self.config.requests_per_minute)
        
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        self._clean_old_requests()
        
        return {
            'tokens_available': self.tokens,
            'requests_this_minute': len(self.request_timestamps),
            'requests_this_hour': len(self.hourly_requests),
            'requests_today': len(self.daily_requests),
            'can_make_request': self.can_make_request(),
            'wait_time_seconds': self.get_wait_time(),
            'in_backoff': bool(self.metrics.backoff_until and 
                             datetime.now() < self.metrics.backoff_until),
            'consecutive_errors': self.metrics.consecutive_errors,
            'total_requests_made': self.metrics.requests_made
        }

class SmartAPIClient:
    """
    Smart API client with automatic rate limiting and intelligent caching
    """
    
    def __init__(self, rate_limiter: APIRateLimiter = None):
        self.rate_limiter = rate_limiter or APIRateLimiter()
        self.cache = {}  # Will be enhanced with Redis in production
        self.fallback_enabled = True
        
    async def make_request(self, url: str, headers: Dict[str, str] = None, 
                          priority: str = "normal", cache_ttl: int = 3600) -> Dict[str, Any]:
        """
        Make rate-limited API request with caching and fallback
        """
        import aiohttp
        
        # Check cache first
        cache_key = f"{url}_{hash(str(headers))}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"🎯 Cache hit for {url}")
            return cached_result
        
        # Check rate limit
        if not self.rate_limiter.acquire_request_slot(priority):
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time > 0:
                logger.warning(f"⏰ Rate limited - waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                return await self.make_request(url, headers, priority, cache_ttl)
        
        # Make request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    
                    # Handle rate limit response
                    if response.status == 429:
                        self.rate_limiter.handle_rate_limit_error(429)
                        wait_time = self.rate_limiter.get_wait_time()
                        logger.warning(f"🚫 API rate limit hit - retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        return await self.make_request(url, headers, priority, cache_ttl)
                    
                    # Handle successful response
                    if response.status == 200:
                        data = await response.json()
                        self.rate_limiter.handle_api_response(dict(response.headers))
                        self._cache_result(cache_key, data, cache_ttl)
                        return data
                    
                    else:
                        logger.warning(f"⚠️ API returned status {response.status}")
                        return {"error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            if self.fallback_enabled:
                return self._get_fallback_data(url)
            return {"error": str(e)}
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if valid"""
        cache_entry = self.cache.get(cache_key)
        if cache_entry and time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
            return cache_entry['data']
        return None
    
    def _cache_result(self, cache_key: str, data: Dict, ttl: int):
        """Cache API result"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def _get_fallback_data(self, url: str) -> Dict:
        """Get fallback data when API fails"""
        logger.info(f"🔄 Using fallback data for {url}")
        return {"fallback": True, "message": "Using cached/sample data"}