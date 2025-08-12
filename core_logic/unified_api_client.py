#!/usr/bin/env python3
"""
UNIFIED API CLIENT - Single, Robust Implementation
Consolidates api_client.py, smart_api_manager.py, and async_api_client.py
Provides both sync and async capabilities with intelligent caching and rate limiting
"""

import requests
import asyncio
import aiohttp
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import os
import threading
from collections import defaultdict

# Configuration
API_BASE_URL = "https://cricbuzz-cricket.p.rapidapi.com"

@dataclass
class APIConfig:
    """Centralized API configuration"""
    base_url: str = API_BASE_URL
    timeout: int = 10
    max_retries: int = 3
    rate_limit_per_minute: int = 8
    rate_limit_per_hour: int = 400
    rate_limit_per_day: int = 8000
    cache_ttl_seconds: int = 1800  # 30 minutes default
    
class RateLimiter:
    """Intelligent rate limiting for API calls"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.call_times = []
        self.daily_calls = 0
        self.hourly_calls = 0
        self.last_reset_day = datetime.now().date()
        self.last_reset_hour = datetime.now().hour
        self.lock = threading.Lock()
        
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting limits"""
        with self.lock:
            self._reset_counters_if_needed()
            
            # Check daily limit
            if self.daily_calls >= self.config.rate_limit_per_day:
                return False
                
            # Check hourly limit
            if self.hourly_calls >= self.config.rate_limit_per_hour:
                return False
                
            # Check per-minute limit
            now = time.time()
            recent_calls = [t for t in self.call_times if now - t < 60]
            if len(recent_calls) >= self.config.rate_limit_per_minute:
                return False
                
            return True
    
    def record_request(self):
        """Record a successful API request"""
        with self.lock:
            now = time.time()
            self.call_times.append(now)
            self.daily_calls += 1
            self.hourly_calls += 1
            
            # Keep only last minute of calls
            self.call_times = [t for t in self.call_times if now - t < 60]
    
    def get_wait_time(self) -> float:
        """Get seconds to wait before next request is allowed"""
        with self.lock:
            now = time.time()
            
            # Check minute limit
            recent_calls = [t for t in self.call_times if now - t < 60]
            if len(recent_calls) >= self.config.rate_limit_per_minute:
                oldest_call = min(recent_calls)
                return max(0, 60 - (now - oldest_call))
                
            return 0.0
    
    def _reset_counters_if_needed(self):
        """Reset hourly/daily counters when appropriate"""
        now = datetime.now()
        
        # Reset daily counter
        if now.date() > self.last_reset_day:
            self.daily_calls = 0
            self.last_reset_day = now.date()
            
        # Reset hourly counter
        if now.hour != self.last_reset_hour:
            self.hourly_calls = 0
            self.last_reset_hour = now.hour

class SmartCache:
    """Intelligent caching system with multiple strategies"""
    
    def __init__(self, cache_dir: str = "api_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.lock = threading.Lock()
        
        # Cache strategies by endpoint
        self.strategies = {
            'upcoming_matches': {'ttl': 1800, 'strategy': 'time_based'},  # 30 min
            'match_center': {'ttl': 900, 'strategy': 'time_based'},       # 15 min
            'recent_matches': {'ttl': 3600, 'strategy': 'time_based'},    # 1 hour
            'squads': {'ttl': 86400, 'strategy': 'time_based'},           # 24 hours
            'player_stats': {'ttl': 7200, 'strategy': 'time_based'},      # 2 hours
        }
    
    def get(self, key: str, endpoint: str = 'default') -> Optional[Any]:
        """Get cached data if valid"""
        with self.lock:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_valid(entry, endpoint):
                    return entry['data']
                else:
                    del self.memory_cache[key]
            
            # Try file cache
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                    
                    if self._is_valid(entry, endpoint):
                        # Load back to memory cache
                        self.memory_cache[key] = entry
                        return entry['data']
                    else:
                        cache_file.unlink()  # Remove expired file
                except Exception:
                    pass
                    
            return None
    
    def set(self, key: str, data: Any, endpoint: str = 'default'):
        """Cache data with appropriate strategy"""
        with self.lock:
            entry = {
                'data': data,
                'timestamp': time.time(),
                'endpoint': endpoint
            }
            
            # Store in memory
            self.memory_cache[key] = entry
            
            # Store in file for persistence
            cache_file = self._get_cache_file(key)
            try:
                with open(cache_file, 'w') as f:
                    json.dump(entry, f, default=str)
            except Exception:
                pass  # Don't fail if file cache fails
    
    def _is_valid(self, entry: Dict, endpoint: str) -> bool:
        """Check if cache entry is still valid"""
        strategy = self.strategies.get(endpoint, {'ttl': 1800})
        age = time.time() - entry['timestamp']
        return age < strategy['ttl']
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"
    
    def clear_expired(self):
        """Clear expired cache entries"""
        with self.lock:
            # Clear memory cache
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if not self._is_valid(entry, entry.get('endpoint', 'default')):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            # Clear file cache
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                    if not self._is_valid(entry, entry.get('endpoint', 'default')):
                        cache_file.unlink()
                except Exception:
                    cache_file.unlink()  # Remove corrupted files

class UnifiedAPIClient:
    """
    Single, comprehensive API client for all Dream11 AI needs
    Replaces api_client.py, smart_api_manager.py, and async_api_client.py
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.rate_limiter = RateLimiter(self.config)
        self.cache = SmartCache()
        
        # Load API key
        self._load_api_key()
        
        # Headers
        self.headers = {
            'x-rapidapi-host': 'cricbuzz-cricket.p.rapidapi.com',
            'User-Agent': 'Dream11AI/2.0'
        }
        
        if self.api_key:
            self.headers['x-rapidapi-key'] = self.api_key
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Async session (created when needed)
        self._async_session = None
    
    def _load_api_key(self):
        """Load API key from environment or .env file"""
        self.api_key = os.getenv('RAPIDAPI_KEY')
        
        if not self.api_key:
            # Try loading from .env file
            env_file = Path('.env')
            if env_file.exists():
                try:
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('RAPIDAPI_KEY='):
                                self.api_key = line.split('=', 1)[1].strip().strip('"\'')
                                break
                except Exception:
                    pass
    
    # SYNC METHODS
    def fetch_upcoming_matches(self) -> Dict[str, Any]:
        """Fetch upcoming matches with caching and rate limiting"""
        return self._make_request('upcoming_matches', '/matches/v1/upcoming')
    
    def fetch_match_center(self, match_id: Union[str, int]) -> Dict[str, Any]:
        """Fetch match center data"""
        endpoint = f'match_center_{match_id}'
        url = f'/match/{match_id}'
        return self._make_request(endpoint, url)
    
    def fetch_recent_matches(self) -> Dict[str, Any]:
        """Fetch recent/completed matches"""
        return self._make_request('recent_matches', '/matches/v1/recent')
    
    def fetch_player_stats(self, player_id: Union[str, int]) -> Dict[str, Any]:
        """Fetch player statistics"""
        endpoint = f'player_stats_{player_id}'
        url = f'/player/{player_id}'
        return self._make_request(endpoint, url)
    
    def fetch_squads(self, series_id: Union[str, int]) -> Dict[str, Any]:
        """Fetch team squads for a series"""
        endpoint = f'squads_{series_id}'
        url = f'/series/{series_id}/squads'
        return self._make_request(endpoint, url)
    
    # ASYNC METHODS
    async def fetch_upcoming_matches_async(self) -> Dict[str, Any]:
        """Async version of fetch_upcoming_matches"""
        return await self._make_request_async('upcoming_matches', '/matches/v1/upcoming')
    
    async def fetch_match_center_async(self, match_id: Union[str, int]) -> Dict[str, Any]:
        """Async version of fetch_match_center"""
        endpoint = f'match_center_{match_id}'
        url = f'/match/{match_id}'
        return await self._make_request_async(endpoint, url)
    
    # CORE REQUEST METHODS
    def _make_request(self, endpoint: str, url_path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a synchronous API request with all protections"""
        cache_key = f"{endpoint}_{hashlib.md5(str(params or {}).encode()).hexdigest()}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key, endpoint)
        if cached_data:
            return cached_data
        
        # Check if API key is available
        if not self.api_key:
            return self._get_fallback_data(endpoint)
        
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time > 0:
                print(f"⏰ Rate limited, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            if not self.rate_limiter.can_make_request():
                return self._get_fallback_data(endpoint)
        
        # Make the request
        try:
            url = f"{self.config.base_url}{url_path}"
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Record successful request
                self.rate_limiter.record_request()
                
                # Cache the response
                self.cache.set(cache_key, data, endpoint)
                
                return data
            else:
                print(f"⚠️ API returned status {response.status_code}")
                return self._get_fallback_data(endpoint)
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout for {endpoint}")
            return self._get_fallback_data(endpoint)
        except Exception as e:
            print(f"❌ Request failed for {endpoint}: {e}")
            return self._get_fallback_data(endpoint)
    
    async def _make_request_async(self, endpoint: str, url_path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an asynchronous API request"""
        cache_key = f"{endpoint}_{hashlib.md5(str(params or {}).encode()).hexdigest()}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key, endpoint)
        if cached_data:
            return cached_data
        
        if not self.api_key:
            return self._get_fallback_data(endpoint)
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            if not self.rate_limiter.can_make_request():
                return self._get_fallback_data(endpoint)
        
        # Create async session if needed
        if not self._async_session:
            self._async_session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        
        try:
            url = f"{self.config.base_url}{url_path}"
            async with self._async_session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Record successful request
                    self.rate_limiter.record_request()
                    
                    # Cache the response
                    self.cache.set(cache_key, data, endpoint)
                    
                    return data
                else:
                    return self._get_fallback_data(endpoint)
                    
        except Exception as e:
            print(f"❌ Async request failed for {endpoint}: {e}")
            return self._get_fallback_data(endpoint)
    
    def _get_fallback_data(self, endpoint: str) -> Dict[str, Any]:
        """Get fallback data when API calls fail"""
        fallback_data = {
            'upcoming_matches': {
                'typeMatches': [
                    {
                        'seriesMatches': [
                            {
                                'seriesAdWrapper': {
                                    'seriesName': 'Sample T20 Series',
                                    'matches': [
                                        {
                                            'matchInfo': {
                                                'matchId': 105780,
                                                'team1': {'teamName': 'Team A'},
                                                'team2': {'teamName': 'Team B'},
                                                'matchFormat': 'T20',
                                                'startDate': int(time.time() * 1000)
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
            'recent_matches': {'typeMatches': []},
            'match_center': {'status': 'fallback'},
            'player_stats': {'error': 'fallback_mode'},
            'squads': {'error': 'fallback_mode'}
        }
        
        for key in fallback_data:
            if endpoint.startswith(key):
                return fallback_data[key]
        
        return {'error': 'fallback_mode', 'endpoint': endpoint}
    
    # UTILITY METHODS
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        return {
            'daily_calls': self.rate_limiter.daily_calls,
            'hourly_calls': self.rate_limiter.hourly_calls,
            'minute_calls': len([t for t in self.rate_limiter.call_times if time.time() - t < 60]),
            'can_make_request': self.rate_limiter.can_make_request(),
            'wait_time': self.rate_limiter.get_wait_time(),
            'cache_entries': len(self.cache.memory_cache)
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.memory_cache.clear()
        for cache_file in self.cache.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def __del__(self):
        """Cleanup when client is destroyed"""
        if hasattr(self, 'session'):
            self.session.close()
        
        if hasattr(self, '_async_session') and self._async_session:
            # Note: In real async code, you should properly close this
            # This is just for cleanup in case of improper shutdown
            try:
                asyncio.create_task(self._async_session.close())
            except:
                pass

# Global instance for backward compatibility
_global_client = None

def get_global_client() -> UnifiedAPIClient:
    """Get or create global API client instance"""
    global _global_client
    if _global_client is None:
        _global_client = UnifiedAPIClient()
    return _global_client

# Backward compatibility functions
def fetch_upcoming_matches():
    """Backward compatibility wrapper"""
    return get_global_client().fetch_upcoming_matches()

def fetch_match_center(match_id):
    """Backward compatibility wrapper"""
    return get_global_client().fetch_match_center(match_id)

def fetch_recent_matches():
    """Backward compatibility wrapper"""
    return get_global_client().fetch_recent_matches()

def fetch_player_stats(player_id):
    """Backward compatibility wrapper"""
    return get_global_client().fetch_player_stats(player_id)

def fetch_squads(series_id):
    """Backward compatibility wrapper"""
    return get_global_client().fetch_squads(series_id)
