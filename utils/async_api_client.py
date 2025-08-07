#!/usr/bin/env python3
"""
Enhanced Async API Client - High-Performance API Layer
Provides async/await support while maintaining backward compatibility
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from .api_client import (
    API_BASE_URL, API_HEADERS, 
    _get_sample_matches_data, _get_sample_squads_data, 
    _get_sample_player_stats, _get_sample_venue_stats
)

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncAPIClient:
    """
    High-performance async API client with connection pooling,
    caching, and intelligent fallback to sync client
    """
    
    def __init__(self, cache_ttl: int = 3600, max_connections: int = 100):
        self.cache = {}  # Simple in-memory cache (will upgrade to Redis)
        self.cache_ttl = cache_ttl
        self.session = None
        self.max_connections = max_connections
        self.fallback_enabled = True
        
        # Performance metrics
        self.metrics = {
            'async_calls': 0,
            'cache_hits': 0,
            'fallback_calls': 0,
            'total_time_saved': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
    
    async def initialize_session(self):
        """Initialize aiohttp session with connection pooling"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=30,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=API_HEADERS
            )
            logger.info("🚀 Async API session initialized with connection pooling")
    
    async def close_session(self):
        """Clean up session resources"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("✅ Async API session closed")
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for endpoint and parameters"""
        if params:
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{endpoint}_{param_str}"
        return endpoint
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    def _cache_result(self, key: str, data: Any):
        """Cache API result with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        cache_entry = self.cache.get(key)
        if self._is_cache_valid(cache_entry):
            self.metrics['cache_hits'] += 1
            logger.debug(f"🎯 Cache hit for {key}")
            return cache_entry['data']
        return None
    
    async def _make_async_request(self, url: str, params: Dict = None) -> Dict:
        """Make async HTTP request with error handling"""
        start_time = time.time()
        
        try:
            if not self.session:
                await self.initialize_session()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.metrics['async_calls'] += 1
                    self.metrics['total_time_saved'] += time.time() - start_time
                    logger.debug(f"✅ Async request successful: {url}")
                    return data
                else:
                    logger.warning(f"⚠️ API returned status {response.status} for {url}")
                    return {"error": f"HTTP {response.status}"}
        
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout for {url}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"❌ Async request failed for {url}: {e}")
            return {"error": str(e)}
    
    async def fetch_upcoming_matches(self) -> Dict:
        """Fetch upcoming matches with caching and fallback"""
        cache_key = self._get_cache_key("upcoming_matches")
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Make async request
        url = f"{API_BASE_URL}/matches/v1/upcoming"
        result = await self._make_async_request(url)
        
        # Fallback to sync method if async fails
        if result.get('error') and self.fallback_enabled:
            logger.info("🔄 Falling back to sync API for upcoming matches")
            from .api_client import fetch_upcoming_matches
            result = fetch_upcoming_matches()
            self.metrics['fallback_calls'] += 1
        
        # Cache successful results
        if not result.get('error'):
            self._cache_result(cache_key, result)
        
        return result
    
    async def fetch_recent_matches(self) -> Dict:
        """Fetch recent matches with caching and fallback"""
        cache_key = self._get_cache_key("recent_matches")
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        url = f"{API_BASE_URL}/matches/v1/recent"
        result = await self._make_async_request(url)
        
        if result.get('error') and self.fallback_enabled:
            logger.info("🔄 Falling back to sync API for recent matches")
            from .api_client import fetch_recent_matches
            result = fetch_recent_matches()
            self.metrics['fallback_calls'] += 1
        
        if not result.get('error'):
            self._cache_result(cache_key, result)
        
        return result
    
    async def fetch_squads(self, series_id: int) -> Dict:
        """Fetch squad information with caching"""
        cache_key = self._get_cache_key("squads", {"series_id": series_id})
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        url = f"{API_BASE_URL}/series/v1/{series_id}/squads"
        result = await self._make_async_request(url)
        
        if result.get('error') and self.fallback_enabled:
            logger.info(f"🔄 Falling back to sync API for squads {series_id}")
            from .api_client import fetch_squads
            result = fetch_squads(series_id)
            self.metrics['fallback_calls'] += 1
        
        if not result.get('error'):
            self._cache_result(cache_key, result)
        
        return result
    
    async def fetch_player_stats_batch(self, player_ids: List[int]) -> Dict[int, Dict]:
        """Fetch multiple player stats concurrently"""
        logger.info(f"🔄 Fetching stats for {len(player_ids)} players concurrently")
        
        # Check cache for each player
        tasks = []
        cached_results = {}
        
        for player_id in player_ids:
            cache_key = self._get_cache_key("player_stats", {"player_id": player_id})
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                cached_results[player_id] = cached_result
            else:
                tasks.append(self._fetch_single_player_stats(player_id))
        
        # Execute remaining requests concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for player_id, result in zip(
                [pid for pid in player_ids if pid not in cached_results], 
                results
            ):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error fetching stats for player {player_id}: {result}")
                    cached_results[player_id] = {"error": str(result)}
                else:
                    cached_results[player_id] = result
                    
                    # Cache successful results
                    if not result.get('error'):
                        cache_key = self._get_cache_key("player_stats", {"player_id": player_id})
                        self._cache_result(cache_key, result)
        
        logger.info(f"✅ Completed batch fetch: {len(cached_results)} players processed")
        return cached_results
    
    async def _fetch_single_player_stats(self, player_id: int) -> Dict:
        """Fetch single player stats with fallback"""
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}"
        result = await self._make_async_request(url)
        
        if result.get('error') and self.fallback_enabled:
            from .api_client import fetch_player_stats
            result = fetch_player_stats(player_id)
            self.metrics['fallback_calls'] += 1
        
        return result
    
    async def fetch_match_center(self, match_id: int) -> Dict:
        """Fetch match center data with caching"""
        cache_key = self._get_cache_key("match_center", {"match_id": match_id})
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}"
        result = await self._make_async_request(url)
        
        if result.get('error') and self.fallback_enabled:
            logger.info(f"🔄 Falling back to sync API for match center {match_id}")
            from .api_client import fetch_match_center
            result = fetch_match_center(match_id)
            self.metrics['fallback_calls'] += 1
        
        if not result.get('error'):
            self._cache_result(cache_key, result)
        
        return result
    
    async def fetch_venue_data_batch(self, venue_ids: List[int]) -> Dict[int, Dict]:
        """Fetch multiple venue data concurrently"""
        logger.info(f"🔄 Fetching venue data for {len(venue_ids)} venues concurrently")
        
        tasks = []
        for venue_id in venue_ids:
            tasks.append(self._fetch_single_venue_stats(venue_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        venue_data = {}
        for venue_id, result in zip(venue_ids, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error fetching venue {venue_id}: {result}")
                venue_data[venue_id] = {"error": str(result)}
            else:
                venue_data[venue_id] = result
        
        return venue_data
    
    async def _fetch_single_venue_stats(self, venue_id: int) -> Dict:
        """Fetch single venue stats with fallback"""
        cache_key = self._get_cache_key("venue_stats", {"venue_id": venue_id})
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        url = f"{API_BASE_URL}/stats/v1/venue/{venue_id}"
        result = await self._make_async_request(url)
        
        if result.get('error') and self.fallback_enabled:
            from .api_client import fetch_venue_stats
            result = fetch_venue_stats(venue_id)
            self.metrics['fallback_calls'] += 1
        
        if not result.get('error'):
            self._cache_result(cache_key, result)
        
        return result
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for monitoring"""
        total_calls = self.metrics['async_calls'] + self.metrics['fallback_calls']
        cache_hit_rate = (self.metrics['cache_hits'] / max(total_calls, 1)) * 100
        
        return {
            **self.metrics,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'avg_time_per_call': round(
                self.metrics['total_time_saved'] / max(self.metrics['async_calls'], 1), 3
            )
        }
    
    def clear_cache(self):
        """Clear the cache manually"""
        self.cache.clear()
        logger.info("🗑️ Cache cleared")

# Global async client instance
async_client = AsyncAPIClient()

# Convenience functions for backward compatibility
async def async_fetch_upcoming_matches():
    """Async wrapper for upcoming matches"""
    async with AsyncAPIClient() as client:
        return await client.fetch_upcoming_matches()

async def async_fetch_recent_matches():
    """Async wrapper for recent matches"""
    async with AsyncAPIClient() as client:
        return await client.fetch_recent_matches()

async def async_fetch_squads(series_id: int):
    """Async wrapper for squad data"""
    async with AsyncAPIClient() as client:
        return await client.fetch_squads(series_id)

async def async_fetch_match_center(match_id: int):
    """Async wrapper for match center data"""
    async with AsyncAPIClient() as client:
        return await client.fetch_match_center(match_id)

# Batch processing functions (NEW CAPABILITY)
async def async_fetch_multiple_player_stats(player_ids: List[int]) -> Dict[int, Dict]:
    """Fetch multiple player stats concurrently - significant performance improvement"""
    async with AsyncAPIClient() as client:
        return await client.fetch_player_stats_batch(player_ids)

async def async_fetch_multiple_venues(venue_ids: List[int]) -> Dict[int, Dict]:
    """Fetch multiple venue data concurrently"""
    async with AsyncAPIClient() as client:
        return await client.fetch_venue_data_batch(venue_ids)

if __name__ == "__main__":
    # Test the async client
    async def test_async_client():
        async with AsyncAPIClient() as client:
            print("🧪 Testing async API client...")
            
            # Test upcoming matches
            matches = await client.fetch_upcoming_matches()
            print(f"✅ Fetched upcoming matches: {len(str(matches))} chars")
            
            # Test performance metrics
            metrics = client.get_performance_metrics()
            print(f"📊 Performance metrics: {metrics}")
    
    asyncio.run(test_async_client())