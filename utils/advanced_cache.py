#!/usr/bin/env python3
"""
Advanced Caching System - Reduce API Calls with Smart Caching
Multi-level caching with intelligent invalidation and persistence
"""

import json
import time
import os
import pickle
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

CACHE_MAGIC = b"D11CACHE\x00\x01"  # magic + version 1

def _safe_pickle_dump(obj: Any, file_obj):
    file_obj.write(CACHE_MAGIC)
    pickle.dump(obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

def _safe_pickle_load(file_obj):
    header = file_obj.read(len(CACHE_MAGIC))
    if header != CACHE_MAGIC:
        # Legacy file or tampered; rewind and try json, then pickle in restricted way
        file_obj.seek(0)
        try:
            return json.load(file_obj)
        except Exception:
            file_obj.seek(0)
            # Last resort: load pickle but only if created by our structure (dict with expected keys)
            data = pickle.load(file_obj)
            if isinstance(data, dict) and {'key', 'data', 'timestamp', 'ttl'}.issubset(set(data.keys())):
                return data
            raise ValueError("Unrecognized cache file format")
    return pickle.load(file_obj)

@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = None
    tags: List[str] = None
    size_bytes: int = 0

class SmartCache:
    """
    Multi-level smart cache with:
    - Memory cache (L1) - Fastest access
    - Disk cache (L2) - Persistent across restarts
    - Intelligent TTL management
    - LRU eviction
    - Tag-based invalidation
    """
    
    def __init__(self, 
                 memory_limit_mb: int = 100,
                 disk_cache_dir: str = ".cache",
                 default_ttl: int = 3600):
        
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_cache_dir = disk_cache_dir
        self.default_ttl = default_ttl
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'writes': 0
        }
        
        # Create cache directory
        os.makedirs(disk_cache_dir, exist_ok=True)
        
        # Cache configurations for different data types
        self.cache_configs = {
            'matches': {'ttl': 1800, 'tags': ['matches', 'live_data']},     # 30 min
            'squads': {'ttl': 7200, 'tags': ['squads', 'team_data']},      # 2 hours
            'player_stats': {'ttl': 3600, 'tags': ['players', 'stats']},   # 1 hour
            'venue_info': {'ttl': 86400, 'tags': ['venues', 'static']},    # 24 hours
            'series_info': {'ttl': 43200, 'tags': ['series', 'static']},   # 12 hours
            'weather': {'ttl': 1800, 'tags': ['weather', 'live_data']},    # 30 min
        }
    
    def _get_cache_key(self, prefix: str, identifier: str, params: Dict = None) -> str:
        """Generate consistent cache key"""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            key = f"{prefix}:{identifier}:{hashlib.md5(param_str.encode()).hexdigest()}"
        else:
            key = f"{prefix}:{identifier}"
        return key
    
    def _get_memory_usage(self) -> int:
        """Calculate current memory cache usage"""
        return sum(entry.size_bytes for entry in self.memory_cache.values())
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes"""
        try:
            return len(pickle.dumps(data))
        except Exception:
            return len(str(data).encode('utf-8'))
    
    def _evict_lru(self):
        """Evict least recently used items from memory cache"""
        if not self.memory_cache:
            return
        
        # Sort by last access time
        items = [(key, entry) for key, entry in self.memory_cache.items()]
        items.sort(key=lambda x: x[1].last_accessed or 0)
        
        # Remove oldest 25% of items
        to_remove = len(items) // 4 or 1
        for i in range(to_remove):
            key = items[i][0]
            self._move_to_disk(key)
            del self.memory_cache[key]
            self.stats['evictions'] += 1
    
    def _move_to_disk(self, key: str):
        """Move entry from memory to disk cache"""
        if key not in self.memory_cache:
            return
        
        entry = self.memory_cache[key]
        disk_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
        
        try:
            cache_data = {
                'key': key,
                'data': entry.data,
                'timestamp': entry.timestamp,
                'ttl': entry.ttl,
                'access_count': entry.access_count,
                'tags': entry.tags or []
            }
            
            with open(disk_path, 'wb') as f:
                _safe_pickle_dump(cache_data, f)
            
            logger.debug(f"💾 Moved {key} to disk cache")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to move {key} to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load entry from disk cache"""
        disk_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
        
        if not os.path.exists(disk_path):
            return None
        
        try:
            with open(disk_path, 'rb') as f:
                cache_data = _safe_pickle_load(f)
            
            # Check if expired
            if time.time() - cache_data['timestamp'] > cache_data['ttl']:
                os.remove(disk_path)
                return None
            
            entry = CacheEntry(
                data=cache_data['data'],
                timestamp=cache_data['timestamp'],
                ttl=cache_data['ttl'],
                access_count=cache_data.get('access_count', 0),
                last_accessed=time.time(),
                tags=cache_data.get('tags', []),
                size_bytes=self._estimate_size(cache_data['data'])
            )
            
            logger.debug(f"💽 Loaded {key} from disk cache")
            return entry
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {key} from disk: {e}")
            try:
                os.remove(disk_path)
            except Exception:
                pass
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)"""
        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if expired
            if time.time() - entry.timestamp > entry.ttl:
                del self.memory_cache[key]
            else:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.stats['memory_hits'] += 1
                logger.debug(f"🎯 Memory cache hit: {key}")
                return entry.data
        
        # Try disk cache
        entry = self._load_from_disk(key)
        if entry:
            # Move back to memory if there's space
            if self._get_memory_usage() + entry.size_bytes < self.memory_limit_bytes:
                self.memory_cache[key] = entry
            self.stats['disk_hits'] += 1
            logger.debug(f"💽 Disk cache hit: {key}")
            return entry.data
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None, tags: List[str] = None):
        """Set item in cache"""
        ttl = ttl or self.default_ttl
        size_bytes = self._estimate_size(data)
        
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl,
            last_accessed=time.time(),
            tags=tags or [],
            size_bytes=size_bytes
        )
        
        # Check if we need to evict items
        if self._get_memory_usage() + size_bytes > self.memory_limit_bytes:
            self._evict_lru()
        
        # If still too big, store directly to disk
        if size_bytes > self.memory_limit_bytes // 2:
            self._move_to_disk_direct(key, entry)
        else:
            self.memory_cache[key] = entry
        
        self.stats['writes'] += 1
        logger.debug(f"💾 Cached {key} (TTL: {ttl}s, Size: {size_bytes} bytes)")
    
    def _move_to_disk_direct(self, key: str, entry: CacheEntry):
        """Move large entry directly to disk"""
        disk_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
        
        try:
            cache_data = {
                'key': key,
                'data': entry.data,
                'timestamp': entry.timestamp,
                'ttl': entry.ttl,
                'access_count': entry.access_count,
                'tags': entry.tags
            }
            
            with open(disk_path, 'wb') as f:
                _safe_pickle_dump(cache_data, f)
            
            logger.debug(f"💾 Large item stored directly to disk: {key}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to store {key} to disk: {e}")
    
    def delete(self, key: str):
        """Delete item from cache"""
        # Remove from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from disk
        disk_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
        try:
            if os.path.exists(disk_path):
                os.remove(disk_path)
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete {key} from disk: {e}")
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate all cache entries with specified tags"""
        to_delete = []
        
        # Check memory cache
        for key, entry in self.memory_cache.items():
            if entry.tags and any(tag in entry.tags for tag in tags):
                to_delete.append(key)
        
        # Check disk cache
        for filename in os.listdir(self.disk_cache_dir):
            if filename.endswith('.cache'):
                disk_path = os.path.join(self.disk_cache_dir, filename)
                try:
                    with open(disk_path, 'rb') as f:
                        cache_data = _safe_pickle_load(f)
                    
                    if cache_data.get('tags') and any(tag in cache_data['tags'] for tag in tags):
                        key = cache_data['key']
                        to_delete.append(key)
                        
                except Exception:
                    continue
        
        # Delete found items
        for key in to_delete:
            self.delete(key)
        
        logger.info(f"🗑️ Invalidated {len(to_delete)} cache entries with tags: {tags}")
    
    def clear_expired(self):
        """Remove all expired entries"""
        current_time = time.time()
        to_delete = []
        
        # Check memory cache
        for key, entry in self.memory_cache.items():
            if current_time - entry.timestamp > entry.ttl:
                to_delete.append(key)
        
        # Check disk cache
        for filename in os.listdir(self.disk_cache_dir):
            if filename.endswith('.cache'):
                disk_path = os.path.join(self.disk_cache_dir, filename)
                try:
                    with open(disk_path, 'rb') as f:
                        cache_data = _safe_pickle_load(f)
                    
                    if current_time - cache_data['timestamp'] > cache_data['ttl']:
                        to_delete.append(cache_data['key'])
                        
                except Exception:
                    # Remove corrupted files
                    try:
                        os.remove(disk_path)
                    except Exception:
                        pass
        
        # Delete expired items
        for key in to_delete:
            self.delete(key)
        
        logger.info(f"🧹 Cleaned {len(to_delete)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_usage = self._get_memory_usage()
        disk_files = len([f for f in os.listdir(self.disk_cache_dir) if f.endswith('.cache')])
        
        total_requests = self.stats['memory_hits'] + self.stats['disk_hits'] + self.stats['misses']
        hit_rate = (self.stats['memory_hits'] + self.stats['disk_hits']) / max(total_requests, 1) * 100
        
        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': disk_files,
            'memory_usage_mb': memory_usage / (1024 * 1024),
            'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024),
            'hit_rate_percent': hit_rate,
            'memory_hits': self.stats['memory_hits'],
            'disk_hits': self.stats['disk_hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'writes': self.stats['writes']
        }

# Enhanced cache for Dream11 AI
class Dream11Cache(SmartCache):
    """
    Specialized cache for Dream11 AI with cricket-specific optimizations
    """
    
    def cache_match_data(self, match_id: str, data: Dict, live: bool = False):
        """Cache match data with appropriate TTL"""
        ttl = 600 if live else 3600  # 10 min for live, 1 hour for completed
        tags = ['matches', 'live_data' if live else 'static_data']
        key = self._get_cache_key('match', match_id)
        self.set(key, data, ttl, tags)
    
    def cache_player_stats(self, player_id: str, stats: Dict, recent: bool = False):
        """Cache player statistics"""
        ttl = 1800 if recent else 7200  # 30 min for recent, 2 hours for career
        tags = ['players', 'stats', 'recent' if recent else 'career']
        key = self._get_cache_key('player_stats', player_id)
        self.set(key, stats, ttl, tags)
    
    def cache_squad_data(self, series_id: str, team_id: str, squad: Dict):
        """Cache squad/team data"""
        key = self._get_cache_key('squad', f"{series_id}_{team_id}")
        self.set(key, squad, ttl=7200, tags=['squads', 'team_data'])
    
    def cache_venue_data(self, venue_id: str, venue_info: Dict):
        """Cache venue information (rarely changes)"""
        key = self._get_cache_key('venue', venue_id)
        self.set(key, venue_info, ttl=86400, tags=['venues', 'static'])  # 24 hours
    
    def invalidate_live_data(self):
        """Invalidate all live/dynamic data"""
        self.invalidate_by_tags(['live_data', 'recent'])
    
    def get_match_data(self, match_id: str) -> Optional[Dict]:
        """Get cached match data"""
        key = self._get_cache_key('match', match_id)
        return self.get(key)
    
    def get_player_stats(self, player_id: str) -> Optional[Dict]:
        """Get cached player stats"""
        key = self._get_cache_key('player_stats', player_id)
        return self.get(key)
    
    def get_squad_data(self, series_id: str, team_id: str) -> Optional[Dict]:
        """Get cached squad data"""
        key = self._get_cache_key('squad', f"{series_id}_{team_id}")
        return self.get(key)