#!/usr/bin/env python3
"""
API Request Optimizer with Deduplication
Smart request batching, deduplication, and unified API management
"""

import hashlib
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .intelligent_api_cache import get_api_cache, cached_api_call
    from .api_rate_limiter import get_rate_limiter, can_make_api_request, record_api_request
except ImportError:
    from intelligent_api_cache import get_api_cache, cached_api_call
    from api_rate_limiter import get_rate_limiter, can_make_api_request, record_api_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RequestGroup:
    """Group related requests for optimization"""
    group_id: str
    requests: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 1
    batch_size: int = 5
    delay_between_requests: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PendingRequest:
    """Pending API request awaiting optimization"""
    request_id: str
    endpoint: str
    method: str = 'GET'
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    dedupe_key: str = ""

class APIRequestOptimizer:
    """
    Unified API request optimizer with intelligent deduplication and batching
    """
    
    def __init__(self):
        self.api_cache = get_api_cache()
        self.rate_limiter = get_rate_limiter()
        
        # Request deduplication
        self.pending_requests = {}  # request_id -> PendingRequest
        self.dedupe_map = {}  # dedupe_key -> request_id
        self.request_groups = defaultdict(RequestGroup)
        
        # Optimization settings
        self.batch_enabled = True
        self.deduplication_enabled = True
        self.intelligent_grouping = True
        self.max_concurrent_requests = 5
        self.request_timeout = 30
        
        # Performance tracking
        self.optimization_stats = {
            'total_requests': 0,
            'deduplicated_requests': 0,
            'batched_requests': 0,
            'cache_hits': 0,
            'api_calls_made': 0,
            'total_time_saved': 0.0,
            'total_cost_saved': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._processing_lock = threading.Lock()
        
        # Background processor
        self.processing_enabled = True
        self.processing_interval = 2.0  # Process every 2 seconds
        self._start_background_processor()
    
    def _generate_dedupe_key(self, endpoint: str, method: str, params: Dict, headers: Dict) -> str:
        """Generate deduplication key for request"""
        # Remove authentication and user-specific headers
        filtered_headers = {
            k: v for k, v in headers.items() 
            if k.lower() not in ['authorization', 'user-agent', 'x-api-key']
        }
        
        dedupe_data = {
            'endpoint': endpoint,
            'method': method.upper(),
            'params': params or {},
            'headers': filtered_headers
        }
        
        dedupe_string = json.dumps(dedupe_data, sort_keys=True)
        return hashlib.md5(dedupe_string.encode()).hexdigest()
    
    def _determine_request_group(self, endpoint: str) -> str:
        """Determine which group a request belongs to"""
        if 'match-center' in endpoint.lower() or 'match_center' in endpoint.lower():
            return 'match_data'
        elif 'upcoming' in endpoint.lower():
            return 'upcoming_matches'
        elif 'player' in endpoint.lower() or 'squad' in endpoint.lower():
            return 'player_data'
        elif 'live' in endpoint.lower() or 'score' in endpoint.lower():
            return 'live_data'
        else:
            return 'general'
    
    def queue_request(self, endpoint: str, method: str = 'GET', params: Dict = None,
                     headers: Dict = None, callback: Callable = None, priority: int = 1) -> str:
        """
        Queue an API request for optimization
        
        Returns:
            request_id: Unique identifier for the request
        """
        params = params or {}
        headers = headers or {}
        
        # Generate unique request ID
        request_id = hashlib.sha256(
            f"{endpoint}_{method}_{time.time()}_{threading.current_thread().ident}".encode()
        ).hexdigest()[:16]
        
        # Generate deduplication key
        dedupe_key = self._generate_dedupe_key(endpoint, method, params, headers)
        
        with self._lock:
            # Check for deduplication
            if self.deduplication_enabled and dedupe_key in self.dedupe_map:
                existing_request_id = self.dedupe_map[dedupe_key]
                if existing_request_id in self.pending_requests:
                    logger.info(f"🔄 Deduplicated request: {endpoint}")
                    self.optimization_stats['deduplicated_requests'] += 1
                    return existing_request_id
            
            # Create pending request
            request = PendingRequest(
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                params=params,
                headers=headers,
                callback=callback,
                priority=priority,
                dedupe_key=dedupe_key
            )
            
            # Store request
            self.pending_requests[request_id] = request
            if self.deduplication_enabled:
                self.dedupe_map[dedupe_key] = request_id
            
            # Add to appropriate group
            group_id = self._determine_request_group(endpoint)
            if group_id not in self.request_groups:
                self.request_groups[group_id] = RequestGroup(
                    group_id=group_id,
                    priority=priority,
                    batch_size=self._get_batch_size_for_group(group_id)
                )
            
            self.request_groups[group_id].requests.append(request_id)
            self.optimization_stats['total_requests'] += 1
            
            logger.debug(f"📝 Queued request: {request_id} ({group_id})")
            
            return request_id
    
    def _get_batch_size_for_group(self, group_id: str) -> int:
        """Get optimal batch size for request group"""
        batch_sizes = {
            'match_data': 3,      # Match data can be batched efficiently
            'player_data': 5,     # Player data benefits from larger batches
            'upcoming_matches': 2, # Upcoming matches need fresher data
            'live_data': 1,       # Live data should not be batched
            'general': 3
        }
        return batch_sizes.get(group_id, 3)
    
    def execute_request_sync(self, endpoint: str, method: str = 'GET', params: Dict = None,
                           headers: Dict = None, priority: int = 1) -> Dict[str, Any]:
        """
        Execute request synchronously with optimization
        """
        # Check cache first
        cached_response = self.api_cache.get_cached_response(endpoint, params, headers)
        if cached_response:
            self.optimization_stats['cache_hits'] += 1
            return cached_response
        
        # Check rate limits
        can_make, reason, wait_time = can_make_api_request(endpoint)
        if not can_make:
            if wait_time > 0:
                logger.warning(f"⏳ Rate limited: {reason}, waiting {wait_time:.1f}s")
                time.sleep(min(wait_time, 60))  # Cap wait time at 1 minute
        
        # Make request using cached API call
        start_time = time.time()
        
        try:
            response = cached_api_call(endpoint, method, params=params, headers=headers, timeout=self.request_timeout)
            response_time = time.time() - start_time
            
            # Record request for rate limiting
            record_api_request(endpoint, response_time, 200)
            
            self.optimization_stats['api_calls_made'] += 1
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            record_api_request(endpoint, response_time, 500, str(e))
            
            logger.error(f"❌ API request failed: {e}")
            return {'error': str(e), 'endpoint': endpoint}
    
    def _start_background_processor(self):
        """Start background request processor"""
        def processor_worker():
            while self.processing_enabled:
                try:
                    self._process_pending_requests()
                    time.sleep(self.processing_interval)
                except Exception as e:
                    logger.error(f"❌ Error in request processor: {e}")
                    time.sleep(5)
        
        processor_thread = threading.Thread(target=processor_worker, daemon=True)
        processor_thread.start()
        logger.info("🔄 Started background request processor")
    
    def _process_pending_requests(self):
        """Process pending requests in batches"""
        if not self.pending_requests:
            return
        
        with self._processing_lock:
            # Group requests by priority and age
            groups_to_process = []
            
            for group_id, group in self.request_groups.items():
                if not group.requests:
                    continue
                
                # Check if group is ready for processing
                oldest_request_time = min(
                    self.pending_requests[req_id].created_at 
                    for req_id in group.requests 
                    if req_id in self.pending_requests
                )
                
                age_seconds = (datetime.now() - oldest_request_time).total_seconds()
                
                # Process if batch is full or requests are old enough
                if len(group.requests) >= group.batch_size or age_seconds > 10:
                    groups_to_process.append(group_id)
            
            # Process groups
            for group_id in groups_to_process:
                self._process_request_group(group_id)
    
    def _process_request_group(self, group_id: str):
        """Process a group of related requests"""
        group = self.request_groups[group_id]
        
        if not group.requests:
            return
        
        logger.info(f"🔄 Processing request group: {group_id} ({len(group.requests)} requests)")
        
        # Sort requests by priority
        request_ids = sorted(
            group.requests,
            key=lambda req_id: self.pending_requests[req_id].priority if req_id in self.pending_requests else 999
        )
        
        # Process requests in batches
        batch_size = group.batch_size
        
        for i in range(0, len(request_ids), batch_size):
            batch = request_ids[i:i + batch_size]
            self._process_request_batch(batch, group_id)
            
            # Delay between batches
            if i + batch_size < len(request_ids):
                time.sleep(group.delay_between_requests)
        
        # Clear processed requests from group
        group.requests.clear()
    
    def _process_request_batch(self, request_ids: List[str], group_id: str):
        """Process a batch of requests"""
        batch_start = time.time()
        
        valid_requests = [
            self.pending_requests[req_id] for req_id in request_ids 
            if req_id in self.pending_requests
        ]
        
        if not valid_requests:
            return
        
        logger.info(f"⚡ Processing batch: {len(valid_requests)} requests ({group_id})")
        
        # Execute requests with threading
        with ThreadPoolExecutor(max_workers=min(len(valid_requests), self.max_concurrent_requests)) as executor:
            futures = {}
            
            for request in valid_requests:
                future = executor.submit(
                    self.execute_request_sync,
                    request.endpoint,
                    request.method,
                    request.params,
                    request.headers,
                    request.priority
                )
                futures[future] = request
            
            # Process completed requests
            for future in as_completed(futures, timeout=self.request_timeout * 2):
                request = futures[future]
                
                try:
                    response = future.result()
                    
                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(response)
                        except Exception as e:
                            logger.error(f"❌ Callback error for {request.request_id}: {e}")
                    
                    # Clean up request
                    self._cleanup_request(request.request_id)
                    
                except Exception as e:
                    logger.error(f"❌ Batch request failed: {e}")
                    self._cleanup_request(request.request_id)
        
        batch_time = time.time() - batch_start
        self.optimization_stats['batched_requests'] += len(valid_requests)
        
        logger.info(f"✅ Batch completed in {batch_time:.2f}s")
    
    def _cleanup_request(self, request_id: str):
        """Clean up completed request"""
        with self._lock:
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                
                # Remove from deduplication map
                if request.dedupe_key in self.dedupe_map:
                    del self.dedupe_map[request.dedupe_key]
                
                # Remove from pending requests
                del self.pending_requests[request_id]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get request optimization statistics"""
        with self._lock:
            total_requests = self.optimization_stats['total_requests']
            
            cache_hit_rate = (self.optimization_stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            deduplication_rate = (self.optimization_stats['deduplicated_requests'] / total_requests * 100) if total_requests > 0 else 0
            batch_efficiency = (self.optimization_stats['batched_requests'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'total_requests': total_requests,
                'api_calls_made': self.optimization_stats['api_calls_made'],
                'cache_hits': self.optimization_stats['cache_hits'],
                'deduplicated_requests': self.optimization_stats['deduplicated_requests'],
                'batched_requests': self.optimization_stats['batched_requests'],
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'deduplication_rate': f"{deduplication_rate:.1f}%",
                'batch_efficiency': f"{batch_efficiency:.1f}%",
                'requests_saved': self.optimization_stats['cache_hits'] + self.optimization_stats['deduplicated_requests'],
                'pending_requests': len(self.pending_requests),
                'active_groups': len([g for g in self.request_groups.values() if g.requests])
            }
    
    def optimize_request_sequence(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a sequence of requests for maximum efficiency
        
        Args:
            requests: List of request dictionaries with 'endpoint', 'method', 'params', etc.
            
        Returns:
            List of optimized request results
        """
        logger.info(f"🎯 Optimizing sequence of {len(requests)} requests")
        
        # Group requests by endpoint similarity
        request_groups = defaultdict(list)
        
        for i, req in enumerate(requests):
            endpoint = req.get('endpoint', '')
            group_key = self._determine_request_group(endpoint)
            request_groups[group_key].append((i, req))
        
        # Execute groups in optimal order (high priority first)
        group_priorities = {
            'live_data': 1,
            'match_data': 2,
            'player_data': 3,
            'upcoming_matches': 4,
            'general': 5
        }
        
        results = [None] * len(requests)
        
        for group_key in sorted(request_groups.keys(), key=lambda x: group_priorities.get(x, 999)):
            group_requests = request_groups[group_key]
            
            # Process group requests
            for original_index, request in group_requests:
                result = self.execute_request_sync(
                    request.get('endpoint', ''),
                    request.get('method', 'GET'),
                    request.get('params'),
                    request.get('headers'),
                    request.get('priority', 1)
                )
                results[original_index] = result
                
                # Small delay between requests in same group
                time.sleep(0.2)
        
        return results
    
    def preload_common_endpoints(self, endpoints: List[str]):
        """Preload common endpoints to cache"""
        logger.info(f"🔥 Preloading {len(endpoints)} common endpoints...")
        
        for endpoint in endpoints:
            try:
                self.queue_request(endpoint, priority=5)  # Low priority
            except Exception as e:
                logger.error(f"❌ Error queuing preload request: {e}")
        
        # Wait for processing
        time.sleep(len(endpoints) * 0.5)
    
    def shutdown(self):
        """Shutdown the optimizer"""
        self.processing_enabled = False
        logger.info("🔄 API request optimizer shutting down...")
        
        # Process remaining requests
        if self.pending_requests:
            logger.info(f"⚡ Processing {len(self.pending_requests)} remaining requests...")
            self._process_pending_requests()

# Global optimizer instance
api_optimizer = APIRequestOptimizer()

def get_api_optimizer() -> APIRequestOptimizer:
    """Get global API optimizer instance"""
    return api_optimizer

def optimized_api_call(endpoint: str, method: str = 'GET', **kwargs) -> Dict[str, Any]:
    """Make optimized API call"""
    return api_optimizer.execute_request_sync(endpoint, method, **kwargs)

def queue_api_request(endpoint: str, method: str = 'GET', callback: Callable = None, **kwargs) -> str:
    """Queue API request for batch processing"""
    return api_optimizer.queue_request(endpoint, method, callback=callback, **kwargs)

def get_api_optimization_stats() -> Dict[str, Any]:
    """Get API optimization statistics"""
    return api_optimizer.get_optimization_stats()