#!/usr/bin/env python3
"""
Smart API Manager - Intelligent rate limiting and optimization for auto-prediction system
Ensures we never hit API limits while maximizing learning efficiency
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import threading
import queue

class SmartAPIManager:
    """
    Intelligent API management system that:
    1. Tracks API usage across all calls
    2. Prioritizes high-value API calls
    3. Uses intelligent caching strategies
    4. Prevents rate limit violations
    5. Optimizes for maximum learning with minimum API usage
    """
    
    def __init__(self):
        self.api_db_path = "api_usage_tracking.db"
        self.setup_database()
        
        # API Limits (Conservative estimates for safety)
        self.api_limits = {
            'requests_per_minute': 8,  # Safe limit (RapidAPI typically allows 10)
            'requests_per_hour': 400,  # Conservative hourly limit
            'requests_per_day': 8000,  # Daily limit with buffer
            'priority_reserve': 50     # Reserve calls for high-priority
        }
        
        # Request queue with priority
        self.request_queue = queue.PriorityQueue()
        self.api_lock = threading.Lock()
        
        # Cache optimization
        self.cache_strategies = {
            'upcoming_matches': 3600,    # Cache for 1 hour
            'match_center': 1800,       # Cache for 30 minutes  
            'recent_matches': 7200,     # Cache for 2 hours
            'squads': 86400,           # Cache for 24 hours
            'scorecard': 3600          # Cache for 1 hour after match
        }
    
    def setup_database(self):
        """Setup API usage tracking database"""
        conn = sqlite3.connect(self.api_db_path)
        cursor = conn.cursor()
        
        # API usage log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT NOT NULL,
                match_id TEXT,
                priority INTEGER DEFAULT 5,
                cache_hit BOOLEAN DEFAULT FALSE,
                response_time REAL,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT
            )
        ''')
        
        # Daily API quotas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_quotas (
                date DATE PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                cached_requests INTEGER DEFAULT 0,
                priority_requests INTEGER DEFAULT 0,
                auto_prediction_requests INTEGER DEFAULT 0,
                manual_requests INTEGER DEFAULT 0,
                quota_remaining INTEGER,
                efficiency_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ API usage tracking database initialized")
    
    def get_current_usage(self) -> Dict[str, int]:
        """Get current API usage for rate limiting"""
        conn = sqlite3.connect(self.api_db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        today = now.date()
        
        # Get usage counts
        cursor.execute('''
            SELECT COUNT(*) FROM api_usage 
            WHERE timestamp > ? AND cache_hit = FALSE
        ''', (one_minute_ago,))
        requests_last_minute = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM api_usage 
            WHERE timestamp > ? AND cache_hit = FALSE
        ''', (one_hour_ago,))
        requests_last_hour = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT total_requests FROM daily_quotas 
            WHERE date = ?
        ''', (today,))
        result = cursor.fetchone()
        requests_today = result[0] if result else 0
        
        conn.close()
        
        return {
            'requests_last_minute': requests_last_minute,
            'requests_last_hour': requests_last_hour,
            'requests_today': requests_today,
            'quota_remaining': self.api_limits['requests_per_day'] - requests_today
        }
    
    def can_make_request(self, priority: int = 5) -> bool:
        """Check if we can make an API request without hitting limits"""
        usage = self.get_current_usage()
        
        # Check minute limit
        if usage['requests_last_minute'] >= self.api_limits['requests_per_minute']:
            return False
        
        # Check hourly limit
        if usage['requests_last_hour'] >= self.api_limits['requests_per_hour']:
            return False
        
        # Check daily limit with priority reserve
        if priority <= 3:  # High priority
            if usage['requests_today'] >= self.api_limits['requests_per_day']:
                return False
        else:  # Normal/low priority
            daily_limit_with_reserve = self.api_limits['requests_per_day'] - self.api_limits['priority_reserve']
            if usage['requests_today'] >= daily_limit_with_reserve:
                return False
        
        return True
    
    def calculate_wait_time(self) -> float:
        """Calculate optimal wait time before next request"""
        usage = self.get_current_usage()
        
        # If we're at minute limit, wait until minute resets
        if usage['requests_last_minute'] >= self.api_limits['requests_per_minute']:
            return 65  # Wait 65 seconds to be safe
        
        # If we're at hourly limit, calculate wait time
        if usage['requests_last_hour'] >= self.api_limits['requests_per_hour']:
            return 3700  # Wait ~1 hour
        
        # Otherwise, use intelligent spacing
        requests_per_minute_limit = self.api_limits['requests_per_minute']
        optimal_interval = 60 / requests_per_minute_limit  # Spread requests evenly
        return optimal_interval
    
    def log_api_request(self, endpoint: str, match_id: str = None, priority: int = 5, 
                       cache_hit: bool = False, response_time: float = None, 
                       success: bool = True, error_message: str = None):
        """Log API request for tracking and optimization"""
        conn = sqlite3.connect(self.api_db_path)
        cursor = conn.cursor()
        
        # Log the request
        cursor.execute('''
            INSERT INTO api_usage 
            (endpoint, match_id, priority, cache_hit, response_time, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (endpoint, match_id, priority, cache_hit, response_time, success, error_message))
        
        # Update daily quota if not a cache hit
        if not cache_hit:
            today = datetime.now().date()
            cursor.execute('''
                INSERT OR IGNORE INTO daily_quotas (date, total_requests) VALUES (?, 0)
            ''', (today,))
            
            cursor.execute('''
                UPDATE daily_quotas 
                SET total_requests = total_requests + 1,
                    quota_remaining = ? - total_requests - 1
                WHERE date = ?
            ''', (self.api_limits['requests_per_day'], today))
        
        conn.commit()
        conn.close()
    
    def optimize_api_calls_for_auto_prediction(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize API calls for auto-prediction system"""
        print("🎯 OPTIMIZING API CALLS FOR AUTO-PREDICTION")
        print("-" * 50)
        
        usage = self.get_current_usage()
        available_calls = usage['quota_remaining']
        
        print(f"📊 API Status:")
        print(f"   • Remaining quota: {available_calls} calls")
        print(f"   • Matches to predict: {len(matches)}")
        
        # Calculate API calls needed per match
        calls_per_match = 3  # match_center + squads + venue_data (approximately)
        total_calls_needed = len(matches) * calls_per_match
        
        print(f"   • Estimated calls needed: {total_calls_needed}")
        
        if total_calls_needed > available_calls:
            # Prioritize matches
            prioritized_matches = self.prioritize_matches_for_api_efficiency(matches, available_calls)
            print(f"   ⚡ Optimized to {len(prioritized_matches)} highest priority matches")
            return prioritized_matches
        else:
            print(f"   ✅ Sufficient quota for all matches")
            return matches
    
    def prioritize_matches_for_api_efficiency(self, matches: List[Dict[str, Any]], 
                                            available_calls: int) -> List[Dict[str, Any]]:
        """Prioritize matches based on learning value and API efficiency"""
        
        # Scoring criteria for match priority
        def calculate_match_priority(match):
            score = 0
            
            # Format priority (higher learning value)
            format_scores = {
                'T20': 10, 'T20I': 10,
                'The Hundred': 9,
                'ODI': 8, 'ODIM': 8,
                'Test': 6  # Lower priority due to longer duration
            }
            score += format_scores.get(match['format'], 5)
            
            # Series importance (international > domestic)
            series_name = match.get('series_name', '').lower()
            if any(term in series_name for term in ['world cup', 'championship', 'final']):
                score += 5
            elif any(term in series_name for term in ['international', 'tour']):
                score += 3
            
            # Timing priority (sooner = higher priority)
            time_until_match = (match['start_time'] - datetime.now()).total_seconds() / 3600
            if time_until_match < 2:  # Starting soon
                score += 8
            elif time_until_match < 6:
                score += 5
            elif time_until_match < 24:
                score += 2
            
            return score
        
        # Calculate priorities and sort
        match_priorities = [(match, calculate_match_priority(match)) for match in matches]
        match_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Select top matches that fit within API quota
        calls_per_match = 3
        max_matches = available_calls // calls_per_match
        selected_matches = [match for match, _ in match_priorities[:max_matches]]
        
        print(f"   🎯 Selected {len(selected_matches)} highest priority matches:")
        for i, match in enumerate(selected_matches[:5]):  # Show top 5
            priority_score = match_priorities[i][1]
            print(f"      {i+1}. {match['teams']} ({match['format']}) - Score: {priority_score}")
        
        return selected_matches
    
    def smart_api_call(self, api_function, endpoint_name: str, *args, priority: int = 5, 
                      match_id: str = None, **kwargs):
        """Make API call with smart rate limiting and caching"""
        
        # Check cache first
        cache_key = f"{endpoint_name}_{hash(str(args) + str(kwargs))}"
        cached_result = self.get_cached_result(cache_key, endpoint_name)
        
        if cached_result:
            self.log_api_request(endpoint_name, match_id, priority, cache_hit=True)
            print(f"💾 Cache hit: {endpoint_name}")
            return cached_result
        
        # Check if we can make the request
        if not self.can_make_request(priority):
            wait_time = self.calculate_wait_time()
            print(f"⏰ Rate limit reached, waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Make the API call
        start_time = time.time()
        try:
            with self.api_lock:
                result = api_function(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Cache the result
                self.cache_result(cache_key, endpoint_name, result)
                
                # Log successful request
                self.log_api_request(endpoint_name, match_id, priority, 
                                   cache_hit=False, response_time=response_time, success=True)
                
                print(f"🌐 API call: {endpoint_name} ({response_time:.2f}s)")
                return result
                
        except Exception as e:
            response_time = time.time() - start_time
            self.log_api_request(endpoint_name, match_id, priority, 
                               cache_hit=False, response_time=response_time, 
                               success=False, error_message=str(e))
            print(f"❌ API error: {endpoint_name} - {e}")
            raise e
    
    def get_cached_result(self, cache_key: str, endpoint_name: str) -> Optional[Any]:
        """Get cached API result if still valid"""
        cache_file = f"api_cache/{cache_key}.json"
        
        try:
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_data['timestamp'])
            cache_duration = self.cache_strategies.get(endpoint_name, 3600)
            
            if (datetime.now() - cache_time).total_seconds() < cache_duration:
                return cached_data['result']
            else:
                # Cache expired
                os.remove(cache_file)
                return None
                
        except Exception:
            return None
    
    def cache_result(self, cache_key: str, endpoint_name: str, result: Any):
        """Cache API result for future use"""
        import os
        cache_dir = "api_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = f"{cache_dir}/{cache_key}.json"
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint_name,
                'result': result
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            print(f"⚠️ Failed to cache result: {e}")
    
    def get_api_efficiency_report(self) -> Dict[str, Any]:
        """Generate API efficiency report"""
        conn = sqlite3.connect(self.api_db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Get today's stats
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   COUNT(CASE WHEN cache_hit = TRUE THEN 1 END) as cached,
                   AVG(response_time) as avg_response_time
            FROM api_usage 
            WHERE DATE(timestamp) = ?
        ''', (today,))
        
        stats = cursor.fetchone()
        total_requests = stats[0] or 0
        cached_requests = stats[1] or 0
        avg_response_time = stats[2] or 0
        
        actual_api_calls = total_requests - cached_requests
        cache_hit_rate = (cached_requests / max(total_requests, 1)) * 100
        
        usage = self.get_current_usage()
        
        conn.close()
        
        return {
            'total_requests': total_requests,
            'actual_api_calls': actual_api_calls,
            'cached_requests': cached_requests,
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time': avg_response_time,
            'quota_used': usage['requests_today'],
            'quota_remaining': usage['quota_remaining'],
            'efficiency_score': cache_hit_rate * 0.7 + (100 - min(usage['requests_today']/80, 100)) * 0.3
        }

def integrate_with_auto_prediction():
    """Integrate smart API manager with auto-prediction system"""
    
    # Modify the auto_prediction_system.py to use smart API manager
    integration_code = '''
# Add this to auto_prediction_system.py

from smart_api_manager import SmartAPIManager

class EnhancedAutoPredictionSystem(AutoPredictionSystem):
    def __init__(self):
        super().__init__()
        self.api_manager = SmartAPIManager()
    
    def get_all_upcoming_matches(self) -> List[Dict[str, Any]]:
        """Enhanced version with smart API management"""
        
        # Use smart API call instead of direct API call
        upcoming_data = self.api_manager.smart_api_call(
            api_function=fetch_upcoming_matches,
            endpoint_name='upcoming_matches',
            priority=3  # High priority for match discovery
        )
        
        # Rest of the logic remains the same...
        # Process matches and return optimized list
        matches = self.process_upcoming_matches(upcoming_data)
        
        # Optimize API calls for the day
        optimized_matches = self.api_manager.optimize_api_calls_for_auto_prediction(matches)
        
        return optimized_matches
    
    def run_prediction_for_match(self, match: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Enhanced prediction with API optimization"""
        
        # Check API quota before starting prediction
        if not self.api_manager.can_make_request(priority=4):
            print(f"⏰ Skipping match {match['match_id']} due to API limits")
            return None
        
        # Proceed with normal prediction
        return super().run_prediction_for_match(match)
    
    def daily_prediction_job(self):
        """Enhanced daily job with API efficiency reporting"""
        
        # Generate efficiency report
        efficiency = self.api_manager.get_api_efficiency_report()
        print(f"📊 API Efficiency: {efficiency['cache_hit_rate']:.1f}% cache hit rate")
        print(f"🔋 Quota remaining: {efficiency['quota_remaining']} calls")
        
        # Run normal daily job
        super().daily_prediction_job()
'''
    
    # Save the integration code
    with open("enhanced_auto_prediction_integration.py", "w") as f:
        f.write(integration_code)
    
    print("✅ Integration code created: enhanced_auto_prediction_integration.py")

def main():
    """Test and demonstrate the smart API manager"""
    api_manager = SmartAPIManager()
    
    # Show current status
    usage = api_manager.get_current_usage()
    print("\n🔋 CURRENT API STATUS")
    print("="*30)
    print(f"Requests last minute: {usage['requests_last_minute']}")
    print(f"Requests last hour: {usage['requests_last_hour']}")
    print(f"Requests today: {usage['requests_today']}")
    print(f"Quota remaining: {usage['quota_remaining']}")
    
    # Generate efficiency report
    efficiency = api_manager.get_api_efficiency_report()
    print(f"\n📊 API EFFICIENCY REPORT")
    print("="*30)
    print(f"Cache hit rate: {efficiency['cache_hit_rate']:.1f}%")
    print(f"Efficiency score: {efficiency['efficiency_score']:.1f}/100")
    print(f"Avg response time: {efficiency['avg_response_time']:.2f}s")
    
    # Create integration
    integrate_with_auto_prediction()

if __name__ == "__main__":
    main()
