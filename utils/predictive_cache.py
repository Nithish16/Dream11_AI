#!/usr/bin/env python3
"""
Predictive Caching System - Pre-load Data for Upcoming Matches
Automatically cache data for matches in the next 24 hours to reduce API calls
"""

import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PredictiveCache:
    """
    Intelligent predictive caching system that pre-loads data for upcoming matches
    """
    
    def __init__(self):
        self.cache_warming_enabled = True
        self.max_matches_to_cache = 10  # Limit to avoid excessive API usage
        self.cache_warming_interval = 3600  # 1 hour between warming cycles
        
    async def warm_cache_for_upcoming_matches(self) -> Dict[str, Any]:
        """
        Pre-cache data for upcoming matches in the next 24 hours
        """
        try:
            from utils.api_client import (
                fetch_upcoming_matches, fetch_match_center, fetch_squads,
                _rate_limiter, _cache, RATE_LIMITING_ENABLED
            )
            
            if not RATE_LIMITING_ENABLED:
                print("⚠️ Predictive caching requires API optimization to be enabled")
                return {"success": False, "message": "API optimization not available"}
            
            print("🔮 Starting predictive cache warming...")
            
            # Get upcoming matches
            upcoming_data = fetch_upcoming_matches()
            upcoming_matches = self._extract_upcoming_matches(upcoming_data)
            
            if not upcoming_matches:
                print("📅 No upcoming matches found for caching")
                return {"success": True, "matches_cached": 0}
            
            # Limit to prevent excessive API usage
            matches_to_cache = upcoming_matches[:self.max_matches_to_cache]
            cached_count = 0
            
            for match in matches_to_cache:
                if not _rate_limiter.can_make_request():
                    print(f"🚫 Rate limit reached - stopping cache warming")
                    break
                
                try:
                    await self._cache_match_data(match)
                    cached_count += 1
                    print(f"✅ Cached data for match {match['match_id']}")
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"⚠️ Failed to cache match {match.get('match_id', 'Unknown')}: {e}")
                    continue
            
            print(f"🎯 Predictive caching complete: {cached_count}/{len(matches_to_cache)} matches cached")
            
            return {
                "success": True,
                "matches_cached": cached_count,
                "total_matches_found": len(upcoming_matches),
                "rate_limit_status": _rate_limiter.get_status()
            }
            
        except Exception as e:
            print(f"❌ Predictive caching failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_upcoming_matches(self, upcoming_data: Dict) -> List[Dict]:
        """Extract match information from upcoming matches data"""
        matches = []
        
        try:
            type_matches = upcoming_data.get('typeMatches', [])
            
            for type_match in type_matches:
                series_matches = type_match.get('seriesMatches', [])
                
                for series_match in series_matches:
                    series_wrapper = series_match.get('seriesAdWrapper', {})
                    series_id = series_wrapper.get('seriesId')
                    series_matches_list = series_wrapper.get('matches', [])
                    
                    for match in series_matches_list:
                        match_info = match.get('matchInfo', {})
                        
                        # Check if match is in next 24 hours
                        if self._is_match_upcoming(match_info):
                            match_data = {
                                'match_id': match_info.get('matchId'),
                                'series_id': series_id,
                                'team1_id': match_info.get('team1', {}).get('teamId'),
                                'team2_id': match_info.get('team2', {}).get('teamId'),
                                'venue_id': match_info.get('venueInfo', {}).get('id'),
                                'start_date': match_info.get('startDate'),
                                'format': match_info.get('matchFormat', 'Unknown')
                            }
                            
                            if match_data['match_id']:
                                matches.append(match_data)
            
            return sorted(matches, key=lambda x: x['start_date'] or '')
            
        except Exception as e:
            print(f"⚠️ Error extracting upcoming matches: {e}")
            return []
    
    def _is_match_upcoming(self, match_info: Dict) -> bool:
        """Check if match is in the next 24 hours"""
        try:
            start_date = match_info.get('startDate')
            if not start_date:
                return False
            
            # Handle different date formats
            if isinstance(start_date, str):
                if start_date.isdigit():
                    # Unix timestamp in milliseconds
                    match_time = datetime.fromtimestamp(int(start_date) / 1000)
                else:
                    # ISO format
                    match_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                return False
            
            now = datetime.now()
            time_until_match = (match_time - now).total_seconds()
            
            # Match is in next 24 hours and hasn't started yet
            return 0 < time_until_match < 86400  # 24 hours in seconds
            
        except Exception:
            return False
    
    async def _cache_match_data(self, match: Dict):
        """Cache all relevant data for a specific match"""
        from utils.api_client import fetch_match_center, fetch_squads, _cache
        
        match_id = match['match_id']
        series_id = match['series_id']
        
        # Cache match center data
        if match_id:
            match_center_data = fetch_match_center(match_id)
            if match_center_data and 'error' not in match_center_data:
                _cache.cache_match_data(str(match_id), match_center_data, live=False)
        
        # Cache squad data
        if series_id:
            squad_data = fetch_squads(series_id)
            if squad_data and 'error' not in squad_data:
                cache_key = f"series_squads_{series_id}"
                _cache.set(cache_key, squad_data, ttl=14400, tags=['squads', 'team_data'])
        
        # Cache team-specific squad data
        for team_key in ['team1_id', 'team2_id']:
            team_id = match.get(team_key)
            if team_id and series_id:
                _cache.cache_squad_data(str(series_id), str(team_id), {
                    'team_id': team_id,
                    'series_id': series_id,
                    'cached_at': datetime.now().isoformat()
                })

def start_predictive_caching_service():
    """Start the predictive caching service as a background task"""
    async def caching_loop():
        predictor = PredictiveCache()
        
        while True:
            try:
                await predictor.warm_cache_for_upcoming_matches()
                
                # Wait for next caching cycle
                await asyncio.sleep(predictor.cache_warming_interval)
                
            except Exception as e:
                print(f"❌ Predictive caching service error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retry
    
    # Run the caching loop in the background
    return asyncio.create_task(caching_loop())

def run_manual_cache_warming():
    """Run predictive caching manually (for testing)"""
    async def manual_warming():
        predictor = PredictiveCache()
        result = await predictor.warm_cache_for_upcoming_matches()
        print(f"\n🔮 Manual Cache Warming Results:")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        
        if result['success']:
            print(f"   Matches cached: {result.get('matches_cached', 0)}")
            print(f"   Total matches found: {result.get('total_matches_found', 0)}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    asyncio.run(manual_warming())

if __name__ == "__main__":
    print("🔮 Dream11 AI - Predictive Cache Warming")
    print("=" * 50)
    run_manual_cache_warming()