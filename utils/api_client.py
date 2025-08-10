import requests
import json
from datetime import datetime, timedelta

# Cricbuzz API Configuration
API_BASE_URL = "https://cricbuzz-cricket.p.rapidapi.com"

# Enhanced API Configuration with Environment Variables
import os

def _get_api_key() -> str:
    return os.getenv('RAPIDAPI_KEY', '').strip()

def _get_api_headers() -> dict:
    api_key = _get_api_key()
    headers = {
        'x-rapidapi-host': 'cricbuzz-cricket.p.rapidapi.com',
    }
    if api_key:
        headers['x-rapidapi-key'] = api_key
    return headers

API_HEADERS = _get_api_headers()

# Import rate limiting and caching
try:
    from .api_rate_limiter import APIRateLimiter, SmartAPIClient
    from .advanced_cache import Dream11Cache
    RATE_LIMITING_ENABLED = True
except ImportError:
    RATE_LIMITING_ENABLED = False

# Global instances
if RATE_LIMITING_ENABLED:
    _rate_limiter = APIRateLimiter()
    _cache = Dream11Cache()
    _smart_client = SmartAPIClient(_rate_limiter)

def _missing_key() -> bool:
    return 'x-rapidapi-key' not in API_HEADERS or not API_HEADERS['x-rapidapi-key']

def fetch_upcoming_matches():
    """
    Fetches upcoming matches from Cricbuzz API with rate limiting and caching
    """
    if RATE_LIMITING_ENABLED:
        # Check cache first
        cache_key = "upcoming_matches"
        cached_data = _cache.get(cache_key)
        if cached_data:
            print("🎯 Cache hit: upcoming matches")
            return cached_data

        # Check rate limit
        if not _rate_limiter.can_make_request():
            wait_time = _rate_limiter.get_wait_time()
            print(f"⏰ Rate limited - using cached/fallback data (wait: {wait_time:.1f}s)")
            return _get_sample_matches_data()

        # Acquire rate limit slot
        if not _rate_limiter.acquire_request_slot("high"):
            print("🚫 Rate limit hit - using fallback data")
            return _get_sample_matches_data()

    if _missing_key():
        print("⚠️ RAPIDAPI_KEY not set; returning fallback upcoming matches data")
        return _get_sample_matches_data()

    try:
        url = f"{API_BASE_URL}/matches/v1/upcoming"
        response = requests.get(url, headers=API_HEADERS, timeout=10)

        if RATE_LIMITING_ENABLED:
            _rate_limiter.handle_api_response(dict(response.headers))

        if response.status_code == 200:
            try:
                data = response.json()
                if data and len(str(data)) > 10:  # Valid response check
                    # Cache successful response
                    if RATE_LIMITING_ENABLED:
                        _cache.set(cache_key, data, ttl=1800, tags=['matches', 'live_data'])  # 30 min
                        print("💾 Cached upcoming matches")
                    return data
                else:
                    print("API returned empty/invalid data, using fallback")
                    return _get_sample_matches_data()
            except json.JSONDecodeError:
                print("API returned invalid JSON, using fallback")
                return _get_sample_matches_data()
        else:
            if RATE_LIMITING_ENABLED and response.status_code == 429:
                _rate_limiter.handle_rate_limit_error(429)
            print(f"API returned status {response.status_code}, using fallback")
            return _get_sample_matches_data()

    except requests.exceptions.Timeout:
        print("API request timed out, using fallback data")
        return _get_sample_matches_data()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching upcoming matches: {e}")
        return _get_sample_matches_data()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return _get_sample_matches_data()

def fetch_live_matches():
    """
    Fetches live matches from Cricbuzz API
    """
    if _missing_key():
        print("⚠️ RAPIDAPI_KEY not set; returning error for live matches")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/matches/v1/live"
        response = requests.get(url, headers=API_HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live matches: {e}")
        return {"error": str(e)}

def fetch_recent_matches():
    """
    Fetches recent/completed matches from Cricbuzz API with improved error handling
    """
    if _missing_key():
        print("⚠️ RAPIDAPI_KEY not set; returning fallback recent matches data")
        return _get_sample_recent_matches_data()
    try:
        url = f"{API_BASE_URL}/matches/v1/recent"
        response = requests.get(url, headers=API_HEADERS, timeout=15)

        if response.status_code == 200:
            try:
                data = response.json()
                if data and isinstance(data, dict):
                    return data
                else:
                    print(f"Invalid recent matches data, using fallback")
                    return _get_sample_recent_matches_data()
            except json.JSONDecodeError:
                print(f"Invalid JSON in recent matches response, using fallback")
                return _get_sample_recent_matches_data()
        else:
            print(f"Recent matches API returned status {response.status_code}, using fallback")
            return _get_sample_recent_matches_data()

    except requests.exceptions.Timeout:
        print(f"Recent matches API request timed out, using fallback")
        return _get_sample_recent_matches_data()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recent matches: {e}")
        return _get_sample_recent_matches_data()
    except Exception as e:
        print(f"Unexpected error fetching recent matches: {e}")
        return _get_sample_recent_matches_data()

def _get_sample_recent_matches_data():
    """
    Returns sample recent matches data as fallback
    """
    return {
        "typeMatches": [
            {
                "matchType": "International",
                "seriesMatches": [
                    {
                        "seriesAdWrapper": {
                            "seriesId": 9408,
                            "seriesName": "Australia tour of West Indies, 2025",
                            "matches": [
                                {
                                    "matchInfo": {
                                        "matchId": 114627,
                                        "seriesId": 9408,
                                        "seriesName": "Australia tour of West Indies, 2025",
                                        "matchDesc": "5th T20I",
                                        "matchFormat": "T20",
                                        "startDate": "1753743600000",
                                        "endDate": "1753756200000",
                                        "state": "Complete",
                                        "status": "Australia won by 3 wkts",
                                        "team1": {
                                            "teamId": 10,
                                            "teamName": "West Indies",
                                            "teamSName": "WI"
                                        },
                                        "team2": {
                                            "teamId": 4,
                                            "teamName": "Australia",
                                            "teamSName": "AUS"
                                        },
                                        "venueInfo": {
                                            "id": 96,
                                            "ground": "Warner Park",
                                            "city": "Basseterre, St Kitts"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

def _get_sample_matches_data():
    """
    Returns sample match data as fallback when API is unavailable
    """
    return {
        "typeMatches": [
            {
                "matchType": "International",
                "seriesMatches": [
                    {
                        "seriesAdWrapper": {
                            "seriesId": 6732,
                            "seriesName": "Australia vs India T20 Series 2024",
                            "matches": [
                                {
                                    "matchInfo": {
                                        "matchId": 74648,
                                        "seriesId": 6732,
                                        "team1": {
                                            "teamId": 2,
                                            "teamName": "India",
                                            "teamSName": "IND"
                                        },
                                        "team2": {
                                            "teamId": 1,
                                            "teamName": "Australia",
                                            "teamSName": "AUS"
                                        },
                                        "venueInfo": {
                                            "id": 25,
                                            "ground": "Melbourne Cricket Ground",
                                            "city": "Melbourne"
                                        },
                                        "startDate": "2024-01-26T14:30:00.000Z",
                                        "matchFormat": "T20I"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

def fetch_squads(series_id):
    """
    Fetches squad information for a series from Cricbuzz API with rate limiting and caching
    """
    if RATE_LIMITING_ENABLED:
        # Check cache first
        cache_key = f"series_squads_{series_id}"
        cached_data = _cache.get(cache_key)
        if cached_data:
            print(f"🎯 Cache hit: squads for series {series_id}")
            return cached_data

        # Check rate limit
        if not _rate_limiter.acquire_request_slot("normal"):
            print(f"🚫 Rate limited: squads for series {series_id}")
            return _get_sample_squads_data(series_id)

    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning fallback squads for series {series_id}")
        return _get_sample_squads_data(series_id)

    try:
        url = f"{API_BASE_URL}/series/v1/{series_id}/squads"
        response = requests.get(url, headers=API_HEADERS, timeout=15)

        if RATE_LIMITING_ENABLED:
            _rate_limiter.handle_api_response(dict(response.headers))

        if response.status_code == 200:
            try:
                data = response.json()
                if data and isinstance(data, dict):
                    # Cache squad data for 4 hours
                    if RATE_LIMITING_ENABLED:
                        _cache.set(cache_key, data, ttl=14400, tags=['squads', 'team_data'])  # 4 hours
                        print(f"💾 Cached squads for series {series_id}")
                    return data
                else:
                    print(f"Invalid squad data for series {series_id}, using fallback")
                    return _get_sample_squads_data(series_id)
            except json.JSONDecodeError:
                print(f"Invalid JSON in squads response for series {series_id}, using fallback")
                return _get_sample_squads_data(series_id)
        else:
            if RATE_LIMITING_ENABLED and response.status_code == 429:
                _rate_limiter.handle_rate_limit_error(429)
            print(f"Squad API returned status {response.status_code} for series {series_id}, using fallback")
            return _get_sample_squads_data(series_id)

    except requests.exceptions.Timeout:
        print(f"Squad API request timed out for series {series_id}, using fallback")
        return _get_sample_squads_data(series_id)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching squads for series {series_id}: {e}")
        return _get_sample_squads_data(series_id)
    except Exception as e:
        print(f"Unexpected error fetching squads for series {series_id}: {e}")
        return _get_sample_squads_data(series_id)

def fetch_team_squad(series_id, team_id):
    """
    Fetches specific team squad for a series from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for team squad {team_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/series/v1/{series_id}/squads/{team_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching squad for team {team_id} in series {series_id}: {e}")
        return {"error": str(e)}

def _get_sample_squads_data(series_id):
    """
    Returns sample squad data as fallback
    """
    return {
        "squads": [
            {
                "squadId": 15826,
                "squadType": "squad",
                "teamId": 2,
                "teamName": "India",
                "players": [
                    {"id": 1001, "name": "Virat Kohli", "role": "Batsman"},
                    {"id": 1002, "name": "Rohit Sharma", "role": "Batsman"},
                    {"id": 1003, "name": "Jasprit Bumrah", "role": "Bowler"},
                    {"id": 1004, "name": "Hardik Pandya", "role": "All-rounder"},
                    {"id": 1005, "name": "KL Rahul", "role": "Wicket-keeper"}
                ]
            }
        ]
    }

def fetch_player_stats(player_id):
    """
    Fetches player statistics from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning fallback player stats for {player_id}")
        return _get_sample_player_stats(player_id)
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for player {player_id}: {e}")
        return _get_sample_player_stats(player_id)

def fetch_player_career_stats(player_id):
    """
    Fetches player career statistics from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for player career stats {player_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}/career"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching career stats for player {player_id}: {e}")
        return {"error": str(e)}

def fetch_player_batting_stats(player_id):
    """
    Fetches player batting statistics from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for player batting stats {player_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}/batting"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching batting stats for player {player_id}: {e}")
        return {"error": str(e)}

def fetch_player_bowling_stats(player_id):
    """
    Fetches player bowling statistics from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for player bowling stats {player_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}/bowling"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching bowling stats for player {player_id}: {e}")
        return {"error": str(e)}

def _get_sample_player_stats(player_id):
    """
    Returns sample player stats as fallback
    """
    return {
        "playerId": player_id,
        "playerName": "Sample Player",
        "recentMatches": [
            {"matchId": "74640", "runs": 85, "balls": 58, "fours": 8, "sixes": 2},
            {"matchId": "74641", "runs": 45, "balls": 32, "fours": 4, "sixes": 1}
        ],
        "careerStats": {
            "matches": 150,
            "runs": 6500,
            "average": 45.5,
            "strikeRate": 88.5
        }
    }

def fetch_venue_stats(venue_id):
    """
    Fetches venue statistics from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning fallback venue stats for {venue_id}")
        return _get_sample_venue_stats(venue_id)
    try:
        url = f"{API_BASE_URL}/stats/v1/venue/{venue_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for venue {venue_id}: {e}")
        return _get_sample_venue_stats(venue_id)

def fetch_venue_info(venue_id):
    """
    Fetches venue information from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for venue info {venue_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/venues/v1/{venue_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching info for venue {venue_id}: {e}")
        return {"error": str(e)}

def fetch_venue_matches(venue_id):
    """
    Fetches matches at a venue from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for venue matches {venue_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/venues/v1/{venue_id}/matches"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches for venue {venue_id}: {e}")
        return {"error": str(e)}

def fetch_match_center(match_id):
    """
    Fetches match center information from Cricbuzz API with optimization
    """
    if RATE_LIMITING_ENABLED:
        # Check cache first
        cached_data = _cache.get_match_data(str(match_id))
        if cached_data:
            print(f"🎯 Cache hit: match center {match_id}")
            return cached_data

        # Check rate limit
        if not _rate_limiter.acquire_request_slot("high"):
            print(f"🚫 Rate limited: match center {match_id}")
            return {"error": "Rate limited", "fallback": True}

    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for match center {match_id}")
        return {"error": "API key missing"}

    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)

        if RATE_LIMITING_ENABLED:
            _rate_limiter.handle_api_response(dict(response.headers))

        if response.status_code == 200:
            data = response.json()
            # Cache match data (30 min for live, 2 hours for completed)
            if RATE_LIMITING_ENABLED:
                is_live = data.get('matchHeader', {}).get('state', '').lower() in ['live', 'in progress']
                _cache.cache_match_data(str(match_id), data, live=is_live)
                print(f"💾 Cached match center {match_id}")
            return data
        elif response.status_code == 429 and RATE_LIMITING_ENABLED:
            _rate_limiter.handle_rate_limit_error(429)
            return {"error": "Rate limited", "status_code": 429}
        else:
            response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching match center for match {match_id}: {e}")
        return {"error": str(e)}

def fetch_match_scorecard(match_id):
    """
    Fetches match scorecard from Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for scorecard {match_id}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}/scard"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching scorecard for match {match_id}: {e}")
        return {"error": str(e)}

def search_player(player_name):
    """
    Searches for a player by name using Cricbuzz API
    """
    if _missing_key():
        print(f"⚠️ RAPIDAPI_KEY not set; returning error for player search {player_name}")
        return {"error": "API key missing"}
    try:
        url = f"{API_BASE_URL}/stats/v1/player/search?plrN={player_name}"
        response = requests.get(url, headers=API_HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error searching for player {player_name}: {e}")
        return {"error": str(e)}

def _get_sample_venue_stats(venue_id):
    """
    Returns sample venue stats as fallback
    """
    return {
        "venueId": venue_id,
        "venueName": "Sample Venue",
        "averageScore": 165,
        "wicketTendency": "Balanced",
        "recentMatches": [
            {"matchId": "74635", "team1Score": 178, "team2Score": 162},
            {"matchId": "74636", "team1Score": 155, "team2Score": 148}
        ],
        "conditions": {
            "pitch": "Good for batting",
            "weather": "Clear",
            "temperature": "22°C"
        }
    }

# API Optimization Monitoring Functions
def get_api_optimization_status():
    """Get current API optimization status and metrics"""
    if not RATE_LIMITING_ENABLED:
        return {
            "optimization_enabled": False,
            "message": "API optimization not available - install aiohttp and restart"
        }

    rate_limit_status = _rate_limiter.get_status()
    cache_stats = _cache.get_stats()

    return {
        "optimization_enabled": True,
        "rate_limiting": {
            "can_make_request": rate_limit_status['can_make_request'],
            "tokens_available": rate_limit_status['tokens_available'],
            "requests_this_minute": rate_limit_status['requests_this_minute'],
            "requests_this_hour": rate_limit_status['requests_this_hour'],
            "requests_today": rate_limit_status['requests_today'],
            "wait_time_seconds": rate_limit_status['wait_time_seconds'],
            "total_requests_made": rate_limit_status['total_requests_made']
        },
        "caching": {
            "hit_rate_percent": cache_stats['hit_rate_percent'],
            "memory_entries": cache_stats['memory_entries'],
            "disk_entries": cache_stats['disk_entries'],
            "memory_usage_mb": cache_stats['memory_usage_mb'],
            "memory_hits": cache_stats['memory_hits'],
            "disk_hits": cache_stats['disk_hits'],
            "misses": cache_stats['misses']
        },
        "performance_summary": _calculate_performance_summary(rate_limit_status, cache_stats)
    }

def _calculate_performance_summary(rate_status, cache_stats):
    """Calculate performance improvement summary"""
    total_requests = cache_stats['memory_hits'] + cache_stats['disk_hits'] + cache_stats['misses']

    if total_requests == 0:
        return {
            "api_calls_saved": 0,
            "cost_savings_percent": 0,
            "performance_improvement": "No data yet"
        }

    api_calls_saved = cache_stats['memory_hits'] + cache_stats['disk_hits']
    cost_savings_percent = (api_calls_saved / total_requests) * 100 if total_requests > 0 else 0

    # Estimate monthly cost savings
    monthly_api_calls_saved = api_calls_saved * 30  # Rough estimate
    cost_per_1000 = 5.0  # $5 per 1000 requests typical
    monthly_savings = (monthly_api_calls_saved / 1000) * cost_per_1000

    return {
        "api_calls_saved": api_calls_saved,
        "cost_savings_percent": round(cost_savings_percent, 1),
        "estimated_monthly_savings_usd": round(monthly_savings, 2),
        "performance_improvement": f"{cache_stats['hit_rate_percent']:.1f}% faster responses"
    }

def print_optimization_report():
    """Print a detailed optimization report"""
    status = get_api_optimization_status()

    if not status["optimization_enabled"]:
        print("❌ API Optimization Status: DISABLED")
        print(f"   {status['message']}")
        return

    rate_limit = status["rate_limiting"]
    cache = status["caching"]
    perf = status["performance_summary"]

    print("🚀 API OPTIMIZATION STATUS REPORT")
    print("=" * 50)
    print(f"✅ Optimization: ENABLED")
    print()
    print("🛡️ RATE LIMITING:")
    print(f"   Status: {'✅ Available' if rate_limit['can_make_request'] else '🚫 Limited'}")
    print(f"   Tokens available: {rate_limit['tokens_available']:.1f}")
    print(f"   Requests today: {rate_limit['requests_today']}")
    print(f"   This hour: {rate_limit['requests_this_hour']}")
    print(f"   This minute: {rate_limit['requests_this_minute']}")
    if rate_limit['wait_time_seconds'] > 0:
        print(f"   ⏰ Wait time: {rate_limit['wait_time_seconds']:.1f}s")
    print()
    print("⚡ SMART CACHING:")
    print(f"   Hit rate: {cache['hit_rate_percent']:.1f}%")
    print(f"   Memory cache: {cache['memory_entries']} entries ({cache['memory_usage_mb']:.2f}MB)")
    print(f"   Disk cache: {cache['disk_entries']} entries")
    print(f"   Cache hits: {cache['memory_hits']} memory + {cache['disk_hits']} disk")
    print(f"   Cache misses: {cache['misses']}")
    print()
    print("💰 PERFORMANCE GAINS:")
    print(f"   API calls saved: {perf['api_calls_saved']}")
    print(f"   Cost reduction: {perf['cost_savings_percent']:.1f}%")
    print(f"   Monthly savings: ${perf.get('estimated_monthly_savings_usd', 0):.2f}")
    print(f"   Speed improvement: {perf.get('performance_improvement', 'No data yet')}")
    print("=" * 50)

def clear_cache():
    """Clear all cached data"""
    if RATE_LIMITING_ENABLED:
        _cache.clear_expired()
        print("🧹 Cleared expired cache entries")
    else:
        print("❌ Cache not available")

def invalidate_live_data():
    """Invalidate all live/dynamic data that should be refreshed"""
    if RATE_LIMITING_ENABLED:
        _cache.invalidate_live_data()
        print("🔄 Invalidated live data cache")
    else:
        print("❌ Cache not available")
