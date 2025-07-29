import requests
import json
from datetime import datetime, timedelta

# Cricbuzz API Configuration
API_BASE_URL = "https://cricbuzz-cricket.p.rapidapi.com"
API_HEADERS = {
    'x-rapidapi-host': 'cricbuzz-cricket.p.rapidapi.com',
    'x-rapidapi-key': 'dffdea8894mshfa97b71e0282550p18895bjsn5f7c318f35d1'
}

def fetch_upcoming_matches():
    """
    Fetches upcoming matches from Cricbuzz API with improved error handling
    """
    try:
        url = f"{API_BASE_URL}/matches/v1/upcoming"
        response = requests.get(url, headers=API_HEADERS, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data and len(str(data)) > 10:  # Valid response check
                    return data
                else:
                    print("API returned empty/invalid data, using fallback")
                    return _get_sample_matches_data()
            except json.JSONDecodeError:
                print("API returned invalid JSON, using fallback")
                return _get_sample_matches_data()
        else:
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
    try:
        url = f"{API_BASE_URL}/matches/v1/live"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live matches: {e}")
        return {"error": str(e)}

def fetch_recent_matches():
    """
    Fetches recent/completed matches from Cricbuzz API with improved error handling
    """
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
    Fetches squad information for a series from Cricbuzz API with improved error handling
    """
    try:
        url = f"{API_BASE_URL}/series/v1/{series_id}/squads"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data and isinstance(data, dict):
                    return data
                else:
                    print(f"Invalid squad data for series {series_id}, using fallback")
                    return _get_sample_squads_data(series_id)
            except json.JSONDecodeError:
                print(f"Invalid JSON in squads response for series {series_id}, using fallback")
                return _get_sample_squads_data(series_id)
        else:
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
    try:
        url = f"{API_BASE_URL}/series/v1/{series_id}/squads/{team_id}"
        response = requests.get(url, headers=API_HEADERS)
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
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for player {player_id}: {e}")
        return _get_sample_player_stats(player_id)

def fetch_player_career_stats(player_id):
    """
    Fetches player career statistics from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}/career"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching career stats for player {player_id}: {e}")
        return {"error": str(e)}

def fetch_player_batting_stats(player_id):
    """
    Fetches player batting statistics from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}/batting"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching batting stats for player {player_id}: {e}")
        return {"error": str(e)}

def fetch_player_bowling_stats(player_id):
    """
    Fetches player bowling statistics from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/stats/v1/player/{player_id}/bowling"
        response = requests.get(url, headers=API_HEADERS)
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
    try:
        url = f"{API_BASE_URL}/stats/v1/venue/{venue_id}"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for venue {venue_id}: {e}")
        return _get_sample_venue_stats(venue_id)

def fetch_venue_info(venue_id):
    """
    Fetches venue information from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/venues/v1/{venue_id}"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching info for venue {venue_id}: {e}")
        return {"error": str(e)}

def fetch_venue_matches(venue_id):
    """
    Fetches matches at a venue from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/venues/v1/{venue_id}/matches"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches for venue {venue_id}: {e}")
        return {"error": str(e)}

def fetch_match_center(match_id):
    """
    Fetches match center information from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching match center for match {match_id}: {e}")
        return {"error": str(e)}

def fetch_match_scorecard(match_id):
    """
    Fetches match scorecard from Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}/scard"
        response = requests.get(url, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching scorecard for match {match_id}: {e}")
        return {"error": str(e)}

def search_player(player_name):
    """
    Searches for a player by name using Cricbuzz API
    """
    try:
        url = f"{API_BASE_URL}/stats/v1/player/search?plrN={player_name}"
        response = requests.get(url, headers=API_HEADERS)
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