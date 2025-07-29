from datetime import datetime
from utils.api_client import fetch_upcoming_matches, fetch_match_center, fetch_recent_matches

def resolve_match_by_id(match_id):
    """
    Resolves match details directly from match ID using match center API.
    
    Args:
        match_id (int): The match ID to fetch details for
        
    Returns:
        dict: Match details if found, None otherwise
    """
    try:
        print(f"🔍 Fetching match details for Match ID: {match_id}")
        
        # First try: Search in upcoming matches (has complete seriesId info)
        print(f"🔍 Searching for Match ID {match_id} in upcoming matches...")
        upcoming_matches = fetch_upcoming_matches()
        
        if upcoming_matches and not upcoming_matches.get('error'):
            found_match = search_match_in_data(match_id, upcoming_matches)
            if found_match:
                print(f"✅ Found match in upcoming: {found_match['team1Name']} vs {found_match['team2Name']}")
                return found_match
        
        # Second try: Search in recent/completed matches
        print(f"🔍 Searching for Match ID {match_id} in recent matches...")
        recent_matches = fetch_recent_matches()
        
        if recent_matches and not recent_matches.get('error'):
            found_match = search_match_in_data(match_id, recent_matches)
            if found_match:
                print(f"✅ Found match in recent: {found_match['team1Name']} vs {found_match['team2Name']}")
                return found_match
        
        # Third try: Direct match center API (fallback for special matches like Champions League)
        match_data = fetch_match_center(match_id)
        
        if match_data and not match_data.get('error'):
            # Extract match info from match center response
            match_info = match_data.get('matchInfo', {})
            if match_info:
                # Extract team information - handle both regular and Champions League formats
                team1_info = match_info.get('team1', {})
                team2_info = match_info.get('team2', {})
                
                # Handle different team data structures
                # Regular format: teamId, teamName
                # Champions League format: id, name
                team1_id = team1_info.get('teamId') or team1_info.get('id')
                team2_id = team2_info.get('teamId') or team2_info.get('id')
                team1_name = team1_info.get('teamName') or team1_info.get('name')
                team2_name = team2_info.get('teamName') or team2_info.get('name')
                
                # Extract venue information - handle both formats
                venue_info = match_info.get('venueInfo', {}) or match_info.get('venue', {})
                venue_id = venue_info.get('id')
                venue_name = venue_info.get('ground') or venue_info.get('name', 'Unknown Venue')
                venue_city = venue_info.get('city', 'Unknown City')
                
                # Extract series information - handle both formats
                series_info = match_info.get('series', {})
                series_id = match_info.get('seriesId') or series_info.get('id')
                series_name = match_info.get('seriesName') or series_info.get('name', 'Unknown Series')
                
                # Extract date information - handle both formats
                match_date = (match_info.get('startDate') or 
                            match_info.get('matchStartTimestamp') or 
                            match_info.get('date'))
                
                # Build resolved match data structure
                resolved_match = {
                    'matchId': match_id,
                    'seriesId': series_id,
                    'team1Id': team1_id,
                    'team2Id': team2_id,
                    'team1Name': team1_name,
                    'team2Name': team2_name,
                    'venueId': venue_id,
                    'venue': venue_name,
                    'city': venue_city,
                    'matchFormat': match_info.get('matchFormat', 'T20I'),
                    'seriesName': series_name,
                    'matchDate': match_date,
                    'matchState': match_info.get('state', 'Unknown')
                }
                
                # For Champions League matches, seriesId might not be present, so validate differently
                essential_fields_present = all([team1_id, team2_id, team1_name, team2_name])
                if essential_fields_present:
                    print(f"✅ Successfully resolved match: {resolved_match['team1Name']} vs {resolved_match['team2Name']}")
                    return resolved_match
        
        
        print(f"❌ Could not resolve Match ID {match_id} from any source")
        return None
        
    except Exception as e:
        print(f"❌ Error resolving match ID {match_id}: {e}")
        return None

def search_match_in_data(target_match_id, matches_data):
    """
    Search for a match ID in matches data (works for both upcoming and recent)
    """
    try:
        target_id = int(target_match_id)
        
        # Navigate through the matches data structure (works for both upcoming and recent)
        if 'typeMatches' in matches_data:
            for type_match in matches_data['typeMatches']:
                if 'seriesMatches' in type_match:
                    for series_match in type_match['seriesMatches']:
                        if 'seriesAdWrapper' in series_match and 'matches' in series_match['seriesAdWrapper']:
                            for match in series_match['seriesAdWrapper']['matches']:
                                if 'matchInfo' in match:
                                    match_info = match['matchInfo']
                                    if match_info.get('matchId') == target_id:
                                        # Found the match! Extract details
                                        team1_info = match_info.get('team1', {})
                                        team2_info = match_info.get('team2', {})
                                        venue_info = match_info.get('venueInfo', {})
                                        
                                        return {
                                            'matchId': target_id,
                                            'seriesId': match_info.get('seriesId'),
                                            'team1Id': team1_info.get('teamId'),
                                            'team2Id': team2_info.get('teamId'),
                                            'team1Name': team1_info.get('teamName'),
                                            'team2Name': team2_info.get('teamName'),
                                            'venueId': venue_info.get('id'),
                                            'venue': venue_info.get('ground', 'Unknown Venue'),
                                            'city': venue_info.get('city', 'Unknown City'),
                                            'matchFormat': match_info.get('matchFormat', 'T20I'),
                                            'seriesName': series_match['seriesAdWrapper'].get('seriesName', 'Unknown Series'),
                                            'matchDate': match_info.get('startDate'),
                                            'matchState': match_info.get('state', 'Unknown')
                                        }
    except Exception as e:
        print(f"Error searching in upcoming matches: {e}")
    
    return None

def resolve_match_ids(team_a, team_b, match_date):
    """
    Resolves match IDs based on team names and match date.
    
    Args:
        team_a (str): First team name
        team_b (str): Second team name  
        match_date (str): Match date in YYYY-MM-DD format
        
    Returns:
        dict: Match details if found, None otherwise
    """
    # Normalize inputs for flexible matching
    team_a_lower = team_a.lower().strip()
    team_b_lower = team_b.lower().strip()
    
    try:
        target_date = datetime.strptime(match_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {match_date}. Expected YYYY-MM-DD")
        return None
    
    # Fetch upcoming matches
    matches_data = fetch_upcoming_matches()
    
    if not matches_data:
        print("No matches data available")
        return None
    
    # Handle both real API response and fallback data structures
    matches_list = []
    
    if 'typeMatches' in matches_data:
        # Real API response structure
        for type_match in matches_data['typeMatches']:
            if 'seriesMatches' in type_match:
                for series_match in type_match['seriesMatches']:
                    if 'seriesAdWrapper' in series_match and 'matches' in series_match['seriesAdWrapper']:
                        for match in series_match['seriesAdWrapper']['matches']:
                            if 'matchInfo' in match:
                                matches_list.append(match['matchInfo'])
    elif 'matches' in matches_data:
        # Fallback structure
        matches_list = matches_data['matches']
    else:
        print("Unknown matches data structure")
        return None
    
    for match in matches_list:
        try:
            # Parse match date - handle both API formats
            start_date = match.get('startDate') or match.get('date')
            if not start_date:
                continue
            
            # Handle Unix timestamp (milliseconds) format from real API
            if isinstance(start_date, str) and start_date.isdigit():
                # Convert milliseconds to seconds
                timestamp = int(start_date) / 1000
                match_datetime = datetime.fromtimestamp(timestamp)
            elif isinstance(start_date, int):
                # Already in milliseconds
                timestamp = start_date / 1000
                match_datetime = datetime.fromtimestamp(timestamp)
            else:
                # Handle ISO format string (fallback data)
                match_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            
            match_date_only = match_datetime.date()
            
            # Get team names for comparison - handle both API formats
            team1_info = match.get('team1', {})
            team2_info = match.get('team2', {})
            
            team1_name = team1_info.get('teamName', '').lower().strip()
            team1_short = team1_info.get('teamSName', '').lower().strip()
            team2_name = team2_info.get('teamName', '').lower().strip() 
            team2_short = team2_info.get('teamSName', '').lower().strip()
            
            if not all([team1_name, team2_name]):
                continue
                
            # Check if teams match (allow for full names or short names)
            team_a_match = (team_a_lower in [team1_name, team1_short, team2_name, team2_short])
            team_b_match = (team_b_lower in [team1_name, team1_short, team2_name, team2_short])
            
            # Also check if team names contain the input (partial matching)
            if not team_a_match:
                team_a_match = (team_a_lower in team1_name or team_a_lower in team2_name or
                              team1_name in team_a_lower or team2_name in team_a_lower)
            
            if not team_b_match:
                team_b_match = (team_b_lower in team1_name or team_b_lower in team2_name or
                              team1_name in team_b_lower or team2_name in team_b_lower)
            
            # Check if date matches and both teams are found
            if match_date_only == target_date and team_a_match and team_b_match:
                # Handle different venue formats
                venue_info = match.get('venueInfo', {})
                venue_name = venue_info.get('ground') or match.get('venue', 'Unknown Venue')
                venue_id = venue_info.get('id') or match.get('venueId', 0)
                
                return {
                    'matchId': match.get('matchId'),
                    'seriesId': match.get('seriesId'), 
                    'team1Id': team1_info.get('teamId'),
                    'team2Id': team2_info.get('teamId'),
                    'venueId': venue_id,
                    'matchFormat': match.get('matchFormat') or match.get('format', 'Unknown'),
                    'team1Name': team1_info.get('teamName'),
                    'team2Name': team2_info.get('teamName'),
                    'venue': venue_name,
                    'seriesName': match.get('seriesName', 'Unknown Series')
                }
        except Exception as e:
            print(f"Error processing match: {e}")
            continue
    
    print(f"No match found for teams '{team_a}' vs '{team_b}' on {match_date}")
    return None

def get_match_summary(match_info):
    """
    Returns a formatted summary of match information with safe key access
    """
    if not match_info:
        return "No match information available"
    
    # Safe access to all fields with defaults
    match_id = match_info.get('matchId', 'Unknown')
    team1_name = match_info.get('team1Name', 'Team 1')
    team2_name = match_info.get('team2Name', 'Team 2')
    match_format = match_info.get('matchFormat', match_info.get('format', 'TEST'))  # Try both keys
    venue = match_info.get('venue', match_info.get('venueName', 'Unknown Venue'))
    series_name = match_info.get('seriesName', 'Unknown Series')
    series_id = match_info.get('seriesId', 'Unknown')
    match_date = match_info.get('matchDate', match_info.get('date', 'TBD'))
    
    return f"""
Match Found:
- Match ID: {match_id}
- Teams: {team1_name} vs {team2_name}
- Format: {match_format}
- Venue: {venue}
- Date: {match_date}
- Series: {series_name}
- Series ID: {series_id}
"""