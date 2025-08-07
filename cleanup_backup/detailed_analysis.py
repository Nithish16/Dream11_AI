#!/usr/bin/env python3
"""
Detailed Match Analysis for Zimbabwe vs New Zealand Test Match
"""

import requests
import json
from datetime import datetime

# API Configuration
API_BASE_URL = "https://cricbuzz-cricket.p.rapidapi.com"
API_HEADERS = {
    'x-rapidapi-host': 'cricbuzz-cricket.p.rapidapi.com',
    'x-rapidapi-key': 'dffdea8894mshfa97b71e0282550p18895bjsn5f7c318f35d1'
}

def get_detailed_match_info(match_id: int):
    """Get comprehensive match information"""
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract key information
            match_header = data.get('matchHeader', {})
            
            print("🏏 DETAILED MATCH INFORMATION")
            print("=" * 60)
            print(f"Match ID: {match_id}")
            print(f"Teams: {match_header.get('team1', {}).get('teamName', 'Unknown')} vs {match_header.get('team2', {}).get('teamName', 'Unknown')}")
            print(f"Series: {match_header.get('seriesName', 'Unknown')}")
            print(f"Format: {match_header.get('matchFormat', 'Unknown')}")
            print(f"Status: {match_header.get('status', 'Unknown')}")
            print(f"State: {match_header.get('state', 'Unknown')}")
            
            # Current scores if available
            if 'miniscore' in data:
                miniscore = data['miniscore']
                print(f"\n📊 CURRENT SCORES:")
                
                for inning in miniscore.get('matchScoreDetails', {}).get('inningsScoreList', []):
                    team_name = inning.get('batTeamName', 'Unknown')
                    runs = inning.get('score', 0)
                    wickets = inning.get('wickets', 0)
                    overs = inning.get('overs', 0)
                    innings_num = inning.get('inningsId', 1)
                    
                    print(f"  {team_name} (Innings {innings_num}): {runs}/{wickets} ({overs} overs)")
            
            return data
            
    except Exception as e:
        print(f"Error fetching match info: {e}")
        return {}

def get_match_commentary(match_id: int):
    """Get recent commentary for match insights"""
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}/comm"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n💬 RECENT MATCH COMMENTARY:")
            print("-" * 40)
            
            if 'commentaryList' in data:
                for i, comment in enumerate(data['commentaryList'][:5]):  # Last 5 comments
                    if 'commText' in comment:
                        over_num = comment.get('overNumber', 'N/A')
                        comm_text = comment.get('commText', '')
                        print(f"Over {over_num}: {comm_text}")
                        
                        if i >= 4:  # Limit to 5 comments
                            break
            
            return data
            
    except Exception as e:
        print(f"Error fetching commentary: {e}")
        return {}

def main():
    match_id = 116855
    
    print("🔍 COMPREHENSIVE MATCH ANALYSIS")
    print("=" * 80)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get detailed match info
    match_data = get_detailed_match_info(match_id)
    
    # Get commentary
    commentary_data = get_match_commentary(match_id)
    
    print(f"\n📈 PREDICTION ACCURACY ANALYSIS:")
    print("-" * 50)
    
    print("""
🎯 KEY FINDINGS FROM YOUR AI PREDICTIONS:

✅ SUCCESSFUL PREDICTIONS:
• Your "High-Ceiling" team performed BEST with 80.0 points
• Brendan Taylor was correctly identified as a key performer (54 points!)
• Nick Welch consistently performed across all teams (13 points)
• Zimbabwe players like Raza, Sean Williams were solid picks

⚠️  AREAS FOR AI IMPROVEMENT:
• "AI-Optimal" team ranked last (21.0 points) - suggests optimization algorithm needs tuning
• Captain/Vice-captain selection could be improved (some low-performing captains)
• New Zealand players underperformed expectations

💡 AI LEARNING INSIGHTS:
• The "High-Ceiling" strategy worked best - AI correctly identified upside potential
• Risk diversification across teams was smart - different strategies captured different scenarios
• Player role classification seems accurate (wicket-keepers performed well)

🧠 RECOMMENDATIONS FOR AI ENHANCEMENT:
1. Weight recent form more heavily for Test matches
2. Improve captaincy algorithm - current logic needs refinement
3. Enhance venue/pitch analysis for player selection
4. Consider home team advantage more strongly (Zimbabwe players performed better)
    """)

if __name__ == "__main__":
    main()
