#!/usr/bin/env python3
"""
Match Performance Analysis Script
Analyzes how the predicted teams performed against actual match results
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# API Configuration
API_BASE_URL = "https://cricbuzz-cricket.p.rapidapi.com"
API_HEADERS = {
    'x-rapidapi-host': 'cricbuzz-cricket.p.rapidapi.com',
    'x-rapidapi-key': 'dffdea8894mshfa97b71e0282550p18895bjsn5f7c318f35d1'
}

def fetch_match_scorecard(match_id: int) -> Dict[str, Any]:
    """Fetch detailed scorecard for a match"""
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}/scard"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error fetching scorecard: HTTP {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error fetching scorecard: {e}")
        return {}

def fetch_match_center(match_id: int) -> Dict[str, Any]:
    """Fetch match center data for current status"""
    try:
        url = f"{API_BASE_URL}/mcenter/v1/{match_id}"
        response = requests.get(url, headers=API_HEADERS, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error fetching match center: HTTP {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error fetching match center: {e}")
        return {}

def analyze_player_performance(scorecard: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract player performance data from scorecard"""
    player_performances = {}
    
    if not scorecard or 'scoreCard' not in scorecard:
        return player_performances
    
    # Extract batting and bowling performances
    for innings_data in scorecard.get('scoreCard', []):
        if 'batTeamDetails' in innings_data:
            bat_team = innings_data['batTeamDetails']
            
            # Batting performance
            for batsman_data in bat_team.get('batsmenData', {}).values():
                if isinstance(batsman_data, dict) and 'batName' in batsman_data:
                    player_name = batsman_data['batName']
                    player_performances[player_name] = {
                        'batting': {
                            'runs': batsman_data.get('runs', 0),
                            'balls': batsman_data.get('balls', 0),
                            'fours': batsman_data.get('fours', 0),
                            'sixes': batsman_data.get('sixes', 0),
                            'strike_rate': batsman_data.get('strikeRate', 0),
                            'dismissal': batsman_data.get('outDesc', 'Not Out')
                        },
                        'bowling': {},
                        'fielding': {}
                    }
        
        if 'bowlTeamDetails' in innings_data:
            bowl_team = innings_data['bowlTeamDetails']
            
            # Bowling performance
            for bowler_data in bowl_team.get('bowlersData', {}).values():
                if isinstance(bowler_data, dict) and 'bowlName' in bowler_data:
                    player_name = bowler_data['bowlName']
                    if player_name not in player_performances:
                        player_performances[player_name] = {'batting': {}, 'bowling': {}, 'fielding': {}}
                    
                    player_performances[player_name]['bowling'] = {
                        'overs': bowler_data.get('overs', 0),
                        'maidens': bowler_data.get('maidens', 0),
                        'runs': bowler_data.get('runs', 0),
                        'wickets': bowler_data.get('wickets', 0),
                        'economy': bowler_data.get('economy', 0),
                        'wides': bowler_data.get('wides', 0),
                        'noballs': bowler_data.get('noballs', 0)
                    }
    
    return player_performances

def calculate_dream11_points(player_name: str, performance: Dict[str, Any]) -> float:
    """Calculate Dream11 points based on player performance"""
    points = 0.0
    
    # Batting points
    batting = performance.get('batting', {})
    if batting:
        runs = batting.get('runs', 0)
        balls = batting.get('balls', 0)
        fours = batting.get('fours', 0)
        sixes = batting.get('sixes', 0)
        
        # Base points for runs
        points += runs
        
        # Boundary bonus
        points += fours * 1  # 1 point per four
        points += sixes * 2  # 2 points per six
        
        # Strike rate bonus (for Test cricket, different criteria)
        if balls > 20:  # Minimum balls faced
            sr = (runs / balls) * 100 if balls > 0 else 0
            if runs >= 50:
                if sr >= 100:
                    points += 8  # Fast fifty bonus
                elif sr >= 80:
                    points += 4
                elif sr >= 60:
                    points += 2
        
        # Milestone bonuses
        if runs >= 100:
            points += 16  # Century
        elif runs >= 50:
            points += 8   # Half century
        elif runs >= 30:
            points += 4   # 30+ runs
        
        # Duck penalty
        if runs == 0 and batting.get('dismissal', '').lower() not in ['not out', 'retired']:
            points -= 2
    
    # Bowling points
    bowling = performance.get('bowling', {})
    if bowling:
        wickets = bowling.get('wickets', 0)
        overs = bowling.get('overs', 0)
        runs_conceded = bowling.get('runs', 0)
        maidens = bowling.get('maidens', 0)
        
        # Base points for wickets
        points += wickets * 25  # 25 points per wicket
        
        # Maiden over bonus
        points += maidens * 12  # 12 points per maiden
        
        # Economy rate bonus (Test cricket specific)
        if overs >= 5:  # Minimum overs bowled
            economy = runs_conceded / overs if overs > 0 else 0
            if economy <= 2.0:
                points += 12
            elif economy <= 2.5:
                points += 6
            elif economy <= 3.0:
                points += 4
        
        # Wicket milestone bonuses
        if wickets >= 5:
            points += 16  # 5-wicket haul
        elif wickets >= 4:
            points += 8
        elif wickets >= 3:
            points += 4
    
    # Fielding points (basic - catches, stumpings, run-outs)
    # This would need to be extracted separately from detailed scorecard
    
    return round(points, 1)

def analyze_team_performance(predicted_teams: List[Dict[str, Any]], 
                           player_performances: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how each predicted team performed"""
    team_analysis = {}
    
    for i, team in enumerate(predicted_teams, 1):
        team_name = team.get('strategy', f'Team {i}')
        players = team.get('players', [])
        captain = team.get('captain', '')
        vice_captain = team.get('vice_captain', '')
        
        total_points = 0.0
        player_points = {}
        players_found = 0
        
        for player in players:
            player_name = player.get('name', '')
            
            # Try to match player in performance data
            matched_performance = None
            for perf_name, perf_data in player_performances.items():
                if player_name.lower() in perf_name.lower() or perf_name.lower() in player_name.lower():
                    matched_performance = perf_data
                    players_found += 1
                    break
            
            if matched_performance:
                points = calculate_dream11_points(player_name, matched_performance)
                player_points[player_name] = {
                    'points': points,
                    'performance': matched_performance
                }
                
                # Captain gets 2x points, Vice-captain gets 1.5x
                if player_name == captain:
                    total_points += points * 2
                elif player_name == vice_captain:
                    total_points += points * 1.5
                else:
                    total_points += points
            else:
                player_points[player_name] = {
                    'points': 0,
                    'performance': None,
                    'note': 'Player not found in match data'
                }
        
        team_analysis[team_name] = {
            'total_points': round(total_points, 1),
            'players_found': players_found,
            'total_players': len(players),
            'captain': captain,
            'vice_captain': vice_captain,
            'player_breakdown': player_points
        }
    
    return team_analysis

def get_match_status(match_center: Dict[str, Any]) -> str:
    """Get current match status"""
    if not match_center:
        return "Unknown"
    
    match_header = match_center.get('matchHeader', {})
    state = match_header.get('state', 'Unknown')
    status = match_header.get('status', 'Unknown')
    
    return f"{state} - {status}"

def main():
    """Main analysis function"""
    match_id = 116855
    
    print("🏏 DREAM11 TEAM PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"📊 Analyzing Match ID: {match_id}")
    print(f"🕐 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Fetch current match data
    print("📡 Fetching current match status...")
    match_center = fetch_match_center(match_id)
    
    if match_center:
        status = get_match_status(match_center)
        print(f"📈 Match Status: {status}")
        
        # Extract team names
        match_header = match_center.get('matchHeader', {})
        team1 = match_header.get('team1', {}).get('teamName', 'Team 1')
        team2 = match_header.get('team2', {}).get('teamName', 'Team 2')
        print(f"🏏 Teams: {team1} vs {team2}")
    
    print("\n📊 Fetching detailed scorecard...")
    scorecard = fetch_match_scorecard(match_id)
    
    if not scorecard:
        print("❌ Could not fetch scorecard. Match might not have started or API issues.")
        return
    
    # Analyze player performances
    print("🔍 Analyzing player performances...")
    player_performances = analyze_player_performance(scorecard)
    
    if not player_performances:
        print("⚠️  No player performance data found. Match might not have significant play yet.")
        return
    
    print(f"✅ Found performance data for {len(player_performances)} players")
    
    # Sample predicted teams structure (this would come from your AI system)
    predicted_teams = [
        {
            'strategy': 'AI-Optimal',
            'captain': 'Raza',
            'vice_captain': 'Matt Henry',
            'players': [
                {'name': 'Conway'}, {'name': 'Henry Nicholls'}, {'name': 'Will Young'},
                {'name': 'Nick Welch'}, {'name': 'Matt Henry'}, {'name': 'Trevor Gwandu'},
                {'name': 'Matthew Fisher'}, {'name': 'Raza'}, {'name': 'Jacob Duffy'},
                {'name': 'Brian Bennett'}, {'name': 'Tom Blundell'}
            ]
        },
        {
            'strategy': 'Risk-Balanced',
            'captain': 'Daryl Mitchell',
            'vice_captain': 'Tanaka Chivanga',
            'players': [
                {'name': 'Conway'}, {'name': 'Nick Welch'}, {'name': 'Henry Nicholls'},
                {'name': 'Craig Ervine'}, {'name': 'Tanaka Chivanga'}, {'name': 'Matt Henry'},
                {'name': 'Muzarabani'}, {'name': 'Daryl Mitchell'}, {'name': 'Brian Bennett'},
                {'name': 'Tom Blundell'}, {'name': 'Jacob Duffy'}
            ]
        },
        {
            'strategy': 'High-Ceiling',
            'captain': 'Raza',
            'vice_captain': 'Santner',
            'players': [
                {'name': 'Brendan Taylor'}, {'name': 'Will Young'}, {'name': 'Craig Ervine'},
                {'name': 'Nick Welch'}, {'name': 'Muzarabani'}, {'name': 'Matt Henry'},
                {'name': 'Matthew Fisher'}, {'name': 'Raza'}, {'name': 'Rachin Ravindra'},
                {'name': 'Santner'}, {'name': 'Brian Bennett'}
            ]
        },
        {
            'strategy': 'Value-Optimal',
            'captain': 'Raza',
            'vice_captain': 'Zakary Foulkes',
            'players': [
                {'name': 'Tafadzwa Tsiga'}, {'name': 'Henry Nicholls'}, {'name': 'Will Young'},
                {'name': 'Craig Ervine'}, {'name': 'Trevor Gwandu'}, {'name': 'Tanaka Chivanga'},
                {'name': 'Jacob Duffy'}, {'name': 'Raza'}, {'name': 'Brian Bennett'},
                {'name': 'Tom Blundell'}, {'name': 'Zakary Foulkes'}
            ]
        },
        {
            'strategy': 'Conditions-Based',
            'captain': 'Vincent Masekesa',
            'vice_captain': 'Brian Bennett',
            'players': [
                {'name': 'Tafadzwa Tsiga'}, {'name': 'Nick Welch'}, {'name': 'Henry Nicholls'},
                {'name': 'Craig Ervine'}, {'name': 'Matthew Fisher'}, {'name': 'Matt Henry'},
                {'name': 'Tanaka Chivanga'}, {'name': 'Vincent Masekesa'}, {'name': 'Brian Bennett'},
                {'name': 'Santner'}, {'name': 'Sean Williams'}
            ]
        }
    ]
    
    # Analyze team performance
    print("\n🎯 Analyzing predicted team performances...")
    team_analysis = analyze_team_performance(predicted_teams, player_performances)
    
    # Display results
    print("\n" + "=" * 80)
    print("🏆 TEAM PERFORMANCE RESULTS")
    print("=" * 80)
    
    # Sort teams by points
    sorted_teams = sorted(team_analysis.items(), key=lambda x: x[1]['total_points'], reverse=True)
    
    for rank, (team_name, analysis) in enumerate(sorted_teams, 1):
        print(f"\n🏅 RANK {rank}: {team_name}")
        print(f"💎 Total Points: {analysis['total_points']}")
        print(f"👥 Players Found: {analysis['players_found']}/{analysis['total_players']}")
        print(f"👑 Captain: {analysis['captain']} (2x points)")
        print(f"🥈 Vice-Captain: {analysis['vice_captain']} (1.5x points)")
        
        # Show top performers
        top_performers = sorted(
            [(name, data) for name, data in analysis['player_breakdown'].items() 
             if data['points'] > 0],
            key=lambda x: x[1]['points'], reverse=True
        )[:3]
        
        if top_performers:
            print(f"⭐ Top Performers:")
            for name, data in top_performers:
                print(f"   • {name}: {data['points']} points")
        
        print("-" * 60)
    
    # Summary
    if sorted_teams:
        best_team = sorted_teams[0]
        print(f"\n🥇 BEST PERFORMING TEAM: {best_team[0]}")
        print(f"📊 Score: {best_team[1]['total_points']} points")
        
        # Show individual player performances for context
        print(f"\n📋 Individual Player Performances:")
        for player_name, perf_data in player_performances.items():
            points = calculate_dream11_points(player_name, perf_data)
            if points > 0:
                print(f"   • {player_name}: {points} points")
    
    print(f"\n✅ Analysis Complete!")

if __name__ == "__main__":
    main()
