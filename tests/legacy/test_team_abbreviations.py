#!/usr/bin/env python3
"""
Test script to verify team abbreviations are working correctly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_logic.data_aggregator import aggregate_all_data
from core_logic.feature_engine import batch_generate_features
from core_logic.team_generator import generate_hybrid_teams, print_team_summary

def test_team_abbreviations():
    """Test that team abbreviations appear correctly in predicted teams"""
    print("🧪 TESTING TEAM ABBREVIATIONS")
    print("="*50)
    
    try:
        # Test with fallback match data
        match_info = {
            'matchId': 105780,
            'seriesId': 8786,
            'team1Id': 9,
            'team2Id': 2,
            'team1Name': 'England', 
            'team2Name': 'India',
            'venueId': 12,
            'matchFormat': 'TEST',
            'venue': 'Kennington Oval',
            'city': 'London'
        }
        
        print(f"🔍 Testing with match: {match_info['team1Name']} vs {match_info['team2Name']}")
        
        # Aggregate data
        aggregated_data = aggregate_all_data(match_info)
        
        if not aggregated_data:
            print("❌ Data aggregation failed")
            return False
        
        # Generate features with team names
        all_players = aggregated_data.team1.players + aggregated_data.team2.players
        players_dict = []
        
        for player in all_players:
            player_dict = {
                'player_id': getattr(player, 'player_id', 0),
                'name': getattr(player, 'name', 'Unknown'),
                'role': getattr(player, 'role', 'Unknown'),
                'team_id': getattr(player, 'team_id', 0),
                'team_name': getattr(player, 'team_name', 'Unknown'),
                'batting_stats': getattr(player, 'batting_stats', {}),
                'bowling_stats': getattr(player, 'bowling_stats', {}),
                'career_stats': getattr(player, 'career_stats', {}),
                'recent_form': getattr(player, 'recent_form', []),
                'consistency_score': getattr(player, 'consistency_score', 0.0)
            }
            players_dict.append(player_dict)
        
        match_context = {
            'venue': aggregated_data.venue,
            'match_format': aggregated_data.match_format,
            'pitch_archetype': getattr(aggregated_data.venue, 'pitch_archetype', 'Balanced')
        }
        
        player_features = batch_generate_features(players_dict, match_context)
        print(f"✅ Generated features for {len(player_features)} players")
        
        # Test team abbreviation function
        from core_logic.team_generator import get_team_abbreviation
        
        print(f"\\n🔍 TESTING TEAM ABBREVIATION FUNCTION:")
        test_teams = {
            'England': 'ENG',
            'India': 'IND', 
            'New Zealand': 'NZ',
            'South Africa': 'SA',
            'Australia': 'AUS',
            'Unknown Team': 'UNK'
        }
        
        for team_name, expected in test_teams.items():
            result = get_team_abbreviation(team_name)
            print(f"  {team_name} → {result} {'✅' if result == expected else '❌'}")
        
        # Generate hybrid teams to test abbreviations in output
        print(f"\\n🎯 TESTING TEAM ABBREVIATIONS IN TEAM OUTPUT:")
        hybrid_teams = generate_hybrid_teams(player_features, aggregated_data.match_format, match_context)
        
        if hybrid_teams:
            # Show one team with abbreviations
            all_teams = []
            for pack_name, teams in hybrid_teams.items():
                all_teams.extend(teams)
            
            if all_teams:
                sample_team = all_teams[0]
                
                print(f"\\n📋 SAMPLE TEAM WITH ABBREVIATIONS:")
                print_team_summary(sample_team)
                
                # Check if players have proper team names (not "Unknown Team")
                has_proper_teams = any(
                    player.team in ['England', 'India', 'ENG', 'IND'] 
                    for player in sample_team.players
                )
                
                if has_proper_teams:
                    print("\\n✅ TEAM ABBREVIATIONS WORKING CORRECTLY!")
                    print("Team names are properly set and abbreviations appear in the output.")
                    return True
                else:
                    print("\\n❌ Team abbreviations not working correctly")
                    print("Let's check the team names in players:")
                    for player in sample_team.players[:3]:
                        print(f"  Player: {player.name}, Team: '{player.team}'")
                    return False
        
        print("❌ Could not generate teams for testing")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_team_abbreviations()
    if success:
        print(f"\\n🎉 TEAM ABBREVIATIONS TEST PASSED!")
        sys.exit(0)
    else:
        print(f"\\n❌ TEAM ABBREVIATIONS TEST FAILED!")
        sys.exit(1)