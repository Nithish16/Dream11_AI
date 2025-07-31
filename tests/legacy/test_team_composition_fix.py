#!/usr/bin/env python3
"""
Test script to verify the team composition fix
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_logic.data_aggregator import aggregate_all_data, print_aggregation_summary

def test_team_composition_fix():
    """Test that team composition now shows correct values"""
    print("🧪 TESTING TEAM COMPOSITION FIX")
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
        
        # Print summary to see if team composition now shows correct values
        print_aggregation_summary(aggregated_data)
        
        # Verify that team composition is not all zeros
        team1_total = (len(aggregated_data.team1.batsmen) + 
                      len(aggregated_data.team1.bowlers) + 
                      len(aggregated_data.team1.all_rounders) + 
                      len(aggregated_data.team1.wicket_keepers))
        
        team2_total = (len(aggregated_data.team2.batsmen) + 
                      len(aggregated_data.team2.bowlers) + 
                      len(aggregated_data.team2.all_rounders) + 
                      len(aggregated_data.team2.wicket_keepers))
        
        print(f"\n🔍 VERIFICATION:")
        print(f"Team 1 ({aggregated_data.team1.team_name}) categorized players: {team1_total}")
        print(f"Team 2 ({aggregated_data.team2.team_name}) categorized players: {team2_total}")
        print(f"Team 1 total players: {len(aggregated_data.team1.players)}")
        print(f"Team 2 total players: {len(aggregated_data.team2.players)}")
        
        if team1_total > 0 and team2_total > 0:
            print("✅ TEAM COMPOSITION FIX SUCCESSFUL!")
            print("Players are now properly categorized by role.")
            return True
        else:
            print("❌ TEAM COMPOSITION STILL SHOWING ZEROS")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_team_composition_fix()
    if success:
        print(f"\n🎉 TEAM COMPOSITION FIX VERIFIED!")
        sys.exit(0)
    else:
        print(f"\n❌ TEAM COMPOSITION FIX FAILED!")
        sys.exit(1)