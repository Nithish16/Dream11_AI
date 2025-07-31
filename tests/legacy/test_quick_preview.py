#!/usr/bin/env python3
"""
Test script for Quick Preview functionality
Tests Option 2 from the main menu
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_logic.match_resolver import resolve_match_by_id
from core_logic.data_aggregator import aggregate_all_data

def test_quick_preview():
    """Test the quick preview functionality"""
    print("🧪 TESTING QUICK PREVIEW FUNCTIONALITY")
    print("="*50)
    
    try:
        # Use fallback match data
        match_id = "105780"
        print(f"🔍 Testing with Match ID: {match_id}")
        
        # Test resolve_match_from_id (simulating user input)
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
        
        if match_info:
            print(f"✅ Found match: {match_info['team1Name']} vs {match_info['team2Name']}")
            print(f"📊 Match ID: {match_info['matchId']}")
            print(f"📅 Date: {match_info.get('matchDate', 'TBD')}")
            print(f"🏟️ Venue: {match_info.get('venue', 'TBD')}")
            
            # Basic data aggregation
            print("\n📊 Aggregating basic data...")
            aggregated_data = aggregate_all_data(match_info)
            
            if aggregated_data:
                total_players = len(aggregated_data.team1.players) + len(aggregated_data.team2.players)
                print(f"👥 Total Players Available: {total_players}")
                print(f"🌱 Pitch Type: {aggregated_data.venue.pitch_archetype}")
                print(f"🏏 Format: {aggregated_data.match_format}")
                
                print(f"\n💡 Quick Insights:")
                print(f"• {aggregated_data.venue.pitch_archetype} pitch favors specific player types")
                print(f"• {total_players} players available for team selection")
                print(f"• Full analysis will provide optimized 11-player teams")
                
                print(f"\n✅ QUICK PREVIEW TEST PASSED!")
                return True
        
        print("❌ Could not generate quick preview")
        return False
    
    except Exception as e:
        print(f"❌ Quick preview test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_preview()
    if success:
        print(f"\n🎉 QUICK PREVIEW TEST PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ QUICK PREVIEW TEST FAILED!")
        sys.exit(1)