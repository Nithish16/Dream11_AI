#!/usr/bin/env python3
"""
Test script for the new hybrid workflow
"""

from core_logic.match_resolver import resolve_match_by_id
from core_logic.data_aggregator import aggregate_all_data
from core_logic.feature_engine import batch_generate_features
from core_logic.team_generator import generate_hybrid_teams, print_hybrid_teams_summary

def test_hybrid_workflow():
    """Test the complete hybrid workflow with a sample match ID"""
    print("🧪 TESTING HYBRID WORKFLOW")
    print("=" * 50)
    
    # Test with fallback match ID
    test_match_id = 105780
    print(f"🔍 Testing with Match ID: {test_match_id}")
    
    try:
        # Step 1: Match Resolution
        print("\n📍 Step 1: Match Resolution")
        match_info = resolve_match_by_id(test_match_id)
        
        if not match_info:
            print("❌ Match resolution failed - using fallback")
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
        
        print(f"✅ Match: {match_info.get('team1Name')} vs {match_info.get('team2Name')}")
        
        # Step 2: Data Aggregation
        print("\n📊 Step 2: Data Aggregation")
        aggregated_data = aggregate_all_data(match_info)
        
        if not aggregated_data:
            print("❌ Data aggregation failed")
            return False
        
        print(f"✅ Data aggregated: {len(aggregated_data.team1.players + aggregated_data.team2.players)} players")
        
        # Step 3: Feature Engineering
        print("\n🧠 Step 3: Feature Engineering")
        all_players = aggregated_data.team1.players + aggregated_data.team2.players
        
        # Convert to dict format for feature engine
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
        
        if not player_features:
            print("❌ Feature engineering failed")
            return False
        
        print(f"✅ Features generated for {len(player_features)} players")
        
        # Step 4: Hybrid Team Generation
        print("\n🎯 Step 4: Hybrid Team Generation")
        hybrid_teams = generate_hybrid_teams(player_features, aggregated_data.match_format, match_context)
        
        if not hybrid_teams:
            print("❌ Hybrid team generation failed")
            return False
        
        # Step 5: Results
        print("\n🏆 Step 5: Results")
        print_hybrid_teams_summary(hybrid_teams)
        
        pack1_count = len(hybrid_teams.get('Pack-1', []))
        pack2_count = len(hybrid_teams.get('Pack-2', []))
        total_teams = pack1_count + pack2_count
        
        print(f"\n✅ TEST SUCCESSFUL!")
        print(f"📦 Pack-1 Teams: {pack1_count}")
        print(f"📦 Pack-2 Teams: {pack2_count}")
        print(f"🏆 Total Teams: {total_teams}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_hybrid_workflow()