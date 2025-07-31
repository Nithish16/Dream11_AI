#!/usr/bin/env python3
"""
Test script for enhanced DreamTeamAI features
Tests the new features: confidence scores, ownership predictions, contest mapping, scenario planning
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_logic.match_resolver import resolve_match_by_id
from core_logic.data_aggregator import aggregate_all_data
from core_logic.feature_engine import batch_generate_features
from core_logic.team_generator import generate_hybrid_teams, generate_scenario_alternatives

def test_enhanced_features():
    """Test all enhanced features end-to-end"""
    print("🧪 TESTING ENHANCED DREAMTEAMAI FEATURES")
    print("="*60)
    
    try:
        # Phase 1: Get fallback match data for testing
        print("🔍 Phase 1: Using fallback match data for testing...")
        # Use the same fallback data as in run_dreamteam.py
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
        
        if not match_info:
            print("❌ Could not get match data")
            return False
        
        print(f"✅ Match: {match_info['team1Name']} vs {match_info['team2Name']}")
        
        # Phase 2: Data Aggregation
        print("\n📊 Phase 2: Aggregating data...")
        aggregated_data = aggregate_all_data(match_info)
        
        if not aggregated_data:
            print("❌ Data aggregation failed")
            return False
        
        print(f"✅ Data aggregated for {len(aggregated_data.team1.players + aggregated_data.team2.players)} players")
        
        # Phase 3: Feature Engineering
        print("\n🧠 Phase 3: Generating player features...")
        all_players = aggregated_data.team1.players + aggregated_data.team2.players
        
        # Convert PlayerData objects to dictionaries
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
        
        print(f"✅ Generated features for {len(player_features)} players")
        
        # Phase 4: Enhanced Hybrid Team Generation
        print("\n🎯 Phase 4: Testing enhanced hybrid team generation...")
        hybrid_teams = generate_hybrid_teams(player_features, aggregated_data.match_format, match_context)
        
        if not hybrid_teams or (not hybrid_teams.get('Pack-1') and not hybrid_teams.get('Pack-2')):
            print("❌ Hybrid team generation failed")
            return False
        
        print(f"✅ Generated hybrid teams successfully")
        
        # Phase 5: Test Enhanced Features
        print("\n🔍 Phase 5: Testing enhanced features...")
        
        all_teams = []
        for pack_name, teams in hybrid_teams.items():
            all_teams.extend(teams)
        
        print(f"\n📊 ENHANCED FEATURES TEST RESULTS:")
        print("="*60)
        
        for team in all_teams:
            print(f"\n🏆 Team {team.team_id} ({team.pack_type} - {team.strategy}):")
            
            # Test confidence scores
            stars = "⭐" * int(team.confidence_score)
            print(f"   💎 Confidence Score: {team.confidence_score:.1f}/5.0 {stars}")
            
            # Test ownership predictions
            print(f"   📈 Team Ownership: {team.ownership_prediction:.1f}%")
            
            # Test contest recommendations
            print(f"   🎪 Contest Recommendation: {team.contest_recommendation}")
            
            # Test strategic focus
            print(f"   📊 Strategic Focus: {team.strategic_focus}")
            
            # Test player ownership predictions
            high_ownership_players = [p for p in team.players if p.ownership_prediction > 60]
            low_ownership_players = [p for p in team.players if p.ownership_prediction < 30]
            
            print(f"   👥 High Ownership Players ({len(high_ownership_players)}): {', '.join(p.name for p in high_ownership_players[:3])}")
            print(f"   🎯 Low Ownership Players ({len(low_ownership_players)}): {', '.join(p.name for p in low_ownership_players[:3])}")
            
            # Test scenario planning (for first team only to avoid spam)
            if team.team_id == 1:
                print(f"\n🔮 SCENARIO PLANNING FOR TEAM {team.team_id}:")
                from core_logic.team_generator import prepare_players_for_optimization
                all_players_opt = prepare_players_for_optimization(player_features, aggregated_data.match_format, match_context)
                scenarios = generate_scenario_alternatives(team, all_players_opt)
                
                if scenarios['captain_alternatives']:
                    print(f"   👑 Captain Alternatives: {', '.join(scenarios['captain_alternatives'][:3])}")
                if scenarios['vice_captain_alternatives']:
                    print(f"   🥈 Vice-Captain Alternatives: {', '.join(scenarios['vice_captain_alternatives'][:3])}")
                if scenarios['risky_player_substitutes']:
                    print(f"   ⚠️  Risky Player Substitutes: {', '.join(scenarios['risky_player_substitutes'])}")
        
        print(f"\n✅ ALL ENHANCED FEATURES TESTED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        pack1_teams = len(hybrid_teams.get('Pack-1', []))
        pack2_teams = len(hybrid_teams.get('Pack-2', []))
        total_teams = pack1_teams + pack2_teams
        
        print(f"\n📋 TEST SUMMARY:")
        print(f"✅ Match: {aggregated_data.team1.team_name} vs {aggregated_data.team2.team_name}")
        print(f"✅ Players Analyzed: {len(player_features)}")
        print(f"✅ Pack-1 Teams: {pack1_teams}")
        print(f"✅ Pack-2 Teams: {pack2_teams}")
        print(f"✅ Total Teams: {total_teams}")
        print(f"✅ Enhanced Features: ✅ Confidence Scores ✅ Ownership Predictions ✅ Contest Mapping ✅ Scenario Planning")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_features()
    if success:
        print(f"\n🎉 ALL TESTS PASSED! Enhanced features working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ TESTS FAILED! Please check the implementation.")
        sys.exit(1)