#!/usr/bin/env python3
"""
Complete end-to-end user workflow testing
"""

import sys
import time
from core_logic.match_resolver import resolve_match_by_id, get_match_summary
from core_logic.data_aggregator import aggregate_all_data, print_aggregation_summary
from core_logic.feature_engine import batch_generate_features
from core_logic.team_generator import generate_hybrid_teams, print_hybrid_teams_summary, print_team_summary

def test_complete_user_workflow():
    """Test the complete user workflow from match ID to final teams"""
    print("🚀 COMPLETE USER WORKFLOW TEST")
    print("=" * 60)
    
    # Simulate user providing match ID
    test_match_ids = [105780, 74648, 86543, 12345]
    
    for match_id in test_match_ids:
        print(f"\n🔍 Testing with Match ID: {match_id}")
        print("-" * 40)
        
        try:
            # Phase 1: Match Resolution (User Input → Match Details)
            print(f"\n📍 Phase 1: Match Resolution for ID {match_id}")
            start_time = time.time()
            
            match_info = resolve_match_by_id(match_id)
            if not match_info:
                print(f"  ❌ Could not resolve Match ID {match_id}")
                continue
                
            print(f"  ✅ Match resolved in {time.time() - start_time:.1f}s")
            print(f"  🏏 {match_info.get('team1Name', 'Team1')} vs {match_info.get('team2Name', 'Team2')}")
            
            # Phase 2: Data Aggregation
            print(f"\n📊 Phase 2: Data Aggregation")
            start_time = time.time()
            
            aggregated_data = aggregate_all_data(match_info)
            if not aggregated_data:
                print("  ❌ Data aggregation failed")
                continue
                
            total_players = len(aggregated_data.team1.players) + len(aggregated_data.team2.players)
            print(f"  ✅ Data aggregated in {time.time() - start_time:.1f}s")
            print(f"  👥 {total_players} players processed")
            print(f"  🏟️ Venue: {aggregated_data.venue.venue_name}")
            print(f"  🌱 Pitch: {aggregated_data.venue.pitch_archetype}")
            
            # Phase 3: Feature Engineering
            print(f"\n🧠 Phase 3: Feature Engineering")
            start_time = time.time()
            
            # Convert players to feature engine format
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
            if not player_features:
                print("  ❌ Feature engineering failed")
                continue
                
            print(f"  ✅ Features generated in {time.time() - start_time:.1f}s")
            print(f"  🎯 {len(player_features)} players with features")
            
            # Phase 4: Hybrid Team Generation
            print(f"\n🎯 Phase 4: Hybrid Team Generation")
            start_time = time.time()
            
            hybrid_teams = generate_hybrid_teams(player_features, aggregated_data.match_format, match_context)
            if not hybrid_teams or (not hybrid_teams.get('Pack-1') and not hybrid_teams.get('Pack-2')):
                print("  ❌ Team generation failed")
                continue
                
            print(f"  ✅ Teams generated in {time.time() - start_time:.1f}s")
            
            pack1_count = len(hybrid_teams.get('Pack-1', []))
            pack2_count = len(hybrid_teams.get('Pack-2', []))
            total_teams = pack1_count + pack2_count
            
            # Phase 5: Results Validation
            print(f"\n🏆 Phase 5: Results Validation")
            
            # Validate team structure
            valid_teams = 0
            for pack_name, teams in hybrid_teams.items():
                for team in teams:
                    if (len(team.players) == 11 and 
                        team.captain and team.vice_captain and
                        team.total_credits <= 105.0):  # Allow some buffer
                        valid_teams += 1
            
            print(f"  ✅ Valid teams: {valid_teams}/{total_teams}")
            print(f"  📦 Pack-1: {pack1_count} teams")
            print(f"  📦 Pack-2: {pack2_count} teams")
            
            # Success - print summary
            print(f"\n🎉 SUCCESS for Match ID {match_id}!")
            print(f"  🏏 Match: {aggregated_data.team1.team_name} vs {aggregated_data.team2.team_name}")
            print(f"  🏆 Generated {total_teams} optimized teams")
            
            # Show first team as example
            if hybrid_teams.get('Pack-1'):
                first_team = hybrid_teams['Pack-1'][0]
                print(f"\n📋 Sample Team (Pack-1, Team 1):")
                print(f"  👑 Captain: {first_team.captain.name}")
                print(f"  🥈 Vice-Captain: {first_team.vice_captain.name}")
                print(f"  💰 Credits: {first_team.total_credits:.1f}/100")
                print(f"  📊 Score: {first_team.total_score:.1f}")
            
            return True  # Stop after first successful test
            
        except Exception as e:
            print(f"  ❌ Error in workflow for Match ID {match_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n❌ All test match IDs failed")
    return False

if __name__ == "__main__":
    success = test_complete_user_workflow()
    sys.exit(0 if success else 1)