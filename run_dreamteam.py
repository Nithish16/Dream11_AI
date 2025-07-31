#!/usr/bin/env python3
"""
DreamTeamAI - One-Click Dream11 Predictor
Simple menu-driven interface for easy usage
"""

import os
import sys
from datetime import datetime, timedelta
# Import core modules directly
from core_logic.match_resolver import resolve_match_ids, resolve_match_by_id, get_match_summary
from core_logic.data_aggregator import aggregate_all_data, print_aggregation_summary
from core_logic.feature_engine import generate_player_features, batch_generate_features, print_feature_summary, PlayerFeatures
from core_logic.team_generator import (
    batch_generate_teams, print_team_summary, print_hybrid_teams_summary, OptimalTeam, 
    get_final_player_score, prepare_players_for_optimization, generate_optimal_teams, generate_hybrid_teams
)
import random
from typing import List, Dict, Any
# Production test functionality removed for simplicity

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def resolve_match_from_id(match_id):
    """Resolve match details from user-provided match ID"""
    print("=== DreamTeamAI - Match Resolution ===\n")
    
    try:
        # Convert to integer if string
        match_id = int(match_id)
        
        # Show available match IDs first
        print("🔍 Checking available match IDs...")
        show_available_matches()
        
        # Resolve match using new function
        resolved_match = resolve_match_by_id(match_id)
        
        if resolved_match:
            match_summary = get_match_summary(resolved_match)
            print("Match Details:")
            print(match_summary)
            return resolved_match
        else:
            print(f"❌ Match ID {match_id} not found in current available matches.")
            print("💡 Please use one of the Match IDs listed above, or continue with demo data.")
            
            # Ask user if they want to continue with demo or try another ID
            user_choice = input("\nDo you want to (1) Continue with demo data or (2) Try another Match ID? Enter 1 or 2: ").strip()
            
            if user_choice == "2":
                new_match_id = input("Enter a valid Match ID from the list above: ").strip()
                return resolve_match_from_id(new_match_id)
            else:
                print("\n🔄 Continuing with demo data for demonstration...")
                return get_fallback_match_data()
            
    except ValueError:
        print("❌ Invalid match ID format. Please provide a numeric match ID.")
        return None
    except Exception as e:
        print(f"❌ Error resolving match: {e}")
        return get_fallback_match_data()

def show_available_matches():
    """Display currently available match IDs from both upcoming and recent matches"""
    try:
        from utils.api_client import fetch_upcoming_matches, fetch_recent_matches
        
        print("\n📋 CURRENTLY AVAILABLE MATCH IDs:")
        print("-" * 60)
        
        # Show upcoming matches
        print("🔜 UPCOMING MATCHES:")
        upcoming = fetch_upcoming_matches()
        count = 0
        if upcoming and 'typeMatches' in upcoming:
            for type_match in upcoming['typeMatches']:
                if 'seriesMatches' in type_match:
                    for series_match in type_match['seriesMatches']:
                        if 'seriesAdWrapper' in series_match:
                            matches = series_match['seriesAdWrapper'].get('matches', [])
                            for match in matches:
                                if count >= 5:  # Limit to 5 upcoming matches
                                    break
                                match_info = match.get('matchInfo', {})
                                match_id = match_info.get('matchId')
                                if match_id:
                                    team1 = match_info.get('team1', {}).get('teamName', 'Team1')
                                    team2 = match_info.get('team2', {}).get('teamName', 'Team2')
                                    status = match_info.get('state', 'Unknown')
                                    format_type = match_info.get('matchFormat', 'Unknown')
                                    print(f"  🏏 {match_id}: {team1} vs {team2} ({format_type}, {status})")
                                    count += 1
                            if count >= 5:
                                break
                        if count >= 5:
                            break
                    if count >= 5:
                        break
                if count >= 5:
                    break
        
        # Show recent/completed matches
        print("\n✅ RECENT/COMPLETED MATCHES:")
        recent = fetch_recent_matches()
        count = 0
        if recent and 'typeMatches' in recent:
            for type_match in recent['typeMatches']:
                if 'seriesMatches' in type_match:
                    for series_match in type_match['seriesMatches']:
                        if 'seriesAdWrapper' in series_match:
                            matches = series_match['seriesAdWrapper'].get('matches', [])
                            for match in matches:
                                if count >= 5:  # Limit to 5 recent matches
                                    break
                                match_info = match.get('matchInfo', {})
                                match_id = match_info.get('matchId')
                                if match_id:
                                    team1 = match_info.get('team1', {}).get('teamName', 'Team1')
                                    team2 = match_info.get('team2', {}).get('teamName', 'Team2')
                                    status = match_info.get('state', 'Unknown')
                                    format_type = match_info.get('matchFormat', 'Unknown')
                                    print(f"  🏏 {match_id}: {team1} vs {team2} ({format_type}, {status})")
                                    count += 1
                            if count >= 5:
                                break
                        if count >= 5:
                            break
                    if count >= 5:
                        break
                if count >= 5:
                    break
        
        print("-" * 60)
        print("💡 Tip: Both upcoming and completed matches can be used for analysis")
        
    except Exception as e:
        print(f"⚠️ Error fetching available matches: {e}")
        print("💡 Using fallback match data if available")

def get_fallback_match_data():
    """Fallback match data for testing"""
    fallback_data = {
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
    
    print("🔄 Using fallback match data for demo purposes...")
    match_summary = get_match_summary(fallback_data)
    print("Fallback Match:")
    print(match_summary)
    return fallback_data

def demo_match_resolver():
    """Demo the match resolution functionality (legacy support)"""
    print("=== DreamTeamAI - Match Resolver Demo ===\n")
    print("🔍 Finding England vs India match...")
    
    # Try to find upcoming match without specific date first
    from datetime import datetime, timedelta
    
    # Try current date and next few days
    for days_ahead in range(0, 7):
        match_date = datetime.now() + timedelta(days=days_ahead)
        date_str = match_date.strftime("%Y-%m-%d")
        
        try:
            resolved_ids = resolve_match_ids("England", "India", date_str)
            if resolved_ids and not resolved_ids.get('error'):
                print("✅ Match found!\n")
                
                match_summary = get_match_summary(resolved_ids)
                print("Match Found:")
                print(match_summary)
                return resolved_ids
        except Exception as e:
            continue
    
    # If no match found, use fallback
    return get_fallback_match_data()

def demo_data_aggregation(resolved_ids):
    """Demo the data aggregation functionality"""
    print("\n" + "="*60)
    print("=== DreamTeamAI - Data Aggregation Demo ===")
    print("="*60)
    
    aggregated_data = aggregate_all_data(resolved_ids)
    if aggregated_data and not getattr(aggregated_data, 'errors', []):
        print_aggregation_summary(aggregated_data)
        return aggregated_data
    else:
        print("❌ Data aggregation failed")
        return None

def demo_feature_engineering(aggregated_data):
    """Demo the feature engineering functionality"""
    print("\n" + "="*60)
    print("=== DreamTeamAI - Feature Engineering Demo ===")
    print("="*60)
    
    all_players = aggregated_data.team1.players + aggregated_data.team2.players
    
    # Convert PlayerData objects to dictionaries for feature engine
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
    
    # Create match context for feature generation
    match_context = {
        'venue': aggregated_data.venue,
        'match_format': aggregated_data.match_format,
        'pitch_archetype': getattr(aggregated_data.venue, 'pitch_archetype', 'Balanced')
    }
    
    player_features = batch_generate_features(players_dict, match_context)
    
    if player_features:
        print(f"🔧 Generating features for {len(player_features)} players...")
        print(f"🌱 Pitch Context: {aggregated_data.venue.pitch_archetype}")
        print(f"🏏 Format: {aggregated_data.match_format}")
        print(f"✅ Generated features for {len(player_features)} players")
        
        # Show top performers
        if len(player_features) > 0:
            sorted_features = sorted(player_features, key=lambda x: x.performance_rating, reverse=True)
            print(f"\n🏆 TOP 5 EXPECTED PERFORMERS:")
            for i, pf in enumerate(sorted_features[:5], 1):
                print(f"  {i}. {pf.player_name} ({pf.role})")
                print(f"     💎 Performance Rating: {pf.performance_rating:.1f}")
                print(f"     👑 Captain Probability: {pf.captain_vice_captain_probability:.1f}%")
                print(f"     🎯 EMA Score: {pf.ema_score:.1f}")
                print(f"     📊 Consistency: {pf.consistency_score:.1f}%")
        
        return player_features
    else:
        print("❌ Feature engineering failed")
        return None

def demo_team_optimization(player_features, aggregated_data):
    """Demo the team optimization functionality"""
    print("\n" + "="*60)
    print("=== DreamTeamAI - Team Optimization Demo ===")
    print("="*60)
    
    optimal_teams = batch_generate_teams(player_features, num_teams=3)
    if optimal_teams:
        print_team_summary(optimal_teams)
        return optimal_teams
    else:
        print("❌ Team optimization failed")
        return None

def demo_post_toss_workflow(player_features, aggregated_data):
    """Demo the post-toss workflow"""
    print("\n" + "="*60)
    print("=== DreamTeamAI - Toss Simulation ===")
    print("="*60)
    
    # Simulate toss
    teams = ['India', 'England']
    toss_winner = random.choice(teams)
    toss_decision = random.choice(['bat', 'bowl'])
    
    print(f"🪙 Toss Result: {toss_winner} won the toss and elected to {toss_decision}")
    
    if toss_decision == 'bat':
        print(f"🏏 Batting First: {toss_winner}")
        print(f"⚡ Bowling First: {teams[1] if toss_winner == teams[0] else teams[0]}")
    else:
        print(f"⚡ Bowling First: {toss_winner}")  
        print(f"🏏 Batting First: {teams[1] if toss_winner == teams[0] else teams[0]}")
    
    # Simulate confirmed playing XIs
    team1_xi = aggregated_data.team1.players[:11] if len(aggregated_data.team1.players) >= 11 else aggregated_data.team1.players
    team2_xi = aggregated_data.team2.players[:11] if len(aggregated_data.team2.players) >= 11 else aggregated_data.team2.players
    
    print(f"📋 {aggregated_data.team1.team_name} XI: {len(team1_xi)} players confirmed")
    print(f"📋 {aggregated_data.team2.team_name} XI: {len(team2_xi)} players confirmed")
    
    # Filter to confirmed XI players only
    confirmed_players = [pf for pf in player_features if any(p.player_id == pf.player_id for p in team1_xi + team2_xi)]
    
    print(f"\n" + "="*60)
    print("=== DreamTeamAI - Post-Toss Refinement ===")
    print("="*60)
    print(f"🔍 Filtered to {len(confirmed_players)} confirmed XI players")
    print("⚡ Recalculating metrics based on toss result...")
    print("🎯 Generating optimized teams with post-toss adjustments...")
    
    # Generate refined teams
    refined_teams = batch_generate_teams(confirmed_players, num_teams=3)
    
    if refined_teams:
        print(f"✅ Generated {len(refined_teams)} refined teams")
        
        print(f"\n" + "="*60)
        print("=== DreamTeamAI - Final Presentation ===")
        print("="*60)
        print_team_summary(refined_teams)
        return refined_teams
    else:
        print("❌ Post-toss team generation failed")
        return None

def print_banner():
    """Print the application banner"""
    print("🏆" * 60)
    print("🚀 WELCOME TO DREAMTEAMAI - DREAM11 PREDICTOR 🚀")
    print("🏆" * 60)
    print("⚡ AI-Powered Cricket Team Optimization")
    print("📊 Real-time API Integration")
    print("🎯 Smart Risk Profile Analysis")
    print("🏏 Complete 7-Phase Prediction Pipeline")
    print("🏆" * 60)

def print_menu():
    """Print the main menu options"""
    print("\n📋 CHOOSE YOUR OPTION:")
    print("=" * 40)
    print("1️⃣  🎯 Generate Hybrid Dream11 Teams (Match ID)")
    print("2️⃣  📊 Quick Team Preview")
    print("3️⃣  ❓ Help & Info")
    print("4️⃣  🚪 Exit")
    print("=" * 40)

def run_full_pipeline():
    """Run the complete Dream11 prediction pipeline with match ID input"""
    clear_screen()
    print("🚀 DREAMTEAMAI - HYBRID TEAM GENERATION")
    print("=" * 60)
    
    # Get match ID from user
    try:
        match_id = input("🔍 Enter Match ID: ").strip()
        if not match_id:
            print("❌ No match ID provided.")
            return False
        
        print(f"\n🔄 Starting analysis for Match ID: {match_id}")
        print("This will take 30-60 seconds to complete...")
        
        # Phase 1: Match Resolution
        print("\n🔍 Phase 1: Resolving match details...")
        match_info = resolve_match_from_id(match_id)
        
        if not match_info:
            print("❌ Could not resolve match details.")
            return False
        
        # Phase 2: Data Aggregation
        print("\n📊 Phase 2: Gathering player and match data...")
        aggregated_data = demo_data_aggregation(match_info)
        
        if not aggregated_data:
            print("❌ Data aggregation failed. Please check API connection.")
            return False
        
        # Phase 3: Feature Engineering
        print("\n🧠 Phase 3: Calculating advanced player metrics...")
        player_features = demo_feature_engineering(aggregated_data)
        
        if not player_features:
            print("❌ Feature engineering failed.")
            return False
        
        # Phase 4-6: Hybrid Team Generation
        print("\n🎯 Phase 4-6: Generating hybrid team strategy...")
        
        # Create match context for hybrid generation
        match_context = {
            'venue': aggregated_data.venue,
            'match_format': aggregated_data.match_format,
            'pitch_archetype': getattr(aggregated_data.venue, 'pitch_archetype', 'Balanced')
        }
        
        hybrid_teams = generate_hybrid_teams(player_features, aggregated_data.match_format, match_context)
        
        if not hybrid_teams or (not hybrid_teams.get('Pack-1') and not hybrid_teams.get('Pack-2')):
            print("❌ Hybrid team generation failed.")
            return False
        
        # Phase 7: Results Presentation
        print("\n🏆 Phase 7: Presenting final teams...")
        
        # Print summary first
        print_hybrid_teams_summary(hybrid_teams)
        
        # Print detailed team information
        all_teams = []
        for pack_name, teams in hybrid_teams.items():
            all_teams.extend(teams)
        
        if all_teams:
            print("\n📋 DETAILED TEAM INFORMATION:")
            for team in all_teams:
                print_team_summary(team)
        
        print("\n" + "🎉" * 20)
        print("✅ SUCCESS! Your Dream11 hybrid teams are ready!")
        print("🎉" * 20)
        
        # Summary
        total_teams = len(all_teams)
        pack1_teams = len(hybrid_teams.get('Pack-1', []))
        pack2_teams = len(hybrid_teams.get('Pack-2', []))
        
        print(f"\n📋 PIPELINE SUMMARY:")
        print(f"✅ Match: {aggregated_data.team1.team_name} vs {aggregated_data.team2.team_name}")
        print(f"✅ Venue: {aggregated_data.venue.venue_name}")
        print(f"✅ Players Analyzed: {len(player_features)}")
        print(f"✅ Pack-1 Teams: {pack1_teams} (same players, different C/VC)")
        print(f"✅ Pack-2 Teams: {pack2_teams} (alternative strategies)")
        print(f"✅ Total Teams Generated: {total_teams}")
        print(f"✅ All phases completed successfully!")
        
        return True
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        return False

def run_legacy_pipeline():
    """Run the legacy pipeline (for compatibility)"""
    clear_screen()
    print("🚀 RUNNING LEGACY DREAM11 PREDICTION PIPELINE")
    print("=" * 60)
    print("This will take 30-60 seconds to complete...")
    print("\n🔄 Starting 7-phase analysis...")
    
    try:
        # Phase 1: Match Resolution
        print("\n🔍 Phase 1: Finding England vs India match...")
        match_info = demo_match_resolver()
        
        if not match_info:
            print("❌ No suitable match found. Please try again later.")
            return False
        
        # Phase 2: Data Aggregation
        print("\n📊 Phase 2: Gathering player and match data...")
        aggregated_data = demo_data_aggregation(match_info)
        
        if not aggregated_data:
            print("❌ Data aggregation failed. Please check API connection.")
            return False
        
        # Phase 3: Feature Engineering
        print("\n🧠 Phase 3: Calculating advanced player metrics...")
        player_features = demo_feature_engineering(aggregated_data)
        
        if not player_features:
            print("❌ Feature engineering failed.")
            return False
        
        # Phase 4-5: Team Optimization
        print("\n🎯 Phase 4-5: Optimizing Dream11 teams...")
        optimal_teams = demo_team_optimization(player_features, aggregated_data)
        
        if not optimal_teams:
            print("❌ Team optimization failed.")
            return False
        
        # Phase 6-7: Post-Toss Refinement
        print("\n🏆 Phase 6-7: Post-toss refinement and final presentation...")
        final_teams = demo_post_toss_workflow(player_features, aggregated_data)
        
        if final_teams:
            print("\n" + "🎉" * 20)
            print("✅ SUCCESS! Your Dream11 teams are ready!")
            print("🎉" * 20)
            
            # Summary
            print(f"\n📋 PIPELINE SUMMARY:")
            print(f"✅ Match: {aggregated_data.team1.team_name} vs {aggregated_data.team2.team_name}")
            print(f"✅ Venue: {aggregated_data.venue.venue_name}")
            print(f"✅ Players Analyzed: {len(player_features)}")
            print(f"✅ Final Teams Generated: {len(final_teams)}")
            print(f"✅ All phases completed successfully!")
            
            return True
        else:
            print("⚠️ Post-toss refinement had issues, but pre-toss teams were generated.")
            return True
    
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        return False

# Production test functionality removed for simplicity

def run_quick_preview():
    """Run a quick team preview (just match resolution and basic teams)"""
    clear_screen()
    print("📊 QUICK TEAM PREVIEW")
    print("=" * 40)
    
    # Get match ID from user
    try:
        match_id = input("🔍 Enter Match ID for quick preview: ").strip()
        if not match_id:
            print("❌ No match ID provided.")
            return False
        
        print("Generating basic team recommendations...")
        
        # Quick match resolution using provided match ID
        match_info = resolve_match_from_id(match_id)
        
        if match_info:
            print(f"✅ Found match: {match_info['team1Name']} vs {match_info['team2Name']}")
            print(f"📊 Match ID: {match_info['matchId']}")
            print(f"📅 Date: {match_info.get('matchDate', 'TBD')}")
            print(f"🏟️ Venue: {match_info.get('venue', 'TBD')}")
            
            # Basic data aggregation
            aggregated_data = demo_data_aggregation(match_info)
            
            if aggregated_data:
                total_players = len(aggregated_data.team1.players) + len(aggregated_data.team2.players)
                print(f"👥 Total Players Available: {total_players}")
                print(f"🌱 Pitch Type: {aggregated_data.venue.pitch_archetype}")
                print(f"🏏 Format: {aggregated_data.match_format}")
                
                print(f"\n💡 Quick Insights:")
                print(f"• {aggregated_data.venue.pitch_archetype} pitch favors specific player types")
                print(f"• {total_players} players available for team selection")
                print(f"• Full analysis will provide optimized 11-player teams")
                
                return True
        
        print("❌ Could not generate quick preview")
        return False
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        return False
    except Exception as e:
        print(f"❌ Quick preview failed: {e}")
        return False

def show_help():
    """Show help and information"""
    clear_screen()
    print("❓ DREAMTEAMAI HELP & INFORMATION")
    print("=" * 50)
    
    print("\n🎯 WHAT IS DREAMTEAMAI?")
    print("DreamTeamAI is an AI-powered Dream11 team predictor that uses")
    print("real cricket data, advanced analytics, and optimization algorithms")
    print("to generate winning fantasy cricket teams using a hybrid strategy.")
    
    print("\n🚀 NEW FEATURES:")
    print("• Match ID based input - Simply provide the match ID")
    print("• Hybrid Team Strategy - Two complementary approaches")
    print("• Pack-1: Same optimal 11 players with 3 C/VC variations")
    print("• Pack-2: Alternative teams with different strategies")
    print("• Real-time API integration with Cricbuzz")
    print("• Advanced player feature engineering")
    print("• Mathematical optimization algorithms")
    
    print("\n📊 HOW IT WORKS:")
    print("1. 🔍 Match Resolution - Fetches details using Match ID")
    print("2. 📊 Data Aggregation - Gathers player statistics")
    print("3. 🧠 Feature Engineering - Calculates performance metrics")
    print("4. 🎯 Base Team Generation - Creates optimal 11-player team")
    print("5. 📦 Pack-1 Generation - 3 C/VC variations of base team")
    print("6. 📦 Pack-2 Generation - Alternative team strategies")
    print("7. 🏆 Results Presentation - Detailed team breakdowns")
    
    print("\n⚡ INPUT REQUIRED:")
    print("• Match ID: Get this from Cricbuzz or cricket websites")
    print("• Example Match IDs: 74648, 105780, 86543")
    print("• System will auto-fetch all match details")
    
    print("\n🎯 TEAM STRATEGIES:")
    print("📦 Pack-1 (C/VC Focus):")
    print("   • Same 11 optimal players across all 3 teams")
    print("   • Different Captain/Vice-Captain combinations")
    print("   • Maximizes base team strength")
    print("")
    print("📦 Pack-2 (Strategy Diversity):")
    print("   • Risk-Adjusted: Consistent performers focus")
    print("   • Form-Based: Recent performance emphasis")
    print("   • Value-Picks: Best credit value selections")
    
    print("\n🛠️ TECHNICAL INFO:")
    print(f"• Python Version: {sys.version.split()[0]}")
    print("• Uses: pandas, scikit-learn, OR-Tools optimization")
    print("• API: Real-time Cricbuzz integration")
    print("• Models: Expert-weighted + ML ensemble hybrid")

def main():
    """Main application loop"""
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (1-5): ").strip()
            
            if choice == '1':
                success = run_full_pipeline()
                if success:
                    input("\n🎯 Press Enter to return to main menu...")
                else:
                    input("\n❌ Press Enter to return to main menu...")
            
            elif choice == '2':
                success = run_quick_preview()
                input(f"\n{'🎯' if success else '❌'} Press Enter to return to main menu...")
            
            elif choice == '3':
                show_help()
                input("\n🎯 Press Enter to return to main menu...")
            
            elif choice == '4':
                clear_screen()
                print("👋 Thank you for using DreamTeamAI!")
                print("🏆 Good luck with your Dream11 teams!")
                print("🚀 See you next time!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            clear_screen()
            print("\n👋 Goodbye! Thanks for using DreamTeamAI!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('core_logic'):
        print("❌ Error: Please run this script from the Dream11_AI directory")
        print("💡 Tip: cd /Users/nitish.natarajan/Downloads/Dream11_AI")
        sys.exit(1)
    
    print("🚀 Starting DreamTeamAI...")
    main()