#!/usr/bin/env python3
"""
🗄️ Smart 15 Database Viewer
Query and display Smart 15 predictions from database without JSON files
"""

import sys
import argparse
from database_storage_manager import get_storage_manager
from datetime import datetime

class Smart15DatabaseViewer:
    """View Smart 15 predictions from database"""
    
    def __init__(self):
        self.storage_manager = get_storage_manager()
    
    def list_recent_predictions(self, limit: int = 10):
        """List recent predictions"""
        predictions = self.storage_manager.list_predictions(limit)
        
        print("\n" + "="*80)
        print("📋 RECENT SMART 15 PREDICTIONS")
        print("="*80)
        
        if not predictions:
            print("No predictions found in database.")
            return
        
        for i, pred in enumerate(predictions, 1):
            timestamp = datetime.fromisoformat(pred['prediction_timestamp']).strftime('%Y-%m-%d %H:%M')
            
            print(f"\n{i}. PREDICTION ID: {pred['id']}")
            print(f"   🏏 Match: {pred['match_id']}")
            print(f"   ⏰ Generated: {timestamp}")
            print(f"   📊 Format: {pred['match_format'] or 'Unknown'}")
            print(f"   🏟️ Venue: {pred['venue'] or 'Unknown'}")
            print(f"   ⚽ Teams: {pred['teams'] or 'Unknown vs Unknown'}")
            print(f"   🎯 Diversification: {pred['diversification_score']:.1%}")
            print(f"   👑 Unique Captains: {pred['unique_captains']}")
            print(f"   💰 Budget: ₹{pred['total_budget']:.0f}")
    
    def show_prediction_details(self, prediction_id: int):
        """Show detailed prediction information"""
        summary = self.storage_manager.get_prediction_summary(prediction_id=prediction_id)
        
        if not summary:
            print(f"❌ Prediction ID {prediction_id} not found")
            return
        
        teams = self.storage_manager.get_detailed_teams(prediction_id)
        captain_analysis = self.storage_manager.get_captain_analysis(prediction_id)
        player_usage = self.storage_manager.get_player_usage_analysis(prediction_id)
        
        print("\n" + "="*80)
        print(f"🏆 SMART 15 PREDICTION DETAILS - ID: {prediction_id}")
        print("="*80)
        
        # Summary information
        timestamp = datetime.fromisoformat(summary['prediction_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"🆔 Match ID: {summary['match_id']}")
        print(f"⏰ Generated: {timestamp}")
        print(f"🔧 System: {summary['system_version']}")
        print(f"💰 Budget: ₹{summary['total_budget']:.0f}")
        print(f"📊 Format: {summary['match_format'] or 'Unknown'}")
        print(f"🏟️ Venue: {summary['venue'] or 'Unknown'}")
        print(f"⚽ Teams: {summary['teams'] or 'Unknown vs Unknown'}")
        
        # Portfolio metrics
        print(f"\n🎯 PORTFOLIO METRICS:")
        print(f"✅ Diversification Score: {summary['diversification_score']:.1%}")
        print(f"✅ Captain Diversity: {summary['captain_diversity_score']:.1%}")
        print(f"✅ Unique Captains: {summary['unique_captains']}")
        print(f"✅ Unique Players: {summary['unique_players']}")
        print(f"✅ Avg Team Overlap: {summary['avg_team_overlap']:.1%}")
        print(f"✅ Max Captain Usage: {summary['max_captain_usage']}")
        
        # Captain analysis
        print(f"\n👑 CAPTAIN DISTRIBUTION:")
        for captain in captain_analysis:
            print(f"  {captain['captain_name']}: {captain['usage_count']} teams ({captain['usage_percentage']:.1f}%)")
            print(f"    Tiers: {', '.join(captain['tiers_used'])}")
        
        # Player usage
        print(f"\n👥 PLAYER USAGE ANALYSIS:")
        for category, data in player_usage.items():
            print(f"  🔹 {category.title()} Players: {data['count']}")
            if len(data['players']) <= 10:
                print(f"    {', '.join(data['players'])}")
            else:
                print(f"    {', '.join(data['players'][:5])} + {len(data['players'])-5} more...")
        
        # Teams by tier
        print(f"\n🛡️ TIER 1 - CORE TEAMS:")
        core_teams = [t for t in teams if t['tier'] == 1]
        for team in core_teams:
            print(f"  {team['team_number']}. {team['strategy']}")
            print(f"     👑 Captain: {team['captain']} | 🥈 VC: {team['vice_captain']}")
            print(f"     📊 Confidence: {team['confidence_level']} | 🎯 Diversity: {team['diversification_score']:.2f}")
        
        print(f"\n⚖️ TIER 2 - DIVERSIFIED TEAMS:")
        div_teams = [t for t in teams if t['tier'] == 2]
        for team in div_teams:
            print(f"  {team['team_number']}. {team['strategy']}")
            print(f"     👑 Captain: {team['captain']} | 🥈 VC: {team['vice_captain']}")
            print(f"     🎯 Diversity: {team['diversification_score']:.2f}")
        
        print(f"\n🚀 TIER 3 - MOONSHOT TEAMS:")
        moon_teams = [t for t in teams if t['tier'] == 3]
        for team in moon_teams:
            print(f"  {team['team_number']}. {team['strategy']}")
            print(f"     👑 Captain: {team['captain']} | 🥈 VC: {team['vice_captain']}")
            print(f"     🎯 Diversity: {team['diversification_score']:.2f}")
    
    def show_team_composition(self, prediction_id: int, team_number: int):
        """Show detailed team composition"""
        teams = self.storage_manager.get_detailed_teams(prediction_id)
        
        target_team = None
        for team in teams:
            if team['team_number'] == team_number:
                target_team = team
                break
        
        if not target_team:
            print(f"❌ Team {team_number} not found in prediction {prediction_id}")
            return
        
        print("\n" + "="*60)
        print(f"🎯 TEAM {team_number} COMPOSITION - Prediction ID: {prediction_id}")
        print("="*60)
        
        print(f"\n📊 TEAM DETAILS:")
        print(f"🎯 Strategy: {target_team['strategy']}")
        print(f"🏆 Tier: {target_team['tier_name']} (Tier {target_team['tier']})")
        print(f"👑 Captain: {target_team['captain']}")
        print(f"🥈 Vice Captain: {target_team['vice_captain']}")
        print(f"📊 Confidence: {target_team['confidence_level']}")
        print(f"⚠️ Risk Level: {target_team['risk_level']}")
        print(f"🎯 Diversity Score: {target_team['diversification_score']:.3f}")
        print(f"💰 Budget Weight: {target_team['budget_weight']:.3f}")
        
        print(f"\n👥 PLAYING XI:")
        for i, player in enumerate(target_team['players'], 1):
            role = "👑 CAPTAIN" if player == target_team['captain'] else "🥈 VC" if player == target_team['vice_captain'] else ""
            print(f"  {i:2d}. {player} {role}")
        
        if target_team['reasoning']:
            print(f"\n💡 REASONING:")
            print(f"  {target_team['reasoning']}")
        
        if target_team['intelligence_applied']:
            print(f"\n🧠 AI INTELLIGENCE APPLIED:")
            for intelligence in target_team['intelligence_applied']:
                print(f"  • {intelligence}")
    
    def search_predictions_by_match(self, match_id: str):
        """Search predictions for a specific match"""
        summary = self.storage_manager.get_prediction_summary(match_id=match_id)
        
        if not summary:
            print(f"❌ No predictions found for match {match_id}")
            return
        
        print(f"\n🔍 SEARCH RESULTS FOR MATCH {match_id}:")
        print("-" * 50)
        
        timestamp = datetime.fromisoformat(summary['prediction_timestamp']).strftime('%Y-%m-%d %H:%M')
        
        print(f"📋 Prediction ID: {summary['id']}")
        print(f"⏰ Generated: {timestamp}")
        print(f"🎯 Diversification: {summary['diversification_score']:.1%}")
        print(f"👑 Unique Captains: {summary['unique_captains']}")
        print(f"💰 Budget: ₹{summary['total_budget']:.0f}")
        
        print(f"\n💡 Use 'python3 smart15_database_viewer.py --details {summary['id']}' for full details")
    
    def export_prediction(self, prediction_id: int, output_file: str = None):
        """Export prediction to JSON file"""
        filename = self.storage_manager.export_prediction_to_json(prediction_id, output_file)
        print(f"✅ Exported prediction {prediction_id} to: {filename}")
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old predictions"""
        self.storage_manager.cleanup_old_predictions(days)
        print(f"✅ Cleaned up predictions older than {days} days")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="🗄️ Smart 15 Database Viewer")
    parser.add_argument("--list", "-l", type=int, default=10, help="List recent predictions (default: 10)")
    parser.add_argument("--details", "-d", type=int, help="Show detailed prediction by ID")
    parser.add_argument("--team", "-t", type=int, nargs=2, metavar=('PRED_ID', 'TEAM_NUM'), 
                       help="Show team composition (prediction_id team_number)")
    parser.add_argument("--match", "-m", type=str, help="Search predictions by match ID")
    parser.add_argument("--export", "-e", type=int, help="Export prediction to JSON file")
    parser.add_argument("--output", "-o", type=str, help="Output filename for export")
    parser.add_argument("--cleanup", type=int, help="Clean up predictions older than N days")
    
    args = parser.parse_args()
    
    viewer = Smart15DatabaseViewer()
    
    try:
        if args.details:
            viewer.show_prediction_details(args.details)
        elif args.team:
            viewer.show_team_composition(args.team[0], args.team[1])
        elif args.match:
            viewer.search_predictions_by_match(args.match)
        elif args.export:
            viewer.export_prediction(args.export, args.output)
        elif args.cleanup:
            viewer.cleanup_old_data(args.cleanup)
        else:
            viewer.list_recent_predictions(args.list)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()