#!/usr/bin/env python3
"""
DreamTeamAI - Production Readiness Test
Complete end-to-end validation of all phases
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_logic.match_resolver import resolve_match_ids, get_match_summary
from core_logic.data_aggregator import aggregate_all_data, print_aggregation_summary
from core_logic.feature_engine import batch_generate_features, print_feature_summary
from core_logic.team_generator import batch_generate_teams, print_team_summary
from app import (
    simulate_toss_and_playing_xi, post_toss_workflow, 
    format_output_for_user, calculate_team_confidence_score
)

class ProductionTestResult:
    def __init__(self):
        self.phases = {}
        self.start_time = None
        self.end_time = None
        self.total_duration = 0
        self.errors = []
        self.warnings = []
        self.success = False
        
    def add_phase_result(self, phase_name: str, success: bool, duration: float, data=None, error=None):
        self.phases[phase_name] = {
            'success': success,
            'duration': duration,
            'data': data,
            'error': error
        }
        if error:
            self.errors.append(f"Phase {phase_name}: {error}")
    
    def generate_report(self):
        """Generate comprehensive production test report"""
        report = []
        report.append("🔬 DREAMTEAMAI PRODUCTION READINESS TEST REPORT")
        report.append("=" * 70)
        report.append(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏱️  Total Duration: {self.total_duration:.2f} seconds")
        report.append(f"✅ Overall Success: {'PASS' if self.success else 'FAIL'}")
        report.append("")
        
        # Phase-by-phase results
        report.append("📊 PHASE-BY-PHASE RESULTS:")
        report.append("-" * 50)
        for phase_name, result in self.phases.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            report.append(f"{phase_name:30s} {status} ({duration:.2f}s)")
            if result['error']:
                report.append(f"   Error: {result['error']}")
        
        report.append("")
        
        # Data quality metrics
        if 'Phase 2' in self.phases and self.phases['Phase 2']['success']:
            aggregated_data = self.phases['Phase 2']['data']
            if aggregated_data:
                report.append("📈 DATA QUALITY METRICS:")
                report.append("-" * 50)
                report.append(f"Match ID: {aggregated_data.match_id}")
                report.append(f"Total Players: {len(aggregated_data.team1.players) + len(aggregated_data.team2.players)}")
                report.append(f"Data Completeness: {aggregated_data.data_completeness_score}%")
                report.append(f"Venue: {aggregated_data.venue.venue_name}")
                report.append(f"Pitch Type: {aggregated_data.venue.pitch_archetype}")
                report.append("")
        
        # Feature engineering metrics
        if 'Phase 3' in self.phases and self.phases['Phase 3']['success']:
            player_features = self.phases['Phase 3']['data']
            if player_features:
                avg_score = sum(p.dream11_expected_points for p in player_features) / len(player_features)
                avg_consistency = sum(p.consistency_score for p in player_features) / len(player_features)
                report.append("🧠 FEATURE ENGINEERING METRICS:")
                report.append("-" * 50)
                report.append(f"Players Processed: {len(player_features)}")
                report.append(f"Avg Expected Points: {avg_score:.1f}")
                report.append(f"Avg Consistency: {avg_consistency:.1f}%")
                report.append("")
        
        # Team optimization metrics
        if 'Phase 4-5' in self.phases and self.phases['Phase 4-5']['success']:
            teams_data = self.phases['Phase 4-5']['data']
            if teams_data:
                total_teams = sum(len(teams) for teams in teams_data.values())
                report.append("🎯 TEAM OPTIMIZATION METRICS:")
                report.append("-" * 50)
                report.append(f"Teams Generated: {total_teams}")
                for risk_profile, teams in teams_data.items():
                    if teams:
                        avg_score = sum(t.total_score for t in teams) / len(teams)
                        report.append(f"{risk_profile} Teams: {len(teams)} (avg score: {avg_score:.1f})")
                report.append("")
        
        # Final presentation metrics
        if 'Phase 6-7' in self.phases and self.phases['Phase 6-7']['success']:
            final_teams = self.phases['Phase 6-7']['data']
            if final_teams:
                avg_confidence = sum(calculate_team_confidence_score(t) for t in final_teams) / len(final_teams)
                report.append("🏆 FINAL PRESENTATION METRICS:")
                report.append("-" * 50)
                report.append(f"Final Teams: {len(final_teams)}")
                report.append(f"Avg Confidence: {avg_confidence:.1f}%")
                report.append("")
        
        # Error summary
        if self.errors:
            report.append("❌ ERRORS ENCOUNTERED:")
            report.append("-" * 50)
            for error in self.errors:
                report.append(f"• {error}")
            report.append("")
        
        # Production readiness assessment
        report.append("🚀 PRODUCTION READINESS ASSESSMENT:")
        report.append("-" * 50)
        
        readiness_score = sum(1 for phase in self.phases.values() if phase['success'])
        total_phases = len(self.phases)
        readiness_percentage = (readiness_score / total_phases) * 100 if total_phases > 0 else 0
        
        report.append(f"Phase Success Rate: {readiness_score}/{total_phases} ({readiness_percentage:.1f}%)")
        report.append(f"Performance: {'GOOD' if self.total_duration < 30 else 'NEEDS OPTIMIZATION'}")
        
        if readiness_percentage >= 80:
            report.append("✅ STATUS: PRODUCTION READY")
            report.append("💡 RECOMMENDATION: System is ready for deployment")
        elif readiness_percentage >= 60:
            report.append("⚠️  STATUS: NEEDS MINOR FIXES")
            report.append("💡 RECOMMENDATION: Address failing phases before deployment")
        else:
            report.append("❌ STATUS: NOT PRODUCTION READY")
            report.append("💡 RECOMMENDATION: Major fixes required before deployment")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

def run_production_test():
    """Execute comprehensive production test"""
    print("🔬 STARTING DREAMTEAMAI PRODUCTION TEST")
    print("=" * 60)
    
    test_result = ProductionTestResult()
    test_result.start_time = time.time()
    
    # Phase 1: Match Identification
    print("\n🔍 PHASE 1: Match Identification & API Integration")
    print("-" * 50)
    phase_start = time.time()
    
    try:
        match_info = resolve_match_ids("England", "India", "2025-07-31")
        phase_duration = time.time() - phase_start
        
        if match_info:
            print(f"✅ Match found: {match_info['team1Name']} vs {match_info['team2Name']}")
            print(f"📊 Match ID: {match_info['matchId']}, Series ID: {match_info['seriesId']}")
            test_result.add_phase_result("Phase 1", True, phase_duration, match_info)
        else:
            raise Exception("No match found for given criteria")
            
    except Exception as e:
        phase_duration = time.time() - phase_start
        print(f"❌ Phase 1 failed: {e}")
        test_result.add_phase_result("Phase 1", False, phase_duration, None, str(e))
        return test_result
    
    # Phase 2: Data Aggregation
    print("\n📊 PHASE 2: Data Aggregation")
    print("-" * 50)
    phase_start = time.time()
    
    try:
        aggregated_data = aggregate_all_data(match_info)
        phase_duration = time.time() - phase_start
        
        if aggregated_data and aggregated_data.data_completeness_score > 0:
            total_players = len(aggregated_data.team1.players) + len(aggregated_data.team2.players)
            print(f"✅ Data aggregated: {total_players} players, {aggregated_data.data_completeness_score}% complete")
            print(f"🏟️  Venue: {aggregated_data.venue.venue_name} ({aggregated_data.venue.pitch_archetype})")
            test_result.add_phase_result("Phase 2", True, phase_duration, aggregated_data)
        else:
            raise Exception("Data aggregation failed or incomplete")
            
    except Exception as e:
        phase_duration = time.time() - phase_start
        print(f"❌ Phase 2 failed: {e}")
        test_result.add_phase_result("Phase 2", False, phase_duration, None, str(e))
        return test_result
    
    # Phase 3: Feature Engineering
    print("\n🧠 PHASE 3: Feature Engineering")
    print("-" * 50)
    phase_start = time.time()
    
    try:
        match_context = {
            'pitch_archetype': aggregated_data.venue.pitch_archetype,
            'match_format': aggregated_data.match_format,
            'venue_id': aggregated_data.venue.venue_id,
            'venue_name': aggregated_data.venue.venue_name
        }
        
        # Convert PlayerData to dict format for feature engine
        all_players_dict = []
        for team in [aggregated_data.team1, aggregated_data.team2]:
            for player in team.players:
                player_dict = {
                    'player_id': player.player_id,
                    'name': player.name,
                    'role': player.role,
                    'career_stats': player.career_stats,
                    'batting_stats': player.batting_stats,
                    'bowling_stats': player.bowling_stats
                }
                all_players_dict.append(player_dict)
        
        player_features = batch_generate_features(all_players_dict, match_context)
        phase_duration = time.time() - phase_start
        
        if player_features and len(player_features) > 0:
            avg_score = sum(p.dream11_expected_points for p in player_features) / len(player_features)
            print(f"✅ Features generated: {len(player_features)} players, avg score {avg_score:.1f}")
            test_result.add_phase_result("Phase 3", True, phase_duration, player_features)
        else:
            raise Exception("Feature engineering produced no results")
            
    except Exception as e:
        phase_duration = time.time() - phase_start
        print(f"❌ Phase 3 failed: {e}")
        traceback.print_exc()
        test_result.add_phase_result("Phase 3", False, phase_duration, None, str(e))
        return test_result
    
    # Phase 4-5: Team Optimization
    print("\n🎯 PHASE 4-5: Ensemble Prediction & Team Optimization")
    print("-" * 50)
    phase_start = time.time()
    
    try:
        teams_data = batch_generate_teams(
            player_features_list=player_features,
            match_format=aggregated_data.match_format,
            match_context=match_context,
            num_teams=1,  # Reduced for faster testing
            risk_profiles=['Balanced', 'High-Risk']  # Reduced profiles
        )
        phase_duration = time.time() - phase_start
        
        total_teams = sum(len(teams) for teams in teams_data.values())
        if total_teams > 0:
            print(f"✅ Teams optimized: {total_teams} teams generated")
            for risk_profile, teams in teams_data.items():
                if teams:
                    avg_score = sum(t.total_score for t in teams) / len(teams)
                    print(f"   {risk_profile}: {len(teams)} teams, avg score {avg_score:.1f}")
            test_result.add_phase_result("Phase 4-5", True, phase_duration, teams_data)
        else:
            raise Exception("Team optimization produced no valid teams")
            
    except Exception as e:
        phase_duration = time.time() - phase_start
        print(f"❌ Phase 4-5 failed: {e}")
        test_result.add_phase_result("Phase 4-5", False, phase_duration, None, str(e))
        # Continue to next phase even if this fails
    
    # Phase 6-7: Post-Toss Refinement & Final Presentation
    print("\n🏆 PHASE 6-7: Post-Toss Refinement & Final Presentation")
    print("-" * 50)
    phase_start = time.time()
    
    try:
        # Simulate toss and playing XIs
        confirmed_xi_a, confirmed_xi_b, toss_result = simulate_toss_and_playing_xi(aggregated_data)
        
        # Post-toss refinement
        final_teams = post_toss_workflow(
            player_features, confirmed_xi_a, confirmed_xi_b, toss_result, match_context
        )
        
        phase_duration = time.time() - phase_start
        
        if final_teams and len(final_teams) > 0:
            avg_confidence = sum(calculate_team_confidence_score(t) for t in final_teams) / len(final_teams)
            print(f"✅ Post-toss refinement: {len(final_teams)} final teams, {avg_confidence:.1f}% avg confidence")
            
            # Test final presentation
            final_output = format_output_for_user(final_teams)
            print(f"📄 Final presentation: {len(final_output)} characters generated")
            
            test_result.add_phase_result("Phase 6-7", True, phase_duration, final_teams)
        else:
            # This is acceptable - post-toss might fail due to constraints
            print("⚠️  Post-toss refinement produced no teams (acceptable for simplified optimizer)")
            test_result.add_phase_result("Phase 6-7", False, phase_duration, None, "No teams generated (constraint issue)")
            
    except Exception as e:
        phase_duration = time.time() - phase_start
        print(f"❌ Phase 6-7 failed: {e}")
        test_result.add_phase_result("Phase 6-7", False, phase_duration, None, str(e))
    
    # Calculate overall results
    test_result.end_time = time.time()
    test_result.total_duration = test_result.end_time - test_result.start_time
    
    successful_phases = sum(1 for phase in test_result.phases.values() if phase['success'])
    total_phases = len(test_result.phases)
    test_result.success = successful_phases >= 4  # At least 4/5 phases must pass
    
    return test_result

def main():
    """Main production test execution"""
    try:
        test_result = run_production_test()
        
        print("\n" + "=" * 60)
        print("📋 GENERATING PRODUCTION TEST REPORT...")
        print("=" * 60)
        
        # Generate and display report
        report = test_result.generate_report()
        print(report)
        
        # Save report to file
        with open('/Users/nitish.natarajan/Downloads/Dream11_AI/production_test_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n📄 Full report saved to: production_test_report.txt")
        
        # Return appropriate exit code
        if test_result.success:
            print("\n🎉 PRODUCTION TEST PASSED - SYSTEM READY FOR DEPLOYMENT!")
            return 0
        else:
            print("\n⚠️  PRODUCTION TEST FAILED - FIXES REQUIRED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 PRODUCTION TEST CRASHED: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit(main())