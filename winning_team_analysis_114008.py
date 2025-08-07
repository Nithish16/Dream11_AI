#!/usr/bin/env python3
"""
Winning Team Analysis - Analyze the actual 1082-point winning team vs our AI predictions
"""

import json
from datetime import datetime
from typing import Dict, List, Any

# ACTUAL WINNING TEAM (1082 points)
WINNING_TEAM = {
    'players': [
        {'name': 'Tammy Beaumont', 'role': 'WK-Batsman', 'team': 'Welsh Fire Women'},
        {'name': 'Armitage', 'role': 'Batting Allrounder', 'team': 'Northern Superchargers Women'},
        {'name': 'Hayley Matthews', 'role': 'Batting Allrounder', 'team': 'Welsh Fire Women'},
        {'name': 'Annabel Sutherland', 'role': 'Bowling Allrounder', 'team': 'Northern Superchargers Women'},
        {'name': 'Georgia Wareham', 'role': 'Bowling Allrounder', 'team': 'Northern Superchargers Women'},
        {'name': 'Shabnim Ismail', 'role': 'Bowler', 'team': 'Welsh Fire Women'},
        {'name': 'Kate Cross', 'role': 'Bowler', 'team': 'Northern Superchargers Women'},
        {'name': 'Linsey Smith', 'role': 'Bowler', 'team': 'Northern Superchargers Women'},
        {'name': 'F Davies', 'role': 'Bowler', 'team': 'Welsh Fire Women'},
        {'name': 'Katie Levick', 'role': 'Bowler', 'team': 'Welsh Fire Women'},
        {'name': 'Grace Ballinger', 'role': 'Bowler', 'team': 'Northern Superchargers Women'}
    ],
    'captain': 'Georgia Wareham',
    'vice_captain': 'Annabel Sutherland',
    'total_score': 1082
}

# Our AI predicted teams
AI_PREDICTED_TEAMS = [
    {
        'strategy': 'AI-Optimal',
        'score': 558,
        'captain': 'Georgia Wareham',
        'vice_captain': 'Sophia Dunkley',
        'players': ['Tammy Beaumont', 'Davina Perrin', 'Emily Windsor', 'Phoebe Litchfield',
                   'F Davies', 'Linsey Smith', 'Shabnim Ismail', 'Georgia Wareham',
                   'Katie Levick', 'Sophia Dunkley', 'Hayley Matthews']
    },
    {
        'strategy': 'Risk-Balanced',
        'score': 351.5,
        'captain': 'G Elwiss',
        'vice_captain': 'Hayley Matthews',
        'players': ['Sarah Bryce', 'Davina Perrin', 'Phoebe Litchfield', 'Emily Windsor',
                   'Grace Ballinger', 'F Davies', 'Katie Levick', 'G Elwiss',
                   'Hayley Matthews', 'Lucy Higham', 'Georgia Wareham']
    },
    {
        'strategy': 'High-Ceiling',
        'score': 310.5,
        'captain': 'Katie George',
        'vice_captain': 'Davidson Richards',
        'players': ['Tammy Beaumont', 'Emily Windsor', 'Phoebe Litchfield', 'Davina Perrin',
                   'Jonassen', 'F Davies', 'Katie Levick', 'Katie George',
                   'Davidson Richards', 'Linsey Smith', 'Sophia Dunkley']
    },
    {
        'strategy': 'Value-Optimal',
        'score': 410,
        'captain': 'Sophia Dunkley',
        'vice_captain': 'Georgia Wareham',
        'players': ['Bess Heath', 'Davina Perrin', 'Phoebe Litchfield', 'Emily Windsor',
                   'Linsey Smith', 'Grace Ballinger', 'Katie Levick', 'Tammy Beaumont',
                   'Sophia Dunkley', 'Georgia Wareham', 'Armitage']
    },
    {
        'strategy': 'Conditions-Based',
        'score': 484.5,
        'captain': 'Georgia Wareham',
        'vice_captain': 'Hayley Matthews',
        'players': ['Sarah Bryce', 'Emily Windsor', 'Davina Perrin', 'Phoebe Litchfield',
                   'Shabnim Ismail', 'F Davies', 'Jonassen', 'Georgia Wareham',
                   'Hayley Matthews', 'Davidson Richards', 'Katie George']
    }
]

def analyze_team_overlap():
    """Analyze overlap between winning team and our AI predictions"""
    print("🏆 WINNING TEAM ANALYSIS - Match 114008")
    print("="*80)
    
    winning_players = [p['name'] for p in WINNING_TEAM['players']]
    winning_captain = WINNING_TEAM['captain']
    winning_vice_captain = WINNING_TEAM['vice_captain']
    
    print(f"🏆 ACTUAL WINNING TEAM (1082 points):")
    print("-" * 50)
    for i, player in enumerate(WINNING_TEAM['players'], 1):
        role_indicator = ""
        if player['name'] == winning_captain:
            role_indicator = " 👑 (C)"
        elif player['name'] == winning_vice_captain:
            role_indicator = " 🥈 (VC)"
        print(f"{i:2d}. {player['name']:18s} ({player['role']}) - {player['team']}{role_indicator}")
    
    print(f"\n📊 AI PREDICTIONS vs WINNING TEAM ANALYSIS:")
    print("="*80)
    
    best_overlap = 0
    best_strategy = ""
    
    for ai_team in AI_PREDICTED_TEAMS:
        ai_players = ai_team['players']
        
        # Calculate player overlap
        common_players = set(winning_players) & set(ai_players)
        missed_players = set(winning_players) - set(ai_players)
        extra_players = set(ai_players) - set(winning_players)
        
        overlap_percentage = (len(common_players) / 11) * 100
        
        # Check captain accuracy
        captain_correct = ai_team['captain'] == winning_captain
        vice_captain_correct = ai_team['vice_captain'] == winning_vice_captain
        
        if len(common_players) > best_overlap:
            best_overlap = len(common_players)
            best_strategy = ai_team['strategy']
        
        print(f"\n🎯 {ai_team['strategy']} (Our Score: {ai_team['score']} points)")
        print(f"   Player Overlap: {len(common_players)}/11 ({overlap_percentage:.1f}%)")
        print(f"   Captain: {'✅' if captain_correct else '❌'} ({ai_team['captain']} vs {winning_captain})")
        print(f"   Vice-Captain: {'✅' if vice_captain_correct else '❌'} ({ai_team['vice_captain']} vs {winning_vice_captain})")
        
        if common_players:
            print(f"   ✅ Correct Picks: {', '.join(sorted(common_players))}")
        
        if missed_players:
            print(f"   ❌ Missed Winners: {', '.join(sorted(missed_players))}")
    
    print(f"\n🏅 BEST AI PERFORMANCE: {best_strategy} with {best_overlap}/11 players correct")
    return best_overlap, best_strategy

def analyze_key_insights():
    """Analyze key learning insights"""
    print(f"\n🧠 CRITICAL LEARNING INSIGHTS")
    print("="*80)
    
    winning_players = [p['name'] for p in WINNING_TEAM['players']]
    
    # Captain analysis
    captain_hits = 0
    for ai_team in AI_PREDICTED_TEAMS:
        if ai_team['captain'] == WINNING_TEAM['captain']:
            captain_hits += 1
    
    print(f"👑 CAPTAIN ANALYSIS:")
    print(f"   Winning Captain: {WINNING_TEAM['captain']}")
    print(f"   AI Teams that got it right: {captain_hits}/5")
    if captain_hits > 0:
        print(f"   ✅ Georgia Wareham was correctly identified in some teams")
    else:
        print(f"   🚨 MAJOR MISS: None of our teams had the right captain!")
    
    # Vice-Captain analysis
    vice_captain_hits = 0
    for ai_team in AI_PREDICTED_TEAMS:
        if ai_team['vice_captain'] == WINNING_TEAM['vice_captain']:
            vice_captain_hits += 1
    
    print(f"\n🥈 VICE-CAPTAIN ANALYSIS:")
    print(f"   Winning Vice-Captain: {WINNING_TEAM['vice_captain']}")
    print(f"   AI Teams that got it right: {vice_captain_hits}/5")
    if vice_captain_hits == 0:
        print(f"   🚨 MAJOR MISS: We never considered Annabel Sutherland as VC!")
    
    # Find most missed players
    all_ai_players = []
    for ai_team in AI_PREDICTED_TEAMS:
        all_ai_players.extend(ai_team['players'])
    
    from collections import Counter
    ai_player_frequency = Counter(all_ai_players)
    
    print(f"\n🎯 PLAYER SELECTION ANALYSIS:")
    print(f"   Players in winning team that we NEVER picked:")
    never_picked = []
    rarely_picked = []
    
    for player in winning_players:
        frequency = ai_player_frequency.get(player, 0)
        if frequency == 0:
            never_picked.append(player)
        elif frequency <= 2:
            rarely_picked.append(player)
    
    if never_picked:
        print(f"   🚨 NEVER PICKED: {', '.join(never_picked)}")
    if rarely_picked:
        print(f"   ⚠️  RARELY PICKED: {', '.join(rarely_picked)}")
    
    # Team composition analysis
    winning_team_composition = {}
    for player in WINNING_TEAM['players']:
        team = player['team']
        winning_team_composition[team] = winning_team_composition.get(team, 0) + 1
    
    print(f"\n🏏 TEAM COMPOSITION ANALYSIS:")
    for team, count in winning_team_composition.items():
        print(f"   {team}: {count} players")
    
    # Role analysis
    winning_role_composition = {}
    for player in WINNING_TEAM['players']:
        role = player['role']
        winning_role_composition[role] = winning_role_composition.get(role, 0) + 1
    
    print(f"\n🎭 ROLE COMPOSITION ANALYSIS:")
    for role, count in winning_role_composition.items():
        print(f"   {role}: {count} players")

def generate_improvement_plan():
    """Generate specific improvement recommendations"""
    print(f"\n🚀 AI IMPROVEMENT ACTION PLAN")
    print("="*80)
    
    print(f"🚨 URGENT FIXES NEEDED:")
    print(f"1. CAPTAIN SELECTION ALGORITHM:")
    print(f"   - Georgia Wareham was the RIGHT choice (2-3 teams got this)")
    print(f"   - Need to weight all-rounder performance more heavily")
    print(f"   - Analyze Georgia's specific performance patterns")
    
    print(f"\n2. VICE-CAPTAIN DETECTION:")
    print(f"   - Annabel Sutherland was NEVER considered")
    print(f"   - Need better analysis of bowling all-rounders")
    print(f"   - Consider matchup-specific performance")
    
    print(f"\n3. PLAYER IDENTIFICATION GAPS:")
    
    # Check which key players we missed
    winning_players = [p['name'] for p in WINNING_TEAM['players']]
    all_ai_players = []
    for ai_team in AI_PREDICTED_TEAMS:
        all_ai_players.extend(ai_team['players'])
    
    from collections import Counter
    ai_frequency = Counter(all_ai_players)
    
    never_picked = [p for p in winning_players if ai_frequency.get(p, 0) == 0]
    if never_picked:
        print(f"   - NEVER picked: {', '.join(never_picked)}")
        print(f"   - Research why these players were overlooked")
    
    print(f"\n💡 ALGORITHM IMPROVEMENTS:")
    print(f"1. Enhanced All-Rounder Modeling:")
    print(f"   - Georgia Wareham + Annabel Sutherland were key")
    print(f"   - Better dual-role performance prediction")
    
    print(f"2. Bowling Selection Refinement:")
    print(f"   - 5 bowlers in winning team (including all-rounders)")
    print(f"   - Our models may under-value bowling in The Hundred")
    
    print(f"3. Team Balance Optimization:")
    print(f"   - Winning team: 6 Northern + 5 Welsh")
    print(f"   - We may have over-weighted Welsh Fire")
    
    print(f"\n📊 SUCCESS METRICS TO TRACK:")
    print(f"   - Captain accuracy: Currently 2-3/5, target 4+/5")
    print(f"   - Player overlap: Currently 6-8/11, target 9+/11")
    print(f"   - Score gap: Currently 524 points, target <200 points")

def save_analysis():
    """Save the analysis for future reference"""
    analysis_data = {
        'match_id': '114008',
        'analysis_date': datetime.now().isoformat(),
        'winning_team': WINNING_TEAM,
        'ai_predictions': AI_PREDICTED_TEAMS,
        'performance_gap': 1082 - 558,  # Best AI score
        'key_findings': {
            'captain_accuracy': 'Georgia Wareham was correct choice, 2-3/5 teams got it',
            'vice_captain_miss': 'Annabel Sutherland never considered',
            'best_ai_strategy': 'AI-Optimal with highest overlap',
            'major_gaps': 'All-rounder selection and bowling value'
        }
    }
    
    filename = f"winning_team_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {filename}")
    return filename

def main():
    """Run the complete analysis"""
    best_overlap, best_strategy = analyze_team_overlap()
    analyze_key_insights()
    generate_improvement_plan()
    filename = save_analysis()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"🎯 Key Takeaway: Our AI got {best_overlap}/11 players right")
    print(f"🏆 Best Strategy: {best_strategy}")
    print(f"📈 Main Gap: Captain/Vice-Captain selection and all-rounder analysis")
    print(f"📁 Data saved: {filename}")

if __name__ == "__main__":
    main()
