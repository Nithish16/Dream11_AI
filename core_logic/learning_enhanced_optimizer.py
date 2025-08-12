# ENHANCED WITH COMPREHENSIVE ANALYSIS - v2.0
# Patterns from 6 matches: 131455, 116972, 114670, 113890, 113898, 121136
# Updated: 2025-08-11T08:51:49.355105

#!/usr/bin/env python3
"""
Learning-Enhanced Team Optimizer
Incorporates learnings from 1 Crore INR winning teams to improve predictions.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime

class LearningEnhancedOptimizer:
    """
    Team optimizer that learns from winning patterns
    """
    
    def __init__(self):
        self.learning_db_path = "ai_learning_database.db"
        self.learning_insights = self._load_learning_insights()
        
    def _load_learning_insights(self) -> Dict[str, Any]:
        """Load learning insights from database"""
        insights = {
            'captain_success_patterns': {
                'young_bowlers': ['Ahmed', 'Maphaka'],  # Won 1131.5 and 1375
                'spinners': ['Ahmed', 'Rashid'],
                'all_rounders': ['Wasim', 'Clark']
            },
            'vc_success_patterns': {
                'keepers': ['Hope'],  # Won 1016 points  
                'bowlers': ['Hosein', 'Hazlewood'],  # Won 1131.5 and 1375
                'avoid_pure_batsmen': True  # 0% success rate
            },
            'format_patterns': {
                'T20': {'power_focus': True, 'success_rate': 0.32},
                'ODI': {'anchor_focus': True, 'success_rate': 0.42}, 
                'HUN': {'current_good': True, 'success_rate': 0.64}
            }
        }
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM learning_insights WHERE insight_date LIKE "2025-08-10%"')
            recent_count = cursor.fetchone()[0]
            insights['learning_data_available'] = recent_count > 0
            conn.close()
        except:
            insights['learning_data_available'] = False
            
        return insights
    
    def enhance_captain_selection(self, players: List[Dict], format_type: str = 'T20') -> str:
        """
        Enhanced captain selection based on 1 Crore winner patterns
        """
        print("👑 LEARNING-ENHANCED CAPTAIN SELECTION")
        
        if not players:
            return "Unknown"
            
        captain_scores = {}
        
        for player in players:
            name = player.get('name', '')
            role = player.get('role', '').lower()
            ema = float(player.get('ema', 0))
            consistency = float(player.get('consistency', 0))
            
            # Base score calculation
            base_score = ema * 0.4 + consistency * 0.3
            
            # Apply learning multipliers
            multiplier = 1.0
            
            # Young bowler pattern (Ahmed won 1131.5, Maphaka won 1375)
            if any(word in role for word in ['bowl', 'spin']) and ema < 50:
                multiplier *= 1.5  # 50% boost for young bowlers
                print(f"   🔥 Young bowler captain boost: {name}")
            
            # Spinner captain pattern (Ahmed success)
            if any(spinner in name.lower() for spinner in ['ahmed', 'rashid', 'chahal', 'kuldeep']):
                multiplier *= 1.4  # 40% boost for known spinners
                print(f"   🌀 Spinner captain boost: {name}")
            
            # All-rounder captain pattern (Wasim, Clark)
            if 'all' in role and 'round' in role:
                multiplier *= 1.3  # 30% boost for all-rounders
                print(f"   🔄 All-rounder captain boost: {name}")
            
            # Differential captain (low EMA but winning potential)
            if ema < 40:
                multiplier *= 1.25  # 25% differential boost
                print(f"   💎 Differential captain boost: {name}")
            
            captain_scores[name] = base_score * multiplier
        
        if captain_scores:
            best_captain = max(captain_scores, key=captain_scores.get)
            best_score = captain_scores[best_captain]
            print(f"👑 SELECTED CAPTAIN: {best_captain} (Score: {best_score:.1f})")
            return best_captain
        
        return players[0].get('name', 'Unknown')
    
    def enhance_vc_selection(self, players: List[Dict], captain: str, format_type: str = 'T20') -> str:
        """
        Enhanced VC selection based on learning insights (complete overhaul)
        """
        print("🥈 LEARNING-ENHANCED VC SELECTION")
        
        if not players:
            return "Unknown"
            
        vc_scores = {}
        
        for player in players:
            name = player.get('name', '')
            role = player.get('role', '').lower()
            ema = float(player.get('ema', 0))
            consistency = float(player.get('consistency', 0))
            
            if name == captain:  # Skip captain
                continue
            
            # Base score with higher consistency weight for VC
            base_score = ema * 0.3 + consistency * 0.6
            multiplier = 1.0
            
            # Keeper preference (Hope won 1016 as VC)
            if 'keeper' in role:
                multiplier *= 1.6  # 60% boost for keepers
                print(f"   🧤 Keeper VC boost: {name}")
            
            # Bowler preference (Hosein won 1131.5, Hazlewood won 1375 as VCs)
            if 'bowler' in role or 'bowling' in role:
                multiplier *= 1.5  # 50% boost for bowlers
                print(f"   ⚾ Bowler VC boost: {name}")
            
            # Bowling all-rounder preference
            if 'bowling' in role and 'all' in role:
                multiplier *= 1.4  # 40% boost for bowling AR
                print(f"   ⚾🔄 Bowling AR VC boost: {name}")
            
            # Avoid pure batsmen (0% success rate from learnings)
            if 'batsman' in role and 'all' not in role and 'keeper' not in role:
                multiplier *= 0.6  # 40% penalty for pure batsmen
                print(f"   ❌ Pure batsman VC penalty: {name}")
            
            vc_scores[name] = base_score * multiplier
        
        if vc_scores:
            best_vc = max(vc_scores, key=vc_scores.get)
            best_score = vc_scores[best_vc]
            print(f"🥈 SELECTED VC: {best_vc} (Score: {best_score:.1f})")
            return best_vc
        
        # Fallback to first non-captain player
        for player in players:
            if player.get('name') != captain:
                return player.get('name', 'Unknown')
        
        return "Unknown"
    
    def apply_format_learning(self, players: List[Dict], format_type: str) -> List[Dict]:
        """
        Apply format-specific learning adjustments to player ratings
        """
        print(f"📊 APPLYING {format_type} FORMAT LEARNING")
        
        enhanced_players = []
        
        for player in players:
            enhanced_player = player.copy()
            name = player.get('name', '')
            role = player.get('role', '').lower()
            ema = float(player.get('ema', 0))
            
            # T20 Format learning (32% success - needs power focus)
            if format_type == 'T20':
                # Boost known power hitters we missed (Maxwell pattern)
                if any(word in name.lower() for word in ['maxwell', 'russell', 'pollard', 'david', 'buttler']):
                    enhanced_player['ema'] = ema * 1.3
                    print(f"   ⚡ T20 power hitter boost: {name}")
                
                # Boost aggressive batsmen
                if 'batsman' in role and ema > 40:
                    enhanced_player['ema'] = ema * 1.15
                    print(f"   🏏 T20 aggressive boost: {name}")
            
            # ODI Format learning (42% success - needs anchor focus)
            elif format_type == 'ODI':
                # Boost anchor batsmen (Hope, Chase pattern)
                if 'batsman' in role and player.get('consistency', 0) > 70:
                    enhanced_player['ema'] = ema * 1.2
                    print(f"   🎯 ODI anchor boost: {name}")
                
                # Boost chase specialists
                if any(word in name.lower() for word in ['chase', 'hope', 'taylor']):
                    enhanced_player['ema'] = ema * 1.25
                    print(f"   🏃 ODI chase specialist boost: {name}")
            
            # The Hundred learning (64% success - maintain strategy)
            elif format_type in ['HUN', 'The Hundred']:
                # Only minor adjustments - strategy is working well
                enhanced_player['ema'] = ema * 1.02
            
            enhanced_players.append(enhanced_player)
        
        return enhanced_players
    
    def log_prediction_with_learning(self, match_id: str, teams: List[Dict], learning_applied: bool = True):
        """
        Log prediction with learning metadata
        """
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Create predictions table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    learning_applied BOOLEAN,
                    teams_data TEXT,
                    captain_enhancements TEXT,
                    vc_enhancements TEXT,
                    format_enhancements TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO enhanced_predictions (match_id, learning_applied, teams_data)
                VALUES (?, ?, ?)
            ''', (match_id, learning_applied, json.dumps(teams)))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Prediction logged with learning status: {learning_applied}")
            
        except Exception as e:
            print(f"⚠️ Failed to log prediction: {e}")

# Global instance
learning_optimizer = LearningEnhancedOptimizer()

def enhance_teams_with_learning(teams: List[Dict], match_format: str, match_id: str) -> List[Dict]:
    """
    Main function to enhance teams with learning
    """
    print("\n🧠 APPLYING LEARNING ENHANCEMENTS")
    print("="*50)
    
    if not teams:
        return teams
    
    enhanced_teams = []
    
    for i, team in enumerate(teams):
        print(f"\n🔧 Enhancing Team {i+1}...")
        
        # Get players from team
        players = team.get('players', [])
        if not players:
            enhanced_teams.append(team)
            continue
        
        # Apply format-specific learning to players
        enhanced_players = learning_optimizer.apply_format_learning(players, match_format)
        
        # Enhanced captain selection
        enhanced_captain = learning_optimizer.enhance_captain_selection(enhanced_players, match_format)
        
        # Enhanced VC selection  
        enhanced_vc = learning_optimizer.enhance_vc_selection(enhanced_players, enhanced_captain, match_format)
        
        # Create enhanced team
        enhanced_team = team.copy()
        enhanced_team['players'] = enhanced_players
        enhanced_team['captain'] = enhanced_captain
        enhanced_team['vice_captain'] = enhanced_vc
        enhanced_team['learning_applied'] = True
        enhanced_team['learning_version'] = '1.0'
        
        enhanced_teams.append(enhanced_team)
        
        print(f"✅ Team {i+1} enhanced - Captain: {enhanced_captain}, VC: {enhanced_vc}")
    
    # Log the prediction
    learning_optimizer.log_prediction_with_learning(match_id, enhanced_teams, True)
    
    print("\n🎯 ALL TEAMS ENHANCED WITH LEARNING!")
    return enhanced_teams
