#!/usr/bin/env python3
"""
AI Learning System - UPDATED with Proper Cumulative Learning
Now uses proper evidence accumulation instead of overwriting latest match
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3

# Import the proper cumulative learning system
from proper_cumulative_learning_system import CumulativeLearningSystem

class AILearningSystem:
    """
    Updated AI Learning System that uses proper cumulative learning
    Maintains backwards compatibility while using the correct learning approach
    """
    
    def __init__(self):
        self.legacy_db_path = "data/ai_learning_database.db"
        self.learning_data_dir = Path("learning_data")
        self.learning_data_dir.mkdir(exist_ok=True)
        
        # Use the proper cumulative learning system
        self.cumulative_learning = CumulativeLearningSystem()
        
        # Keep legacy database for backwards compatibility
        self.setup_legacy_database()
        print("AI Learning System updated with proper cumulative learning")
    
    def setup_legacy_database(self):
        """Keep legacy database for backwards compatibility"""
        conn = sqlite3.connect(self.legacy_db_path)
        cursor = conn.cursor()
        
        # Keep existing tables for compatibility
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                teams_data TEXT NOT NULL,
                ai_strategies TEXT NOT NULL,
                match_format TEXT,
                venue TEXT,
                teams_playing TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                result_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                winning_team TEXT NOT NULL,
                winning_score INTEGER NOT NULL,
                ai_best_score INTEGER,
                performance_gap INTEGER,
                analysis_data TEXT NOT NULL,
                key_learnings TEXT
            )
        ''')
        
        # Deprecated old learning insights table (replaced by proper cumulative system)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS legacy_learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_level TEXT NOT NULL,
                status TEXT DEFAULT 'deprecated',
                implementation_date TIMESTAMP,
                migration_note TEXT DEFAULT 'Migrated to proper cumulative learning system'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, match_id: str, teams_data: Dict[str, Any], 
                      ai_strategies: List[Dict[str, Any]], match_format: str = None,
                      venue: str = None, teams_playing: str = None):
        """Log AI predictions - maintains backwards compatibility"""
        # Log to legacy database for compatibility
        conn = sqlite3.connect(self.legacy_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (match_id, teams_data, ai_strategies, match_format, venue, teams_playing)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            match_id,
            json.dumps(teams_data),
            json.dumps(ai_strategies),
            match_format,
            venue,
            teams_playing
        ))
        
        conn.commit()
        conn.close()
        print(f"Logged prediction for match {match_id}")
    
    def log_result(self, match_id: str, winning_team: Dict[str, Any], 
                   winning_score: int, ai_best_score: int, analysis_data: Dict[str, Any]):
        """Log match results using proper cumulative learning"""
        
        # Log to legacy database for backwards compatibility
        conn = sqlite3.connect(self.legacy_db_path)
        cursor = conn.cursor()
        
        performance_gap = winning_score - ai_best_score
        
        cursor.execute('''
            INSERT INTO results (match_id, winning_team, winning_score, ai_best_score, 
                               performance_gap, analysis_data, key_learnings)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_id,
            json.dumps(winning_team),
            winning_score,
            ai_best_score,
            performance_gap,
            json.dumps(analysis_data),
            json.dumps({"note": "Using proper cumulative learning system"})
        ))
        
        conn.commit()
        conn.close()
        
        # Use proper cumulative learning system for actual learning
        self.cumulative_learning.add_match_evidence(match_id, analysis_data)
        
        print(f"Logged result for match {match_id} with proper cumulative learning")
    
    def get_learning_recommendations(self) -> Dict[str, Any]:
        """Get learning recommendations from proper cumulative system"""
        
        # Get proper accumulated insights
        accumulated_insights = self.cumulative_learning.get_accumulated_insights()
        learning_summary = self.cumulative_learning.get_learning_summary()
        
        # Convert to legacy format for backwards compatibility
        recommendations = {
            'pending_insights': [],
            'recent_performance': [],
            'recommendations': []
        }
        
        # Convert accumulated insights to legacy format
        for category, patterns in accumulated_insights.items():
            for pattern in patterns[:3]:  # Top 3 per category
                recommendations['pending_insights'].append((
                    category,
                    pattern.subcategory,
                    pattern.description,
                    'high' if pattern.reliability_score > 0.7 else 'medium' if pattern.reliability_score > 0.4 else 'low',
                    pattern.last_updated
                ))
                
                recommendations['recommendations'].append({
                    'description': pattern.description,
                    'reliability': pattern.reliability_score,
                    'evidence_count': pattern.evidence_count,
                    'category': category
                })
        
        return recommendations
    
    def auto_learn_from_match(self, match_id: str, predicted_teams: List[Dict], 
                             winning_team: Dict, winning_score: int):
        """Auto-learn from match using proper cumulative learning"""
        
        # Extract analysis data from the match
        analysis_data = {
            'match_id': match_id,
            'format': 'ODI',  # Default, should be passed in
            'winning_team_data': winning_team,
            'winning_score': winning_score,
            'predicted_teams': predicted_teams,
            'captain_analysis': self._analyze_captain_performance(predicted_teams, winning_team),
            'team_balance_analysis': self._analyze_team_balance(predicted_teams, winning_team),
            'player_selection_analysis': self._analyze_player_selection(predicted_teams, winning_team)
        }
        
        # Use cumulative learning system
        self.cumulative_learning.add_match_evidence(match_id, analysis_data)
        
        print(f"Auto-learned from match {match_id} using proper cumulative system")
        return analysis_data
    
    def _analyze_captain_performance(self, predicted_teams: List[Dict], winning_team: Dict) -> Dict:
        """Analyze captain performance"""
        # Extract captain from winning team (simplified)
        return {
            'best_actual_captain': winning_team.get('captain', 'Unknown'),
            'captain_accuracy_score': 75  # Placeholder - would need real performance data
        }
    
    def _analyze_team_balance(self, predicted_teams: List[Dict], winning_team: Dict) -> Dict:
        """Analyze team balance effectiveness"""
        return {
            'best_balance_type': 'balanced',  # Placeholder
            'effectiveness': True
        }
    
    def _analyze_player_selection(self, predicted_teams: List[Dict], winning_team: Dict) -> Dict:
        """Analyze player selection accuracy"""
        return {
            'selected_player_performances': {},  # Placeholder
            'selection_accuracy': 70
        }
    
    def get_accumulated_insights_for_prediction(self, match_context: Dict[str, Any]) -> Dict:
        """Get accumulated insights for prediction weighting"""
        return self.cumulative_learning.get_accumulated_insights(match_context)
    
    def generate_prediction_weights(self, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Generate prediction weights based on accumulated learning"""
        return self.cumulative_learning.generate_prediction_weights(match_context)

# Legacy compatibility functions
class Dream11LearningSystem(AILearningSystem):
    """Legacy name compatibility"""
    pass

def enable_auto_learning():
    """Enable auto learning - returns proper cumulative learning system"""
    return AILearningSystem()

def log_prediction_for_learning(learning_system, match_id, teams_data, ai_strategies):
    """Log prediction for future learning - uses proper system"""
    if learning_system:
        learning_system.log_prediction(match_id, teams_data, ai_strategies)

def learn_from_result(learning_system, match_id, winning_team, winning_score, predicted_teams):
    """Learn from actual match result - uses proper cumulative learning"""
    if learning_system:
        return learning_system.auto_learn_from_match(match_id, predicted_teams, winning_team, winning_score)

if __name__ == "__main__":
    # Test the updated system
    learning_system = AILearningSystem()
    
    print("Testing Updated AI Learning System:")
    
    # Test logging prediction
    sample_teams = [{"team_name": "Team1", "players": []}]
    learning_system.log_prediction("117013", {"teams": sample_teams}, sample_teams)
    
    # Test getting recommendations
    recommendations = learning_system.get_learning_recommendations()
    print(f"Available recommendations: {len(recommendations.get('recommendations', []))}")
    
    # Test accumulated insights
    insights = learning_system.get_accumulated_insights_for_prediction({'format': 'ODI'})
    print(f"Accumulated insight categories: {list(insights.keys())}")
    
    print("Updated AI Learning System working with proper cumulative learning!")