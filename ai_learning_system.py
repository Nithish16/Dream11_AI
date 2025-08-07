#!/usr/bin/env python3
"""
AI Learning System - Persistent learning from every prediction and result
This system will continuously improve the AI across all matches, automatically.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3

class Dream11LearningSystem:
    """
    Persistent learning system that improves AI predictions based on actual results
    """
    
    def __init__(self):
        self.learning_db_path = "ai_learning_database.db"
        self.learning_data_dir = Path("learning_data")
        self.learning_data_dir.mkdir(exist_ok=True)
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for persistent learning storage"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Predictions table
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
        
        # Results table
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
        
        # Learning insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_level TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                implementation_date TIMESTAMP
            )
        ''')
        
        # Algorithm improvements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                improvement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                old_logic TEXT,
                new_logic TEXT NOT NULL,
                reason TEXT NOT NULL,
                performance_impact TEXT,
                matches_tested TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Learning database initialized")
    
    def log_prediction(self, match_id: str, teams_data: Dict[str, Any], 
                      ai_strategies: List[Dict[str, Any]], match_format: str = None,
                      venue: str = None, teams_playing: str = None):
        """Log AI predictions for a match"""
        conn = sqlite3.connect(self.learning_db_path)
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
        print(f"✅ Logged prediction for match {match_id}")
    
    def log_result(self, match_id: str, winning_team: Dict[str, Any], 
                   winning_score: int, ai_best_score: int, analysis_data: Dict[str, Any]):
        """Log match results and analysis"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        performance_gap = winning_score - ai_best_score
        
        # Extract key learnings
        key_learnings = self._extract_key_learnings(analysis_data)
        
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
            json.dumps(key_learnings)
        ))
        
        conn.commit()
        conn.close()
        print(f"✅ Logged result for match {match_id}")
        
        # Generate learning insights
        self._generate_learning_insights(match_id, analysis_data)
    
    def _extract_key_learnings(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key learning points from analysis"""
        return {
            'captain_accuracy': analysis_data.get('captain_accuracy', 0),
            'vice_captain_accuracy': analysis_data.get('vice_captain_accuracy', 0),
            'player_overlap': analysis_data.get('best_overlap', 0),
            'major_misses': analysis_data.get('never_picked_players', []),
            'team_balance_insights': analysis_data.get('team_composition', {}),
            'performance_gap_size': analysis_data.get('performance_gap', 0)
        }
    
    def _generate_learning_insights(self, match_id: str, analysis_data: Dict[str, Any]):
        """Generate learning insights and improvement recommendations"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        insights = []
        
        # Captain selection insights
        if analysis_data.get('captain_accuracy', 0) < 3:  # Less than 3/5 teams got it right
            insights.append({
                'category': 'captain_selection',
                'insight_type': 'algorithm_fix',
                'description': f"Captain selection needs improvement for match {match_id}",
                'impact_level': 'high'
            })
        
        # Player identification insights
        never_picked = analysis_data.get('never_picked_players', [])
        if never_picked:
            insights.append({
                'category': 'player_identification',
                'insight_type': 'missing_players',
                'description': f"Never picked winning players: {', '.join(never_picked)}",
                'impact_level': 'high'
            })
        
        # Performance gap insights
        gap = analysis_data.get('performance_gap', 0)
        if gap > 400:  # Significant gap
            insights.append({
                'category': 'overall_performance',
                'insight_type': 'major_gap',
                'description': f"Large performance gap of {gap} points needs systematic improvement",
                'impact_level': 'critical'
            })
        
        # Insert insights
        for insight in insights:
            cursor.execute('''
                INSERT INTO learning_insights (category, insight_type, description, impact_level)
                VALUES (?, ?, ?, ?)
            ''', (
                insight['category'],
                insight['insight_type'],
                insight['description'],
                insight['impact_level']
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Generated {len(insights)} learning insights")
    
    def get_learning_recommendations(self) -> Dict[str, Any]:
        """Get current learning recommendations based on all historical data"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Get recent insights
        cursor.execute('''
            SELECT category, insight_type, description, impact_level, insight_date
            FROM learning_insights 
            WHERE status = 'pending'
            ORDER BY impact_level DESC, insight_date DESC
            LIMIT 10
        ''')
        
        insights = cursor.fetchall()
        
        # Get historical performance trends
        cursor.execute('''
            SELECT match_id, performance_gap, key_learnings
            FROM results
            ORDER BY result_date DESC
            LIMIT 5
        ''')
        
        recent_results = cursor.fetchall()
        
        conn.close()
        
        return {
            'pending_insights': insights,
            'recent_performance': recent_results,
            'recommendations': self._generate_recommendations(insights, recent_results)
        }
    
    def _generate_recommendations(self, insights: List[tuple], recent_results: List[tuple]) -> List[Dict[str, Any]]:
        """Generate specific algorithm improvement recommendations"""
        recommendations = []
        
        # Count insight types
        insight_categories = {}
        for insight in insights:
            category = insight[0]
            insight_categories[category] = insight_categories.get(category, 0) + 1
        
        # Generate recommendations based on frequency
        if insight_categories.get('captain_selection', 0) >= 2:
            recommendations.append({
                'priority': 'high',
                'component': 'captain_selection_algorithm',
                'action': 'complete_overhaul',
                'reason': 'Multiple captain selection failures detected'
            })
        
        if insight_categories.get('player_identification', 0) >= 2:
            recommendations.append({
                'priority': 'high',
                'component': 'player_analysis_engine',
                'action': 'enhance_detection',
                'reason': 'Consistently missing key winning players'
            })
        
        return recommendations
    
    def implement_learning(self, component: str, improvement_description: str, 
                          old_logic: str = None, new_logic: str = None):
        """Log implementation of a learning-based improvement"""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO algorithm_improvements (component, old_logic, new_logic, reason)
            VALUES (?, ?, ?, ?)
        ''', (component, old_logic, new_logic, improvement_description))
        
        conn.commit()
        conn.close()
        print(f"✅ Logged improvement to {component}")
    
    def auto_learn_from_match(self, match_id: str, predicted_teams: List[Dict[str, Any]], 
                             winning_team: Dict[str, Any], winning_score: int):
        """Automatically learn from a completed match (main entry point)"""
        print(f"🧠 AUTO-LEARNING FROM MATCH {match_id}")
        print("="*60)
        
        # Calculate AI best score (this would come from actual analysis)
        ai_best_score = max([team.get('score', 0) for team in predicted_teams])
        
        # Perform analysis
        analysis = self._analyze_vs_predictions(predicted_teams, winning_team, winning_score)
        
        # Log the result
        self.log_result(match_id, winning_team, winning_score, ai_best_score, analysis)
        
        # Get and display recommendations
        recommendations = self.get_learning_recommendations()
        self._display_recommendations(recommendations)
        
        return analysis
    
    def _analyze_vs_predictions(self, predicted_teams: List[Dict[str, Any]], 
                               winning_team: Dict[str, Any], winning_score: int) -> Dict[str, Any]:
        """Analyze predicted teams vs winning team"""
        winning_players = [p['name'] for p in winning_team['players']]
        winning_captain = winning_team['captain']
        winning_vice_captain = winning_team['vice_captain']
        
        analysis = {
            'captain_accuracy': 0,
            'vice_captain_accuracy': 0,
            'best_overlap': 0,
            'never_picked_players': [],
            'performance_gap': winning_score - max([team.get('score', 0) for team in predicted_teams])
        }
        
        # Calculate accuracies
        for team in predicted_teams:
            if team.get('captain') == winning_captain:
                analysis['captain_accuracy'] += 1
            if team.get('vice_captain') == winning_vice_captain:
                analysis['vice_captain_accuracy'] += 1
            
            overlap = len(set(winning_players) & set(team.get('players', [])))
            analysis['best_overlap'] = max(analysis['best_overlap'], overlap)
        
        # Find never picked players
        all_predicted = []
        for team in predicted_teams:
            all_predicted.extend(team.get('players', []))
        
        from collections import Counter
        predicted_frequency = Counter(all_predicted)
        analysis['never_picked_players'] = [p for p in winning_players if predicted_frequency.get(p, 0) == 0]
        
        return analysis
    
    def _display_recommendations(self, recommendations: Dict[str, Any]):
        """Display current learning recommendations"""
        print(f"\n🚀 CURRENT LEARNING RECOMMENDATIONS:")
        print("-" * 50)
        
        for rec in recommendations['recommendations']:
            print(f"🎯 {rec['priority'].upper()}: {rec['component']}")
            print(f"   Action: {rec['action']}")
            print(f"   Reason: {rec['reason']}\n")
    
    def create_auto_learning_hook(self):
        """Create a hook that can be called from the main prediction system"""
        hook_file = """
# AUTO-LEARNING HOOK
# Add this to your main dream11_ai.py file

def enable_auto_learning():
    '''Enable automatic learning from match results'''
    from ai_learning_system import Dream11LearningSystem
    return Dream11LearningSystem()

def log_prediction_for_learning(learning_system, match_id, teams_data, ai_strategies):
    '''Log prediction for future learning'''
    if learning_system:
        learning_system.log_prediction(match_id, teams_data, ai_strategies)

def learn_from_result(learning_system, match_id, winning_team, winning_score, predicted_teams):
    '''Learn from actual match result'''
    if learning_system:
        return learning_system.auto_learn_from_match(match_id, predicted_teams, winning_team, winning_score)

# Usage in dream11_ai.py:
# 1. learning_system = enable_auto_learning()
# 2. After generating teams: log_prediction_for_learning(learning_system, match_id, teams_data, teams)
# 3. After match completes: learn_from_result(learning_system, match_id, winning_team, score, predicted_teams)
"""
        
        with open("auto_learning_hook.py", "w") as f:
            f.write(hook_file)
        
        print("✅ Created auto-learning hook file: auto_learning_hook.py")

def main():
    """Demo the learning system"""
    learning_system = Dream11LearningSystem()
    
    # Example: Learn from match 114008
    predicted_teams = [
        {'strategy': 'AI-Optimal', 'score': 558, 'captain': 'Georgia Wareham', 'vice_captain': 'Sophia Dunkley',
         'players': ['Tammy Beaumont', 'Davina Perrin', 'Emily Windsor', 'Phoebe Litchfield',
                    'F Davies', 'Linsey Smith', 'Shabnim Ismail', 'Georgia Wareham',
                    'Katie Levick', 'Sophia Dunkley', 'Hayley Matthews']}
    ]
    
    winning_team = {
        'players': [
            {'name': 'Tammy Beaumont'}, {'name': 'Armitage'}, {'name': 'Hayley Matthews'},
            {'name': 'Annabel Sutherland'}, {'name': 'Georgia Wareham'}, {'name': 'Shabnim Ismail'},
            {'name': 'Kate Cross'}, {'name': 'Linsey Smith'}, {'name': 'F Davies'},
            {'name': 'Katie Levick'}, {'name': 'Grace Ballinger'}
        ],
        'captain': 'Georgia Wareham',
        'vice_captain': 'Annabel Sutherland'
    }
    
    # Auto-learn
    analysis = learning_system.auto_learn_from_match("114008", predicted_teams, winning_team, 1082)
    
    # Create the hook for integration
    learning_system.create_auto_learning_hook()
    
    print("\n✅ Learning system demo complete!")
    print("📁 Database created: ai_learning_database.db")
    print("🔗 Integration hook: auto_learning_hook.py")

if __name__ == "__main__":
    main()
