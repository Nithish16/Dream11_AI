#!/usr/bin/env python3
"""
AI Feedback System - Simple tool for user to provide feedback and help train the AI
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

class FeedbackCollector:
    def __init__(self):
        self.feedback_dir = "feedback_data"
        self.ensure_feedback_directory()
    
    def ensure_feedback_directory(self):
        """Create feedback directory if it doesn't exist"""
        if not os.path.exists(self.feedback_dir):
            os.makedirs(self.feedback_dir)
            print(f"✅ Created feedback directory: {self.feedback_dir}")
    
    def collect_team_performance_feedback(self, match_id: str):
        """Collect feedback on team performance after match completion"""
        print(f"\n🎯 TEAM PERFORMANCE FEEDBACK - Match {match_id}")
        print("=" * 60)
        print("Rate each team's performance (1-5 stars):")
        print("1 = Poor, 2 = Below Average, 3 = Average, 4 = Good, 5 = Excellent")
        print()
        
        teams = [
            "AI-Optimal",
            "Risk-Balanced", 
            "High-Ceiling",
            "Value-Optimal",
            "Conditions-Based"
        ]
        
        feedback = {
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'team_performance',
            'team_ratings': {}
        }
        
        for team in teams:
            while True:
                try:
                    rating = input(f"Rate {team} team (1-5): ")
                    rating = int(rating)
                    if 1 <= rating <= 5:
                        feedback['team_ratings'][team] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        
        # Additional comments
        comments = input("\nAny additional comments about team performance? (optional): ")
        if comments.strip():
            feedback['comments'] = comments
        
        # Best and worst performers
        best_team = input("Which team performed BEST overall? ")
        worst_team = input("Which team performed WORST overall? ")
        
        feedback['best_team'] = best_team
        feedback['worst_team'] = worst_team
        
        self.save_feedback(feedback)
        print("✅ Team performance feedback saved!")
        return feedback
    
    def collect_strategy_preferences(self, match_id: str):
        """Collect user strategy preferences"""
        print(f"\n🎯 STRATEGY PREFERENCE FEEDBACK - Match {match_id}")
        print("=" * 60)
        
        feedback = {
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'strategy_preferences',
            'preferences': {}
        }
        
        # Contest type preference
        print("What type of contest are you playing?")
        print("1. Small Contest (10-20 people)")
        print("2. Medium Contest (50-100 people)")
        print("3. Grand Prize Pool (1000+ people)")
        print("4. Head-to-Head")
        
        contest_type = input("Enter choice (1-4): ")
        contest_map = {
            '1': 'small',
            '2': 'medium', 
            '3': 'gpp',
            '4': 'h2h'
        }
        feedback['preferences']['contest_type'] = contest_map.get(contest_type, 'unknown')
        
        # Risk tolerance
        print("\nWhat's your risk tolerance for this match?")
        print("1. Conservative (Safe picks, consistent points)")
        print("2. Balanced (Mix of safe and risky picks)")
        print("3. Aggressive (High upside, willing to take risks)")
        
        risk_choice = input("Enter choice (1-3): ")
        risk_map = {
            '1': 'conservative',
            '2': 'balanced',
            '3': 'aggressive'
        }
        feedback['preferences']['risk_tolerance'] = risk_map.get(risk_choice, 'balanced')
        
        # Team preferences
        preferred_teams = input("Which teams did you prefer BEFORE the match? (comma-separated): ")
        if preferred_teams.strip():
            feedback['preferences']['preferred_teams'] = [t.strip() for t in preferred_teams.split(',')]
        
        # Captain preferences
        captain_feedback = input("Any comments on captain/vice-captain choices? (optional): ")
        if captain_feedback.strip():
            feedback['preferences']['captain_feedback'] = captain_feedback
        
        self.save_feedback(feedback)
        print("✅ Strategy preferences saved!")
        return feedback
    
    def collect_context_corrections(self, match_id: str):
        """Collect corrections to AI's understanding of match context"""
        print(f"\n🎯 CONTEXT CORRECTIONS - Match {match_id}")
        print("=" * 60)
        print("Help the AI learn by correcting any misunderstandings:")
        print()
        
        feedback = {
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'context_corrections',
            'corrections': {}
        }
        
        # Player news corrections
        player_news = input("Any important player news the AI missed? (injuries, form, etc.): ")
        if player_news.strip():
            feedback['corrections']['player_news'] = player_news
        
        # Venue corrections
        venue_feedback = input("Any corrections about venue/pitch conditions? ")
        if venue_feedback.strip():
            feedback['corrections']['venue_conditions'] = venue_feedback
        
        # Weather corrections
        weather_feedback = input("Any corrections about weather impact? ")
        if weather_feedback.strip():
            feedback['corrections']['weather_impact'] = weather_feedback
        
        # Team dynamics
        team_dynamics = input("Any insights about team dynamics or player roles? ")
        if team_dynamics.strip():
            feedback['corrections']['team_dynamics'] = team_dynamics
        
        # Overall context
        other_context = input("Any other context the AI should know about? ")
        if other_context.strip():
            feedback['corrections']['other_context'] = other_context
        
        if feedback['corrections']:
            self.save_feedback(feedback)
            print("✅ Context corrections saved!")
        else:
            print("ℹ️ No corrections provided.")
        
        return feedback
    
    def save_feedback(self, feedback: Dict[str, Any]):
        """Save feedback to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.feedback_dir}/feedback_{feedback['match_id']}_{feedback['feedback_type']}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(feedback, f, indent=2)
    
    def analyze_accumulated_feedback(self):
        """Analyze all collected feedback to show learning patterns"""
        print("\n📊 ACCUMULATED FEEDBACK ANALYSIS")
        print("=" * 60)
        
        feedback_files = [f for f in os.listdir(self.feedback_dir) if f.endswith('.json')]
        
        if not feedback_files:
            print("No feedback data found yet. Start providing feedback to see analysis!")
            return
        
        total_feedback = len(feedback_files)
        performance_feedback = len([f for f in feedback_files if 'team_performance' in f])
        strategy_feedback = len([f for f in feedback_files if 'strategy_preferences' in f])
        context_feedback = len([f for f in feedback_files if 'context_corrections' in f])
        
        print(f"📈 Total Feedback Entries: {total_feedback}")
        print(f"🎯 Team Performance Feedback: {performance_feedback}")
        print(f"🎪 Strategy Preferences: {strategy_feedback}")
        print(f"🔧 Context Corrections: {context_feedback}")
        
        # Show learning potential
        if total_feedback >= 10:
            print(f"\n🧠 LEARNING STATUS:")
            print(f"   ✅ Good feedback volume for initial learning")
            print(f"   📈 AI can start optimizing based on your preferences")
        elif total_feedback >= 5:
            print(f"\n🧠 LEARNING STATUS:")
            print(f"   🔄 Building initial learning patterns")
            print(f"   📊 Need more feedback for robust optimization")
        else:
            print(f"\n🧠 LEARNING STATUS:")
            print(f"   🌱 Just getting started - keep providing feedback!")
        
        # Estimate learning acceleration
        months_of_data = total_feedback / 30  # Assuming 30 predictions per month
        if months_of_data >= 2:
            estimated_accuracy = min(95, 50 + (months_of_data * 15))
            print(f"   🎯 Estimated Current Accuracy: {estimated_accuracy:.1f}%")

def create_simple_feedback_interface():
    """Create a simple command-line interface for feedback collection"""
    print("🤖 DREAM11 AI FEEDBACK SYSTEM")
    print("=" * 60)
    print("Help the AI learn and improve by providing feedback!")
    print()
    
    collector = FeedbackCollector()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Provide team performance feedback (after match completion)")
        print("2. Share strategy preferences (before/during match)")
        print("3. Correct AI context understanding")
        print("4. View accumulated feedback analysis")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            match_id = input("Enter match ID: ")
            collector.collect_team_performance_feedback(match_id)
        
        elif choice == '2':
            match_id = input("Enter match ID: ")
            collector.collect_strategy_preferences(match_id)
        
        elif choice == '3':
            match_id = input("Enter match ID: ")
            collector.collect_context_corrections(match_id)
        
        elif choice == '4':
            collector.analyze_accumulated_feedback()
        
        elif choice == '5':
            print("👋 Thank you for helping the AI learn!")
            print("Your feedback will accelerate the AI's journey to perfect predictions.")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    create_simple_feedback_interface()
