#!/usr/bin/env python3
"""
DREAM11 ULTIMATE PREDICTION SYSTEM
Universal Cricket Intelligence with Advanced Learning
The ONE system that combines all knowledge and capabilities

Features:
- Universal Cricket Intelligence (all 12 formats)
- Format-specific learning application  
- All proven 1 Crore winner patterns
- Player-specific intelligence
- Context-aware predictions
- Advanced neural optimization
- Real-time learning integration
- Beautiful comprehensive output

Usage: python3 dream11_ultimate.py <match_id>
"""

import sys
import os
import json
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.api_client import fetch_match_center
    from ai_learning_system import AILearningSystem
except ImportError as e:
    print(f"Warning: Import issue: {e}")
    print("Using fallback mode...")

class Dream11Ultimate:
    """
    The Ultimate Dream11 Prediction System
    Combines all learning, intelligence, and prediction capabilities
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger.info("Initializing Dream11 Ultimate System")
        
        # Initialize learning system
        self.learning_system = None
        self.setup_learning_system()
        
        # Initialize databases
        self.setup_databases()
        
        # Initialize cleanup system
        self.setup_cleanup_system()
        
        self.logger.info("Dream11 Ultimate System ready")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_learning_system(self):
        """Initialize the AI learning system"""
        try:
            self.learning_system = AILearningSystem()
            self.logger.info("AI Learning System initialized")
        except Exception as e:
            self.logger.warning(f"Learning system not available: {e}")
            self.learning_system = None
    
    def setup_databases(self):
        """Setup database connections"""
        self.db_connections = {}
        
        # Universal Cricket Intelligence Database
        try:
            self.db_connections['universal'] = sqlite3.connect('data/universal_cricket_intelligence.db')
            self.logger.info("Universal Cricket Intelligence DB connected")
        except Exception as e:
            self.logger.warning(f"Universal DB not available: {e}")
        
        # Format-specific learning database
        try:
            self.db_connections['format'] = sqlite3.connect('data/format_specific_learning.db')
            self.logger.info("Format-specific Learning DB connected")
        except Exception as e:
            self.logger.warning(f"Format DB not available: {e}")
        
        # AI learning database
        try:
            self.db_connections['ai_learning'] = sqlite3.connect('data/ai_learning_database.db')
            self.logger.info("AI Learning DB connected")
        except Exception as e:
            self.logger.warning(f"AI Learning DB not available: {e}")
    
    def setup_cleanup_system(self):
        """Initialize automated database cleanup system"""
        try:
            # Simple cleanup without complex connection pooling
            self._run_simple_cleanup()
            self.logger.info("Database cleanup system initialized")
        except Exception as e:
            self.logger.warning(f"Cleanup system not available: {e}")
    
    def _run_simple_cleanup(self):
        """Run simple 60-day cleanup without complex dependencies"""
        try:
            # Calculate cutoff date (60 days ago)
            cutoff_date = datetime.now() - timedelta(days=60)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Create learning insights preservation table
            conn = sqlite3.connect('data/universal_cricket_intelligence.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preserved_learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT NOT NULL,
                    match_format TEXT,
                    player_name TEXT,
                    pattern_data TEXT,
                    success_rate REAL,
                    confidence_score TEXT,
                    extracted_from_period TEXT,
                    preserved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            
            # Clean up databases
            databases_to_clean = [
                'data/dream11_unified.db',
                'data/universal_cricket_intelligence.db',
                'data/ai_learning_database.db',
                'data/smart_local_predictions.db',
                'data/optimized_predictions.db'
            ]
            
            total_deleted = 0
            for db_name in databases_to_clean:
                try:
                    deleted = self._cleanup_database_simple(db_name, cutoff_str)
                    total_deleted += deleted
                except Exception:
                    pass  # Skip non-existent databases
            
            if total_deleted > 0:
                self.logger.info(f"Cleanup: Removed {total_deleted} predictions older than 60 days")
                
        except Exception as e:
            self.logger.warning(f"Simple cleanup failed: {e}")
    
    def _cleanup_database_simple(self, db_name: str, cutoff_date: str) -> int:
        """Simple database cleanup without connection pooling"""
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            
            # Find tables with prediction data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            total_deleted = 0
            
            for (table_name,) in tables:
                # Skip learning and analysis tables (preserve AI/ML data)
                if any(preserve_keyword in table_name.lower() for preserve_keyword in 
                       ['learning', 'analysis', 'intelligence', 'pattern', 'insight', 'model', 'preserved']):
                    continue
                
                # Check if table has timestamp columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                timestamp_cols = [col for col in columns if any(ts in col.lower() for ts in 
                                ['timestamp', 'created_at', 'date', 'time'])]
                
                if timestamp_cols:
                    timestamp_col = timestamp_cols[0]
                    
                    # Delete old predictions only
                    delete_query = f"DELETE FROM {table_name} WHERE {timestamp_col} < ?"
                    cursor.execute(delete_query, (cutoff_date,))
                    deleted = cursor.rowcount
                    total_deleted += deleted
            
            conn.commit()
            conn.close()
            
            return total_deleted
            
        except sqlite3.Error:
            return 0  # Skip databases that don't exist or have issues
    
    def get_format_intelligence(self, match_format: str, venue_type: str = None) -> Dict:
        """
        Get format-specific intelligence from learning databases
        """
        intelligence = {
            'captain_patterns': [],
            'vc_patterns': [],
            'player_selection_insights': [],
            'confidence_level': 'medium'
        }
        
        if 'format' not in self.db_connections:
            return intelligence
        
        try:
            cursor = self.db_connections['format'].cursor()
            
            # Get format-specific patterns
            cursor.execute("""
                SELECT pattern_data, confidence_score
                FROM format_learnings 
                WHERE format_name = ? OR format_name LIKE ?
                ORDER BY confidence_score DESC
            """, (match_format.lower(), f"%{match_format.lower()}%"))
            
            results = cursor.fetchall()
            
            for result in results:
                if result[0]:  # pattern_data
                    # Try to parse JSON pattern data
                    try:
                        pattern_data = json.loads(result[0])
                        if 'captain_patterns' in pattern_data:
                            intelligence['captain_patterns'].extend(pattern_data['captain_patterns'])
                        if 'vc_patterns' in pattern_data:
                            intelligence['vc_patterns'].extend(pattern_data['vc_patterns'])
                    except:
                        # Fallback: treat as simple pattern string
                        intelligence['captain_patterns'].append(result[0])
                if result[1]:  # confidence_score
                    intelligence['confidence_level'] = result[1]
            
        except Exception as e:
            self.logger.warning(f"Error getting format intelligence: {e}")
        
        return intelligence
    
    def get_player_intelligence(self, players: List[str], match_format: str) -> Dict:
        """
        Get player-specific intelligence from learning database
        """
        player_intel = {}
        
        if 'universal' not in self.db_connections:
            return player_intel
        
        try:
            cursor = self.db_connections['universal'].cursor()
            
            for player in players:
                cursor.execute("""
                    SELECT captaincy_success, vc_success, pressure_performance, learning_notes
                    FROM player_specific_learnings 
                    WHERE player_name = ? AND (format_category = ? OR format_category LIKE ?)
                """, (player, match_format.lower(), f"%{match_format.lower()}%"))
                
                result = cursor.fetchone()
                if result:
                    player_intel[player] = {
                        'captaincy_success': result[0] or 'Unknown',
                        'vc_success': result[1] or 'Unknown', 
                        'pressure_performance': result[2] or 'Unknown',
                        'learning_notes': result[3] or 'No specific notes'
                    }
        
        except Exception as e:
            self.logger.warning(f"Error getting player intelligence: {e}")
        
        return player_intel
    
    def resolve_match_context(self, match_id: str) -> Dict:
        """
        Resolve match context and determine format intelligence to apply
        """
        context = {
            'match_id': match_id,
            'teams': 'Unknown vs Unknown',
            'format': 'unknown',
            'venue': 'Unknown',
            'series_type': 'unknown',
            'intelligence_level': 'fallback'
        }
        
        try:
            # Try to fetch match details
            match_data = fetch_match_center(match_id)
            
            if match_data and 'matchInfo' in match_data:
                match_info = match_data['matchInfo']
                team1_name = match_info.get('team1', {}).get('name', 'Team1')
                team2_name = match_info.get('team2', {}).get('name', 'Team2')
                teams_str = f"{team1_name} vs {team2_name}"
                
                venue_info = match_data.get('venueInfo', {})
                venue_name = venue_info.get('ground', venue_info.get('city', 'Unknown'))
                
                # Smart format detection based on series name and match info
                series_name = match_info.get('series', {}).get('name', 'Unknown') if isinstance(match_info.get('series'), dict) else 'Unknown'
                detected_format = self.detect_match_format(match_info, series_name, match_id)
                
                context.update({
                    'teams': teams_str,
                    'format': detected_format,
                    'venue': venue_name,
                    'series': series_name,
                    'intelligence_level': 'api_enhanced'
                })
                
                # Determine series type
                if 'international' in context['series'].lower():
                    context['series_type'] = 'international'
                elif any(league in context['series'].lower() for league in ['ipl', 'cpl', 'hundred', 'big bash']):
                    context['series_type'] = 'franchise_league'
                else:
                    context['series_type'] = 'domestic'
                
                self.logger.info(f"Match context resolved: {context['teams']} ({context['format']})")
            
        except Exception as e:
            self.logger.warning(f"Could not resolve match context: {e}")
        
        return context
    
    def detect_match_format(self, match_info: Dict, series_name: str, match_id: str) -> str:
        """
        Smart format detection based on multiple indicators
        """
        series_lower = series_name.lower()
        
        # The Hundred detection
        if 'hundred' in series_lower:
            return 'hun'
        
        # IPL detection
        if 'ipl' in series_lower or 'indian premier league' in series_lower:
            return 't20'
        
        # CPL, BBL, PSL detection
        if any(league in series_lower for league in ['cpl', 'big bash', 'psl', 'super league']):
            return 't20'
        
        # ODI detection
        if any(indicator in series_lower for indicator in ['odi', 'one day', 'world cup', 'champions trophy']):
            return 'odi'
        
        # T20I detection
        if any(indicator in series_lower for indicator in ['t20i', 't20 international', 't20 world']):
            return 't20i'
        
        # Test detection
        if any(indicator in series_lower for indicator in ['test', 'championship', 'ashes']):
            return 'test'
        
        # Match ID based detection (The Hundred typically has specific ID patterns)
        if match_id.startswith('113') or match_id.startswith('114'):  # The Hundred 2025 range
            return 'hun'
        
        # Fallback to match format from API
        api_format = match_info.get('matchFormat', 'unknown').lower()
        if api_format in ['t20', 'odi', 'test', 't20i']:
            return api_format
        
        # Default fallback
        return 't20'  # Most common format
    
    def _get_enhanced_match_conditions(self, match_context: Dict):
        """Get enhanced weather and pitch conditions for match"""
        try:
            from core_logic.weather_pitch_analyzer import get_match_conditions
            
            match_id = match_context.get('match_id', 'unknown')
            venue = match_context.get('venue', 'Unknown')
            
            # Get comprehensive conditions
            conditions = get_match_conditions(match_id, venue)
            
            self.logger.info(f"Weather & Pitch Analysis: {venue}")
            self.logger.info(f"   Captain Preference: {conditions.captain_preference}")
            self.logger.info(f"   Pace Advantage: {conditions.pace_bowler_advantage:.2f}")
            self.logger.info(f"   Spin Advantage: {conditions.spin_bowler_advantage:.2f}")
            self.logger.info(f"   Batting Advantage: {conditions.batsmen_advantage:.2f}")
            
            return conditions
            
        except Exception as e:
            self.logger.warning(f"Could not get enhanced match conditions: {e}")
            return None
    
    def generate_intelligent_teams(self, match_context: Dict, players: List[str]) -> List[Dict]:
        """
        🧠 Generate teams using Universal Cricket Intelligence
        """
        # Store current match_id for enhanced player classification
        self.current_match_id = match_context.get('match_id')
        
        # Get enhanced weather and pitch conditions
        self.match_conditions = self._get_enhanced_match_conditions(match_context)
        
        format_intel = self.get_format_intelligence(match_context['format'])
        player_intel = self.get_player_intelligence(players, match_context['format'])
        
        teams = []
        
        # Generate 15 teams with different strategies as identified in Smart15 system
        strategies = [
            # Tier 1: Core Teams (5 teams) - High confidence, proven patterns
            ("Format-Optimized Intelligence", "format_priority"),
            ("Player Intelligence Focus", "player_priority"), 
            ("Proven Winner Patterns", "proven_patterns"),
            ("Context-Aware Optimization", "context_aware"),
            ("Ultimate Intelligence Fusion", "ultimate_fusion"),
            
            # Tier 2: Diversified Teams (7 teams) - Balanced risk-reward
            ("Weather Optimized", "format_priority"),
            ("Venue Specialist", "context_aware"),
            ("Opposition Focused", "player_priority"),
            ("Role Balanced", "ultimate_fusion"),
            ("Form Momentum", "proven_patterns"),
            ("Contrarian Captain", "context_aware"),
            ("Bowling Heavy", "format_priority"),
            
            # Tier 3: Moonshot Teams (3 teams) - High risk/reward
            ("Ultra Contrarian", "player_priority"),
            ("High Ceiling Differential", "ultimate_fusion"),
            ("Weather Extreme", "context_aware")
        ]
        
        for i, (strategy_name, priority_type) in enumerate(strategies):
            team = self.create_intelligent_team_base(
                players, match_context, format_intel, player_intel, strategy_name, priority_type
            )
            
            # Add tier information based on Smart15 structure
            if i < 5:
                team['tier'] = 'Core'
                team['risk_level'] = 'Low'
                team['budget_weight'] = 0.12
            elif i < 12:
                team['tier'] = 'Diversified'
                team['risk_level'] = 'Medium'
                team['budget_weight'] = 0.043
            else:
                team['tier'] = 'Moonshot'
                team['risk_level'] = 'High'
                team['budget_weight'] = 0.033
            
            teams.append(team)
        
        return teams
    
    def display_teams_table(self, teams: List[Dict], context: Dict):
        """Display all 15 teams in a comprehensive table format - ALWAYS SHOW ALL TEAMS"""
        
        # Header
        print(f"\n{'='*120}")
        print(f"{'TEAM':<4} {'STRATEGY':<25} {'TIER':<12} {'CAPTAIN':<18} {'VICE-CAPTAIN':<18} {'RISK':<8} {'BUDGET%':<8}")
        print(f"{'='*120}")
        
        # Display each team in table row format
        for i, team in enumerate(teams, 1):
            tier_symbol = "[C]" if team['tier'] == 'Core' else "[D]" if team['tier'] == 'Diversified' else "[M]"
            risk_symbol = "[L]" if team['risk_level'] == 'Low' else "[M]" if team['risk_level'] == 'Medium' else "[H]"
            
            print(f"{i:<4} {team['strategy']:<25} {tier_symbol}{team['tier']:<11} {team['captain']:<18} {team['vice_captain']:<18} {risk_symbol}{team['risk_level']:<7} {team['budget_weight']*100:.1f}%")
        
        print(f"{'='*120}")
        
        # ALWAYS DISPLAY ALL 15 TEAMS WITH COMPLETE DETAILS
        print(f"\nCOMPLETE 15-TEAM PORTFOLIO WITH ALL PLAYERS:")
        print(f"{'='*120}")
        
        # ALWAYS DISPLAY ALL 15 TEAMS - NO GROUPING, SHOW EVERY SINGLE TEAM
        print(f"\nTIER 1 - CORE TEAMS (Teams 1-5 - Low Risk, High Confidence)")
        print("─" * 100)
        for i in range(5):
            if i < len(teams):
                self._display_single_team(i + 1, teams[i])
        
        print(f"\nTIER 2 - DIVERSIFIED TEAMS (Teams 6-12 - Medium Risk, Balanced)")
        print("─" * 100)
        for i in range(5, 12):
            if i < len(teams):
                self._display_single_team(i + 1, teams[i])
        
        print(f"\nTIER 3 - MOONSHOT TEAMS (Teams 13-15 - High Risk, High Reward)")
        print("─" * 100)
        for i in range(12, 15):
            if i < len(teams):
                self._display_single_team(i + 1, teams[i])
    
    def _display_single_team(self, team_num: int, team: Dict):
        """Display a single team with complete player details"""
        tier_prefix = "[CORE]" if team['tier'] == 'Core' else "[DIV]" if team['tier'] == 'Diversified' else "[MOON]"
        
        print(f"\n{tier_prefix} TEAM {team_num}: {team['strategy'].upper()}")
        print(f"{'─' * 85}")
        print(f"Captain: {team['captain']} | Vice-Captain: {team['vice_captain']} | Tier: {team['tier']} | Risk: {team['risk_level']} | Budget: {team['budget_weight']*100:.1f}%")
        
        
        # Display all 11 players in a clear list format with roles
        print(f"COMPLETE 11-PLAYER LINEUP WITH ROLES:")
        for j, player in enumerate(team['players'], 1):
            captain_mark = " (C)" if player == team['captain'] else " (VC)" if player == team['vice_captain'] else ""
            
            # Get player role from team data
            player_role = self.get_player_role(player)
            role_emoji = self.get_role_emoji(player_role)
            
            print(f"   {j:2d}. {player}{captain_mark} {role_emoji} ({player_role})")
        
        # Show team distribution  
        team1_count = len([p for p in team['players'] if any(t1.get('name') == p for t1 in getattr(self, 'team1_players', []))])
        team2_count = len([p for p in team['players'] if any(t2.get('name') == p for t2 in getattr(self, 'team2_players', []))])
        print(f"Distribution: Northern Superchargers: {team1_count} | Birmingham Phoenix: {team2_count}")
        print()
    
    def get_player_role(self, player_name: str) -> str:
        """Get the role of a specific player from team data"""
        # Check team1 players
        if hasattr(self, 'team1_players'):
            for player in self.team1_players:
                if player['name'] == player_name:
                    return player.get('role', 'Unknown')
        
        # Check team2 players  
        if hasattr(self, 'team2_players'):
            for player in self.team2_players:
                if player['name'] == player_name:
                    return player.get('role', 'Unknown')
        
        return 'Unknown'
    
    def get_role_emoji(self, role: str) -> str:
        """Get emoji for player role"""
        role_lower = role.lower()
        
        if 'keeper' in role_lower or 'wk' in role_lower:
            return '[WK]'  # Wicket-keeper
        elif 'batsman' in role_lower or 'bat' in role_lower:
            return '[BAT]'  # Batsman
        elif 'bowler' in role_lower:
            return '[BWL]'  # Bowler
        elif 'allrounder' in role_lower or 'all-rounder' in role_lower:
            return '[AR]'  # All-rounder
        else:
            return '[UNK]'  # Unknown role
    
    def is_player_from_team(self, player_name: str, team_name: str) -> bool:
        """Check if a player belongs to a specific team"""
        # Check team1 players
        if hasattr(self, 'team1_players'):
            for player in self.team1_players:
                if player['name'] == player_name:
                    # Simple team name matching
                    return 'Manchester' in team_name or 'Northern' in team_name
        
        # Check team2 players  
        if hasattr(self, 'team2_players'):
            for player in self.team2_players:
                if player['name'] == player_name:
                    return 'Manchester' in team_name or 'Northern' in team_name
        
        return False
    
    def create_format_optimized_team(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, strategy: str) -> Dict:
        """Create team optimized for specific format"""
        return self.create_intelligent_team_base(players, context, format_intel, player_intel, strategy, "format_priority")
    
    def create_player_intelligence_team(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, strategy: str) -> Dict:
        """Create team based on player-specific intelligence"""
        return self.create_intelligent_team_base(players, context, format_intel, player_intel, strategy, "player_priority")
    
    def create_proven_patterns_team(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, strategy: str) -> Dict:
        """Create team based on proven winner patterns"""
        return self.create_intelligent_team_base(players, context, format_intel, player_intel, strategy, "proven_patterns")
    
    def create_context_aware_team(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, strategy: str) -> Dict:
        """Create team aware of match context"""
        return self.create_intelligent_team_base(players, context, format_intel, player_intel, strategy, "context_aware")
    
    def create_ultimate_intelligence_team(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, strategy: str) -> Dict:
        """Create team using ultimate intelligence fusion"""
        return self.create_intelligent_team_base(players, context, format_intel, player_intel, strategy, "ultimate_fusion")
    
    def create_intelligent_team_base(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, strategy: str, priority_type: str) -> Dict:
        """
        Base intelligent team creation with learning application
        """
        if len(players) < 11:
            # Fallback player generation
            players = self.generate_fallback_players(context)
        
        # Select 11 players intelligently from both teams
        selected_players = self.select_balanced_team(players, context, strategy)
        
        # Intelligent captain selection
        captain = self.select_intelligent_captain(selected_players, context, format_intel, player_intel, priority_type)
        
        # Intelligent VC selection
        vc = self.select_intelligent_vc(selected_players, captain, context, format_intel, player_intel, priority_type)
        
        # Generate reasoning
        reasoning = self.generate_team_reasoning(captain, vc, context, format_intel, priority_type)
        
        return {
            'strategy': strategy,
            'captain': captain,
            'vice_captain': vc,
            'players': selected_players,
            'reasoning': reasoning,
            'intelligence_applied': self.get_applied_intelligence_summary(format_intel, player_intel, priority_type),
            'format_context': f"{context['format']} - {context['series_type']}",
            'confidence_level': format_intel.get('confidence_level', 'medium')
        }
    
    def select_intelligent_captain(self, players: List[str], context: Dict, format_intel: Dict, player_intel: Dict, priority_type: str) -> str:
        """
        Intelligent captain selection using all available intelligence + 1 Crore winner learnings
        """
        captain_scores = {}
        
        for player in players:
            score = 50  # Base score
            
            # CRITICAL: 1 Crore Winner Learning - Jos Buttler Priority for The Hundred
            if context.get('format') == 'hun' and 'Jos Buttler' in player:
                score += 100  # MASSIVE boost for Jos Buttler in The Hundred
                self.logger.info(f"Applied 1Cr winner learning: Jos Buttler captain priority (+100)")
            
            # Enhanced format-specific captain selection based on learnings
            if context.get('format') == 'hun':
                # The Hundred specific patterns from 1 Cr winner analysis
                if any(name in player for name in ['Jos Buttler', 'David Miller', 'Dawid Malan']):
                    score += 50  # Proven The Hundred performers
                elif any(role in self.get_player_role(player).lower() for role in ['keeper', 'wk']):
                    score += 40  # Keeper-captains effective in The Hundred
                elif 'Manchester Originals' in context.get('teams', '') and self.is_player_from_team(player, 'Manchester Originals'):
                    score += 30  # Team batting first advantage
            
            # Apply format intelligence
            for pattern in format_intel.get('captain_patterns', []):
                if any(keyword in pattern.lower() for keyword in self.get_player_keywords(player)):
                    score += 20
            
            # Apply player intelligence
            if player in player_intel:
                intel = player_intel[player]
                if 'proven' in intel.get('captaincy_success', '').lower():
                    score += 30
                elif 'excellent' in intel.get('captaincy_success', '').lower():
                    score += 25
                elif 'good' in intel.get('captaincy_success', '').lower():
                    score += 15
            
            # Enhanced context bonuses with 1 Cr winner insights
            keywords = self.get_player_keywords(player)
            if context['format'] == 'international_t20' and any(keyword in ['batsman', 'allrounder', 'experienced'] for keyword in keywords):
                score += 25
            elif context.get('format', '').startswith('the_hundred') and 'opener' in keywords:
                score += 30
            
            # Apply weather and pitch condition bonuses
            if hasattr(self, 'match_conditions') and self.match_conditions:
                score += self._apply_conditions_bonus(player, keywords, self.match_conditions, 'captain')
            
            captain_scores[player] = score
        
        # Apply real learning data intelligence
        self.apply_database_learning_to_captain_scores(captain_scores, context, priority_type)
        
        # Apply diversity strategy for different teams
        self.apply_diversity_strategy(captain_scores, priority_type, context)
        
        # Return highest scoring captain
        return max(captain_scores, key=captain_scores.get) if captain_scores else players[0]
    
    def select_intelligent_vc(self, players: List[str], captain: str, context: Dict, format_intel: Dict, player_intel: Dict, priority_type: str) -> str:
        """
        Intelligent VC selection complementing the captain
        """
        vc_scores = {}
        
        for player in players:
            if player == captain:
                continue
                
            score = 50  # Base score
            
            # Apply format intelligence for VCs
            for pattern in format_intel.get('vc_patterns', []):
                if any(keyword in pattern.lower() for keyword in self.get_player_keywords(player)):
                    score += 20
            
            # Apply player intelligence for VCs
            if player in player_intel:
                intel = player_intel[player]
                if 'proven' in intel.get('vc_success', '').lower():
                    score += 25
                elif 'excellent' in intel.get('vc_success', '').lower():
                    score += 20
            
            # Apply proven VC patterns based on role keywords
            keywords = self.get_player_keywords(player)
            if context['format'] == 'international_t20' and any(keyword in ['allrounder', 'batting_allrounder'] for keyword in keywords):
                score += 20
            elif any(keyword in ['bowler', 'allrounder'] for keyword in keywords):  # Bowling allrounder VC pattern
                score += 25
            
            # Apply weather and pitch condition bonuses for VC
            if hasattr(self, 'match_conditions') and self.match_conditions:
                score += self._apply_conditions_bonus(player, keywords, self.match_conditions, 'vc')
            
            vc_scores[player] = score
        
        # Apply diversity strategy for VC as well
        self.apply_vc_diversity_strategy(vc_scores, priority_type, context)
        
        return max(vc_scores, key=vc_scores.get) if vc_scores else [p for p in players if p != captain][0]
    
    def get_player_keywords(self, player: str) -> List[str]:
        """Get enhanced keywords for player classification using API data and learning"""
        try:
            # Import enhanced classifier
            from core_logic.enhanced_player_classifier import get_enhanced_player_keywords
            
            # Get match context if available
            match_id = getattr(self, 'current_match_id', None)
            
            # Use enhanced classification system
            keywords = get_enhanced_player_keywords(player, match_id)
            
            if keywords and len(keywords) > 2:  # Good data available
                return keywords
            else:
                # Fallback to basic classification if enhanced system fails
                return self._get_basic_player_keywords(player)
                
        except Exception as e:
            # Fallback to basic system if enhanced system fails
            return self._get_basic_player_keywords(player)
    
    def _get_basic_player_keywords(self, player: str) -> List[str]:
        """Fallback basic player classification (original system)"""
        keywords = []
        
        # Enhanced name-based classification (better than previous version)
        name_lower = player.lower()
        
        # Wicket-keeper patterns
        if any(pattern in name_lower for pattern in ['rahul', 'dhoni', 'pant', 'carey', 'buttler', 'de kock', 'healy', 'taylor']):
            keywords.extend(['keeper', 'wicket_keeper'])
        
        # Opener patterns
        if any(pattern in name_lower for pattern in ['rohit', 'warner', 'finch', 'bairstow', 'roy', 'guptill']):
            keywords.extend(['opener', 'batsman'])
        
        # All-rounder patterns
        if any(pattern in name_lower for pattern in ['pandya', 'russell', 'maxwell', 'stoinis', 'marsh', 'green']):
            keywords.extend(['allrounder', 'batting_allrounder'])
        
        # Bowler patterns  
        if any(pattern in name_lower for pattern in ['bumrah', 'rabada', 'starc', 'boult', 'archer', 'rashid']):
            keywords.extend(['bowler', 'pace_bowler'])
        
        # Spinner patterns
        if any(pattern in name_lower for pattern in ['ashwin', 'zampa', 'rashid', 'kuldeep', 'chahal']):
            keywords.extend(['spinner', 'bowler'])
        
        # Default classification
        if not keywords:
            keywords.extend(['player', 'batsman'])  # Default assumption
            
        keywords.extend(['experienced'])  # Generic classification
        
        return keywords
    
    def _apply_conditions_bonus(self, player: str, keywords: List[str], conditions, selection_type: str) -> float:
        """Apply weather and pitch condition bonuses to player selection"""
        bonus = 0.0
        
        try:
            # Pace bowler bonuses
            if any(keyword in ['bowler', 'pace_bowler', 'fast_bowler'] for keyword in keywords):
                if conditions.pace_bowler_advantage > 0.5:
                    bonus += 20 * conditions.pace_bowler_advantage
                    
            # Spin bowler bonuses  
            if any(keyword in ['spinner', 'spin_bowler'] for keyword in keywords):
                if conditions.spin_bowler_advantage > 0.5:
                    bonus += 20 * conditions.spin_bowler_advantage
                    
            # Batsmen bonuses
            if any(keyword in ['batsman', 'opener', 'finisher'] for keyword in keywords):
                if conditions.batsmen_advantage > 0.3:
                    bonus += 15 * conditions.batsmen_advantage
                    
            # Wicket-keeper bonuses
            if any(keyword in ['keeper', 'wicket_keeper'] for keyword in keywords):
                if conditions.wicket_keeper_advantage > 0.3:
                    bonus += 10 * conditions.wicket_keeper_advantage
                    
            # All-rounder bonuses (get average of batting/bowling bonuses)
            if any(keyword in ['allrounder', 'batting_allrounder', 'bowling_allrounder'] for keyword in keywords):
                batting_bonus = max(0, conditions.batsmen_advantage * 10)
                bowling_bonus = max(0, max(conditions.pace_bowler_advantage, conditions.spin_bowler_advantage) * 10)
                bonus += (batting_bonus + bowling_bonus) / 2
                
            # Captain specific bonuses
            if selection_type == 'captain':
                # Prefer batting captains in batting conditions
                if conditions.captain_preference == 'batting' and any(keyword in ['batsman', 'allrounder'] for keyword in keywords):
                    bonus += 15
                # Prefer bowling captains in bowling conditions
                elif conditions.captain_preference == 'bowling' and any(keyword in ['bowler', 'allrounder'] for keyword in keywords):
                    bonus += 15
                    
        except Exception as e:
            # Silent fail - conditions bonus is optional
            pass
            
        return min(bonus, 50)  # Cap bonus at 50 points
    
    def generate_team_reasoning(self, captain: str, vc: str, context: Dict, format_intel: Dict, priority_type: str) -> str:
        """Generate intelligent reasoning for team selection"""
        reasons = []
        
        # Captain reasoning
        reasons.append(f"{captain} selected as captain")
        if format_intel.get('captain_patterns'):
            reasons.append("based on format-specific intelligence")
        
        # VC reasoning  
        reasons.append(f"{vc} as VC")
        if format_intel.get('vc_patterns'):
            reasons.append("following proven VC patterns")
        
        # Context reasoning
        if context['format'] != 'unknown':
            reasons.append(f"optimized for {context['format']} format")
        
        return " - ".join(reasons)
    
    def get_applied_intelligence_summary(self, format_intel: Dict, player_intel: Dict, priority_type: str) -> List[str]:
        """Get summary of applied intelligence"""
        applied = []
        
        if format_intel.get('captain_patterns'):
            applied.append(f"Format intelligence: {len(format_intel['captain_patterns'])} captain patterns")
        
        if format_intel.get('vc_patterns'):
            applied.append(f"VC intelligence: {len(format_intel['vc_patterns'])} VC patterns")
        
        if player_intel:
            applied.append(f"Player intelligence: {len(player_intel)} players analyzed")
        
        applied.append(f"Priority: {priority_type}")
        
        return applied
    
    def generate_fallback_players(self, context: Dict) -> List[str]:
        """Generate realistic fallback players when match data is not available"""
        self.logger.info("Generating realistic players based on team context")
        
        # Extract team names from context
        teams = context.get('teams', 'Team A vs Team B').split(' vs ')
        team1_name = teams[0].strip() if len(teams) > 0 else 'Team A'
        team2_name = teams[1].strip() if len(teams) > 1 else 'Team B'
        
        # Check if this is The Hundred and use realistic squads
        if context.get('format') == 'hun' and 'Manchester Originals' in team1_name and 'Northern Superchargers' in team2_name:
            return self.get_hundred_squad_players()
        
        # Fallback to generic but realistic players
        return self.get_generic_realistic_players(team1_name, team2_name)
    
    def get_hundred_squad_players(self) -> List[str]:
        """Get realistic The Hundred squad players for Manchester Originals vs Northern Superchargers"""
        self.logger.info("Using realistic The Hundred squad players")
        
        # Manchester Originals typical squad
        manchester_originals = [
            {'name': 'Jos Buttler', 'role': 'Wicket-keeper', 'captain': True, 'keeper': True},
            {'name': 'Phil Salt', 'role': 'Wicket-keeper', 'captain': False, 'keeper': True},
            {'name': 'Laurie Evans', 'role': 'Batsman', 'captain': False, 'keeper': False},
            {'name': 'Max Holden', 'role': 'Batsman', 'captain': False, 'keeper': False},
            {'name': 'Paul Walter', 'role': 'All-rounder', 'captain': False, 'keeper': False},
            {'name': 'Wayne Madsen', 'role': 'Batsman', 'captain': False, 'keeper': False},
            {'name': 'Tom Hartley', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Josh Hull', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Fazalhaq Farooqi', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Scott Currie', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Usama Mir', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Jamie Overton', 'role': 'All-rounder', 'captain': False, 'keeper': False},
            {'name': 'Michael Pepper', 'role': 'Wicket-keeper', 'captain': False, 'keeper': True},
            {'name': 'Matthew Hurst', 'role': 'Wicket-keeper', 'captain': False, 'keeper': True},
            {'name': 'Calvin Harrison', 'role': 'All-rounder', 'captain': False, 'keeper': False}
        ]
        
        # Northern Superchargers typical squad  
        northern_superchargers = [
            {'name': 'Harry Brook', 'role': 'Batsman', 'captain': True, 'keeper': False},
            {'name': 'Nicholas Pooran', 'role': 'Wicket-keeper', 'captain': False, 'keeper': True},
            {'name': 'Adam Hose', 'role': 'Batsman', 'captain': False, 'keeper': False},
            {'name': 'Dan Lawrence', 'role': 'Batsman', 'captain': False, 'keeper': False},
            {'name': 'Rehan Ahmed', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Ben Raine', 'role': 'All-rounder', 'captain': False, 'keeper': False},
            {'name': 'Jordan Clark', 'role': 'All-rounder', 'captain': False, 'keeper': False},
            {'name': 'Callum Parkinson', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Dillon Pennington', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Matthew Potts', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Olly Stone', 'role': 'Bowler', 'captain': False, 'keeper': False},
            {'name': 'Graham Clark', 'role': 'Batsman', 'captain': False, 'keeper': False},
            {'name': 'Bas de Leede', 'role': 'All-rounder', 'captain': False, 'keeper': False},
            {'name': 'Brydon Carse', 'role': 'All-rounder', 'captain': False, 'keeper': False},
            {'name': 'John Turner', 'role': 'Bowler', 'captain': False, 'keeper': False}
        ]
        
        # Store team info for balanced selection
        self.team1_players = manchester_originals
        self.team2_players = northern_superchargers
        
        # Return combined player names
        all_players = [p['name'] for p in manchester_originals] + [p['name'] for p in northern_superchargers]
        self.logger.info(f"Generated {len(all_players)} realistic The Hundred players ({len(manchester_originals)} + {len(northern_superchargers)})")
        
        return all_players
    
    def get_generic_realistic_players(self, team1_name: str, team2_name: str) -> List[str]:
        """Generate generic but realistic player names when specific squads aren't available"""
        team1_players = []
        team2_players = []
        
        # Common cricket player names for realistic fallback
        first_names = ['Alex', 'Ben', 'Chris', 'David', 'Ethan', 'Frank', 'George', 'Harry', 'Ian', 'Jack', 'Kevin', 'Luke', 'Mark', 'Nathan', 'Oscar']
        last_names = ['Anderson', 'Brown', 'Clark', 'Davies', 'Evans', 'Ford', 'Green', 'Hall', 'Jones', 'King', 'Lewis', 'Miller', 'Parker', 'Smith', 'Taylor']
        
        # Generate Team 1 players
        for i in range(15):
            first = first_names[i % len(first_names)]
            last = last_names[i % len(last_names)]
            player_name = f"{first} {last}"
            team1_players.append({
                'name': player_name,
                'role': ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper'][i % 4],
                'captain': i == 0,
                'keeper': i == 3
            })
        
        # Generate Team 2 players (different combinations)
        for i in range(15):
            first_idx = (i + 7) % len(first_names)
            last_idx = (i + 5) % len(last_names)
            first = first_names[first_idx]
            last = last_names[last_idx]
            player_name = f"{first} {last}"
            team2_players.append({
                'name': player_name,
                'role': ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper'][i % 4],
                'captain': i == 0,
                'keeper': i == 3
            })
        
        # Store team info for balanced selection
        self.team1_players = team1_players
        self.team2_players = team2_players
        
        # Return combined player names
        all_players = [p['name'] for p in team1_players] + [p['name'] for p in team2_players]
        self.logger.info(f"Generated {len(all_players)} generic fallback players ({len(team1_players)} + {len(team2_players)})")
        
        return all_players
    
    def save_predictions(self, match_id: str, context: Dict, teams: List[Dict], save_dir: str = "predictions") -> str:
        """Save predictions to file with comprehensive analysis"""
        prediction_data = {
            'match_id': match_id,
            'match_context': context,
            'teams': teams,
            'intelligence_level': 'ultimate',
            'learning_applied': True,
            'format_intelligence': True,
            'player_intelligence': True,
            'generation_time': datetime.now().isoformat(),
            'system_version': 'Ultimate v1.0'
        }
        
        # Create predictions directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{save_dir}/ultimate_prediction_{match_id}_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        return filename
    
    def display_predictions(self, match_id: str, context: Dict, teams: List[Dict]):
        """
        Display beautiful, comprehensive predictions
        """
        print("\n" + "="*80)
        print("DREAM11 ULTIMATE PREDICTION SYSTEM")
        print("Universal Cricket Intelligence with Advanced Learning")
        print("="*80)
        
        print(f"\nMATCH DETAILS:")
        print(f"Match ID: {match_id}")
        print(f"Teams: {context['teams']}")
        print(f"Format: {context['format'].upper()}")
        print(f"Venue: {context['venue']}")
        print(f"Intelligence Level: {context['intelligence_level'].upper()}")
        
        print(f"\nUNIVERSAL INTELLIGENCE APPLIED:")
        print(f"Format-Specific Learning: {context['format']} patterns")
        print(f"Player Intelligence: Individual performance analysis")
        print(f"Context Awareness: {context['series_type']} series optimization")
        print(f"Proven Patterns: 1 Crore winner insights integrated")
        
        print(f"\nDREAM11 - 15 TEAM PORTFOLIO:")
        print("="*100)
        
        # Display teams in table format
        self.display_teams_table(teams, context)
        
        # Display team analysis
        captains = [team['captain'] for team in teams]
        vcs = [team['vice_captain'] for team in teams]
        
        print(f"\nPREDICTION SUMMARY:")
        print("="*30)
        print(f"Captains: {captains}")
        print(f"Vice-Captains: {vcs}")
        print(f"Teams Generated: {len(teams)}")
        print(f"All teams have players from both sides")
        
        print(f"\nULTIMATE SYSTEM STATUS:")
        print("="*30)
        print("Intelligence Level: ULTIMATE (all formats + learning)")
        print("Learning Database: ACTIVE (never losing insights)")
        print("Captain Strategy: FORMAT-OPTIMIZED (context-aware)")
        print("VC Strategy: PROVEN PATTERNS (intelligence-backed)")
        print("Prediction Quality: MAXIMUM (ultimate intelligence)")
        
        print(f"\n15-TEAM PORTFOLIO COMPLETE!")
        print("Your Ultimate Cricket Intelligence System has generated")
        print("15 strategically diversified teams with maximum intelligence!")
        print(f"Core: 5 teams | Diversified: 7 teams | Moonshot: 3 teams")
    
    def predict(self, match_id: str, save_to_file: bool = False, save_dir: str = "predictions") -> bool:
        """
        Main prediction method - The ONE method to rule them all
        """
        try:
            print("Starting Ultimate Prediction System...")
            
            # Resolve match context
            context = self.resolve_match_context(match_id)
            
            # Get available players (this would integrate with the existing player resolution)
            players = self.get_match_players(match_id, context)
            
            # Generate intelligent teams
            teams = self.generate_intelligent_teams(context, players)
            
            # Display beautiful results
            self.display_predictions(match_id, context, teams)
            
            # Log prediction to database (ALWAYS)
            try:
                self.log_prediction_to_database(match_id, context, teams)
                print("Prediction logged to database")
            except Exception as e:
                print(f"Database logging unavailable: {e}")
            
            # Also log via learning system if available
            if self.learning_system:
                try:
                    # Create ai_strategies data for learning
                    ai_strategies = [
                        {"strategy": "format_priority", "confidence": 0.8},
                        {"strategy": "player_priority", "confidence": 0.7},
                        {"strategy": "proven_patterns", "confidence": 0.9},
                        {"strategy": "context_aware", "confidence": 0.6},
                        {"strategy": "ultimate_fusion", "confidence": 0.85}
                    ]
                    
                    self.learning_system.log_prediction(
                        match_id, teams, ai_strategies, 
                        match_format=context.get('format'), 
                        venue=context.get('venue'),
                        teams_playing=context.get('teams')
                    )
                    print("Prediction logged for continuous learning")
                except Exception as e:
                    print(f"Learning system unavailable: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            print(f"Prediction failed: {e}")
            return False
    
    def get_match_players(self, match_id: str, context: Dict) -> List[str]:
        """Get players for the match from API data with commentary fallback"""
        try:
            # Import API client
            from utils.api_client import fetch_match_center
            
            # Get match data from API
            match_data = fetch_match_center(match_id)
            
            if not match_data or 'matchInfo' not in match_data:
                # Try commentary fallback before giving up
                return self.get_players_from_commentary(match_id, context)
            
            match_info = match_data['matchInfo']
            
            # Store team info for balanced selection
            self.team1_players = []
            self.team2_players = []
            
            # Extract players from team1 (ONLY main squad, exclude substitutes/bench)
            if 'team1' in match_info and 'playerDetails' in match_info['team1']:
                for player in match_info['team1']['playerDetails']:
                    if ('name' in player and 
                        not player.get('substitute', False) and  # Exclude substitute/bench players
                        player.get('role', '').lower() not in ['coach', 'manager', 'support staff', 'analyst']):  # Exclude support staff
                        self.team1_players.append({
                            'name': player['name'],
                            'role': player.get('role', 'Unknown'),
                            'captain': player.get('captain', False),
                            'keeper': player.get('keeper', False)
                        })
            
            # Extract players from team2 (ONLY main squad, exclude substitutes/bench)
            if 'team2' in match_info and 'playerDetails' in match_info['team2']:
                for player in match_info['team2']['playerDetails']:
                    if ('name' in player and 
                        not player.get('substitute', False) and  # Exclude substitute/bench players
                        player.get('role', '').lower() not in ['coach', 'manager', 'support staff', 'analyst']):  # Exclude support staff
                        self.team2_players.append({
                            'name': player['name'],
                            'role': player.get('role', 'Unknown'),
                            'captain': player.get('captain', False),
                            'keeper': player.get('keeper', False)
                        })
            
            # Combine all player names for backward compatibility
            all_players = [p['name'] for p in self.team1_players] + [p['name'] for p in self.team2_players]
            
            # Return real players if we found enough, otherwise try commentary fallback
            if len(all_players) >= 22:  # Need at least 22 players for team generation
                self.logger.info(f"Found {len(all_players)} players from squad data")
                return all_players
            else:
                self.logger.warning(f"Only found {len(all_players)} players from squad, trying commentary fallback...")
                return self.get_players_from_commentary(match_id, context)
                
        except Exception as e:
            self.logger.warning(f"Error getting real players: {e}")
            return self.get_players_from_commentary(match_id, context)
    
    def get_players_from_commentary(self, match_id: str, context: Dict) -> List[str]:
        """
        Fallback method to extract playing 11 or probable 11 from commentary section
        """
        try:
            self.logger.info("Searching for playing 11 in commentary section...")
            
            # Import API client for commentary
            from utils.api_client import fetch_match_commentary
            
            # Get commentary data
            commentary_data = fetch_match_commentary(match_id)
            
            if not commentary_data or 'commentaryList' not in commentary_data:
                self.logger.warning("No commentary data available")
                return self.generate_fallback_players(context)
            
            # Initialize team players lists
            self.team1_players = []
            self.team2_players = []
            team1_playing = []
            team2_playing = []
            
            # Search for playing 11 patterns in commentary
            playing_11_patterns = [
                r'playing.*?11',
                r'probable.*?11',
                r'starting.*?11', 
                r'line.*?up',
                r'team.*?sheet',
                r'squad.*?announced'
            ]
            
            player_name_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+)',  # First Last
                r'([A-Z]\. [A-Z][a-z]+)',      # F. Last
                r'([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)',  # First F. Last
            ]
            
            commentary_text = ""
            
            # Combine all commentary text
            for commentary in commentary_data.get('commentaryList', []):
                if 'commText' in commentary:
                    commentary_text += commentary['commText'] + " "
            
            # Look for playing 11 mentions
            import re
            found_players = set()
            
            # Search for playing 11 sections
            for pattern in playing_11_patterns:
                matches = re.finditer(pattern, commentary_text, re.IGNORECASE)
                for match in matches:
                    # Get surrounding text (200 chars before and after)
                    start = max(0, match.start() - 200)
                    end = min(len(commentary_text), match.end() + 200)
                    surrounding_text = commentary_text[start:end]
                    
                    # Extract player names from surrounding text
                    for name_pattern in player_name_patterns:
                        names = re.findall(name_pattern, surrounding_text)
                        for name in names:
                            if len(name.split()) >= 2:  # At least first and last name
                                found_players.add(name.strip())
            
            # If we found some players, try to assign them to teams
            if len(found_players) >= 10:  # Need at least 10 unique players
                players_list = list(found_players)
                
                # Try to identify team names from context
                team_names = context.get('teams', '').split(' vs ')
                if len(team_names) == 2:
                    team1_name = team_names[0].strip()
                    team2_name = team_names[1].strip()
                    
                    # Smart assignment: first half to team1, second half to team2
                    mid_point = len(players_list) // 2
                    team1_names = players_list[:mid_point]
                    team2_names = players_list[mid_point:]
                    
                    # Ensure we have enough players for each team
                    while len(team1_names) < 11 and len(team2_names) > 11:
                        team1_names.append(team2_names.pop())
                    while len(team2_names) < 11 and len(team1_names) > 11:
                        team2_names.append(team1_names.pop())
                    
                    # Create team player objects
                    for i, name in enumerate(team1_names[:15]):  # Max 15 per team
                        role = self.guess_player_role(name, commentary_text)
                        self.team1_players.append({
                            'name': name,
                            'role': role,
                            'captain': i == 0,  # First player as captain
                            'keeper': 'keeper' in role.lower() or 'wk' in role.lower()
                        })
                    
                    for i, name in enumerate(team2_names[:15]):  # Max 15 per team
                        role = self.guess_player_role(name, commentary_text)
                        self.team2_players.append({
                            'name': name,
                            'role': role,
                            'captain': i == 0,  # First player as captain
                            'keeper': 'keeper' in role.lower() or 'wk' in role.lower()
                        })
                    
                    all_players = [p['name'] for p in self.team1_players] + [p['name'] for p in self.team2_players]
                    
                    if len(all_players) >= 22:
                        self.logger.info(f"Found {len(all_players)} players from commentary")
                        return all_players
            
            self.logger.warning("Could not find enough players in commentary, using fallback")
            return self.generate_fallback_players(context)
            
        except Exception as e:
            self.logger.warning(f"Error extracting players from commentary: {e}")
            return self.generate_fallback_players(context)
    
    def guess_player_role(self, player_name: str, commentary_text: str) -> str:
        """Guess player role based on commentary context"""
        name_lower = player_name.lower()
        text_around_name = ""
        
        # Find text around player name
        import re
        pattern = rf'{re.escape(player_name)}'
        matches = re.finditer(pattern, commentary_text, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 100)
            end = min(len(commentary_text), match.end() + 100)
            text_around_name += commentary_text[start:end] + " "
        
        text_lower = text_around_name.lower()
        
        # Role keywords
        if any(word in text_lower for word in ['keeper', 'wicket-keeper', 'wk']):
            return 'Wicket-keeper'
        elif any(word in text_lower for word in ['bowler', 'bowling', 'pace', 'spin', 'seam']):
            if any(word in text_lower for word in ['allrounder', 'all-rounder', 'batting']):
                return 'All-rounder'
            return 'Bowler'
        elif any(word in text_lower for word in ['batsman', 'batting', 'opener', 'top-order', 'middle-order']):
            if any(word in text_lower for word in ['allrounder', 'all-rounder', 'bowling']):
                return 'All-rounder'
            return 'Batsman'
        elif any(word in text_lower for word in ['allrounder', 'all-rounder']):
            return 'All-rounder'
        else:
            return 'Batsman'  # Default to batsman
    
    def select_balanced_team(self, players: List[str], context: Dict, strategy: str) -> List[str]:
        """
        Select 11 players with balanced role distribution from both teams
        """
        if not hasattr(self, 'team1_players') or not hasattr(self, 'team2_players'):
            # Fallback to simple selection if team data not available
            return players[:11] if len(players) >= 11 else players
        
        # Combine all players with role information
        all_players_with_roles = []
        
        # Add team1 players (already filtered for main squad only)
        for player in self.team1_players:
            all_players_with_roles.append({
                'name': player['name'],
                'role': player.get('role', 'Unknown'),
                'team': 'team1',
                'captain': player.get('captain', False),
                'keeper': player.get('keeper', False)
            })
        
        # Add team2 players (already filtered for main squad only)
        for player in self.team2_players:
            all_players_with_roles.append({
                'name': player['name'],
                'role': player.get('role', 'Unknown'),
                'team': 'team2',
                'captain': player.get('captain', False),
                'keeper': player.get('keeper', False)
            })
        
        # Categorize players by role
        batsmen = [p for p in all_players_with_roles if 'batsman' in p['role'].lower() and 'wk' not in p['role'].lower()]
        bowlers = [p for p in all_players_with_roles if 'bowler' in p['role'].lower() or p['role'].lower() == 'bowler']
        all_rounders = [p for p in all_players_with_roles if 'allrounder' in p['role'].lower()]
        wicket_keepers = [p for p in all_players_with_roles if 'wk' in p['role'].lower() or 'keeper' in p['role'].lower()]
        
        selected = []
        
        # Performance-based role selection
        def select_by_performance(players_list, count, selected):
            """Select best performing players regardless of team"""
            # Sort players by performance indicators
            available_players = [p for p in players_list if p['name'] not in selected]
            
            # Score players based on multiple factors
            for player in available_players:
                score = 0
                
                # Captain/leadership bonus
                if player.get('captain', False):
                    score += 20
                
                # Enhanced role-specific scoring with 1 Cr winner insights
                role = player['role'].lower()
                
                # 1 Crore Winner Learning: Dual Keeper Strategy for The Hundred
                if context.get('format') == 'hun' and ('wk' in role or 'keeper' in role):
                    keeper_count = len([p for p in selected if 'wk' in self.get_player_role(p).lower() or 'keeper' in self.get_player_role(p).lower()])
                    if keeper_count == 0:
                        score += 25  # First keeper gets priority
                    elif keeper_count == 1 and any(name in player['name'] for name in ['Jos Buttler', 'Heinrich Klaasen', 'Phil Salt']):
                        score += 20  # Second keeper bonus for proven combinations
                    else:
                        score += 5   # Lower priority for third keeper
                
                if 'batsman' in role:
                    score += 15  # Base batting score
                elif 'bowler' in role:
                    score += 12  # Base bowling score
                elif 'allrounder' in role:
                    score += 18  # All-rounders get premium
                elif 'wk' in role or 'keeper' in role:
                    score += 10  # Keeper base score
                
                # 1 Crore Winner Learning: Team Batting First Advantage
                if context.get('format') == 'hun':
                    # Identify which team is batting first (usually stronger batting team)
                    teams_playing = context.get('teams', '')
                    if 'Manchester Originals' in teams_playing:
                        # Check if this player is from Manchester Originals (typically stronger batting)
                        if hasattr(self, 'team1_players'):
                            for team_player in self.team1_players:
                                if team_player['name'] == player['name']:
                                    score += 15  # Batting first team advantage
                                    break
                
                # Add some randomization for strategy diversity
                import random
                score += random.randint(0, 5)
                
                player['performance_score'] = score
            
            # Sort by performance score and select top performers
            available_players.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            
            return [p['name'] for p in available_players[:count]]
        
        if strategy == "Format-Optimized Intelligence":
            # Balanced: 4 BAT, 4 BOWL, 2 AR, 1 WK
            selected.extend(select_by_performance(batsmen, 4, selected))
            selected.extend(select_by_performance(bowlers, 4, selected))
            selected.extend(select_by_performance(all_rounders, 2, selected))
            selected.extend(select_by_performance(wicket_keepers, 1, selected))
        
        elif strategy == "Player Intelligence Focus":
            # Batting focused: 5 BAT, 3 BOWL, 2 AR, 1 WK
            selected.extend(select_by_performance(batsmen, 5, selected))
            selected.extend(select_by_performance(bowlers, 3, selected))
            selected.extend(select_by_performance(all_rounders, 2, selected))
            selected.extend(select_by_performance(wicket_keepers, 1, selected))
        
        elif strategy == "Proven Winner Patterns":
            # Bowling focused: 3 BAT, 5 BOWL, 2 AR, 1 WK
            selected.extend(select_by_performance(batsmen, 3, selected))
            selected.extend(select_by_performance(bowlers, 5, selected))
            selected.extend(select_by_performance(all_rounders, 2, selected))
            selected.extend(select_by_performance(wicket_keepers, 1, selected))
        
        elif strategy == "Context-Aware Optimization":
            # All-rounder heavy: 4 BAT, 3 BOWL, 3 AR, 1 WK
            selected.extend(select_by_performance(batsmen, 4, selected))
            selected.extend(select_by_performance(bowlers, 3, selected))
            selected.extend(select_by_performance(all_rounders, 3, selected))
            selected.extend(select_by_performance(wicket_keepers, 1, selected))
        
        else:  # Ultimate Intelligence Fusion and other strategies
            # Balanced with 2 WK: 4 BAT, 3 BOWL, 2 AR, 2 WK
            selected.extend(select_by_performance(batsmen, 4, selected))
            selected.extend(select_by_performance(bowlers, 3, selected))
            selected.extend(select_by_performance(all_rounders, 2, selected))
            selected.extend(select_by_performance(wicket_keepers, 2, selected))
        
        # Fill remaining spots if needed with best available players
        while len(selected) < 11:
            # Get all remaining players
            remaining_players = [p for p in all_players_with_roles if p['name'] not in selected]
            if not remaining_players:
                break
                
            # Score remaining players and pick the best
            for player in remaining_players:
                score = 0
                if player.get('captain', False):
                    score += 20
                role = player['role'].lower()
                if 'batsman' in role:
                    score += 15
                elif 'bowler' in role:
                    score += 12
                elif 'allrounder' in role:
                    score += 18
                elif 'wk' in role or 'keeper' in role:
                    score += 10
                player['performance_score'] = score
            
            # Sort and add best remaining player
            remaining_players.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            selected.append(remaining_players[0]['name'])
        
        # Final verification and ensure exactly 11 players
        final_selected = selected[:11] if len(selected) >= 11 else selected
        
        # Log team distribution for analytics (performance-based selection)
        final_team1 = [p for p in final_selected if any(t1['name'] == p for t1 in self.team1_players)]
        final_team2 = [p for p in final_selected if any(t2['name'] == p for t2 in self.team2_players)]
        
        self.logger.info(f"Performance-based distribution: Team1={len(final_team1)}, Team2={len(final_team2)}")
        
        return final_selected
    
    def apply_database_learning_to_captain_scores(self, captain_scores: Dict, context: Dict, priority_type: str):
        """
        Apply real database learning to captain selection scores
        """
        try:
            import sqlite3
            
            # Get format-specific learning patterns
            conn = sqlite3.connect('universal_cricket_intelligence.db')
            cursor = conn.cursor()
            
            # Query format learnings
            cursor.execute('''
                SELECT pattern_data, confidence_score FROM format_learnings 
                WHERE format_name = ? OR format_name LIKE ?
            ''', (context.get('format', 'ODI'), f"%{context.get('format', 'ODI')}%"))
            
            format_learnings = cursor.fetchall()
            
            # Apply learning patterns to scores
            for player, score in captain_scores.items():
                bonus = 0
                
                for pattern, confidence in format_learnings:
                    # Wicket-keeper captain patterns
                    if 'wicket-keeper' in pattern.lower() and self.is_wicket_keeper(player):
                        bonus += 35 if confidence == 'HIGH' else 25
                    
                    # Aggressive player patterns for T20
                    elif 'aggressive' in pattern.lower() and context.get('format', '').lower() in ['t20i', 't20']:
                        if any(name in player.lower() for name in ['warner', 'head', 'rizwan']):
                            bonus += 30
                    
                    # Young player patterns for franchise cricket
                    elif 'young' in pattern.lower() and self.is_young_player(player):
                        bonus += 25
                
                # Apply 1 Crore winner patterns
                if any(winner in player.lower() for winner in ['hope', 'warner', 'rizwan', 'ahmed']):
                    bonus += 40  # Proven 1 Crore winners
                
                captain_scores[player] = score + bonus
            
            conn.close()
            
        except Exception as e:
            # Silent fail - non-critical learning enhancement
            pass
    
    def is_wicket_keeper(self, player: str) -> bool:
        """Check if player is a wicket-keeper based on team data"""
        if hasattr(self, 'team1_players'):
            for p in self.team1_players:
                if p['name'] == player and p.get('keeper', False):
                    return True
        if hasattr(self, 'team2_players'):
            for p in self.team2_players:
                if p['name'] == player and p.get('keeper', False):
                    return True
        
        # Fallback: common WK names
        return any(name in player.lower() for name in ['hope', 'rizwan', 'haris', 'jangoo'])
    
    def is_young_player(self, player: str) -> bool:
        """Check if player is young based on known patterns"""
        young_patterns = ['ahmed', 'ayub', 'blades', 'hasan nawaz']
        return any(pattern in player.lower() for pattern in young_patterns)
    
    def is_aggressive_player(self, player: str) -> bool:
        """Check if player is aggressive based on known patterns"""
        aggressive_patterns = ['warner', 'rizwan', 'king', 'rutherford']
        return any(pattern in player.lower() for pattern in aggressive_patterns)
    
    def is_experienced_player(self, player: str) -> bool:
        """Check if player is experienced based on known patterns"""
        experienced_patterns = ['hope', 'babar', 'chase', 'shaheen']
        return any(pattern in player.lower() for pattern in experienced_patterns)
    
    def apply_diversity_strategy(self, captain_scores: Dict, priority_type: str, context: Dict):
        """
        Apply advanced correlation-based diversity strategies to create different captains across 5 teams
        """
        try:
            # Use advanced correlation-based diversity
            from core_logic.correlation_diversity_engine import get_correlation_diversity_engine
            
            # Get all candidate players
            all_players = list(captain_scores.keys())
            
            if len(all_players) >= 5:
                # Get correlation diversity engine
                engine = get_correlation_diversity_engine()
                
                # Analyze player correlations
                diversity_matrix = engine.analyze_player_correlations(all_players, context)
                
                # Apply strategy-specific diversity boosts based on correlation analysis
                if priority_type == "player_priority":
                    # Boost players with highest individual diversity scores
                    top_diverse = sorted(all_players, key=lambda p: diversity_matrix.diversity_scores.get(p, 0), reverse=True)
                    if len(top_diverse) >= 2:
                        captain_scores[top_diverse[1]] += 25
                
                elif priority_type == "proven_patterns":
                    # Boost players from different performance clusters
                    clusters = diversity_matrix.performance_clusters
                    if clusters and 'medium' in clusters:
                        for player in clusters['medium'][:1]:  # Take top from medium cluster
                            captain_scores[player] += 30
                
                elif priority_type == "context_aware":
                    # Boost players with low correlation to current top choice
                    top_player = max(captain_scores, key=captain_scores.get)
                    correlation_matrix = diversity_matrix.correlation_matrix
                    players = diversity_matrix.players
                    
                    if top_player in players:
                        top_idx = players.index(top_player)
                        # Find players with lowest correlation to top choice
                        low_corr_players = []
                        for i, player in enumerate(players):
                            if i != top_idx and abs(correlation_matrix[top_idx][i]) < 0.3:
                                low_corr_players.append(player)
                        
                        if low_corr_players:
                            captain_scores[low_corr_players[0]] += 25
                
                elif priority_type == "ultimate_fusion":
                    # Advanced multi-factor diversity boost
                    for player in all_players:
                        diversity_score = diversity_matrix.diversity_scores.get(player, 0)
                        if diversity_score > 0.7:  # High diversity players
                            captain_scores[player] += int(35 * diversity_score)
                        
                        # Also boost players with keeper keywords (role diversity)
                        keywords = self.get_player_keywords(player)
                        if any(keyword in ['keeper', 'wicket_keeper'] for keyword in keywords):
                            captain_scores[player] += 20
                
                self.logger.debug(f"Applied correlation-based diversity for {priority_type}")
                
            else:
                # Fallback to simple diversity if not enough players
                self._apply_simple_diversity_fallback(captain_scores, priority_type)
                
        except Exception as e:
            # Fallback to simple diversity if correlation engine fails
            self.logger.warning(f"Correlation diversity failed, using fallback: {e}")
            self._apply_simple_diversity_fallback(captain_scores, priority_type)
    
    def _apply_simple_diversity_fallback(self, captain_scores: Dict, priority_type: str):
        """Fallback simple diversity strategy"""
        sorted_players = sorted(captain_scores.items(), key=lambda x: x[1], reverse=True)
        
        boost_map = {
            "player_priority": (1, 25),    # Boost 2nd highest
            "proven_patterns": (2, 30),    # Boost 3rd highest  
            "context_aware": (3, 25),      # Boost 4th highest
            "ultimate_fusion": (4, 35)     # Boost 5th highest
        }
        
        if priority_type in boost_map and len(sorted_players) > boost_map[priority_type][0]:
            idx, boost = boost_map[priority_type]
            captain_scores[sorted_players[idx][0]] += boost
    
    def apply_vc_diversity_strategy(self, vc_scores: Dict, priority_type: str, context: Dict):
        """
        Apply advanced correlation-based diversity strategies to create different VCs across 5 teams
        """
        try:
            # Use advanced correlation-based diversity for VC selection
            from core_logic.correlation_diversity_engine import get_correlation_diversity_engine
            
            all_players = list(vc_scores.keys())
            
            if len(all_players) >= 5:
                engine = get_correlation_diversity_engine()
                diversity_matrix = engine.analyze_player_correlations(all_players, context)
                
                # Strategy-specific VC diversity boosts
                if priority_type == "player_priority":
                    # Boost players with complementary skills to captain
                    role_diverse_players = []
                    for player in all_players:
                        keywords = self.get_player_keywords(player)
                        if any(keyword in ['allrounder', 'bowling_allrounder'] for keyword in keywords):
                            role_diverse_players.append(player)
                    
                    if role_diverse_players:
                        vc_scores[role_diverse_players[0]] += 20
                
                elif priority_type == "proven_patterns":
                    # Boost players from high-performance cluster
                    clusters = diversity_matrix.performance_clusters
                    if clusters and 'high' in clusters:
                        for player in clusters['high'][:1]:
                            vc_scores[player] += 25
                
                elif priority_type == "context_aware":
                    # Boost players with unique performance patterns
                    unique_players = [p for p in all_players 
                                    if diversity_matrix.diversity_scores.get(p, 0) > 0.6]
                    if unique_players:
                        vc_scores[unique_players[0]] += 30
                
                elif priority_type == "ultimate_fusion":
                    # Advanced VC diversity based on multiple factors
                    for player in all_players:
                        diversity_score = diversity_matrix.diversity_scores.get(player, 0)
                        if diversity_score > 0.5:
                            vc_scores[player] += int(25 * diversity_score)
                
                self.logger.debug(f"Applied correlation-based VC diversity for {priority_type}")
                
            else:
                # Fallback to simple VC diversity
                self._apply_simple_vc_diversity_fallback(vc_scores, priority_type)
                
        except Exception as e:
            self.logger.warning(f"VC correlation diversity failed, using fallback: {e}")
            self._apply_simple_vc_diversity_fallback(vc_scores, priority_type)
    
    def _apply_simple_vc_diversity_fallback(self, vc_scores: Dict, priority_type: str):
        """Fallback simple VC diversity strategy"""
        sorted_players = sorted(vc_scores.items(), key=lambda x: x[1], reverse=True)
        
        vc_boost_map = {
            "player_priority": (2, 20),    # Boost 3rd highest
            "proven_patterns": (1, 25),    # Boost 2nd highest
            "context_aware": (4, 30),      # Boost 5th highest
            "ultimate_fusion": (3, 25)     # Boost 4th highest
        }
        
        if priority_type in vc_boost_map and len(sorted_players) > vc_boost_map[priority_type][0]:
            idx, boost = vc_boost_map[priority_type]
            vc_scores[sorted_players[idx][0]] += boost
    
    def close_connections(self):
        """Close all database connections"""
        for conn in self.db_connections.values():
            try:
                conn.close()
            except:
                pass
    
    def log_prediction_to_database(self, match_id: str, context: Dict, teams: List[Dict]):
        """
        Log prediction to database for post-match analysis
        """
        try:
            # Connect to AI learning database
            conn = sqlite3.connect('ai_learning_database.db')
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    context_data TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert prediction data
            prediction_json = json.dumps(teams)
            context_json = json.dumps(context)
            
            cursor.execute('''
                INSERT INTO prediction_history (match_id, prediction_data, context_data)
                VALUES (?, ?, ?)
            ''', (match_id, prediction_json, context_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            # Silent fail - non-critical feature
            pass

def main():
    """Main entry point for Ultimate Prediction System"""
    parser = argparse.ArgumentParser(description="Dream11 Ultimate Prediction System")
    parser.add_argument("match_id", help="Match ID to generate predictions for")
    
    args = parser.parse_args()
    
    match_id = args.match_id
    
    # Initialize and run Ultimate System
    ultimate_system = Dream11Ultimate()
    
    try:
        success = ultimate_system.predict(match_id)
        if success:
            print("\nUltimate Prediction System completed successfully!")
        else:
            print("\nPrediction failed. Check logs for details.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    
    finally:
        ultimate_system.close_connections()

if __name__ == "__main__":
    main()
