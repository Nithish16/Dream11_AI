#!/usr/bin/env python3
"""
🗄️ Database Storage Manager for Smart 15 Predictions
Efficient database-only storage with advanced querying capabilities
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

class DatabaseStorageManager:
    """
    Manages all Smart 15 predictions in database-only storage
    No JSON files created unless explicitly requested
    """
    
    def __init__(self, db_path: str = "smart15_predictions.db"):
        self.db_path = db_path
        self.setup_logging()
        self.init_database()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def init_database(self):
        """Initialize comprehensive database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS smart15_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                system_version TEXT DEFAULT 'Enhanced Smart15 v1.0',
                total_budget REAL DEFAULT 1000.0,
                
                -- Match context
                match_format TEXT,
                venue TEXT,
                teams TEXT,
                series_type TEXT,
                intelligence_level TEXT,
                
                -- Portfolio metrics
                diversification_score REAL,
                captain_diversity_score REAL,
                unique_captains INTEGER,
                unique_players INTEGER,
                avg_team_overlap REAL,
                max_captain_usage INTEGER,
                
                -- Analysis results
                portfolio_analysis TEXT, -- JSON blob for complex analysis
                generation_notes TEXT
            )
        ''')
        
        # Create indexes separately
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_id ON smart15_predictions(match_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON smart15_predictions(prediction_timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diversification ON smart15_predictions(diversification_score)')
        
        # Individual teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES smart15_predictions(id),
                team_number INTEGER,
                tier INTEGER, -- 1=Core, 2=Diversified, 3=Moonshot
                tier_name TEXT,
                
                -- Team details
                strategy TEXT NOT NULL,
                captain TEXT NOT NULL,
                vice_captain TEXT NOT NULL,
                confidence_level TEXT,
                risk_level TEXT,
                diversification_score REAL,
                budget_weight REAL,
                
                -- Team composition (stored as JSON for efficiency)
                players TEXT, -- JSON array of player names
                reasoning TEXT,
                intelligence_applied TEXT -- JSON array
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_id ON prediction_teams(prediction_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_captain ON prediction_teams(captain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tier ON prediction_teams(tier)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy ON prediction_teams(strategy)')
        
        # Captain analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captain_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES smart15_predictions(id),
                captain_name TEXT NOT NULL,
                usage_count INTEGER,
                usage_percentage REAL,
                tiers_used TEXT, -- JSON array of tiers
                strategies_used TEXT -- JSON array of strategies
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_captain ON captain_analysis(prediction_id, captain_name)')
        
        # Player usage analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_usage_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES smart15_predictions(id),
                player_name TEXT NOT NULL,
                usage_count INTEGER,
                usage_category TEXT, -- core/frequent/differential
                teams_used TEXT -- JSON array of team numbers
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_player ON player_usage_analysis(prediction_id, player_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_category ON player_usage_analysis(usage_category)')
        
        # Quick access view for latest predictions
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS latest_predictions AS
            SELECT 
                sp.*,
                COUNT(pt.id) as total_teams,
                GROUP_CONCAT(DISTINCT pt.captain) as all_captains
            FROM smart15_predictions sp
            LEFT JOIN prediction_teams pt ON sp.id = pt.prediction_id
            GROUP BY sp.id
            ORDER BY sp.prediction_timestamp DESC
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info(f"✅ Database schema initialized: {self.db_path}")
    
    def store_smart15_prediction(self, match_id: str, portfolio: Dict, 
                                system_version: str = "Enhanced Smart15 v1.0") -> int:
        """
        Store complete Smart 15 prediction in database
        Returns prediction_id for reference
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract portfolio data
            match_context = portfolio.get('match_context', {})
            diversification = portfolio.get('diversification_analysis', {})
            captain_dist = portfolio.get('captain_distribution', {})
            budget_allocation = portfolio.get('budget_allocation', {})
            
            # Insert main prediction record
            cursor.execute('''
                INSERT INTO smart15_predictions (
                    match_id, system_version, total_budget,
                    match_format, venue, teams, series_type, intelligence_level,
                    diversification_score, captain_diversity_score, unique_captains,
                    unique_players, avg_team_overlap, max_captain_usage,
                    portfolio_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id, system_version, budget_allocation.get('total_budget', 1000.0),
                match_context.get('format'), match_context.get('venue'), 
                match_context.get('teams'), match_context.get('series_type'),
                match_context.get('intelligence_level'),
                diversification.get('diversification_score', 0),
                captain_dist.get('captain_diversity_score', 0),
                captain_dist.get('unique_captains', 0),
                diversification.get('unique_players_used', 0),
                diversification.get('average_team_overlap', 0),
                captain_dist.get('max_captain_usage', 0),
                json.dumps(portfolio.get('portfolio_summary', {}))
            ))
            
            prediction_id = cursor.lastrowid
            
            # Store individual teams
            all_teams = (
                portfolio.get('tier1_core_teams', []) +
                portfolio.get('tier2_diversified_teams', []) +
                portfolio.get('tier3_moonshot_teams', [])
            )
            
            for team_idx, team in enumerate(all_teams, 1):
                cursor.execute('''
                    INSERT INTO prediction_teams (
                        prediction_id, team_number, tier, tier_name,
                        strategy, captain, vice_captain, confidence_level,
                        risk_level, diversification_score, budget_weight,
                        players, reasoning, intelligence_applied
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_id, team_idx, team.get('tier_number', 1),
                    team.get('tier', 'Unknown'),
                    team.get('strategy', ''), team.get('captain', ''),
                    team.get('vice_captain', ''), team.get('confidence_level', ''),
                    team.get('risk_level', ''), team.get('diversification_score', 0),
                    team.get('budget_weight', 0),
                    json.dumps(team.get('players', [])),
                    team.get('reasoning', ''),
                    json.dumps(team.get('intelligence_applied', []))
                ))
            
            # Store captain analysis
            captain_distribution = captain_dist.get('captain_distribution', {})
            for captain, count in captain_distribution.items():
                percentage = (count / len(all_teams)) * 100 if all_teams else 0
                
                # Find tiers and strategies this captain is used in
                tiers_used = []
                strategies_used = []
                for team in all_teams:
                    if team.get('captain') == captain:
                        tiers_used.append(team.get('tier', ''))
                        strategies_used.append(team.get('strategy', ''))
                
                cursor.execute('''
                    INSERT INTO captain_analysis (
                        prediction_id, captain_name, usage_count, usage_percentage,
                        tiers_used, strategies_used
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_id, captain, count, percentage,
                    json.dumps(list(set(tiers_used))),
                    json.dumps(strategies_used)
                ))
            
            # Store player usage analysis
            all_players = []
            for team in all_teams:
                all_players.extend(team.get('players', []))
            
            from collections import Counter
            player_counts = Counter(all_players)
            
            for player, count in player_counts.items():
                # Categorize player usage
                if count >= 10:
                    category = 'core'
                elif count >= 6:
                    category = 'frequent'
                else:
                    category = 'differential'
                
                # Find which teams this player is in
                teams_used = []
                for team_idx, team in enumerate(all_teams, 1):
                    if player in team.get('players', []):
                        teams_used.append(team_idx)
                
                cursor.execute('''
                    INSERT INTO player_usage_analysis (
                        prediction_id, player_name, usage_count, usage_category, teams_used
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    prediction_id, player, count, category, json.dumps(teams_used)
                ))
            
            conn.commit()
            self.logger.info(f"✅ Stored Smart 15 prediction for match {match_id} (ID: {prediction_id})")
            return prediction_id
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"❌ Error storing prediction: {e}")
            raise
        finally:
            conn.close()
    
    def get_prediction_summary(self, match_id: str = None, prediction_id: int = None) -> Dict:
        """Get prediction summary from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if prediction_id:
                cursor.execute('SELECT * FROM latest_predictions WHERE id = ?', (prediction_id,))
            elif match_id:
                cursor.execute('SELECT * FROM latest_predictions WHERE match_id = ? ORDER BY prediction_timestamp DESC LIMIT 1', (match_id,))
            else:
                cursor.execute('SELECT * FROM latest_predictions ORDER BY prediction_timestamp DESC LIMIT 1')
            
            result = cursor.fetchone()
            if not result:
                return {}
            
            # Convert to dict
            columns = [desc[0] for desc in cursor.description]
            prediction_data = dict(zip(columns, result))
            
            return prediction_data
            
        finally:
            conn.close()
    
    def get_detailed_teams(self, prediction_id: int) -> List[Dict]:
        """Get detailed team information for a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM prediction_teams 
                WHERE prediction_id = ? 
                ORDER BY team_number
            ''', (prediction_id,))
            
            teams = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                team_data = dict(zip(columns, row))
                # Parse JSON fields
                team_data['players'] = json.loads(team_data['players'])
                team_data['intelligence_applied'] = json.loads(team_data['intelligence_applied'])
                teams.append(team_data)
            
            return teams
            
        finally:
            conn.close()
    
    def get_captain_analysis(self, prediction_id: int) -> List[Dict]:
        """Get captain analysis for a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM captain_analysis 
                WHERE prediction_id = ? 
                ORDER BY usage_count DESC
            ''', (prediction_id,))
            
            analysis = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                data['tiers_used'] = json.loads(data['tiers_used'])
                data['strategies_used'] = json.loads(data['strategies_used'])
                analysis.append(data)
            
            return analysis
            
        finally:
            conn.close()
    
    def get_player_usage_analysis(self, prediction_id: int) -> Dict:
        """Get player usage analysis for a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT usage_category, COUNT(*) as count,
                       GROUP_CONCAT(player_name) as players
                FROM player_usage_analysis 
                WHERE prediction_id = ?
                GROUP BY usage_category
                ORDER BY 
                    CASE usage_category 
                        WHEN 'core' THEN 1 
                        WHEN 'frequent' THEN 2 
                        WHEN 'differential' THEN 3 
                    END
            ''', (prediction_id,))
            
            usage_analysis = {}
            for row in cursor.fetchall():
                category, count, players = row
                usage_analysis[category] = {
                    'count': count,
                    'players': players.split(',') if players else []
                }
            
            return usage_analysis
            
        finally:
            conn.close()
    
    def list_predictions(self, limit: int = 10) -> List[Dict]:
        """List recent predictions with summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, match_id, prediction_timestamp, total_budget,
                       match_format, venue, teams, diversification_score,
                       unique_captains, total_teams
                FROM latest_predictions 
                ORDER BY prediction_timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            predictions = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                predictions.append(dict(zip(columns, row)))
            
            return predictions
            
        finally:
            conn.close()
    
    def export_prediction_to_json(self, prediction_id: int, output_file: str = None) -> str:
        """Export specific prediction to JSON file (optional functionality)"""
        # Get complete prediction data
        summary = self.get_prediction_summary(prediction_id=prediction_id)
        teams = self.get_detailed_teams(prediction_id)
        captain_analysis = self.get_captain_analysis(prediction_id)
        player_usage = self.get_player_usage_analysis(prediction_id)
        
        # Construct complete portfolio
        export_data = {
            'prediction_id': prediction_id,
            'match_id': summary.get('match_id'),
            'prediction_timestamp': summary.get('prediction_timestamp'),
            'system_version': summary.get('system_version'),
            'match_context': {
                'format': summary.get('match_format'),
                'venue': summary.get('venue'),
                'teams': summary.get('teams'),
                'series_type': summary.get('series_type')
            },
            'portfolio_metrics': {
                'diversification_score': summary.get('diversification_score'),
                'captain_diversity_score': summary.get('captain_diversity_score'),
                'unique_captains': summary.get('unique_captains'),
                'unique_players': summary.get('unique_players'),
                'avg_team_overlap': summary.get('avg_team_overlap')
            },
            'teams': teams,
            'captain_analysis': captain_analysis,
            'player_usage_analysis': player_usage,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"export_prediction_{prediction_id}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"✅ Exported prediction {prediction_id} to {output_file}")
        return output_file
    
    def cleanup_old_predictions(self, days_to_keep: int = 30):
        """Clean up old predictions to save space"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Get prediction IDs to delete
            cursor.execute('''
                SELECT id FROM smart15_predictions 
                WHERE prediction_timestamp < ?
            ''', (cutoff_date,))
            
            old_prediction_ids = [row[0] for row in cursor.fetchall()]
            
            if old_prediction_ids:
                # Delete related records
                placeholders = ','.join(['?'] * len(old_prediction_ids))
                
                cursor.execute(f'DELETE FROM player_usage_analysis WHERE prediction_id IN ({placeholders})', old_prediction_ids)
                cursor.execute(f'DELETE FROM captain_analysis WHERE prediction_id IN ({placeholders})', old_prediction_ids)
                cursor.execute(f'DELETE FROM prediction_teams WHERE prediction_id IN ({placeholders})', old_prediction_ids)
                cursor.execute(f'DELETE FROM smart15_predictions WHERE id IN ({placeholders})', old_prediction_ids)
                
                conn.commit()
                self.logger.info(f"✅ Cleaned up {len(old_prediction_ids)} old predictions")
            
        finally:
            conn.close()

# Global instance
storage_manager = DatabaseStorageManager()

def get_storage_manager() -> DatabaseStorageManager:
    """Get global storage manager instance"""
    return storage_manager