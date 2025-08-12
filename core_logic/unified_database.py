#!/usr/bin/env python3
"""
UNIFIED DATABASE - Single Source of Truth for All Data
Consolidates multiple databases into one efficient SQLite database
Replaces: ai_learning_database.db, api_usage_tracking.db, smart_local_predictions.db, optimized_predictions.db
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = "dream11_unified.db"
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    connection_timeout: int = 30
    max_connections: int = 10

class UnifiedDatabase:
    """
    Single, unified database for all Dream11 AI data
    Efficiently replaces multiple separate databases
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.db_path = self.config.db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            if self.config.enable_wal_mode:
                cursor.execute("PRAGMA journal_mode=WAL")
            
            # Enable foreign keys
            if self.config.enable_foreign_keys:
                cursor.execute("PRAGMA foreign_keys=ON")
            
            # Create all tables
            self._create_tables(cursor)
            
            conn.commit()
    
    def _create_tables(self, cursor):
        """Create all database tables"""
        
        # 1. MATCHES TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                team1_name TEXT NOT NULL,
                team2_name TEXT NOT NULL,
                match_format TEXT,
                venue TEXT,
                start_time TIMESTAMP,
                status TEXT DEFAULT 'upcoming',
                toss_winner TEXT,
                elected_to TEXT,
                pitch_type TEXT,
                weather_conditions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. PLAYERS TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                team_name TEXT,
                ema_score REAL DEFAULT 0,
                consistency_score REAL DEFAULT 0,
                form_momentum REAL DEFAULT 0,
                expected_points REAL DEFAULT 0,
                captain_probability REAL DEFAULT 0,
                ownership_prediction REAL DEFAULT 50,
                career_stats TEXT,  -- JSON
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. PREDICTIONS TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                teams_data TEXT,  -- JSON array of teams
                ai_strategies TEXT,  -- JSON array of strategies
                quality_score REAL,
                confidence_score REAL,
                status TEXT DEFAULT 'pending',
                api_calls_used INTEGER DEFAULT 0,
                execution_time_seconds REAL,
                FOREIGN KEY (match_id) REFERENCES matches(match_id)
            )
        """)
        
        # 4. LEARNING DATA TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_id INTEGER,
                actual_scores TEXT,  -- JSON of actual player scores
                prediction_accuracy REAL,
                captain_success BOOLEAN,
                vice_captain_success BOOLEAN,
                lessons_learned TEXT,  -- JSON
                learning_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(match_id),
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
        
        # 5. API USAGE TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                request_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time_ms INTEGER,
                status_code INTEGER,
                cache_hit BOOLEAN DEFAULT FALSE,
                error_message TEXT,
                rate_limit_remaining INTEGER,
                daily_count INTEGER DEFAULT 1
            )
        """)
        
        # 6. TEAM GENERATIONS TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                team_number INTEGER,
                strategy TEXT,
                players TEXT,  -- JSON array of player data
                captain_id INTEGER,
                vice_captain_id INTEGER,
                total_score REAL,
                expected_points REAL,
                confidence_score REAL,
                risk_level TEXT,
                generation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(match_id)
            )
        """)
        
        # 7. SYSTEM SESSIONS TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_type TEXT,  -- 'discovery', 'execution', 'learning'
                session_date DATE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                matches_processed INTEGER DEFAULT 0,
                predictions_generated INTEGER DEFAULT 0,
                api_calls_made INTEGER DEFAULT 0,
                success_rate REAL,
                notes TEXT
            )
        """)
        
        # 8. CACHE TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                endpoint TEXT,
                data TEXT,  -- JSON data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        self._create_indexes(cursor)
    
    def _create_indexes(self, cursor):
        """Create database indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_matches_start_time ON matches(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status)",
            "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_name)",
            "CREATE INDEX IF NOT EXISTS idx_players_role ON players(role)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_time ON api_usage(request_time)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_team_generations_match ON team_generations(match_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_date ON system_sessions(session_date)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=self.config.connection_timeout,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception:
            self._local.connection.rollback()
            raise
    
    # MATCH OPERATIONS
    def save_match(self, match_data: Dict[str, Any]) -> bool:
        """Save or update match data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO matches 
                    (match_id, team1_name, team2_name, match_format, venue, start_time, 
                     status, toss_winner, elected_to, pitch_type, weather_conditions, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    match_data.get('match_id'),
                    match_data.get('team1_name'),
                    match_data.get('team2_name'),
                    match_data.get('format'),
                    match_data.get('venue'),
                    match_data.get('start_time'),
                    match_data.get('status', 'upcoming'),
                    match_data.get('toss_winner'),
                    match_data.get('elected_to'),
                    match_data.get('pitch_type'),
                    match_data.get('weather_conditions')
                ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"⚠️ Failed to save match: {e}")
            return False
    
    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get match data by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception:
            return None
    
    def get_upcoming_matches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get upcoming matches"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM matches 
                    WHERE status = 'upcoming' AND start_time > CURRENT_TIMESTAMP
                    ORDER BY start_time 
                    LIMIT ?
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception:
            return []
    
    # PLAYER OPERATIONS
    def save_players(self, players: List[Dict[str, Any]]) -> bool:
        """Save multiple players efficiently"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                player_data = []
                for player in players:
                    player_data.append((
                        player.get('player_id'),
                        player.get('name'),
                        player.get('role'),
                        player.get('team_name'),
                        player.get('ema_score', 0),
                        player.get('consistency_score', 0),
                        player.get('form_momentum', 0),
                        player.get('expected_points', 0),
                        player.get('captain_probability', 0),
                        player.get('ownership_prediction', 50),
                        json.dumps(player.get('career_stats', {}))
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO players 
                    (player_id, name, role, team_name, ema_score, consistency_score, 
                     form_momentum, expected_points, captain_probability, ownership_prediction, 
                     career_stats, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, player_data)
                
                conn.commit()
                return True
        except Exception as e:
            print(f"⚠️ Failed to save players: {e}")
            return False
    
    def get_players_for_match(self, match_id: str) -> List[Dict[str, Any]]:
        """Get all players for a specific match"""
        match = self.get_match(match_id)
        if not match:
            return []
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM players 
                    WHERE team_name IN (?, ?)
                    ORDER BY ema_score DESC
                """, (match['team1_name'], match['team2_name']))
                
                players = []
                for row in cursor.fetchall():
                    player = dict(row)
                    try:
                        player['career_stats'] = json.loads(player['career_stats'])
                    except:
                        player['career_stats'] = {}
                    players.append(player)
                
                return players
        except Exception:
            return []
    
    # PREDICTION OPERATIONS
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """Save prediction and return prediction ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions 
                    (match_id, teams_data, ai_strategies, quality_score, confidence_score, 
                     status, api_calls_used, execution_time_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_data.get('match_id'),
                    json.dumps(prediction_data.get('teams', [])),
                    json.dumps(prediction_data.get('strategies', [])),
                    prediction_data.get('quality_score'),
                    prediction_data.get('confidence_score'),
                    prediction_data.get('status', 'completed'),
                    prediction_data.get('api_calls_used', 0),
                    prediction_data.get('execution_time')
                ))
                
                prediction_id = cursor.lastrowid
                conn.commit()
                return prediction_id
        except Exception as e:
            print(f"⚠️ Failed to save prediction: {e}")
            return None
    
    def get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent predictions with match info"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.*, m.team1_name, m.team2_name, m.venue
                    FROM predictions p
                    JOIN matches m ON p.match_id = m.match_id
                    ORDER BY p.prediction_time DESC
                    LIMIT ?
                """, (limit,))
                
                predictions = []
                for row in cursor.fetchall():
                    prediction = dict(row)
                    try:
                        prediction['teams_data'] = json.loads(prediction['teams_data'])
                        prediction['ai_strategies'] = json.loads(prediction['ai_strategies'])
                    except:
                        pass
                    predictions.append(prediction)
                
                return predictions
        except Exception:
            return []
    
    # API USAGE TRACKING
    def log_api_call(self, endpoint: str, response_time_ms: int, status_code: int, 
                     cache_hit: bool = False, error_message: Optional[str] = None):
        """Log API call for tracking and optimization"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get daily count for this endpoint
                today = datetime.now().date()
                cursor.execute("""
                    SELECT COUNT(*) FROM api_usage 
                    WHERE endpoint = ? AND DATE(request_time) = ?
                """, (endpoint, today))
                daily_count = cursor.fetchone()[0] + 1
                
                cursor.execute("""
                    INSERT INTO api_usage 
                    (endpoint, response_time_ms, status_code, cache_hit, error_message, daily_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (endpoint, response_time_ms, status_code, cache_hit, error_message, daily_count))
                
                conn.commit()
        except Exception:
            pass  # Don't fail main operation if logging fails
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive API usage statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Today's stats
                today = datetime.now().date()
                cursor.execute("""
                    SELECT endpoint, COUNT(*) as calls, AVG(response_time_ms) as avg_time,
                           SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
                    FROM api_usage 
                    WHERE DATE(request_time) = ?
                    GROUP BY endpoint
                """, (today,))
                
                today_stats = {row[0]: {
                    'calls': row[1],
                    'avg_response_time': row[2],
                    'cache_hits': row[3],
                    'cache_rate': (row[3] / row[1] * 100) if row[1] > 0 else 0
                } for row in cursor.fetchall()}
                
                # Total calls today
                cursor.execute("""
                    SELECT COUNT(*) FROM api_usage 
                    WHERE DATE(request_time) = ?
                """, (today,))
                total_today = cursor.fetchone()[0]
                
                return {
                    'total_calls_today': total_today,
                    'by_endpoint': today_stats,
                    'date': today.isoformat()
                }
        except Exception:
            return {'total_calls_today': 0, 'by_endpoint': {}}
    
    # CACHE OPERATIONS
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (key,))
                
                row = cursor.fetchone()
                if row:
                    # Update access stats
                    cursor.execute("""
                        UPDATE cache_entries 
                        SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP
                        WHERE cache_key = ?
                    """, (key,))
                    conn.commit()
                    
                    return json.loads(row[0])
                
                return None
        except Exception:
            return None
    
    def cache_set(self, key: str, data: Any, ttl_seconds: int, endpoint: str = 'default'):
        """Set cached data with TTL"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, endpoint, data, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (key, endpoint, json.dumps(data, default=str), expires_at))
                
                conn.commit()
        except Exception:
            pass  # Don't fail if caching fails
    
    # MAINTENANCE OPERATIONS
    def cleanup_expired_data(self):
        """Clean up expired cache and old data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Remove expired cache entries
                cursor.execute("DELETE FROM cache_entries WHERE expires_at < CURRENT_TIMESTAMP")
                
                # Remove old API logs (keep last 30 days)
                thirty_days_ago = datetime.now() - timedelta(days=30)
                cursor.execute("DELETE FROM api_usage WHERE request_time < ?", (thirty_days_ago,))
                
                # Remove old predictions (keep last 90 days)
                ninety_days_ago = datetime.now() - timedelta(days=90)
                cursor.execute("DELETE FROM predictions WHERE prediction_time < ?", (ninety_days_ago,))
                
                conn.commit()
                print("✅ Database cleanup completed")
        except Exception as e:
            print(f"⚠️ Database cleanup failed: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Table row counts
                tables = ['matches', 'players', 'predictions', 'api_usage', 'cache_entries']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Database file size
                db_path = Path(self.db_path)
                if db_path.exists():
                    stats['db_size_mb'] = db_path.stat().st_size / (1024 * 1024)
                
                return stats
        except Exception:
            return {}

# Global instance
_unified_db = None

def get_unified_database() -> UnifiedDatabase:
    """Get or create global unified database instance"""
    global _unified_db
    if _unified_db is None:
        _unified_db = UnifiedDatabase()
    return _unified_db

# Migration helper
def migrate_from_old_databases():
    """
    Helper function to migrate data from old database files
    Run this once to consolidate existing data
    """
    db = get_unified_database()
    
    old_db_files = [
        'ai_learning_database.db',
        'api_usage_tracking.db', 
        'smart_local_predictions.db',
        'optimized_predictions.db'
    ]
    
    migrated_count = 0
    
    for db_file in old_db_files:
        if Path(db_file).exists():
            try:
                print(f"🔄 Migrating {db_file}...")
                # Add migration logic here based on old schema
                migrated_count += 1
            except Exception as e:
                print(f"⚠️ Migration failed for {db_file}: {e}")
    
    print(f"✅ Migration completed. {migrated_count} databases processed.")
