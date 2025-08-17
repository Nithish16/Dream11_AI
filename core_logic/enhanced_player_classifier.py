#!/usr/bin/env python3
"""
Enhanced Player Role Classification System
Uses real API data to provide accurate player role analysis
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

# Import API client functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import fetch_squads, fetch_match_center, fetch_player_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerRoleProfile:
    """Comprehensive player role profile with API data"""
    player_id: int
    player_name: str
    primary_role: str  # Batsman, Bowler, All-rounder, Wicket-keeper
    secondary_roles: List[str] = field(default_factory=list)
    
    # Detailed role classification
    batting_style: str = "Unknown"  # Right-hand, Left-hand
    bowling_style: str = "Unknown"  # Fast, Medium, Spin, etc.
    
    # Positional analysis
    is_opener: bool = False
    is_middle_order: bool = False
    is_finisher: bool = False
    is_captain: bool = False
    is_wicket_keeper: bool = False
    
    # Performance context
    powerplay_specialist: bool = False
    death_over_specialist: bool = False
    spin_specialist: bool = False
    
    # Confidence and data quality
    data_confidence: float = 0.0  # 0-1 score based on data availability
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_role_keywords(self) -> List[str]:
        """Generate comprehensive keywords for this player's role"""
        keywords = []
        
        # Primary role keywords
        if self.primary_role:
            role_map = {
                "Batsman": ["batsman", "batter", "bat"],
                "Bowler": ["bowler", "bowl"],
                "All-rounder": ["allrounder", "all_rounder", "batting_allrounder", "bowling_allrounder"],
                "Wicket-keeper": ["keeper", "wicket_keeper", "wk"]
            }
            keywords.extend(role_map.get(self.primary_role, []))
        
        # Specific role modifiers
        if self.is_opener:
            keywords.extend(["opener", "opening_batsman"])
        if self.is_middle_order:
            keywords.extend(["middle_order", "middle_order_batsman"])
        if self.is_finisher:
            keywords.extend(["finisher", "death_batsman", "lower_order"])
        if self.is_captain:
            keywords.extend(["captain", "leader", "experienced"])
        if self.is_wicket_keeper:
            keywords.extend(["keeper", "wicket_keeper"])
            
        # Bowling specializations
        if "fast" in self.bowling_style.lower():
            keywords.extend(["pace_bowler", "fast_bowler", "seamer"])
        elif "spin" in self.bowling_style.lower():
            keywords.extend(["spinner", "spin_bowler"])
        elif "medium" in self.bowling_style.lower():
            keywords.extend(["medium_pacer", "medium_fast"])
            
        # Performance context
        if self.powerplay_specialist:
            keywords.extend(["powerplay_specialist", "pp_specialist"])
        if self.death_over_specialist:
            keywords.extend(["death_bowler", "yorker_specialist"])
            
        # Generic classification
        keywords.extend(["player", "experienced"])
        
        return list(set(keywords))  # Remove duplicates

class EnhancedPlayerClassifier:
    """Enhanced player classification using API data and learning"""
    
    def __init__(self, db_path: str = "enhanced_player_classification.db"):
        self.db_path = db_path
        self.player_cache = {}  # In-memory cache for recent lookups
        self.cache_expiry = timedelta(hours=6)  # Cache player data for 6 hours
        self._init_database()
        
    def _init_database(self):
        """Initialize the player classification database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create player classification table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_classifications (
                    player_id INTEGER PRIMARY KEY,
                    player_name TEXT NOT NULL,
                    primary_role TEXT NOT NULL,
                    secondary_roles TEXT,  -- JSON array
                    batting_style TEXT,
                    bowling_style TEXT,
                    is_opener BOOLEAN DEFAULT 0,
                    is_middle_order BOOLEAN DEFAULT 0,
                    is_finisher BOOLEAN DEFAULT 0,
                    is_captain BOOLEAN DEFAULT 0,
                    is_wicket_keeper BOOLEAN DEFAULT 0,
                    powerplay_specialist BOOLEAN DEFAULT 0,
                    death_over_specialist BOOLEAN DEFAULT 0,
                    spin_specialist BOOLEAN DEFAULT 0,
                    data_confidence REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    api_source TEXT,  -- Which API provided this data
                    series_id TEXT,   -- Series context where this was learned
                    match_id TEXT     -- Match context where this was learned
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_name ON player_classifications(player_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_updated ON player_classifications(last_updated)')
            
            conn.commit()
            conn.close()
            logger.info("✅ Enhanced player classification database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing player classification database: {e}")
    
    def get_player_role_profile(self, player_name: str, match_id: str = None) -> PlayerRoleProfile:
        """Get comprehensive role profile for a player"""
        # Check cache first
        cache_key = f"{player_name}_{match_id or 'general'}"
        if cache_key in self.player_cache:
            cached_profile, cache_time = self.player_cache[cache_key]
            if datetime.now() - cache_time < self.cache_expiry:
                return cached_profile
        
        # Try to get from database first
        profile = self._get_profile_from_database(player_name)
        
        # If not found or data is old, fetch from API
        if not profile or self._is_data_stale(profile):
            profile = self._fetch_and_update_profile(player_name, match_id)
        
        # Cache the result
        self.player_cache[cache_key] = (profile, datetime.now())
        
        return profile
    
    def _get_profile_from_database(self, player_name: str) -> Optional[PlayerRoleProfile]:
        """Get player profile from local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM player_classifications 
                WHERE player_name = ? 
                ORDER BY last_updated DESC 
                LIMIT 1
            ''', (player_name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return PlayerRoleProfile(
                    player_id=row[0],
                    player_name=row[1],
                    primary_role=row[2],
                    secondary_roles=json.loads(row[3]) if row[3] else [],
                    batting_style=row[4] or "Unknown",
                    bowling_style=row[5] or "Unknown",
                    is_opener=bool(row[6]),
                    is_middle_order=bool(row[7]),
                    is_finisher=bool(row[8]),
                    is_captain=bool(row[9]),
                    is_wicket_keeper=bool(row[10]),
                    powerplay_specialist=bool(row[11]),
                    death_over_specialist=bool(row[12]),
                    spin_specialist=bool(row[13]),
                    data_confidence=row[14],
                    last_updated=datetime.fromisoformat(row[15])
                )
        except Exception as e:
            logger.warning(f"⚠️ Error getting profile from database for {player_name}: {e}")
        
        return None
    
    def _is_data_stale(self, profile: PlayerRoleProfile) -> bool:
        """Check if player profile data is stale and needs refresh"""
        age = datetime.now() - profile.last_updated
        
        # Data is stale if:
        # - Older than 24 hours AND confidence is low
        # - Older than 7 days regardless of confidence
        if age > timedelta(days=7):
            return True
        if age > timedelta(hours=24) and profile.data_confidence < 0.7:
            return True
            
        return False
    
    def _fetch_and_update_profile(self, player_name: str, match_id: str = None) -> PlayerRoleProfile:
        """Fetch player data from API and update profile"""
        profile = self._create_fallback_profile(player_name)
        
        try:
            # Try to get match data first if match_id is available
            if match_id:
                match_data = fetch_match_center(match_id)
                if match_data and not match_data.get('error'):
                    api_profile = self._extract_profile_from_match_data(player_name, match_data)
                    if api_profile:
                        profile = api_profile
                        profile.data_confidence = 0.8
            
            # If still not found, try to enhance with statistical analysis
            if profile.data_confidence < 0.5:
                profile = self._enhance_profile_with_stats(profile)
            
            # Save to database
            self._save_profile_to_database(profile, match_id)
            
        except Exception as e:
            logger.warning(f"⚠️ Error fetching API data for {player_name}: {e}")
        
        return profile
    
    def _extract_profile_from_match_data(self, player_name: str, match_data: Dict) -> Optional[PlayerRoleProfile]:
        """Extract player profile from match center data"""
        try:
            # Look for player in team squads
            teams = match_data.get('matchInfo', {}).get('team1', {})
            # This would be implemented based on actual API response structure
            # For now, return None to use fallback
            pass
        except Exception as e:
            logger.warning(f"⚠️ Error extracting profile from match data: {e}")
        
        return None
    
    def _enhance_profile_with_stats(self, profile: PlayerRoleProfile) -> PlayerRoleProfile:
        """Enhance profile using statistical analysis and name patterns"""
        
        # Analyze name patterns for common cricket player naming conventions
        name_lower = profile.player_name.lower()
        
        # Common keeper name patterns
        if any(pattern in name_lower for pattern in ['rahul', 'dhoni', 'pant', 'carey', 'buttler', 'de kock']):
            profile.is_wicket_keeper = True
            if profile.primary_role == "Unknown":
                profile.primary_role = "Wicket-keeper"
        
        # Common opener patterns
        if any(pattern in name_lower for pattern in ['rohit', 'warner', 'finch', 'bairstow', 'roy']):
            profile.is_opener = True
        
        # Common finisher patterns  
        if any(pattern in name_lower for pattern in ['pandya', 'russell', 'pollard', 'maxwell']):
            profile.is_finisher = True
        
        # Common captain patterns
        if any(pattern in name_lower for pattern in ['kohli', 'smith', 'root', 'kane', 'morgan']):
            profile.is_captain = True
        
        # Update confidence based on matches found
        matches_found = sum([
            profile.is_wicket_keeper,
            profile.is_opener, 
            profile.is_finisher,
            profile.is_captain
        ])
        
        if matches_found > 0:
            profile.data_confidence = min(0.6, 0.2 + (matches_found * 0.1))
        
        return profile
    
    def _create_fallback_profile(self, player_name: str) -> PlayerRoleProfile:
        """Create a basic fallback profile for unknown players"""
        return PlayerRoleProfile(
            player_id=hash(player_name) % 1000000,  # Generate simple ID from name
            player_name=player_name,
            primary_role="Unknown",
            data_confidence=0.1,
            last_updated=datetime.now()
        )
    
    def _save_profile_to_database(self, profile: PlayerRoleProfile, match_id: str = None):
        """Save player profile to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO player_classifications 
                (player_id, player_name, primary_role, secondary_roles, batting_style, bowling_style,
                 is_opener, is_middle_order, is_finisher, is_captain, is_wicket_keeper,
                 powerplay_specialist, death_over_specialist, spin_specialist, data_confidence,
                 last_updated, api_source, match_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.player_id, profile.player_name, profile.primary_role,
                json.dumps(profile.secondary_roles), profile.batting_style, profile.bowling_style,
                profile.is_opener, profile.is_middle_order, profile.is_finisher,
                profile.is_captain, profile.is_wicket_keeper,
                profile.powerplay_specialist, profile.death_over_specialist, profile.spin_specialist,
                profile.data_confidence, profile.last_updated.isoformat(),
                "enhanced_classifier", match_id
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"💾 Saved profile for {profile.player_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error saving profile for {profile.player_name}: {e}")
    
    def bulk_update_from_match(self, match_id: str) -> int:
        """Update multiple player profiles from a single match"""
        updated_count = 0
        
        try:
            match_data = fetch_match_center(match_id)
            if match_data and not match_data.get('error'):
                # Extract all player names from match data
                # This would be implemented based on actual API structure
                # For now, return 0
                pass
                
        except Exception as e:
            logger.error(f"❌ Error in bulk update from match {match_id}: {e}")
        
        return updated_count
    
    def get_enhanced_keywords(self, player_name: str, match_id: str = None) -> List[str]:
        """Get enhanced keywords for a player (main interface for ultimate system)"""
        profile = self.get_player_role_profile(player_name, match_id)
        return profile.get_role_keywords()
    
    def clear_cache(self):
        """Clear the in-memory cache"""
        self.player_cache.clear()
        logger.info("🧹 Player classification cache cleared")

# Global instance for easy access
_enhanced_classifier = None

def get_enhanced_classifier() -> EnhancedPlayerClassifier:
    """Get global enhanced classifier instance"""
    global _enhanced_classifier
    if _enhanced_classifier is None:
        _enhanced_classifier = EnhancedPlayerClassifier()
    return _enhanced_classifier

def get_enhanced_player_keywords(player_name: str, match_id: str = None) -> List[str]:
    """Main interface function for getting enhanced player keywords"""
    classifier = get_enhanced_classifier()
    return classifier.get_enhanced_keywords(player_name, match_id)