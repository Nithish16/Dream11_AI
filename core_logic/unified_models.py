#!/usr/bin/env python3
"""
UNIFIED MODELS - Single Source of Truth
Consolidates PlayerData, PlayerFeatures, and PlayerForOptimization into one model
Eliminates redundancy and provides clean, consistent data structures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class UnifiedPlayer:
    """
    Single, comprehensive player model replacing all duplicate classes:
    - PlayerData (data_aggregator.py)
    - PlayerFeatures (feature_engine.py) 
    - PlayerForOptimization (team_generator.py)
    """
    # Core Identity
    player_id: int
    name: str
    role: str
    team_id: int
    team_name: str
    
    # Performance Metrics (consolidated from all sources)
    ema_score: float = 0.0
    consistency_score: float = 0.0
    form_momentum: float = 0.0
    expected_points: float = 0.0
    
    # Advanced Analytics
    opportunity_index: float = 1.0
    dynamic_opportunity_index: float = 1.0
    captain_vice_captain_probability: float = 0.0
    ownership_prediction: float = 50.0
    
    # Selection Metadata
    is_captain_candidate: bool = False
    is_vice_captain_candidate: bool = False
    selection_priority: float = 0.0
    
    # Career Data
    career_stats: Dict[str, Any] = field(default_factory=dict)
    batting_stats: Dict[str, Any] = field(default_factory=dict)
    bowling_stats: Dict[str, Any] = field(default_factory=dict)
    recent_form: Dict[str, Any] = field(default_factory=dict)
    
    # Status Information
    injury_status: str = "Fit"
    form_factor: str = "Unknown"
    
    # Fantasy-specific
    credits: float = 8.0  # Default Dream11 credits
    projected_ownership: float = 0.0
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure player_id is positive
        if self.player_id <= 0:
            import hashlib
            unique_string = f"{self.name}_{self.team_id}_{self.team_name}"
            self.player_id = int(hashlib.md5(unique_string.encode()).hexdigest()[:8], 16)
        
        # Normalize role
        self.role = self._normalize_role(self.role)
        
        # Ensure scores are within valid ranges
        self.ema_score = max(0, min(100, self.ema_score))
        self.consistency_score = max(0, min(100, self.consistency_score))
        self.form_momentum = max(0, min(1, self.form_momentum))
        
    def _normalize_role(self, role: str) -> str:
        """Standardize role names across the system"""
        role_lower = role.lower()
        
        if any(keyword in role_lower for keyword in ['wk', 'wicket', 'keeper']):
            return 'Wicket-keeper'
        elif any(keyword in role_lower for keyword in ['allrounder', 'all-rounder', 'all rounder']):
            return 'All-rounder'
        elif any(keyword in role_lower for keyword in ['bowl']):
            return 'Bowler'
        elif any(keyword in role_lower for keyword in ['bat']):
            return 'Batsman'
        else:
            return 'All-rounder'  # Safe default for team balance
    
    @property
    def display_role(self) -> str:
        """Get display-friendly role name"""
        return self.role
    
    @property
    def total_score(self) -> float:
        """Calculate total player score for team selection"""
        return (self.ema_score * 0.4 + 
                self.consistency_score * 0.3 + 
                self.form_momentum * 20 * 0.3)
    
    def get_captain_worthiness(self) -> float:
        """Calculate how worthy this player is as captain (0-100)"""
        base_score = self.total_score
        role_bonus = 10 if self.role == 'All-rounder' else 5
        consistency_bonus = self.consistency_score * 0.2
        
        return min(100, base_score + role_bonus + consistency_bonus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'player_id': self.player_id,
            'name': self.name,
            'role': self.role,
            'team': self.team_name,
            'ema_score': self.ema_score,
            'consistency_score': self.consistency_score,
            'expected_points': self.expected_points,
            'captain_worthiness': self.get_captain_worthiness(),
            'total_score': self.total_score
        }

@dataclass
class UnifiedTeam:
    """Unified team model for all team representations"""
    team_id: int
    players: List[UnifiedPlayer]
    captain: Optional[UnifiedPlayer] = None
    vice_captain: Optional[UnifiedPlayer] = None
    
    # Team Metadata
    strategy: str = "Optimal"
    team_quality: str = "Standard"
    risk_level: str = "Balanced"
    confidence_score: float = 3.0
    
    # Performance Indicators
    total_score: float = 0.0
    total_credits: float = 0.0
    expected_points: float = 0.0
    
    def __post_init__(self):
        """Calculate team metrics after initialization"""
        if self.players:
            self.total_score = sum(p.total_score for p in self.players)
            self.total_credits = sum(p.credits for p in self.players)
            self.expected_points = sum(p.expected_points for p in self.players)
            
            # Auto-select captain and vice-captain if not set
            if not self.captain or not self.vice_captain:
                self._auto_select_leadership()
    
    def _auto_select_leadership(self):
        """Automatically select captain and vice-captain with diversity"""
        if not self.players:
            return
            
        # Sort by captain worthiness
        sorted_players = sorted(self.players, 
                              key=lambda p: p.get_captain_worthiness(), 
                              reverse=True)
        
        if not self.captain and sorted_players:
            # Add diversity based on team_id to avoid same captain
            import random
            
            # Use team_id to create deterministic but diverse captain selection
            captain_index = self.team_id % min(3, len(sorted_players))
            
            # Add some randomness for teams beyond the top 3 players
            if self.team_id > 3 and len(sorted_players) > 3:
                # For teams 4+, pick from top 4 players randomly but deterministically
                random.seed(self.team_id * 42)  # Deterministic seed
                captain_index = random.randint(0, min(3, len(sorted_players) - 1))
            
            self.captain = sorted_players[captain_index]
            
        if not self.vice_captain and len(sorted_players) > 1:
            # Vice-captain should be different from captain
            for i, player in enumerate(sorted_players):
                if player != self.captain:
                    # Add diversity for vice-captain too
                    vc_candidates = [p for p in sorted_players if p != self.captain]
                    if vc_candidates:
                        vc_index = (self.team_id + 1) % len(vc_candidates)
                        self.vice_captain = vc_candidates[vc_index]
                    break
    
    @property
    def is_valid_team(self) -> bool:
        """Check if team meets Dream11 constraints"""
        if len(self.players) != 11:
            return False
            
        # Check role distribution
        role_counts = {}
        team_counts = {}
        
        for player in self.players:
            role_counts[player.role] = role_counts.get(player.role, 0) + 1
            team_counts[player.team_name] = team_counts.get(player.team_name, 0) + 1
        
        # Dream11 constraints
        if role_counts.get('Wicket-keeper', 0) < 1:
            return False
        if role_counts.get('Batsman', 0) < 3:
            return False
        if role_counts.get('All-rounder', 0) < 1:
            return False
        if role_counts.get('Bowler', 0) < 3:
            return False
            
        # Team balance (max 7 from one team)
        if any(count > 7 for count in team_counts.values()):
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'team_id': self.team_id,
            'strategy': self.strategy,
            'quality': self.team_quality,
            'total_score': self.total_score,
            'expected_points': self.expected_points,
            'captain': self.captain.name if self.captain else None,
            'vice_captain': self.vice_captain.name if self.vice_captain else None,
            'players': [p.to_dict() for p in self.players],
            'valid': self.is_valid_team
        }

@dataclass
class UnifiedMatch:
    """Unified match model consolidating all match representations"""
    match_id: str
    team1_name: str
    team2_name: str
    format: str
    venue: str
    start_time: Optional[datetime] = None
    
    # Teams
    team1_players: List[UnifiedPlayer] = field(default_factory=list)
    team2_players: List[UnifiedPlayer] = field(default_factory=list)
    
    # Match Context
    pitch_type: str = "Balanced"
    weather_conditions: str = "Clear"
    toss_winner: Optional[str] = None
    elected_to: Optional[str] = None
    
    @property
    def all_players(self) -> List[UnifiedPlayer]:
        """Get all players from both teams"""
        return self.team1_players + self.team2_players
    
    @property
    def total_players(self) -> int:
        """Get total number of players"""
        return len(self.all_players)
    
    def get_players_by_team(self, team_name: str) -> List[UnifiedPlayer]:
        """Get players for a specific team"""
        return [p for p in self.all_players if p.team_name == team_name]
    
    def get_players_by_role(self, role: str) -> List[UnifiedPlayer]:
        """Get players for a specific role"""
        return [p for p in self.all_players if p.role == role]

# Factory functions for easy migration from old models
def from_player_data(player_data) -> UnifiedPlayer:
    """Convert old PlayerData to UnifiedPlayer"""
    return UnifiedPlayer(
        player_id=player_data.player_id,
        name=player_data.name,
        role=player_data.role,
        team_id=player_data.team_id,
        team_name=player_data.team_name,
        career_stats=player_data.career_stats,
        batting_stats=getattr(player_data, 'batting_stats', {}),
        bowling_stats=getattr(player_data, 'bowling_stats', {}),
        recent_form=getattr(player_data, 'recent_form', {}),
        consistency_score=getattr(player_data, 'consistency_score', 0.0),
        form_factor=getattr(player_data, 'form_factor', 'Unknown'),
        injury_status=getattr(player_data, 'injury_status', 'Fit')
    )

def from_player_features(player_features) -> UnifiedPlayer:
    """Convert old PlayerFeatures to UnifiedPlayer"""
    return UnifiedPlayer(
        player_id=player_features.player_id,
        name=player_features.player_name,
        role=player_features.role,
        team_id=getattr(player_features, 'team_id', 0),
        team_name=player_features.team_name,
        ema_score=player_features.ema_score,
        consistency_score=player_features.consistency_score,
        form_momentum=player_features.form_momentum,
        expected_points=getattr(player_features, 'dream11_expected_points', 0.0),
        opportunity_index=player_features.dynamic_opportunity_index,
        dynamic_opportunity_index=player_features.dynamic_opportunity_index,
        captain_vice_captain_probability=player_features.captain_vice_captain_probability
    )

def from_player_optimization(player_opt) -> UnifiedPlayer:
    """Convert old PlayerForOptimization to UnifiedPlayer"""
    return UnifiedPlayer(
        player_id=player_opt.player_id,
        name=player_opt.name,
        role=player_opt.role,
        team_id=getattr(player_opt, 'team_id', 0),
        team_name=player_opt.team,
        ema_score=player_opt.ema_score,
        consistency_score=player_opt.consistency_score,
        form_momentum=player_opt.form_momentum,
        opportunity_index=player_opt.opportunity_index,
        is_captain_candidate=player_opt.is_captain_candidate,
        is_vice_captain_candidate=player_opt.is_vice_captain_candidate,
        selection_priority=player_opt.selection_priority,
        ownership_prediction=player_opt.ownership_prediction
    )
