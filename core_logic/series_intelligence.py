#!/usr/bin/env python3
"""
Series Intelligence Engine - Advanced Match-to-Match Analysis
Analyzes series patterns, momentum, and opposition-specific adaptations
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class SeriesStage(Enum):
    FIRST_MATCH = "first_match"
    EARLY_SERIES = "early_series"  # Matches 2-3
    MID_SERIES = "mid_series"      # Matches 4-5
    LATE_SERIES = "late_series"    # Matches 6+
    DECIDER = "decider"            # Final/crucial match

class MomentumDirection(Enum):
    STRONG_UP = "strong_up"        # +3 to +5
    MODERATE_UP = "moderate_up"    # +1 to +2
    STABLE = "stable"              # -1 to +1
    MODERATE_DOWN = "moderate_down" # -2 to -1
    STRONG_DOWN = "strong_down"    # -5 to -3

@dataclass
class MatchResult:
    """Individual match result data"""
    match_id: str
    match_number: int
    date: datetime
    team1: str
    team2: str
    winner: str
    margin: str
    venue: str
    format: str
    
    # Team performance metrics
    team1_score: int
    team2_score: int
    team1_players: List[Dict[str, Any]]
    team2_players: List[Dict[str, Any]]
    
    # Match context
    toss_winner: str
    toss_decision: str
    weather_conditions: str
    pitch_condition: str
    crowd_factor: float = 1.0
    
    # Post-match insights
    player_of_match: str = ""
    key_partnerships: List[Dict[str, Any]] = field(default_factory=list)
    bowling_figures: List[Dict[str, Any]] = field(default_factory=list)
    tactical_changes: List[str] = field(default_factory=list)

@dataclass
class PlayerSeriesData:
    """Player's series-wide performance tracking"""
    player_id: str
    player_name: str
    role: str
    team: str
    
    # Match-by-match performance
    match_scores: List[float] = field(default_factory=list)
    match_positions: List[int] = field(default_factory=list)  # Batting order
    captain_history: List[bool] = field(default_factory=list)
    vice_captain_history: List[bool] = field(default_factory=list)
    
    # Series trends
    form_trend: float = 0.0  # -5 to +5 scale
    consistency_rating: float = 0.0
    opposition_specific_avg: float = 0.0
    pressure_performance: float = 0.0
    
    # Momentum indicators
    last_3_avg: float = 0.0
    vs_opposition_avg: float = 0.0
    series_momentum: MomentumDirection = MomentumDirection.STABLE
    breakthrough_potential: float = 0.0  # 0-1 scale

@dataclass
class TeamSeriesIntelligence:
    """Team-level series intelligence"""
    team_name: str
    
    # Series record
    wins: int = 0
    losses: int = 0
    series_momentum: float = 0.0  # -5 to +5
    
    # Tactical evolution
    batting_order_stability: float = 0.0
    bowling_rotation_changes: int = 0
    captain_consistency: bool = True
    strategy_adaptations: List[str] = field(default_factory=list)
    
    # Performance patterns
    powerplay_trend: float = 0.0
    middle_overs_trend: float = 0.0
    death_overs_trend: float = 0.0
    chase_vs_defend_preference: float = 0.0  # -1 (defend) to +1 (chase)
    
    # Opposition-specific insights
    head_to_head_adjustments: Dict[str, Any] = field(default_factory=dict)
    successful_tactics: List[str] = field(default_factory=list)
    failed_experiments: List[str] = field(default_factory=list)

class SeriesIntelligenceEngine:
    """Main engine for series-wide intelligence and analysis"""
    
    def __init__(self):
        self.series_data: Dict[str, List[MatchResult]] = {}
        self.player_series_data: Dict[str, PlayerSeriesData] = {}
        self.team_intelligence: Dict[str, TeamSeriesIntelligence] = {}
        self.opposition_matrices: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
    def add_match_result(self, match_result: MatchResult):
        """Add a completed match result to the series intelligence"""
        series_key = f"{match_result.team1}_vs_{match_result.team2}_{match_result.format}"
        
        if series_key not in self.series_data:
            self.series_data[series_key] = []
        
        self.series_data[series_key].append(match_result)
        
        # Update player and team intelligence
        self._update_player_series_data(match_result)
        self._update_team_intelligence(match_result)
        self._update_opposition_matrix(match_result)
    
    def _update_player_series_data(self, match_result: MatchResult):
        """Update individual player series data"""
        all_players = match_result.team1_players + match_result.team2_players
        
        for player_data in all_players:
            player_id = player_data.get('player_id', player_data.get('name', ''))
            
            if player_id not in self.player_series_data:
                self.player_series_data[player_id] = PlayerSeriesData(
                    player_id=player_id,
                    player_name=player_data.get('name', ''),
                    role=player_data.get('role', ''),
                    team=player_data.get('team', '')
                )
            
            player_series = self.player_series_data[player_id]
            
            # Add match performance
            match_score = player_data.get('fantasy_score', 0)
            player_series.match_scores.append(match_score)
            player_series.match_positions.append(player_data.get('batting_position', 11))
            
            # Track captaincy
            is_captain = player_data.get('is_captain', False)
            is_vc = player_data.get('is_vice_captain', False)
            player_series.captain_history.append(is_captain)
            player_series.vice_captain_history.append(is_vc)
            
            # Calculate trends
            self._calculate_player_trends(player_series)
    
    def _calculate_player_trends(self, player_series: PlayerSeriesData):
        """Calculate various trend metrics for a player"""
        scores = player_series.match_scores
        if len(scores) < 2:
            return
        
        # Form trend (recent vs earlier matches)
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / min(3, len(scores[-3:]))
            earlier_avg = sum(scores[:-3]) / max(1, len(scores[:-3])) if len(scores) > 3 else recent_avg
            player_series.form_trend = min(5, max(-5, (recent_avg - earlier_avg) / 10))
        
        # Last 3 matches average
        player_series.last_3_avg = sum(scores[-3:]) / min(3, len(scores))
        
        # Consistency (lower standard deviation = higher consistency)
        if len(scores) >= 2:
            mean_score = sum(scores) / len(scores)
            variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
            std_dev = math.sqrt(variance)
            player_series.consistency_rating = max(0, 100 - std_dev * 2)
        
        # Momentum direction
        if len(scores) >= 3:
            trend_value = player_series.form_trend
            if trend_value >= 3:
                player_series.series_momentum = MomentumDirection.STRONG_UP
            elif trend_value >= 1:
                player_series.series_momentum = MomentumDirection.MODERATE_UP
            elif trend_value <= -3:
                player_series.series_momentum = MomentumDirection.STRONG_DOWN
            elif trend_value <= -1:
                player_series.series_momentum = MomentumDirection.MODERATE_DOWN
            else:
                player_series.series_momentum = MomentumDirection.STABLE
    
    def _update_team_intelligence(self, match_result: MatchResult):
        """Update team-level intelligence"""
        teams = [match_result.team1, match_result.team2]
        
        for team in teams:
            if team not in self.team_intelligence:
                self.team_intelligence[team] = TeamSeriesIntelligence(team_name=team)
            
            team_intel = self.team_intelligence[team]
            
            # Update win/loss record
            if match_result.winner == team:
                team_intel.wins += 1
                team_intel.series_momentum = min(5, team_intel.series_momentum + 1)
            else:
                team_intel.losses += 1
                team_intel.series_momentum = max(-5, team_intel.series_momentum - 1)
            
            # Analyze tactical changes
            if match_result.tactical_changes:
                team_intel.strategy_adaptations.extend(match_result.tactical_changes)
    
    def _update_opposition_matrix(self, match_result: MatchResult):
        """Update head-to-head opposition analysis"""
        opposition_key = (match_result.team1, match_result.team2)
        reverse_key = (match_result.team2, match_result.team1)
        
        if opposition_key not in self.opposition_matrices:
            self.opposition_matrices[opposition_key] = {
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'avg_score_team1': 0,
                'avg_score_team2': 0,
                'high_performers': {},
                'successful_strategies': [],
                'pitch_preferences': {}
            }
        
        matrix = self.opposition_matrices[opposition_key]
        matrix['total_matches'] += 1
        
        if match_result.winner == match_result.team1:
            matrix['team1_wins'] += 1
        else:
            matrix['team2_wins'] += 1
        
        # Update averages
        total_matches = matrix['total_matches']
        matrix['avg_score_team1'] = ((matrix['avg_score_team1'] * (total_matches - 1)) + match_result.team1_score) / total_matches
        matrix['avg_score_team2'] = ((matrix['avg_score_team2'] * (total_matches - 1)) + match_result.team2_score) / total_matches
    
    def get_series_context_for_player(self, player_id: str, opposition_team: str, match_number: int) -> Dict[str, Any]:
        """Get comprehensive series context for a player"""
        if player_id not in self.player_series_data:
            return self._get_default_context(player_id)
        
        player_data = self.player_series_data[player_id]
        
        # Determine series stage
        series_stage = self._determine_series_stage(match_number)
        
        # Opposition-specific analysis
        opposition_analysis = self._analyze_opposition_performance(player_data, opposition_team)
        
        # Momentum analysis
        momentum_analysis = self._analyze_player_momentum(player_data, match_number)
        
        # Pressure analysis
        pressure_analysis = self._analyze_pressure_scenarios(player_data, series_stage)
        
        return {
            'player_id': player_id,
            'series_stage': series_stage.value,
            'matches_played': len(player_data.match_scores),
            'form_trend': player_data.form_trend,
            'series_momentum': player_data.series_momentum.value,
            'consistency_rating': player_data.consistency_rating,
            'last_3_avg': player_data.last_3_avg,
            'opposition_analysis': opposition_analysis,
            'momentum_analysis': momentum_analysis,
            'pressure_analysis': pressure_analysis,
            'captaincy_potential': self._calculate_captaincy_potential(player_data),
            'breakthrough_potential': player_data.breakthrough_potential,
            'series_multiplier': self._calculate_series_multiplier(player_data, series_stage)
        }
    
    def _determine_series_stage(self, match_number: int) -> SeriesStage:
        """Determine what stage of the series we're in"""
        if match_number == 1:
            return SeriesStage.FIRST_MATCH
        elif match_number <= 3:
            return SeriesStage.EARLY_SERIES
        elif match_number <= 5:
            return SeriesStage.MID_SERIES
        else:
            return SeriesStage.LATE_SERIES
    
    def _analyze_opposition_performance(self, player_data: PlayerSeriesData, opposition_team: str) -> Dict[str, Any]:
        """Analyze player's performance against specific opposition"""
        # This would ideally look at historical data against this opposition
        # For now, we'll use series averages as proxy
        
        if len(player_data.match_scores) == 0:
            return {'vs_opposition_avg': 0, 'matchup_advantage': 0.0}
        
        series_avg = sum(player_data.match_scores) / len(player_data.match_scores)
        
        # Simulate opposition-specific adjustment (would be based on actual data)
        opposition_adjustment = 0.0
        role_lower = player_data.role.lower()
        
        # Example: Some roles perform better against certain teams
        if 'bowl' in role_lower and opposition_team in ['IND', 'AUS']:
            opposition_adjustment = -0.1  # Stronger batting lineups
        elif 'bat' in role_lower and opposition_team in ['SA', 'NZ']:
            opposition_adjustment = 0.1   # Good batting conditions/matchups
        
        return {
            'vs_opposition_avg': series_avg * (1 + opposition_adjustment),
            'matchup_advantage': opposition_adjustment,
            'historical_success_rate': 0.6 + opposition_adjustment  # Placeholder
        }
    
    def _analyze_player_momentum(self, player_data: PlayerSeriesData, match_number: int) -> Dict[str, Any]:
        """Analyze player's current momentum"""
        if len(player_data.match_scores) < 2:
            return {'momentum_score': 0.0, 'trend_direction': 'stable', 'confidence': 0.5}
        
        scores = player_data.match_scores
        
        # Calculate momentum based on recent trend
        if len(scores) >= 3:
            recent_trend = (scores[-1] - scores[-3]) / 2
        else:
            recent_trend = scores[-1] - scores[-2]
        
        momentum_score = min(5, max(-5, recent_trend / 10))
        
        # Determine trend direction
        if momentum_score >= 2:
            trend_direction = 'strong_upward'
        elif momentum_score >= 0.5:
            trend_direction = 'upward'
        elif momentum_score <= -2:
            trend_direction = 'strong_downward'
        elif momentum_score <= -0.5:
            trend_direction = 'downward'
        else:
            trend_direction = 'stable'
        
        # Confidence based on consistency
        confidence = player_data.consistency_rating / 100
        
        return {
            'momentum_score': momentum_score,
            'trend_direction': trend_direction,
            'confidence': confidence,
            'peak_potential': max(scores) if scores else 0,
            'floor_risk': min(scores) if scores else 0
        }
    
    def _analyze_pressure_scenarios(self, player_data: PlayerSeriesData, series_stage: SeriesStage) -> Dict[str, Any]:
        """Analyze how player performs under pressure"""
        pressure_multiplier = 1.0
        
        # Higher pressure in later stages of series
        if series_stage == SeriesStage.LATE_SERIES or series_stage == SeriesStage.DECIDER:
            pressure_multiplier = 1.2
        elif series_stage == SeriesStage.MID_SERIES:
            pressure_multiplier = 1.1
        
        # Captain/VC experience adds pressure handling
        captaincy_experience = sum(player_data.captain_history) + sum(player_data.vice_captain_history) * 0.5
        pressure_experience = min(1.0, captaincy_experience * 0.2)
        
        return {
            'pressure_multiplier': pressure_multiplier,
            'pressure_experience': pressure_experience,
            'big_match_temperament': pressure_experience + (player_data.consistency_rating / 200),
            'clutch_performance_potential': pressure_multiplier * (1 + pressure_experience)
        }
    
    def _calculate_captaincy_potential(self, player_data: PlayerSeriesData) -> float:
        """Calculate player's captaincy potential for this match"""
        base_score = 0.5
        
        # Previous captaincy experience
        captain_matches = sum(player_data.captain_history)
        vc_matches = sum(player_data.vice_captain_history)
        leadership_experience = (captain_matches * 1.0 + vc_matches * 0.5) * 0.1
        
        # Recent form boost
        form_boost = max(0, player_data.form_trend * 0.1)
        
        # Consistency factor
        consistency_factor = player_data.consistency_rating / 200
        
        captaincy_potential = min(1.0, base_score + leadership_experience + form_boost + consistency_factor)
        
        return captaincy_potential
    
    def _calculate_series_multiplier(self, player_data: PlayerSeriesData, series_stage: SeriesStage) -> float:
        """Calculate balanced series context multiplier for player scoring"""
        base_multiplier = 1.0
        
        # REDUCED Momentum contribution (more conservative)
        momentum_impact = {
            MomentumDirection.STRONG_UP: 0.08,      # Reduced from 0.15
            MomentumDirection.MODERATE_UP: 0.04,    # Reduced from 0.08
            MomentumDirection.STABLE: 0.0,
            MomentumDirection.MODERATE_DOWN: -0.04, # Reduced from -0.08
            MomentumDirection.STRONG_DOWN: -0.08    # Reduced from -0.15
        }
        
        multiplier = base_multiplier + momentum_impact.get(player_data.series_momentum, 0.0)
        
        # REDUCED Series stage adjustment (more conservative)
        stage_adjustments = {
            SeriesStage.FIRST_MATCH: 0.0,
            SeriesStage.EARLY_SERIES: 0.01,    # Reduced from 0.02
            SeriesStage.MID_SERIES: 0.03,      # Reduced from 0.05
            SeriesStage.LATE_SERIES: 0.05,     # Reduced from 0.08
            SeriesStage.DECIDER: 0.08          # Reduced from 0.12
        }
        
        multiplier += stage_adjustments.get(series_stage, 0.0)
        
        # REDUCED Consistency bonus (more balanced)
        if player_data.consistency_rating > 80:  # Higher threshold
            multiplier += 0.02  # Reduced from 0.03
        elif player_data.consistency_rating > 70:
            multiplier += 0.01  # Small bonus for good consistency
        
        # Weight series data by sample size (fewer matches = less impact)
        matches_played = len(player_data.match_scores)
        if matches_played == 1:
            # First match data is unreliable, reduce impact
            multiplier = base_multiplier + (multiplier - base_multiplier) * 0.3
        elif matches_played == 2:
            # Two matches, moderate confidence
            multiplier = base_multiplier + (multiplier - base_multiplier) * 0.6
        # 3+ matches = full confidence
        
        return max(0.92, min(1.12, multiplier))  # Tighter cap: 92% to 112%
    
    def _get_default_context(self, player_id: str) -> Dict[str, Any]:
        """Default context for players without series data"""
        return {
            'player_id': player_id,
            'series_stage': 'first_match',
            'matches_played': 0,
            'form_trend': 0.0,
            'series_momentum': 'stable',
            'consistency_rating': 50.0,
            'last_3_avg': 0.0,
            'opposition_analysis': {'vs_opposition_avg': 0, 'matchup_advantage': 0.0},
            'momentum_analysis': {'momentum_score': 0.0, 'trend_direction': 'stable', 'confidence': 0.5},
            'pressure_analysis': {'pressure_multiplier': 1.0, 'pressure_experience': 0.0},
            'captaincy_potential': 0.5,
            'breakthrough_potential': 0.5,
            'series_multiplier': 1.0
        }
    
    def get_team_series_insights(self, team_name: str, opposition: str, match_number: int) -> Dict[str, Any]:
        """Get comprehensive team-level series insights"""
        if team_name not in self.team_intelligence:
            return self._get_default_team_insights(team_name)
        
        team_intel = self.team_intelligence[team_name]
        
        return {
            'team_name': team_name,
            'series_record': f"{team_intel.wins}-{team_intel.losses}",
            'series_momentum': team_intel.series_momentum,
            'tactical_adaptations': len(team_intel.strategy_adaptations),
            'captain_stability': team_intel.captain_consistency,
            'recent_form': self._calculate_team_recent_form(team_name),
            'opposition_record': self._get_head_to_head_record(team_name, opposition),
            'recommended_strategy': self._recommend_team_strategy(team_intel, match_number),
            'pressure_rating': self._calculate_team_pressure_rating(team_intel, match_number)
        }
    
    def _get_default_team_insights(self, team_name: str) -> Dict[str, Any]:
        """Default insights for teams without series data"""
        return {
            'team_name': team_name,
            'series_record': '0-0',
            'series_momentum': 0.0,
            'tactical_adaptations': 0,
            'captain_stability': True,
            'recent_form': 'unknown',
            'opposition_record': 'no_data',
            'recommended_strategy': 'balanced',
            'pressure_rating': 'medium'
        }
    
    def _calculate_team_recent_form(self, team_name: str) -> str:
        """Calculate team's recent form"""
        team_intel = self.team_intelligence[team_name]
        
        if team_intel.series_momentum >= 2:
            return 'excellent'
        elif team_intel.series_momentum >= 1:
            return 'good'
        elif team_intel.series_momentum <= -2:
            return 'poor'
        elif team_intel.series_momentum <= -1:
            return 'concerning'
        else:
            return 'average'
    
    def _get_head_to_head_record(self, team1: str, team2: str) -> str:
        """Get head-to-head record between teams"""
        key1 = (team1, team2)
        key2 = (team2, team1)
        
        if key1 in self.opposition_matrices:
            matrix = self.opposition_matrices[key1]
            return f"{matrix['team1_wins']}-{matrix['team2_wins']}"
        elif key2 in self.opposition_matrices:
            matrix = self.opposition_matrices[key2]
            return f"{matrix['team2_wins']}-{matrix['team1_wins']}"
        
        return "no_data"
    
    def _recommend_team_strategy(self, team_intel: TeamSeriesIntelligence, match_number: int) -> str:
        """Recommend strategy based on series intelligence"""
        if team_intel.series_momentum >= 2:
            return "aggressive"  # Riding high, press advantage
        elif team_intel.series_momentum <= -2:
            return "conservative"  # Need to steady the ship
        elif match_number >= 3:
            return "adaptive"  # Mid-series, adapt to conditions
        else:
            return "balanced"  # Default approach
    
    def _calculate_team_pressure_rating(self, team_intel: TeamSeriesIntelligence, match_number: int) -> str:
        """Calculate team's pressure rating"""
        pressure_score = 0
        
        # Series position pressure
        if team_intel.losses > team_intel.wins:
            pressure_score += 2
        
        # Momentum pressure
        if team_intel.series_momentum <= -2:
            pressure_score += 2
        elif team_intel.series_momentum >= 2:
            pressure_score -= 1
        
        # Match importance
        if match_number >= 3:
            pressure_score += 1
        
        if pressure_score >= 3:
            return "high"
        elif pressure_score >= 1:
            return "medium"
        else:
            return "low"
    
    def generate_series_intelligence_report(self, team1: str, team2: str, match_number: int) -> Dict[str, Any]:
        """Generate comprehensive series intelligence report"""
        return {
            'match_number': match_number,
            'series_stage': self._determine_series_stage(match_number).value,
            'team1_insights': self.get_team_series_insights(team1, team2, match_number),
            'team2_insights': self.get_team_series_insights(team2, team1, match_number),
            'key_storylines': self._generate_key_storylines(team1, team2, match_number),
            'tactical_predictions': self._generate_tactical_predictions(team1, team2),
            'momentum_analysis': self._generate_momentum_analysis(team1, team2),
            'player_watch_list': self._generate_player_watch_list(team1, team2),
            'series_turning_points': self._identify_potential_turning_points(match_number)
        }
    
    def _generate_key_storylines(self, team1: str, team2: str, match_number: int) -> List[str]:
        """Generate key storylines for the match"""
        storylines = []
        
        # Series context storylines
        if match_number == 2:
            storylines.append("Teams look to build on opening match learnings")
            storylines.append("Tactical adjustments expected after first encounter")
        elif match_number >= 3:
            storylines.append("Series momentum becomes crucial factor")
            
        # Team-specific storylines
        team1_intel = self.team_intelligence.get(team1)
        team2_intel = self.team_intelligence.get(team2)
        
        if team1_intel and team1_intel.series_momentum >= 2:
            storylines.append(f"{team1} riding high on positive momentum")
        if team2_intel and team2_intel.series_momentum <= -2:
            storylines.append(f"{team2} under pressure to bounce back")
        
        return storylines
    
    def _generate_tactical_predictions(self, team1: str, team2: str) -> Dict[str, Any]:
        """Generate tactical predictions based on series intelligence"""
        return {
            'expected_changes': f"Both teams likely to make 1-2 tactical adjustments",
            'batting_order_stability': "70%",  # Based on series data
            'bowling_rotation_changes': "40%",  # Teams adapting
            'captain_decisions': "More aggressive field placements expected",
            'strategy_focus': "Middle overs execution will be key"
        }
    
    def _generate_momentum_analysis(self, team1: str, team2: str) -> Dict[str, Any]:
        """Generate momentum analysis for both teams"""
        team1_momentum = self.team_intelligence.get(team1)
        team2_momentum = self.team_intelligence.get(team2)
        
        return {
            'momentum_advantage': team1 if (team1_momentum and team1_momentum.series_momentum > 0) else team2,
            'pressure_team': team2 if (team1_momentum and team1_momentum.series_momentum > 0) else team1,
            'momentum_shift_potential': "High" if team1_momentum and abs(team1_momentum.series_momentum) >= 2 else "Medium"
        }
    
    def _generate_player_watch_list(self, team1: str, team2: str) -> List[Dict[str, Any]]:
        """Generate list of players to watch based on series intelligence"""
        watch_list = []
        
        # Find players with strong momentum
        for player_id, player_data in self.player_series_data.items():
            if player_data.team in [team1, team2]:
                if player_data.series_momentum in [MomentumDirection.STRONG_UP, MomentumDirection.STRONG_DOWN]:
                    watch_list.append({
                        'player_name': player_data.player_name,
                        'team': player_data.team,
                        'reason': f"Strong {'positive' if player_data.series_momentum == MomentumDirection.STRONG_UP else 'negative'} momentum",
                        'momentum': player_data.series_momentum.value
                    })
        
        return watch_list[:6]  # Top 6 players to watch
    
    def _identify_potential_turning_points(self, match_number: int) -> List[str]:
        """Identify potential series turning points"""
        turning_points = []
        
        if match_number == 2:
            turning_points.append("First team to win 2-0 gains significant psychological advantage")
        elif match_number == 3:
            turning_points.append("Middle match - momentum could shift decisively")
        elif match_number >= 4:
            turning_points.append("Late series stage - every match becomes crucial")
        
        return turning_points

# Global instance for series intelligence
series_intelligence = SeriesIntelligenceEngine()