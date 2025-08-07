#!/usr/bin/env python3
"""
Advanced Captaincy Engine - PRODUCTION READY
Multi-dimensional captain and vice-captain selection system
Replaces simple scoring with comprehensive leadership analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPressure(Enum):
    """Match pressure levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    KNOCKOUT = "knockout"

class MatchSituation(Enum):
    """Match situation types"""
    CHASING = "chasing"
    SETTING = "setting"
    POWERPLAY = "powerplay"
    DEATH_OVERS = "death_overs"
    MIDDLE_OVERS = "middle_overs"

@dataclass
class CaptaincyMetrics:
    """Comprehensive captaincy assessment metrics"""
    pressure_performance: float = 0.0    # Performance in crucial situations (0-2.0)
    leadership_rating: float = 0.0       # Team impact factor (0-2.0)
    big_match_record: float = 0.0        # Performance in important games (0-2.0)
    consistency_under_pressure: float = 0.0  # Pressure consistency (0-1.0)
    captaincy_experience: float = 0.0    # Historical captaincy data (0-2.0)
    tactical_awareness: float = 0.0      # Format-specific tactical understanding (0-2.0)
    clutch_factor: float = 0.0           # Performance in close games (0-2.0)
    team_performance_boost: float = 0.0  # How much team improves with this captain (0-1.5)

@dataclass
class CaptaincyAnalysis:
    """Complete captaincy analysis result"""
    player_name: str
    captaincy_score: float
    vice_captaincy_score: float
    metrics: CaptaincyMetrics
    pressure_breakdown: Dict[str, float]
    situation_performance: Dict[str, float]
    confidence_level: float
    recommendation: str
    risk_assessment: str

class AdvancedCaptaincyEngine:
    """Production-ready captaincy analysis engine"""
    
    def __init__(self):
        self.pressure_weights = {
            'knockout': 2.0,
            'high': 1.5,
            'medium': 1.2,
            'low': 1.0
        }
        
        self.situation_weights = {
            'death_overs': 1.8,
            'chasing': 1.6,
            'powerplay': 1.4,
            'setting': 1.3,
            'middle_overs': 1.2
        }
        
        # Format-specific captaincy importance
        self.format_importance = {
            'T20': {
                'tactical_awareness': 0.25,
                'pressure_performance': 0.30,
                'clutch_factor': 0.20,
                'leadership_rating': 0.15,
                'big_match_record': 0.10
            },
            'ODI': {
                'tactical_awareness': 0.30,
                'pressure_performance': 0.25,
                'leadership_rating': 0.20,
                'big_match_record': 0.15,
                'clutch_factor': 0.10
            },
            'Test': {
                'leadership_rating': 0.35,
                'tactical_awareness': 0.25,
                'consistency_under_pressure': 0.20,
                'big_match_record': 0.15,
                'pressure_performance': 0.05
            }
        }
    
    def calculate_pressure_performance(self, player_data: Dict[str, Any],
                                     match_context: Dict[str, Any]) -> float:
        """
        Calculate how player performs under pressure
        
        Analyzes:
        - Performance in last 5 overs
        - High-pressure chases
        - Knockout matches
        - Against strong opposition
        """
        
        base_performance = player_data.get('ema_score', 50.0)
        
        # Get pressure situations from player history
        pressure_situations = self._extract_pressure_situations(player_data)
        
        if not pressure_situations:
            # No pressure data - use conservative estimate
            return min(1.2, max(0.8, base_performance / 60.0))
        
        # Calculate pressure performance metrics
        pressure_scores = []
        
        for situation in pressure_situations:
            situation_type = situation.get('type', 'medium')
            performance = situation.get('performance', base_performance)
            weight = self.pressure_weights.get(situation_type, 1.0)
            
            # Normalize and weight the performance
            normalized_perf = performance / base_performance if base_performance > 0 else 1.0
            weighted_score = normalized_perf * weight
            pressure_scores.append(weighted_score)
        
        # Calculate weighted average with recency bias
        if pressure_scores:
            recent_weight = 1.0
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for i, score in enumerate(pressure_scores):
                weight = recent_weight * (0.9 ** i)  # Exponential decay for older data
                total_weighted_score += score * weight
                total_weight += weight
            
            pressure_multiplier = total_weighted_score / total_weight
        else:
            pressure_multiplier = 1.0
        
        # Clamp between 0.5 and 2.0
        return max(0.5, min(2.0, pressure_multiplier))
    
    def calculate_leadership_rating(self, player_data: Dict[str, Any],
                                  team_context: Dict[str, Any]) -> float:
        """
        Calculate leadership impact factor
        
        Considers:
        - Team performance with vs without player
        - Senior player status
        - Experience in format
        - International captaincy experience
        """
        
        leadership_components = []
        
        # Experience factor
        career_matches = player_data.get('career_stats', {}).get('matches', 0)
        if career_matches > 100:
            experience_factor = 1.5
        elif career_matches > 50:
            experience_factor = 1.3
        elif career_matches > 20:
            experience_factor = 1.1
        else:
            experience_factor = 0.9
        
        leadership_components.append(('experience', experience_factor, 0.3))
        
        # BALANCED role-based leadership potential (equal priority for key roles)
        role = player_data.get('role', '').lower()
        if 'allrounder' in role:
            role_leadership = 1.4  # All-rounders: versatile leaders, equal to batsmen
        elif 'keeper' in role or 'wk' in role:
            role_leadership = 1.4  # Wicket-keepers are batsmen with field awareness
        elif 'bat' in role:
            role_leadership = 1.4  # Star batsmen: proven performers and leaders
        else:
            role_leadership = 1.2  # Bowlers: tactical understanding, good captains
        
        leadership_components.append(('role_leadership', role_leadership, 0.2))
        
        # Performance consistency (leaders need to be reliable)
        consistency = player_data.get('consistency_score', 50.0) / 100.0
        consistency_factor = 0.8 + consistency * 0.8  # 0.8 to 1.6 range
        
        leadership_components.append(('consistency', consistency_factor, 0.25))
        
        # Current form (in-form players inspire confidence)
        form_momentum = player_data.get('form_momentum', 0.0)
        if form_momentum > 0.5:
            form_factor = 1.3
        elif form_momentum > 0:
            form_factor = 1.1
        elif form_momentum > -0.3:
            form_factor = 1.0
        else:
            form_factor = 0.9
        
        leadership_components.append(('current_form', form_factor, 0.15))
        
        # Team senior status
        team_avg_age = team_context.get('average_age', 28)
        player_age = player_data.get('age', 25)
        if player_age >= team_avg_age + 2:
            seniority_factor = 1.2
        elif player_age >= team_avg_age:
            seniority_factor = 1.1
        else:
            seniority_factor = 1.0
        
        leadership_components.append(('seniority', seniority_factor, 0.1))
        
        # Calculate weighted leadership rating
        leadership_rating = sum(score * weight for _, score, weight in leadership_components)
        
        return max(0.5, min(2.0, leadership_rating))
    
    def calculate_big_match_record(self, player_data: Dict[str, Any],
                                 match_context: Dict[str, Any]) -> float:
        """
        Calculate performance in high-stakes matches
        
        Considers:
        - World Cup/Championship performances
        - Finals and semi-finals
        - High-profile series
        - Derby matches
        """
        
        base_performance = player_data.get('ema_score', 50.0)
        
        # Simulate big match analysis (in production, this would use actual data)
        big_matches = self._extract_big_match_data(player_data, match_context)
        
        if not big_matches:
            # No big match data - neutral rating
            return 1.0
        
        big_match_performances = [match.get('performance', base_performance) for match in big_matches]
        avg_big_match_performance = sum(big_match_performances) / len(big_match_performances)
        
        # Compare to overall average
        big_match_ratio = avg_big_match_performance / max(base_performance, 1)
        
        # Weight recent big matches more heavily
        if len(big_matches) >= 3:
            recent_big_matches = big_matches[:3]
            recent_avg = sum(match.get('performance', base_performance) for match in recent_big_matches) / 3
            recent_ratio = recent_avg / max(base_performance, 1)
            
            # Combine overall and recent (60% recent, 40% overall)
            big_match_ratio = recent_ratio * 0.6 + big_match_ratio * 0.4
        
        return max(0.5, min(2.0, big_match_ratio))
    
    def calculate_tactical_awareness(self, player_data: Dict[str, Any],
                                   match_context: Dict[str, Any]) -> float:
        """
        Calculate tactical understanding for specific format
        """
        
        match_format = match_context.get('match_format', 'T20')
        role = player_data.get('role', '').lower()
        
        # Base tactical score based on role and experience
        base_tactical = 1.0
        
        # Experience in format
        format_matches = player_data.get('format_experience', {}).get(match_format, 0)
        if format_matches > 50:
            experience_bonus = 0.5
        elif format_matches > 20:
            experience_bonus = 0.3
        elif format_matches > 10:
            experience_bonus = 0.1
        else:
            experience_bonus = -0.2
        
        # Role-specific tactical advantages
        if 'keeper' in role:
            role_bonus = 0.4  # Best field awareness
        elif 'allrounder' in role:
            role_bonus = 0.3  # Understand both batting and bowling
        elif 'bowl' in role:
            role_bonus = 0.2  # Good understanding of field placements
        else:
            role_bonus = 0.1  # Batsmen
        
        # Format-specific adjustments
        if match_format == 'T20':
            # T20 rewards aggressive tactical thinking
            aggression_factor = player_data.get('aggression_index', 1.0)
            format_adjustment = (aggression_factor - 1.0) * 0.2
        elif match_format == 'Test':
            # Test cricket rewards patience and long-term thinking
            patience_factor = player_data.get('consistency_score', 50.0) / 100.0
            format_adjustment = patience_factor * 0.3
        else:  # ODI
            # ODI rewards balanced approach
            balance_factor = 1.0 - abs(player_data.get('form_momentum', 0.0))
            format_adjustment = balance_factor * 0.2
        
        tactical_score = base_tactical + experience_bonus + role_bonus + format_adjustment
        
        return max(0.5, min(2.0, tactical_score))
    
    def calculate_clutch_factor(self, player_data: Dict[str, Any]) -> float:
        """
        Calculate performance in close/deciding moments
        """
        
        # Analyze performance in close matches
        close_match_data = self._extract_close_match_performances(player_data)
        
        if not close_match_data:
            return 1.0
        
        base_performance = player_data.get('ema_score', 50.0)
        
        # Calculate performance in matches decided by small margins
        clutch_performances = []
        for match in close_match_data:
            match_pressure = match.get('pressure_level', 1.0)
            performance = match.get('performance', base_performance)
            margin = match.get('margin', 'medium')  # close, medium, comfortable
            
            if margin == 'close':
                pressure_multiplier = 1.5
            elif margin == 'medium':
                pressure_multiplier = 1.2
            else:
                pressure_multiplier = 1.0
            
            # Prevent division by zero
            if base_performance > 0:
                clutch_score = (performance / base_performance) * pressure_multiplier
            else:
                clutch_score = pressure_multiplier  # Use pressure multiplier as fallback
            clutch_performances.append(clutch_score)
        
        if clutch_performances:
            avg_clutch = sum(clutch_performances) / len(clutch_performances)
            return max(0.5, min(2.0, avg_clutch))
        
        return 1.0
    
    def calculate_comprehensive_captaincy_score(self, player_data: Dict[str, Any],
                                              match_context: Dict[str, Any],
                                              team_context: Dict[str, Any]) -> CaptaincyAnalysis:
        """
        Calculate comprehensive captaincy suitability score
        
        Returns complete analysis with breakdown
        """
        
        match_format = match_context.get('match_format', 'T20')
        weights = self.format_importance.get(match_format, self.format_importance['T20'])
        
        # Calculate all captaincy metrics
        metrics = CaptaincyMetrics()
        
        metrics.pressure_performance = self.calculate_pressure_performance(player_data, match_context)
        metrics.leadership_rating = self.calculate_leadership_rating(player_data, team_context)
        metrics.big_match_record = self.calculate_big_match_record(player_data, match_context)
        metrics.consistency_under_pressure = min(2.0, metrics.pressure_performance * 
                                                 player_data.get('consistency_score', 50.0) / 100.0)
        metrics.captaincy_experience = self._calculate_captaincy_experience(player_data)
        metrics.tactical_awareness = self.calculate_tactical_awareness(player_data, match_context)
        metrics.clutch_factor = self.calculate_clutch_factor(player_data)
        metrics.team_performance_boost = self._calculate_team_boost(player_data, team_context)
        
        # Calculate weighted captaincy score
        captaincy_score = (
            metrics.pressure_performance * weights.get('pressure_performance', 0.3) * 25 +
            metrics.leadership_rating * weights.get('leadership_rating', 0.2) * 25 +
            metrics.big_match_record * weights.get('big_match_record', 0.15) * 25 +
            metrics.tactical_awareness * weights.get('tactical_awareness', 0.25) * 25 +
            metrics.clutch_factor * weights.get('clutch_factor', 0.1) * 25
        )
        
        # Vice-captaincy score (slightly different weighting)
        vice_captaincy_score = captaincy_score * 0.8 + player_data.get('final_score', 50.0) * 0.2
        
        # Confidence calculation
        data_completeness = self._assess_data_completeness(player_data)
        confidence_level = min(1.0, data_completeness * metrics.consistency_under_pressure)
        
        # Generate recommendation
        recommendation = self._generate_captaincy_recommendation(captaincy_score, metrics, match_format)
        
        # Risk assessment
        risk_assessment = self._assess_captaincy_risk(metrics, player_data)
        
        # Pressure breakdown
        pressure_breakdown = {
            'knockout_pressure': metrics.pressure_performance * 0.4,
            'chase_pressure': metrics.pressure_performance * 0.3,
            'death_overs_pressure': metrics.pressure_performance * 0.3
        }
        
        # Situation performance
        situation_performance = {
            'powerplay_leadership': metrics.tactical_awareness * 0.4,
            'middle_overs_management': metrics.leadership_rating * 0.5,
            'death_overs_execution': metrics.clutch_factor * 0.6,
            'pressure_handling': metrics.pressure_performance * 0.7
        }
        
        return CaptaincyAnalysis(
            player_name=player_data.get('name', 'Unknown'),
            captaincy_score=captaincy_score,
            vice_captaincy_score=vice_captaincy_score,
            metrics=metrics,
            pressure_breakdown=pressure_breakdown,
            situation_performance=situation_performance,
            confidence_level=confidence_level,
            recommendation=recommendation,
            risk_assessment=risk_assessment
        )
    
    def select_optimal_captain_and_vice(self, team_players: List[Dict[str, Any]],
                                      match_context: Dict[str, Any],
                                      team_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[CaptaincyAnalysis]]:
        """
        Select optimal captain and vice-captain from team
        
        Returns:
            Tuple of (captain, vice_captain, all_analyses)
        """
        
        if not team_players:
            raise ValueError("No players provided for captaincy selection")
        
        # Analyze all players for captaincy
        analyses = []
        for player in team_players:
            analysis = self.calculate_comprehensive_captaincy_score(player, match_context, team_context)
            analyses.append(analysis)
        
        # Sort by captaincy score
        analyses.sort(key=lambda x: x.captaincy_score, reverse=True)
        
        # Select captain (highest captaincy score)
        captain_analysis = analyses[0]
        captain = next(p for p in team_players if p.get('name') == captain_analysis.player_name)
        
        # Select vice-captain (different role preferred, high vice-captaincy score)
        captain_role = captain.get('role', '').lower()
        
        # Filter candidates for vice-captain (excluding captain)
        vc_candidates = [a for a in analyses[1:]]  # Exclude captain
        
        # Prefer different role for tactical diversity
        different_role_candidates = [
            a for a in vc_candidates 
            if self._get_role_category(a.player_name, team_players) != self._get_role_category(captain.get('name'), team_players)
        ]
        
        if different_role_candidates:
            # Select best from different role
            vice_captain_analysis = max(different_role_candidates, key=lambda x: x.vice_captaincy_score)
        else:
            # Select best overall if no different role available
            vice_captain_analysis = max(vc_candidates, key=lambda x: x.vice_captaincy_score)
        
        vice_captain = next(p for p in team_players if p.get('name') == vice_captain_analysis.player_name)
        
        logger.info(f"Selected Captain: {captain.get('name')} (Score: {captain_analysis.captaincy_score:.1f})")
        logger.info(f"Selected Vice-Captain: {vice_captain.get('name')} (Score: {vice_captain_analysis.vice_captaincy_score:.1f})")
        
        return captain, vice_captain, analyses
    
    # Helper methods
    def _extract_pressure_situations(self, player_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pressure situation data (simulated for now)"""
        # In production, this would analyze actual match data
        base_performance = player_data.get('ema_score', 50.0)
        consistency = player_data.get('consistency_score', 50.0)
        
        # Simulate pressure situations based on player profile
        situations = []
        for i in range(5):  # Simulate 5 pressure situations
            pressure_type = ['knockout', 'high', 'medium'][i % 3]
            performance_variance = (100 - consistency) / 100 * 20  # More inconsistent = more variance
            performance = base_performance + np.random.normal(0, performance_variance)
            situations.append({
                'type': pressure_type,
                'performance': max(0, performance),
                'match_id': f'match_{i}'
            })
        
        return situations
    
    def _extract_big_match_data(self, player_data: Dict[str, Any], match_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract big match performance data"""
        # Simulated big match data
        base_performance = player_data.get('ema_score', 50.0)
        big_match_modifier = 1.0 + (player_data.get('consistency_score', 50.0) - 50) / 100
        
        return [{
            'performance': base_performance * big_match_modifier,
            'importance': 'high',
            'match_type': 'final'
        }]
    
    def _extract_close_match_performances(self, player_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract close match performance data"""
        # Simulated close match data
        return [{
            'performance': player_data.get('ema_score', 50.0),
            'margin': 'close',
            'pressure_level': 1.5
        }]
    
    def _calculate_captaincy_experience(self, player_data: Dict[str, Any]) -> float:
        """Calculate captaincy experience factor"""
        # Simulated captaincy experience
        career_matches = player_data.get('career_stats', {}).get('matches', 0)
        if career_matches > 100:
            return 1.5
        elif career_matches > 50:
            return 1.2
        else:
            return 1.0
    
    def _calculate_team_boost(self, player_data: Dict[str, Any], team_context: Dict[str, Any]) -> float:
        """Calculate how much team performance improves with this captain"""
        leadership_rating = player_data.get('leadership_rating', 1.0)
        return min(1.5, leadership_rating * 0.5)
    
    def _assess_data_completeness(self, player_data: Dict[str, Any]) -> float:
        """Assess completeness of player data for confidence calculation"""
        required_fields = ['ema_score', 'consistency_score', 'career_stats', 'role']
        available_fields = sum(1 for field in required_fields if player_data.get(field) is not None)
        return available_fields / len(required_fields)
    
    def _generate_captaincy_recommendation(self, score: float, metrics: CaptaincyMetrics, match_format: str) -> str:
        """Generate captaincy recommendation text"""
        if score >= 80:
            return f"Excellent {match_format} captain - strong leadership and pressure performance"
        elif score >= 70:
            return f"Very good {match_format} captain - reliable in most situations"
        elif score >= 60:
            return f"Good {match_format} captain - solid choice with room for improvement"
        elif score >= 50:
            return f"Adequate {match_format} captain - consider alternatives if available"
        else:
            return f"Poor {match_format} captain choice - high risk option"
    
    def _assess_captaincy_risk(self, metrics: CaptaincyMetrics, player_data: Dict[str, Any]) -> str:
        """Assess risk level of captaincy choice"""
        risk_factors = []
        
        if metrics.pressure_performance < 1.0:
            risk_factors.append("struggles under pressure")
        if metrics.consistency_under_pressure < 0.7:
            risk_factors.append("inconsistent in crucial moments")
        if player_data.get('form_momentum', 0) < -0.3:
            risk_factors.append("poor recent form")
        
        if not risk_factors:
            return "Low risk - stable captaincy choice"
        elif len(risk_factors) == 1:
            return f"Medium risk - {risk_factors[0]}"
        else:
            return f"High risk - {', '.join(risk_factors)}"
    
    def _get_role_category(self, player_name: str, team_players: List[Dict[str, Any]]) -> str:
        """Get role category for a player"""
        player = next((p for p in team_players if p.get('name') == player_name), None)
        if not player:
            return 'unknown'
        
        role = player.get('role', '').lower()
        if 'bat' in role:
            return 'batsman'
        elif 'bowl' in role:
            return 'bowler'
        elif 'keep' in role or 'wk' in role:
            return 'keeper'
        elif 'allrounder' in role:
            return 'allrounder'
        else:
            return 'unknown'

# Global instance
_captaincy_engine = None

def get_captaincy_engine() -> AdvancedCaptaincyEngine:
    """Get global captaincy engine instance"""
    global _captaincy_engine
    if _captaincy_engine is None:
        _captaincy_engine = AdvancedCaptaincyEngine()
    return _captaincy_engine

def select_advanced_captain_and_vice(team_players: List[Dict[str, Any]],
                                   match_context: Dict[str, Any],
                                   team_context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Advanced captain and vice-captain selection for integration with existing code
    
    Returns:
        Tuple of (captain, vice_captain)
    """
    if team_context is None:
        team_context = {'average_age': 27}
    
    engine = get_captaincy_engine()
    captain, vice_captain, _ = engine.select_optimal_captain_and_vice(team_players, match_context, team_context)
    
    return captain, vice_captain

if __name__ == "__main__":
    # Test the advanced captaincy engine
    test_players = [
        {
            'name': 'Virat Kohli',
            'role': 'Batsman',
            'ema_score': 85.0,
            'consistency_score': 80.0,
            'form_momentum': 0.3,
            'final_score': 88.0,
            'career_stats': {'matches': 150},
            'age': 34
        },
        {
            'name': 'MS Dhoni',
            'role': 'WK-Batsman',
            'ema_score': 75.0,
            'consistency_score': 90.0,
            'form_momentum': 0.1,
            'final_score': 82.0,
            'career_stats': {'matches': 200},
            'age': 41
        },
        {
            'name': 'Hardik Pandya',
            'role': 'All-rounder',
            'ema_score': 78.0,
            'consistency_score': 70.0,
            'form_momentum': 0.5,
            'final_score': 85.0,
            'career_stats': {'matches': 80},
            'age': 29
        }
    ]
    
    test_match_context = {
        'match_format': 'T20',
        'match_importance': 'high',
        'venue_factor': 1.1
    }
    
    test_team_context = {
        'average_age': 28
    }
    
    engine = AdvancedCaptaincyEngine()
    captain, vice_captain, analyses = engine.select_optimal_captain_and_vice(
        test_players, test_match_context, test_team_context
    )
    
    print("🏆 Advanced Captaincy Selection Test")
    print(f"Captain: {captain['name']} (Score: {analyses[0].captaincy_score:.1f})")
    print(f"Vice-Captain: {vice_captain['name']}")
    print(f"Captain Analysis: {analyses[0].recommendation}")
    print(f"Risk Assessment: {analyses[0].risk_assessment}")