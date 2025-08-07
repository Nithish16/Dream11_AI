#!/usr/bin/env python3
"""
Dynamic Feature Weighting Engine - PRODUCTION READY
Context-aware adaptive feature weighting system
Replaces static weights with intelligent context-based adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchImportance(Enum):
    """Match importance levels"""
    DEAD_RUBBER = "dead_rubber"
    NORMAL = "normal"
    IMPORTANT = "important"
    SEMIFINAL = "semifinal"
    FINAL = "final"
    KNOCKOUT = "knockout"

class PitchType(Enum):
    """Pitch condition types"""
    BATTING_PARADISE = "batting_paradise"
    BALANCED = "balanced"
    BOWLING_FRIENDLY = "bowling_friendly"
    SPINNER_FRIENDLY = "spinner_friendly"
    SEAMER_FRIENDLY = "seamer_friendly"
    UNPREDICTABLE = "unpredictable"

class WeatherImpact(Enum):
    """Weather impact levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    RAIN_EXPECTED = "rain_expected"
    EXTREME_CONDITIONS = "extreme_conditions"

@dataclass
class FeatureWeights:
    """Dynamic feature weights structure"""
    ema_score: float = 0.25
    consistency_score: float = 0.15
    opportunity_index: float = 0.30
    form_momentum: float = 0.20
    role_bonus: float = 0.10
    pressure_performance: float = 0.0  # Additional weight for pressure situations
    matchup_bonus: float = 0.0  # Additional weight for specific matchups
    environmental_factor: float = 0.0  # Additional weight for environmental conditions
    
    def normalize(self) -> 'FeatureWeights':
        """Normalize weights to sum to 1.0"""
        total = (self.ema_score + self.consistency_score + self.opportunity_index + 
                self.form_momentum + self.role_bonus + self.pressure_performance + 
                self.matchup_bonus + self.environmental_factor)
        
        if total <= 0:
            return FeatureWeights()  # Return default weights
        
        return FeatureWeights(
            ema_score=self.ema_score / total,
            consistency_score=self.consistency_score / total,
            opportunity_index=self.opportunity_index / total,
            form_momentum=self.form_momentum / total,
            role_bonus=self.role_bonus / total,
            pressure_performance=self.pressure_performance / total,
            matchup_bonus=self.matchup_bonus / total,
            environmental_factor=self.environmental_factor / total
        )

@dataclass
class WeightingContext:
    """Context information for dynamic weighting"""
    match_format: str = "T20"
    match_importance: MatchImportance = MatchImportance.NORMAL
    pitch_type: PitchType = PitchType.BALANCED
    weather_impact: WeatherImpact = WeatherImpact.NONE
    venue_factor: float = 1.0
    opposition_strength: float = 0.7
    series_stage: str = "league"  # league, playoff, final
    time_of_day: str = "day"  # day, evening, night
    tournament_stage: str = "group"  # group, knockout, final
    historical_venue_data: Dict[str, float] = field(default_factory=dict)

@dataclass
class WeightingAnalysis:
    """Analysis of weighting adjustments"""
    original_weights: FeatureWeights
    adjusted_weights: FeatureWeights
    adjustments_made: List[str]
    confidence_level: float
    reasoning: str
    impact_assessment: str

class DynamicFeatureWeightingEngine:
    """Production-ready dynamic feature weighting system"""
    
    def __init__(self):
        # Base weights for different formats
        self.base_weights = {
            'T20': FeatureWeights(
                ema_score=0.25,
                consistency_score=0.15,
                opportunity_index=0.30,
                form_momentum=0.20,
                role_bonus=0.10
            ),
            'ODI': FeatureWeights(
                ema_score=0.30,
                consistency_score=0.25,
                opportunity_index=0.20,
                form_momentum=0.15,
                role_bonus=0.10
            ),
            'Test': FeatureWeights(
                ema_score=0.20,
                consistency_score=0.35,
                opportunity_index=0.25,
                form_momentum=0.10,
                role_bonus=0.10
            ),
            'T10': FeatureWeights(
                ema_score=0.20,
                consistency_score=0.10,
                opportunity_index=0.35,
                form_momentum=0.25,
                role_bonus=0.10
            )
        }
        
        # Context-specific adjustments
        self.importance_adjustments = {
            MatchImportance.DEAD_RUBBER: {
                'form_momentum': 1.3,
                'consistency_score': 0.8,
                'pressure_performance': 0.5
            },
            MatchImportance.KNOCKOUT: {
                'consistency_score': 1.5,
                'pressure_performance': 1.8,
                'form_momentum': 0.8
            },
            MatchImportance.FINAL: {
                'consistency_score': 1.6,
                'pressure_performance': 2.0,
                'ema_score': 1.2,
                'form_momentum': 0.7
            }
        }
        
        self.pitch_adjustments = {
            PitchType.BATTING_PARADISE: {
                'opportunity_index': 1.3,
                'form_momentum': 1.2,
                'consistency_score': 0.9
            },
            PitchType.BOWLING_FRIENDLY: {
                'consistency_score': 1.4,
                'ema_score': 1.2,
                'opportunity_index': 0.8,
                'form_momentum': 0.9
            },
            PitchType.SPINNER_FRIENDLY: {
                'role_bonus': 1.5,  # Spinners and spin-playing batsmen
                'matchup_bonus': 1.3,
                'opportunity_index': 1.1
            },
            PitchType.UNPREDICTABLE: {
                'consistency_score': 1.6,
                'form_momentum': 0.8,
                'opportunity_index': 0.9
            }
        }
        
        self.weather_adjustments = {
            WeatherImpact.RAIN_EXPECTED: {
                'role_bonus': 1.4,  # All-rounders more valuable
                'form_momentum': 0.9,
                'consistency_score': 1.2,
                'environmental_factor': 1.5
            },
            WeatherImpact.EXTREME_CONDITIONS: {
                'consistency_score': 1.7,
                'ema_score': 1.3,
                'form_momentum': 0.7,
                'environmental_factor': 1.8
            }
        }
    
    def calculate_adaptive_weights(self, context: WeightingContext,
                                 player_role: str = "Batsman") -> WeightingAnalysis:
        """
        Calculate context-adaptive feature weights
        
        Args:
            context: Match and environmental context
            player_role: Player's role for role-specific adjustments
            
        Returns:
            WeightingAnalysis with original weights, adjusted weights, and reasoning
        """
        
        # Start with base weights for format
        base_weights = self.base_weights.get(context.match_format, self.base_weights['T20'])
        working_weights = FeatureWeights(
            ema_score=base_weights.ema_score,
            consistency_score=base_weights.consistency_score,
            opportunity_index=base_weights.opportunity_index,
            form_momentum=base_weights.form_momentum,
            role_bonus=base_weights.role_bonus
        )
        
        adjustments_made = []
        
        # 1. Match importance adjustments
        importance_adj = self.importance_adjustments.get(context.match_importance, {})
        if importance_adj:
            self._apply_adjustments(working_weights, importance_adj, "match importance")
            adjustments_made.append(f"Adjusted for {context.match_importance.value} match")
        
        # 2. Pitch condition adjustments
        pitch_adj = self.pitch_adjustments.get(context.pitch_type, {})
        if pitch_adj:
            self._apply_adjustments(working_weights, pitch_adj, "pitch conditions")
            adjustments_made.append(f"Adjusted for {context.pitch_type.value} pitch")
        
        # 3. Weather impact adjustments
        weather_adj = self.weather_adjustments.get(context.weather_impact, {})
        if weather_adj:
            self._apply_adjustments(working_weights, weather_adj, "weather conditions")
            adjustments_made.append(f"Adjusted for {context.weather_impact.value} weather")
        
        # 4. Opposition strength adjustments
        if context.opposition_strength > 0.8:
            # Strong opposition - favor consistency and proven performers
            working_weights.consistency_score *= 1.3
            working_weights.ema_score *= 1.2
            working_weights.form_momentum *= 0.9
            adjustments_made.append("Adjusted for strong opposition")
        elif context.opposition_strength < 0.5:
            # Weak opposition - form and opportunity matter more
            working_weights.form_momentum *= 1.2
            working_weights.opportunity_index *= 1.1
            working_weights.consistency_score *= 0.9
            adjustments_made.append("Adjusted for weak opposition")
        
        # 5. Venue-specific adjustments
        if context.venue_factor > 1.2:
            # Venue strongly favors certain type of play
            working_weights.environmental_factor += 0.05
            working_weights.matchup_bonus += 0.03
            adjustments_made.append("Adjusted for venue bias")
        
        # 6. Tournament stage adjustments
        if context.tournament_stage == "knockout":
            working_weights.pressure_performance += 0.08
            working_weights.consistency_score *= 1.2
            working_weights.form_momentum *= 0.9
            adjustments_made.append("Adjusted for knockout stage")
        
        # 7. Time of day adjustments
        if context.time_of_day == "night":
            # Night matches often have dew factor
            working_weights.environmental_factor += 0.03
            working_weights.role_bonus *= 1.1  # Spinners may struggle
            adjustments_made.append("Adjusted for night match conditions")
        
        # 8. Role-specific contextual adjustments
        role_adjustments = self._get_role_specific_adjustments(player_role, context)
        if role_adjustments:
            self._apply_adjustments(working_weights, role_adjustments, "role-specific context")
            adjustments_made.append(f"Applied {player_role}-specific adjustments")
        
        # 9. Historical venue performance adjustments
        if context.historical_venue_data:
            venue_performance = context.historical_venue_data.get('avg_performance', 1.0)
            if venue_performance > 1.1:
                working_weights.opportunity_index *= 1.1
                adjustments_made.append("Adjusted for favorable venue history")
            elif venue_performance < 0.9:
                working_weights.consistency_score *= 1.1
                adjustments_made.append("Adjusted for challenging venue history")
        
        # Normalize weights
        normalized_weights = working_weights.normalize()
        
        # Calculate confidence level
        confidence_level = self._calculate_adjustment_confidence(context, adjustments_made)
        
        # Generate reasoning
        reasoning = self._generate_weighting_reasoning(base_weights, normalized_weights, context, adjustments_made)
        
        # Impact assessment
        impact_assessment = self._assess_adjustment_impact(base_weights, normalized_weights)
        
        return WeightingAnalysis(
            original_weights=base_weights,
            adjusted_weights=normalized_weights,
            adjustments_made=adjustments_made,
            confidence_level=confidence_level,
            reasoning=reasoning,
            impact_assessment=impact_assessment
        )
    
    def _apply_adjustments(self, weights: FeatureWeights, adjustments: Dict[str, float], context: str):
        """Apply adjustment multipliers to weights"""
        for weight_name, multiplier in adjustments.items():
            if hasattr(weights, weight_name):
                current_value = getattr(weights, weight_name)
                new_value = current_value * multiplier
                setattr(weights, weight_name, new_value)
    
    def _get_role_specific_adjustments(self, role: str, context: WeightingContext) -> Dict[str, float]:
        """Get role-specific contextual adjustments"""
        role_lower = role.lower()
        adjustments = {}
        
        if 'bat' in role_lower:
            # Batsman adjustments
            if context.pitch_type == PitchType.BATTING_PARADISE:
                adjustments['opportunity_index'] = 1.2
                adjustments['form_momentum'] = 1.1
            elif context.pitch_type == PitchType.BOWLING_FRIENDLY:
                adjustments['consistency_score'] = 1.3
                adjustments['ema_score'] = 1.2
        
        elif 'bowl' in role_lower:
            # Bowler adjustments
            if context.pitch_type == PitchType.BOWLING_FRIENDLY:
                adjustments['opportunity_index'] = 1.3
                adjustments['form_momentum'] = 1.2
            elif context.weather_impact == WeatherImpact.RAIN_EXPECTED:
                adjustments['role_bonus'] = 1.2  # Bowlers crucial in shortened games
        
        elif 'allrounder' in role_lower:
            # All-rounder adjustments
            if context.weather_impact == WeatherImpact.RAIN_EXPECTED:
                adjustments['role_bonus'] = 1.4
                adjustments['opportunity_index'] = 1.2
            if context.match_format == 'T20':
                adjustments['form_momentum'] = 1.1
        
        elif 'wk' in role_lower or 'keeper' in role_lower:
            # Wicket-keeper adjustments
            adjustments['consistency_score'] = 1.1  # Consistency important for keepers
            if context.pitch_type == PitchType.SPINNER_FRIENDLY:
                adjustments['role_bonus'] = 1.2  # Keeper skills more valuable
        
        return adjustments
    
    def _calculate_adjustment_confidence(self, context: WeightingContext, adjustments: List[str]) -> float:
        """Calculate confidence in weight adjustments"""
        base_confidence = 0.8
        
        # Reduce confidence for extreme conditions
        if context.weather_impact == WeatherImpact.EXTREME_CONDITIONS:
            base_confidence -= 0.2
        
        # Reduce confidence for unpredictable pitches
        if context.pitch_type == PitchType.UNPREDICTABLE:
            base_confidence -= 0.1
        
        # Increase confidence for well-known contexts
        if context.match_importance in [MatchImportance.FINAL, MatchImportance.KNOCKOUT]:
            base_confidence += 0.1
        
        # Adjust based on number of adjustments (too many = less confident)
        if len(adjustments) > 5:
            base_confidence -= 0.1
        elif len(adjustments) < 2:
            base_confidence -= 0.05
        
        return max(0.3, min(1.0, base_confidence))
    
    def _generate_weighting_reasoning(self, original: FeatureWeights, adjusted: FeatureWeights,
                                    context: WeightingContext, adjustments: List[str]) -> str:
        """Generate human-readable reasoning for weight adjustments"""
        
        major_changes = []
        
        # Identify significant changes
        if adjusted.consistency_score > original.consistency_score * 1.2:
            major_changes.append("increased consistency importance for reliability")
        
        if adjusted.form_momentum > original.form_momentum * 1.2:
            major_changes.append("emphasized recent form trends")
        elif adjusted.form_momentum < original.form_momentum * 0.8:
            major_changes.append("reduced form emphasis for stability")
        
        if adjusted.opportunity_index > original.opportunity_index * 1.1:
            major_changes.append("boosted opportunity factor for favorable conditions")
        
        if adjusted.pressure_performance > 0.05:
            major_changes.append("added pressure performance weighting")
        
        reasoning_parts = [
            f"Adapted {context.match_format} base weights for {context.match_importance.value} match"
        ]
        
        if major_changes:
            reasoning_parts.append(f"Key changes: {', '.join(major_changes)}")
        
        if context.pitch_type != PitchType.BALANCED:
            reasoning_parts.append(f"Adjusted for {context.pitch_type.value} pitch conditions")
        
        if context.weather_impact != WeatherImpact.NONE:
            reasoning_parts.append(f"Factored in {context.weather_impact.value} weather impact")
        
        return ". ".join(reasoning_parts) + "."
    
    def _assess_adjustment_impact(self, original: FeatureWeights, adjusted: FeatureWeights) -> str:
        """Assess the impact level of adjustments"""
        
        total_change = 0.0
        significant_changes = 0
        
        changes = [
            abs(adjusted.ema_score - original.ema_score),
            abs(adjusted.consistency_score - original.consistency_score),
            abs(adjusted.opportunity_index - original.opportunity_index),
            abs(adjusted.form_momentum - original.form_momentum),
            abs(adjusted.role_bonus - original.role_bonus)
        ]
        
        total_change = sum(changes)
        significant_changes = sum(1 for change in changes if change > 0.05)
        
        if total_change > 0.3 or significant_changes >= 3:
            return "High impact - significant weighting adjustments made"
        elif total_change > 0.15 or significant_changes >= 2:
            return "Medium impact - moderate weighting adjustments applied"
        elif total_change > 0.05:
            return "Low impact - minor weighting refinements made"
        else:
            return "Minimal impact - weights remain largely unchanged"
    
    def apply_dynamic_weights_to_score(self, player_features: Dict[str, Any],
                                     context: WeightingContext,
                                     player_role: str = "Batsman") -> Tuple[float, WeightingAnalysis]:
        """
        Apply dynamic weights to calculate player score
        
        Returns:
            Tuple of (weighted_score, weighting_analysis)
        """
        
        # Get adaptive weights
        analysis = self.calculate_adaptive_weights(context, player_role)
        weights = analysis.adjusted_weights
        
        # Extract player features with defaults
        ema_score = player_features.get('ema_score', 50.0)
        consistency_score = player_features.get('consistency_score', 50.0)
        opportunity_index = player_features.get('opportunity_index', 1.0)
        form_momentum = player_features.get('form_momentum', 0.0)
        role_bonus = self._calculate_role_bonus(player_role, context)
        pressure_performance = player_features.get('pressure_performance', 1.0)
        matchup_bonus = player_features.get('matchup_bonus', 0.0)
        environmental_factor = player_features.get('environmental_factor', 0.0)
        
        # Calculate weighted score
        weighted_score = (
            ema_score * weights.ema_score +
            consistency_score * weights.consistency_score +
            opportunity_index * 20 * weights.opportunity_index +  # Scale to reasonable range
            (form_momentum + 1) * 10 * weights.form_momentum +   # Normalize and scale
            role_bonus * weights.role_bonus +
            pressure_performance * 25 * weights.pressure_performance +
            matchup_bonus * 15 * weights.matchup_bonus +
            environmental_factor * 10 * weights.environmental_factor
        )
        
        return weighted_score, analysis
    
    def _calculate_role_bonus(self, role: str, context: WeightingContext) -> float:
        """Calculate role-specific bonus based on context"""
        role_lower = role.lower()
        match_format = context.match_format.lower()
        
        if match_format == "t20":
            if 'allrounder' in role_lower:
                return 5.0
            elif 'wk' in role_lower or 'keeper' in role_lower:
                return 5.0  # WK-batsmen equally valuable
            elif 'bat' in role_lower:
                return 4.0  # Star batsmen valuable in T20
            else:
                return 3.0  # Bowlers still important
        elif match_format == "odi":
            if 'allrounder' in role_lower:
                return 4.0
            elif 'bat' in role_lower:
                return 4.0  # Equal to all-rounders
            elif 'wk' in role_lower or 'keeper' in role_lower:
                return 4.0  # WK-batsmen equally valuable
            else:
                return 2.0  # Bowlers
        else:  # Test
            if 'bowl' in role_lower:
                return 4.0  # Bowlers crucial in Test
            elif 'allrounder' in role_lower:
                return 4.0
            elif 'bat' in role_lower:
                return 3.0  # Batsmen important in Test
            elif 'wk' in role_lower or 'keeper' in role_lower:
                return 3.0  # WK-batsmen
            else:
                return 0.0

# Global instance
_weighting_engine = None

def get_weighting_engine() -> DynamicFeatureWeightingEngine:
    """Get global weighting engine instance"""
    global _weighting_engine
    if _weighting_engine is None:
        _weighting_engine = DynamicFeatureWeightingEngine()
    return _weighting_engine

def calculate_adaptive_score(player_features: Dict[str, Any],
                           match_context: Dict[str, Any],
                           player_role: str = "Batsman") -> float:
    """
    Calculate player score with dynamic feature weighting
    
    Integration function for existing codebase
    """
    
    # Convert match_context to WeightingContext
    weighting_context = WeightingContext(
        match_format=match_context.get('match_format', 'T20'),
        match_importance=MatchImportance(match_context.get('match_importance', 'normal')),
        pitch_type=PitchType(match_context.get('pitch_type', 'balanced')),
        weather_impact=WeatherImpact(match_context.get('weather_impact', 'none')),
        venue_factor=match_context.get('venue_factor', 1.0),
        opposition_strength=match_context.get('opposition_strength', 0.7),
        series_stage=match_context.get('series_stage', 'league'),
        time_of_day=match_context.get('time_of_day', 'day'),
        tournament_stage=match_context.get('tournament_stage', 'group')
    )
    
    engine = get_weighting_engine()
    score, _ = engine.apply_dynamic_weights_to_score(player_features, weighting_context, player_role)
    
    return score

if __name__ == "__main__":
    # Test the dynamic weighting engine
    test_context = WeightingContext(
        match_format="T20",
        match_importance=MatchImportance.FINAL,
        pitch_type=PitchType.BATTING_PARADISE,
        weather_impact=WeatherImpact.NONE,
        opposition_strength=0.9,
        tournament_stage="knockout"
    )
    
    test_features = {
        'ema_score': 75.0,
        'consistency_score': 80.0,
        'opportunity_index': 1.2,
        'form_momentum': 0.3,
        'pressure_performance': 1.1
    }
    
    engine = DynamicFeatureWeightingEngine()
    analysis = engine.calculate_adaptive_weights(test_context, "Batsman")
    score, _ = engine.apply_dynamic_weights_to_score(test_features, test_context, "Batsman")
    
    print("⚖️ Dynamic Feature Weighting Test")
    print(f"Original weights: EMA={analysis.original_weights.ema_score:.2f}, Consistency={analysis.original_weights.consistency_score:.2f}")
    print(f"Adjusted weights: EMA={analysis.adjusted_weights.ema_score:.2f}, Consistency={analysis.adjusted_weights.consistency_score:.2f}")
    print(f"Adjustments made: {len(analysis.adjustments_made)}")
    print(f"Reasoning: {analysis.reasoning}")
    print(f"Final score: {score:.1f}")
    print(f"Impact: {analysis.impact_assessment}")