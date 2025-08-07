#!/usr/bin/env python3
"""
Format-Specific Team Constraints and Balancing Engine
Advanced constraints for T20, ODI, and TEST cricket team optimization
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ConstraintType(Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXACT = "exact"
    RANGE = "range"

@dataclass
class FormatConstraint:
    """Individual format-specific constraint"""
    name: str
    constraint_type: ConstraintType
    value: Any
    weight: float = 1.0
    description: str = ""

@dataclass
class FormatTeamBalance:
    """Team balance requirements for each format"""
    format_type: str
    constraints: List[FormatConstraint]
    role_preferences: Dict[str, float]
    skill_requirements: Dict[str, float]
    synergy_factors: Dict[str, float]

class FormatSpecificConstraintEngine:
    """Engine for managing format-specific team constraints"""
    
    def __init__(self):
        self.format_constraints = self._initialize_format_constraints()
        self.synergy_matrices = self._initialize_synergy_matrices()
        self.tactical_preferences = self._initialize_tactical_preferences()
    
    def _initialize_format_constraints(self) -> Dict[str, FormatTeamBalance]:
        """Initialize comprehensive format-specific constraints"""
        return {
            'T20': self._create_t20_constraints(),
            'ODI': self._create_odi_constraints(),
            'TEST': self._create_test_constraints()
        }
    
    def _create_t20_constraints(self) -> FormatTeamBalance:
        """T20-specific team constraints (High-Risk, High-Reward)"""
        constraints = [
            # Core team composition
            FormatConstraint(
                name="power_hitters",
                constraint_type=ConstraintType.MINIMUM,
                value=3,
                weight=1.0,
                description="Minimum 3 players with boundary% > 20%"
            ),
            FormatConstraint(
                name="death_bowlers",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=1.0,
                description="Minimum 2 bowlers with death overs economy < 8.5"
            ),
            FormatConstraint(
                name="powerplay_specialists",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=0.8,
                description="Minimum 2 players excelling in powerplay (6 overs)"
            ),
            FormatConstraint(
                name="finishers",
                constraint_type=ConstraintType.MINIMUM,
                value=1,
                weight=0.9,
                description="Minimum 1 player with strike rate > 150 in last 5 overs"
            ),
            FormatConstraint(
                name="fielding_agility",
                constraint_type=ConstraintType.MINIMUM,
                value=7,
                weight=0.7,
                description="Minimum 7 players with above-average fielding"
            ),
            # Advanced T20 constraints
            FormatConstraint(
                name="spin_in_middle_overs",
                constraint_type=ConstraintType.MINIMUM,
                value=1,
                weight=0.6,
                description="At least 1 spinner for middle overs control"
            ),
            FormatConstraint(
                name="left_right_balance",
                constraint_type=ConstraintType.RANGE,
                value=(3, 8),  # 3-8 right-handed batsmen
                weight=0.5,
                description="Balance between left and right-handed batsmen"
            )
        ]
        
        role_preferences = {
            'explosive_batsmen': 0.35,      # High preference for explosive batting
            'bowling_allrounders': 0.25,    # All-rounders provide flexibility
            'specialist_bowlers': 0.20,     # Need specialist bowlers
            'wicket_keepers': 0.20          # Wicket-keeper batting crucial
        }
        
        skill_requirements = {
            'boundary_percentage': 0.30,    # Most important for T20
            'strike_rate': 0.25,           # Critical for T20
            'death_bowling_skill': 0.25,   # Game-changing skill
            'powerplay_performance': 0.20   # Foundation for T20 success
        }
        
        synergy_factors = {
            'powerplay_combination': 1.2,   # Powerplay partnership bonus
            'death_bowling_pair': 1.3,     # Death bowling combination
            'finisher_support': 1.1,       # Support for main finisher
            'spin_pace_balance': 1.1       # Bowling variety bonus
        }
        
        return FormatTeamBalance('T20', constraints, role_preferences, skill_requirements, synergy_factors)
    
    def _create_odi_constraints(self) -> FormatTeamBalance:
        """ODI-specific team constraints (Balanced Approach)"""
        constraints = [
            # Core team composition
            FormatConstraint(
                name="anchors",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=1.0,
                description="Minimum 2 batsmen with average > 40 and SR > 80"
            ),
            FormatConstraint(
                name="middle_order_stabilizers",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=1.0,
                description="Players who can build partnerships in middle overs"
            ),
            FormatConstraint(
                name="death_bowlers",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=0.9,
                description="Bowlers for overs 41-50"
            ),
            FormatConstraint(
                name="partnership_builders",
                constraint_type=ConstraintType.MINIMUM,
                value=3,
                weight=1.0,
                description="Players with high partnership contribution"
            ),
            FormatConstraint(
                name="middle_overs_bowling",
                constraint_type=ConstraintType.MINIMUM,
                value=3,
                weight=0.8,
                description="Bowlers who can bowl effectively in overs 11-40"
            ),
            # Advanced ODI constraints
            FormatConstraint(
                name="chase_specialists",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=0.7,
                description="Players with good record in run chases"
            ),
            FormatConstraint(
                name="pressure_handlers",
                constraint_type=ConstraintType.MINIMUM,
                value=4,
                weight=0.8,
                description="Players who perform under pressure situations"
            )
        ]
        
        role_preferences = {
            'batting_anchors': 0.30,        # Foundation of ODI batting
            'bowling_allrounders': 0.25,    # Balance and depth
            'middle_order_batsmen': 0.25,   # Partnership builders
            'specialist_bowlers': 0.20      # Wicket-taking ability
        }
        
        skill_requirements = {
            'batting_average': 0.30,        # Consistency most important
            'partnership_building': 0.25,   # Critical for ODI success
            'middle_overs_skill': 0.25,     # Most crucial phase
            'pressure_handling': 0.20       # Handling run rate pressure
        }
        
        synergy_factors = {
            'opening_partnership': 1.2,     # Strong opening combination
            'middle_order_stability': 1.3,  # Partnership building bonus
            'bowling_rotation': 1.1,        # Effective bowling changes
            'pace_spin_balance': 1.2        # Bowling variety for 50 overs
        }
        
        return FormatTeamBalance('ODI', constraints, role_preferences, skill_requirements, synergy_factors)
    
    def _create_test_constraints(self) -> FormatTeamBalance:
        """TEST-specific team constraints (Consistency-First)"""
        constraints = [
            # Core team composition
            FormatConstraint(
                name="openers_patience",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=1.0,
                description="Openers with patience score > 70 and technique rating > 65"
            ),
            FormatConstraint(
                name="middle_order_temperament",
                constraint_type=ConstraintType.MINIMUM,
                value=3,
                weight=1.0,
                description="Middle order with temperament rating > 65"
            ),
            FormatConstraint(
                name="session_specialists",
                constraint_type=ConstraintType.MINIMUM,
                value=4,
                weight=0.9,
                description="Players who adapt well to different sessions"
            ),
            FormatConstraint(
                name="reverse_swing_bowlers",
                constraint_type=ConstraintType.MINIMUM,
                value=1,
                weight=0.8,
                description="At least 1 bowler effective with reverse swing"
            ),
            FormatConstraint(
                name="spin_options",
                constraint_type=ConstraintType.MINIMUM,
                value=1,
                weight=0.7,
                description="Spin bowling option for day 4-5"
            ),
            # Advanced TEST constraints
            FormatConstraint(
                name="first_innings_specialists",
                constraint_type=ConstraintType.MINIMUM,
                value=3,
                weight=0.7,
                description="Players who excel in first innings"
            ),
            FormatConstraint(
                name="follow_on_savers",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=0.6,
                description="Batsmen who can save follow-on situations"
            ),
            FormatConstraint(
                name="declaration_bowlers",
                constraint_type=ConstraintType.MINIMUM,
                value=2,
                weight=0.6,
                description="Bowlers who can take quick wickets for declarations"
            )
        ]
        
        role_preferences = {
            'technique_specialists': 0.35,   # Technical soundness crucial
            'bowling_workhorses': 0.25,     # Bowlers who can bowl long spells
            'partnership_anchors': 0.25,    # Building long partnerships
            'session_adapters': 0.15        # Adapting to changing conditions
        }
        
        skill_requirements = {
            'batting_average': 0.35,        # Most important for TEST
            'temperament': 0.30,           # Mental strength crucial
            'technique_rating': 0.25,      # Technical skills for longer format
            'session_adaptability': 0.10   # Adapting to different sessions
        }
        
        synergy_factors = {
            'opening_partnership': 1.4,     # Crucial for TEST success
            'pace_quartet': 1.3,            # Fast bowling combination
            'spin_pace_combo': 1.2,         # Bowling variety over 5 days
            'lower_order_resistance': 1.1   # Tail-end partnerships
        }
        
        return FormatTeamBalance('TEST', constraints, role_preferences, skill_requirements, synergy_factors)
    
    def _initialize_synergy_matrices(self) -> Dict[str, Dict[str, float]]:
        """Initialize player synergy matrices for each format"""
        return {
            'T20': {
                'powerplay_synergy': {
                    ('aggressive_opener', 'anchor_opener'): 1.2,
                    ('power_hitter', 'rotation_striker'): 1.1,
                    ('left_handed', 'right_handed'): 1.1
                },
                'death_synergy': {
                    ('yorker_specialist', 'slower_ball_expert'): 1.3,
                    ('pace_bowler', 'spinner'): 1.1
                }
            },
            'ODI': {
                'partnership_synergy': {
                    ('anchor', 'aggressor'): 1.3,
                    ('experienced', 'young_talent'): 1.1,
                    ('technique_player', 'power_player'): 1.2
                },
                'bowling_synergy': {
                    ('new_ball_bowler', 'first_change'): 1.2,
                    ('pace_bowler', 'spinner'): 1.2,
                    ('wicket_taker', 'economy_bowler'): 1.1
                }
            },
            'TEST': {
                'batting_synergy': {
                    ('patient_opener', 'solid_opener'): 1.4,
                    ('anchor', 'counter_attacker'): 1.2,
                    ('technique_specialist', 'temperament_player'): 1.3
                },
                'bowling_synergy': {
                    ('swing_bowler', 'seam_bowler'): 1.3,
                    ('pace_bowler', 'spinner'): 1.4,
                    ('aggressive_bowler', 'containing_bowler'): 1.2
                }
            }
        }
    
    def _initialize_tactical_preferences(self) -> Dict[str, Dict[str, Any]]:
        """Initialize tactical preferences for each format"""
        return {
            'T20': {
                'batting_order_flexibility': 0.8,  # High flexibility needed
                'bowling_rotation_complexity': 0.9, # Complex bowling rotations
                'field_position_changes': 0.9,     # Frequent field changes
                'risk_appetite': 0.9,              # High risk appetite
                'innovation_factor': 0.8           # Room for innovative tactics
            },
            'ODI': {
                'batting_order_flexibility': 0.6,  # Moderate flexibility
                'bowling_rotation_complexity': 0.7, # Balanced bowling rotations
                'field_position_changes': 0.6,     # Moderate field changes
                'risk_appetite': 0.6,              # Balanced risk appetite
                'partnership_emphasis': 0.9        # High emphasis on partnerships
            },
            'TEST': {
                'batting_order_flexibility': 0.3,  # Low flexibility
                'bowling_rotation_complexity': 0.8, # Complex long-form rotations
                'field_position_changes': 0.7,     # Strategic field changes
                'risk_appetite': 0.3,              # Conservative approach
                'patience_factor': 0.9,            # High patience requirement
                'session_planning': 0.9            # Session-wise planning crucial
            }
        }
    
    def validate_team_constraints(self, team_players: List, format_type: str, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate team against format-specific constraints"""
        format_balance = self.format_constraints.get(format_type, self.format_constraints['T20'])
        
        validation_results = {
            'is_valid': True,
            'constraint_violations': [],
            'constraint_scores': {},
            'overall_balance_score': 0,
            'recommendations': []
        }
        
        total_weighted_score = 0
        total_weight = 0
        
        # Validate each constraint
        for constraint in format_balance.constraints:
            score, violation = self._validate_single_constraint(
                team_players, constraint, format_type
            )
            
            validation_results['constraint_scores'][constraint.name] = score
            total_weighted_score += score * constraint.weight
            total_weight += constraint.weight
            
            if violation:
                validation_results['constraint_violations'].append(violation)
                validation_results['is_valid'] = False
        
        # Calculate overall balance score
        if total_weight > 0:
            validation_results['overall_balance_score'] = total_weighted_score / total_weight
        
        # Calculate synergy bonus
        synergy_bonus = self._calculate_team_synergy(team_players, format_type)
        validation_results['synergy_bonus'] = synergy_bonus
        validation_results['final_balance_score'] = validation_results['overall_balance_score'] * synergy_bonus
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_balance_recommendations(
            validation_results, format_type
        )
        
        return validation_results
    
    def _validate_single_constraint(self, team_players: List, constraint: FormatConstraint, format_type: str) -> Tuple[float, Optional[str]]:
        """Validate a single constraint against the team"""
        constraint_score = 0
        violation_message = None
        
        if constraint.name == "power_hitters":
            # Count players with high boundary percentage
            power_hitters = sum(1 for p in team_players 
                               if getattr(p, 'boundary_percentage', 0.15) > 0.20)
            constraint_score = min(100, (power_hitters / constraint.value) * 100)
            
            if power_hitters < constraint.value:
                violation_message = f"Need {constraint.value - power_hitters} more power hitters"
        
        elif constraint.name == "death_bowlers":
            death_bowlers = sum(1 for p in team_players 
                               if 'bowl' in getattr(p, 'role', '').lower() and 
                               getattr(p, 'death_overs_economy', 9.0) < 8.5)
            constraint_score = min(100, (death_bowlers / constraint.value) * 100)
            
            if death_bowlers < constraint.value:
                violation_message = f"Need {constraint.value - death_bowlers} more death bowlers"
        
        elif constraint.name == "anchors":
            anchors = sum(1 for p in team_players 
                         if 'bat' in getattr(p, 'role', '').lower() and 
                         getattr(p, 'batting_average', 30) > 40 and 
                         getattr(p, 'strike_rate', 80) > 80)
            constraint_score = min(100, (anchors / constraint.value) * 100)
            
            if anchors < constraint.value:
                violation_message = f"Need {constraint.value - anchors} more anchors"
        
        elif constraint.name == "openers_patience":
            patient_openers = sum(1 for p in team_players 
                                 if getattr(p, 'batting_position', 5) <= 2 and 
                                 getattr(p, 'patience_score', 50) > 70)
            constraint_score = min(100, (patient_openers / constraint.value) * 100)
            
            if patient_openers < constraint.value:
                violation_message = f"Need {constraint.value - patient_openers} more patient openers"
        
        # Add more constraint validations as needed
        else:
            # Default validation for unimplemented constraints
            constraint_score = 75  # Assume partial satisfaction
        
        return constraint_score, violation_message
    
    def _calculate_team_synergy(self, team_players: List, format_type: str) -> float:
        """Calculate team synergy bonus based on player combinations"""
        synergy_matrix = self.synergy_matrices.get(format_type, {})
        synergy_bonus = 1.0
        
        # Calculate batting synergy
        if format_type == 'T20' and 'powerplay_synergy' in synergy_matrix:
            # Check for powerplay combinations
            has_aggressive_opener = any(getattr(p, 'powerplay_strike_rate', 1.0) > 1.3 
                                       for p in team_players if getattr(p, 'batting_position', 5) <= 2)
            has_anchor_opener = any(getattr(p, 'consistency_score', 50) > 70 
                                   for p in team_players if getattr(p, 'batting_position', 5) <= 2)
            
            if has_aggressive_opener and has_anchor_opener:
                synergy_bonus *= 1.2
        
        elif format_type == 'ODI' and 'partnership_synergy' in synergy_matrix:
            # Check for partnership combinations
            anchors = sum(1 for p in team_players if getattr(p, 'batting_average', 30) > 40)
            aggressors = sum(1 for p in team_players if getattr(p, 'strike_rate', 80) > 100)
            
            if anchors >= 2 and aggressors >= 2:
                synergy_bonus *= 1.3
        
        elif format_type == 'TEST' and 'batting_synergy' in synergy_matrix:
            # Check for TEST batting combinations
            patient_players = sum(1 for p in team_players if getattr(p, 'patience_score', 50) > 70)
            technique_players = sum(1 for p in team_players if getattr(p, 'technique_rating', 50) > 65)
            
            if patient_players >= 3 and technique_players >= 4:
                synergy_bonus *= 1.4
        
        # Calculate bowling synergy
        pace_bowlers = sum(1 for p in team_players if 'pace' in getattr(p, 'bowling_style', '').lower())
        spinners = sum(1 for p in team_players if 'spin' in getattr(p, 'bowling_style', '').lower())
        
        if pace_bowlers >= 2 and spinners >= 1:
            synergy_bonus *= 1.1
        
        return min(synergy_bonus, 1.5)  # Cap synergy bonus at 1.5x
    
    def _generate_balance_recommendations(self, validation_results: Dict[str, Any], format_type: str) -> List[str]:
        """Generate recommendations for improving team balance"""
        recommendations = []
        
        # Recommendations based on constraint violations
        for violation in validation_results['constraint_violations']:
            if "power hitters" in violation:
                recommendations.append("Consider players with boundary% > 20% for T20 explosive potential")
            elif "death bowlers" in violation:
                recommendations.append("Add bowlers with death overs economy < 8.5 for closing games")
            elif "anchors" in violation:
                recommendations.append("Include batsmen with average > 40 for ODI stability")
            elif "patient openers" in violation:
                recommendations.append("Select openers with patience score > 70 for TEST foundation")
        
        # Recommendations based on balance score
        balance_score = validation_results['overall_balance_score']
        if balance_score < 60:
            recommendations.append(f"Team balance needs significant improvement for {format_type}")
        elif balance_score < 80:
            recommendations.append(f"Consider minor adjustments for better {format_type} balance")
        
        # Synergy recommendations
        synergy_bonus = validation_results.get('synergy_bonus', 1.0)
        if synergy_bonus < 1.1:
            recommendations.append(f"Look for better player combinations to improve {format_type} synergy")
        
        return recommendations
    
    def get_optimal_team_structure(self, format_type: str) -> Dict[str, Any]:
        """Get optimal team structure for a format"""
        format_balance = self.format_constraints.get(format_type, self.format_constraints['T20'])
        tactical_prefs = self.tactical_preferences.get(format_type, {})
        
        return {
            'format_type': format_type,
            'role_preferences': format_balance.role_preferences,
            'skill_requirements': format_balance.skill_requirements,
            'key_constraints': [c.name for c in format_balance.constraints if c.weight >= 0.8],
            'tactical_preferences': tactical_prefs,
            'synergy_opportunities': list(self.synergy_matrices.get(format_type, {}).keys()),
            'balance_philosophy': self._get_format_philosophy(format_type)
        }
    
    def _get_format_philosophy(self, format_type: str) -> str:
        """Get the philosophical approach for each format"""
        philosophies = {
            'T20': "Maximize explosive potential while maintaining death overs control. Prioritize boundary-hitting ability and fielding agility. Accept higher risk for higher reward.",
            'ODI': "Balance consistency with scoring ability. Build partnerships while maintaining run rate. Emphasize middle overs control and pressure handling.",
            'TEST': "Prioritize patience, technique, and temperament. Build for long-term battle with emphasis on session-wise planning and adaptation to conditions."
        }
        
        return philosophies.get(format_type, "Balanced approach focusing on consistency and performance.")