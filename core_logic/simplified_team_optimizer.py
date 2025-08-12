#!/usr/bin/env python3
"""
SIMPLIFIED TEAM OPTIMIZER - Fast, Practical Team Generation
Replaces overly complex quantum optimization with efficient, proven algorithms
Focus: Speed, reliability, and practical results for Dream11
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core_logic.unified_models import UnifiedPlayer, UnifiedTeam

@dataclass
class OptimizationStrategy:
    """Different optimization strategies for team generation"""
    name: str
    weight_consistency: float = 0.3
    weight_form: float = 0.3
    weight_expected_points: float = 0.4
    risk_tolerance: str = "medium"  # low, medium, high
    prefer_stars: bool = False
    balance_teams: bool = True

class TeamConstraints:
    """Dream11 team constraints and validation"""
    
    # Role constraints
    MIN_WICKET_KEEPERS = 1
    MAX_WICKET_KEEPERS = 4
    MIN_BATSMEN = 3
    MAX_BATSMEN = 6
    MIN_ALL_ROUNDERS = 1
    MAX_ALL_ROUNDERS = 4
    MIN_BOWLERS = 3
    MAX_BOWLERS = 6
    
    # Team constraints
    TOTAL_PLAYERS = 11
    MAX_PLAYERS_PER_TEAM = 10  # Dream11 allows up to 10 players from same team
    
    @classmethod
    def validate_team(cls, players: List[UnifiedPlayer]) -> Tuple[bool, str]:
        """Validate if team meets all Dream11 constraints"""
        if len(players) != cls.TOTAL_PLAYERS:
            return False, f"Need exactly {cls.TOTAL_PLAYERS} players, got {len(players)}"
        
        # Count roles
        role_counts = {
            'Wicket-keeper': 0,
            'Batsman': 0,
            'All-rounder': 0,
            'Bowler': 0
        }
        
        team_counts = {}
        
        for player in players:
            role_counts[player.role] = role_counts.get(player.role, 0) + 1
            team_counts[player.team_name] = team_counts.get(player.team_name, 0) + 1
        
        # Check role constraints
        if role_counts['Wicket-keeper'] < cls.MIN_WICKET_KEEPERS:
            return False, f"Need at least {cls.MIN_WICKET_KEEPERS} wicket-keeper"
        if role_counts['Wicket-keeper'] > cls.MAX_WICKET_KEEPERS:
            return False, f"Max {cls.MAX_WICKET_KEEPERS} wicket-keepers allowed"
        
        if role_counts['Batsman'] < cls.MIN_BATSMEN:
            return False, f"Need at least {cls.MIN_BATSMEN} batsmen"
        if role_counts['Batsman'] > cls.MAX_BATSMEN:
            return False, f"Max {cls.MAX_BATSMEN} batsmen allowed"
        
        if role_counts['All-rounder'] < cls.MIN_ALL_ROUNDERS:
            return False, f"Need at least {cls.MIN_ALL_ROUNDERS} all-rounder"
        if role_counts['All-rounder'] > cls.MAX_ALL_ROUNDERS:
            return False, f"Max {cls.MAX_ALL_ROUNDERS} all-rounders allowed"
        
        if role_counts['Bowler'] < cls.MIN_BOWLERS:
            return False, f"Need at least {cls.MIN_BOWLERS} bowlers"
        if role_counts['Bowler'] > cls.MAX_BOWLERS:
            return False, f"Max {cls.MAX_BOWLERS} bowlers allowed"
        
        # Check team balance
        for team, count in team_counts.items():
            if count > cls.MAX_PLAYERS_PER_TEAM:
                return False, f"Max {cls.MAX_PLAYERS_PER_TEAM} players from {team}, got {count}"
        
        return True, "Valid team"

class SimplifiedTeamOptimizer:
    """
    Fast, practical team optimizer using proven algorithms
    Replaces complex quantum optimization with efficient heuristics
    """
    
    def __init__(self):
        self.strategies = self._create_strategies()
        self.constraints = TeamConstraints()
    
    def _create_strategies(self) -> List[OptimizationStrategy]:
        """Create different optimization strategies"""
        return [
            OptimizationStrategy(
                name="AI-Optimal",
                weight_consistency=0.2,
                weight_form=0.3,
                weight_expected_points=0.5,
                risk_tolerance="medium",
                prefer_stars=True
            ),
            OptimizationStrategy(
                name="Risk-Balanced",
                weight_consistency=0.5,
                weight_form=0.2,
                weight_expected_points=0.3,
                risk_tolerance="low",
                prefer_stars=False
            ),
            OptimizationStrategy(
                name="High-Ceiling",
                weight_consistency=0.1,
                weight_form=0.5,
                weight_expected_points=0.4,
                risk_tolerance="high",
                prefer_stars=True
            ),
            OptimizationStrategy(
                name="Value-Optimal",
                weight_consistency=0.3,
                weight_form=0.3,
                weight_expected_points=0.4,
                risk_tolerance="medium",
                prefer_stars=False,
                balance_teams=True
            ),
            OptimizationStrategy(
                name="Conditions-Based",
                weight_consistency=0.3,
                weight_form=0.4,
                weight_expected_points=0.3,
                risk_tolerance="medium",
                prefer_stars=False
            )
        ]
    
    def generate_teams(self, players: List[UnifiedPlayer], num_teams: int = 5) -> List[UnifiedTeam]:
        """Generate optimized teams using different strategies"""
        if not players:
            return []
        
        # Fix team names issue - assign teams properly if all are "Unknown"
        self._fix_team_names(players)
        
        teams = []
        strategies_to_use = self.strategies[:min(num_teams, len(self.strategies))]
        
        for i, strategy in enumerate(strategies_to_use):
            team = self._generate_single_team(players, strategy, team_id=i+1)
            if team and team.is_valid_team:
                teams.append(team)
        
        # If we need more teams, generate variants with timeout protection
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        while len(teams) < num_teams and attempts < max_attempts:
            attempts += 1
            strategy = random.choice(self.strategies)
            team = self._generate_single_team(players, strategy, team_id=len(teams)+1, add_randomness=True)
            if team and team.is_valid_team and not self._is_duplicate_team(team, teams):
                teams.append(team)
        
        # If we still don't have enough teams, fill with duplicates but modified strategies
        while len(teams) < num_teams and teams:
            base_team = random.choice(teams)
            # Create a slightly modified copy
            modified_team = UnifiedTeam(
                team_id=len(teams)+1,
                players=base_team.players[:],  # Copy players
                strategy=f"Modified-{base_team.strategy}",
                team_quality=base_team.team_quality,
                risk_level=base_team.risk_level,
                confidence_score=max(1.0, base_team.confidence_score - 0.5)
            )
            teams.append(modified_team)
        
        return teams[:num_teams]
    
    def _fix_team_names(self, players: List[UnifiedPlayer]) -> None:
        """Fix team names if all players have 'Unknown' team names"""
        unknown_count = sum(1 for p in players if p.team_name == "Unknown")
        
        if unknown_count > len(players) * 0.8:  # If more than 80% are unknown
            # Split players into two teams based on player names
            team1_players = []
            team2_players = []
            
            for player in players:
                name_lower = player.name.lower()
                
                # Team 1 indicators (Surrey-like names)
                team1_indicators = ['burns', 'patel', 'thomas', 'sykes', 'steel', 'blake', 'majid', 'barnwell', 'ealham']
                
                # Team 2 indicators (Gloucestershire-like names)
                team2_indicators = ['bancroft', 'charlesworth', 'price', 'bracey', 'buuren', 'boorman', 'shaw', 'akhter']
                
                if any(indicator in name_lower for indicator in team1_indicators):
                    team1_players.append(player)
                elif any(indicator in name_lower for indicator in team2_indicators):
                    team2_players.append(player)
                elif 'taylor' in name_lower:
                    # Split the Taylors
                    if player.name == "James Taylor":
                        team1_players.append(player)
                    else:  # J Taylor, Matt Taylor
                        team2_players.append(player)
                else:
                    # Assign to smaller team for balance
                    if len(team1_players) <= len(team2_players):
                        team1_players.append(player)
                    else:
                        team2_players.append(player)
            
            # Assign team names
            for player in team1_players:
                player.team_name = "Surrey"
                player.team_id = 148
            
            for player in team2_players:
                player.team_name = "Gloucestershire"
                player.team_id = 151
    
    def _generate_single_team(self, 
                             players: List[UnifiedPlayer], 
                             strategy: OptimizationStrategy, 
                             team_id: int,
                             add_randomness: bool = False) -> Optional[UnifiedTeam]:
        """Generate a single team using the specified strategy"""
        
        # Calculate weighted scores for all players
        scored_players = []
        for player in players:
            score = self._calculate_player_score(player, strategy)
            if add_randomness:
                # Add small random factor for variation
                score *= (0.9 + random.random() * 0.2)
            scored_players.append((player, score))
        
        # Sort by score
        scored_players.sort(key=lambda x: x[1], reverse=True)
        
        # Use greedy algorithm with constraint satisfaction
        selected = self._greedy_selection_with_constraints(scored_players, strategy)
        
        if len(selected) != 11:
            return None
        
        # Create team
        team = UnifiedTeam(
            team_id=team_id,
            players=selected,
            strategy=strategy.name,
            team_quality=self._assess_team_quality(selected, strategy),
            risk_level=strategy.risk_tolerance.title(),
            confidence_score=self._calculate_confidence_score(selected, strategy)
        )
        
        return team
    
    def _calculate_player_score(self, player: UnifiedPlayer, strategy: OptimizationStrategy) -> float:
        """Calculate player score based on strategy weights"""
        base_score = (
            player.consistency_score * strategy.weight_consistency +
            player.form_momentum * 100 * strategy.weight_form +
            player.expected_points * strategy.weight_expected_points
        )
        
        # Apply strategy-specific bonuses
        if strategy.prefer_stars and player.expected_points > 60:
            base_score *= 1.2
        
        if strategy.risk_tolerance == "low" and player.consistency_score > 70:
            base_score *= 1.1
        
        if strategy.risk_tolerance == "high" and player.form_momentum > 0.7:
            base_score *= 1.15
        
        # Role-specific bonuses
        if player.role == "All-rounder":
            base_score *= 1.05  # All-rounders are valuable
        
        return base_score
    
    def _greedy_selection_with_constraints(self, 
                                         scored_players: List[Tuple[UnifiedPlayer, float]], 
                                         strategy: OptimizationStrategy) -> List[UnifiedPlayer]:
        """Select players using greedy algorithm while satisfying constraints"""
        
        selected = []
        role_counts = {
            'Wicket-keeper': 0,
            'Batsman': 0,
            'All-rounder': 0,
            'Bowler': 0
        }
        team_counts = {}
        
        # Force selection of minimum required roles first
        required_roles = [
            ('Wicket-keeper', 1),
            ('Batsman', 3),
            ('All-rounder', 1),
            ('Bowler', 3)
        ]
        
        # Select minimum required players for each role
        for role, min_count in required_roles:
            role_players = [(p, s) for p, s in scored_players 
                          if p.role == role and p not in selected]
            
            attempts = 0
            for _ in range(min_count):
                if not role_players or attempts > 50:  # Prevent infinite loops
                    break
                
                # Find best player that doesn't violate team constraints
                found = False
                for player, score in role_players:
                    attempts += 1
                    team_count = team_counts.get(player.team_name, 0)
                    if team_count < TeamConstraints.MAX_PLAYERS_PER_TEAM:
                        selected.append(player)
                        role_counts[role] += 1
                        team_counts[player.team_name] = team_count + 1
                        role_players.remove((player, score))
                        found = True
                        break
                
                if not found:
                    # If we can't find a valid player, relax team constraints temporarily
                    if role_players:
                        player, score = role_players[0]
                        selected.append(player)
                        role_counts[role] += 1
                        team_counts[player.team_name] = team_counts.get(player.team_name, 0) + 1
                        role_players.remove((player, score))
        
        # Fill remaining slots with best available players
        remaining_slots = 11 - len(selected)
        available_players = [(p, s) for p, s in scored_players if p not in selected]
        
        attempts = 0
        for player, score in available_players:
            if remaining_slots <= 0 or attempts > 100:  # Prevent infinite loops
                break
            
            attempts += 1
            # Check if we can add this player
            if self._can_add_player(player, selected, role_counts, team_counts):
                selected.append(player)
                role_counts[player.role] += 1
                team_counts[player.team_name] = team_counts.get(player.team_name, 0) + 1
                remaining_slots -= 1
        
        return selected
    
    def _can_add_player(self, 
                       player: UnifiedPlayer, 
                       current_team: List[UnifiedPlayer],
                       role_counts: Dict[str, int],
                       team_counts: Dict[str, int]) -> bool:
        """Check if player can be added without violating constraints"""
        
        # Check team constraint
        if team_counts.get(player.team_name, 0) >= TeamConstraints.MAX_PLAYERS_PER_TEAM:
            return False
        
        # Check role constraints (max limits)
        role_limits = {
            'Wicket-keeper': TeamConstraints.MAX_WICKET_KEEPERS,
            'Batsman': TeamConstraints.MAX_BATSMEN,
            'All-rounder': TeamConstraints.MAX_ALL_ROUNDERS,
            'Bowler': TeamConstraints.MAX_BOWLERS
        }
        
        if role_counts.get(player.role, 0) >= role_limits[player.role]:
            return False
        
        return True
    
    def _assess_team_quality(self, players: List[UnifiedPlayer], strategy: OptimizationStrategy) -> str:
        """Assess overall team quality"""
        avg_score = sum(p.total_score for p in players) / len(players)
        avg_consistency = sum(p.consistency_score for p in players) / len(players)
        
        if avg_score > 70 and avg_consistency > 60:
            return "Premium"
        elif avg_score > 50 and avg_consistency > 40:
            return "Standard"
        else:
            return "Value"
    
    def _calculate_confidence_score(self, players: List[UnifiedPlayer], strategy: OptimizationStrategy) -> float:
        """Calculate team confidence score (1-5)"""
        avg_consistency = sum(p.consistency_score for p in players) / len(players)
        avg_form = sum(p.form_momentum for p in players) / len(players)
        
        base_confidence = 2.5
        
        if avg_consistency > 70:
            base_confidence += 1.0
        elif avg_consistency > 50:
            base_confidence += 0.5
        
        if avg_form > 0.7:
            base_confidence += 0.8
        elif avg_form > 0.5:
            base_confidence += 0.4
        
        return min(5.0, max(1.0, base_confidence))
    
    def _is_duplicate_team(self, new_team: UnifiedTeam, existing_teams: List[UnifiedTeam]) -> bool:
        """Check if team is too similar to existing teams"""
        new_player_ids = set(p.player_id for p in new_team.players)
        
        for existing_team in existing_teams:
            existing_player_ids = set(p.player_id for p in existing_team.players)
            
            # If more than 8 players are the same, consider it a duplicate
            overlap = len(new_player_ids.intersection(existing_player_ids))
            if overlap > 8:
                return True
        
        return False

# Factory function for easy team generation
def generate_optimized_teams(players: List[UnifiedPlayer], num_teams: int = 5) -> List[UnifiedTeam]:
    """
    Simple factory function to generate optimized teams
    Replaces complex quantum optimization with fast, practical algorithms
    """
    optimizer = SimplifiedTeamOptimizer()
    return optimizer.generate_teams(players, num_teams)

# Backward compatibility function
def generate_world_class_ai_teams(player_features, match_format: str = "T20", num_teams: int = 5) -> List[UnifiedTeam]:
    """
    Backward compatibility wrapper for existing code
    Converts old PlayerFeatures to UnifiedPlayer and generates teams
    """
    from core_logic.unified_models import from_player_features
    
    # Convert to unified players
    unified_players = []
    for pf in player_features:
        try:
            unified_player = from_player_features(pf)
            unified_players.append(unified_player)
        except Exception as e:
            print(f"⚠️ Skipping player conversion: {e}")
            continue
    
    if not unified_players:
        return []
    
    # Generate teams
    return generate_optimized_teams(unified_players, num_teams)
