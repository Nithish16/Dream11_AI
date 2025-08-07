#!/usr/bin/env python3
"""
Advanced Team Optimizer - Multi-Objective Team Selection
Implements cutting-edge optimization techniques from 2024-2025 research
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import time
from collections import defaultdict
import heapq

# Import base team generation
from .team_generator import generate_world_class_ai_teams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConstraints:
    """Advanced constraints for team optimization"""
    budget: float = 100.0
    min_batsmen: int = 3
    max_batsmen: int = 6
    min_bowlers: int = 3
    max_bowlers: int = 6
    min_all_rounders: int = 1
    max_all_rounders: int = 4
    min_wicket_keepers: int = 1
    max_wicket_keepers: int = 2
    captain_multiplier: float = 2.0
    vice_captain_multiplier: float = 1.5
    max_players_per_team: int = 7
    min_players_per_team: int = 4
    
    # Advanced constraints
    max_ownership_overlap: float = 80.0  # Percentage
    min_ceiling_players: int = 2  # High upside players
    min_floor_players: int = 6   # Consistent performers
    balance_score_weight: float = 0.3
    uniqueness_weight: float = 0.2

@dataclass
class TeamMetrics:
    """Comprehensive team evaluation metrics"""
    expected_points: float = 0.0
    ceiling_score: float = 0.0  # Best case scenario
    floor_score: float = 0.0    # Worst case scenario
    variance: float = 0.0
    ownership_overlap: float = 0.0
    uniqueness_score: float = 0.0
    balance_score: float = 0.0
    captain_suitability: float = 0.0
    risk_score: float = 0.0
    consistency_score: float = 0.0
    
    # Pareto efficiency metrics
    pareto_rank: int = 0
    dominance_count: int = 0

@dataclass
class OptimizedTeam:
    """Enhanced team representation with detailed analytics"""
    players: List[Dict[str, Any]]
    captain: Dict[str, Any]
    vice_captain: Dict[str, Any]
    total_cost: float
    metrics: TeamMetrics
    generation_method: str = "advanced_optimizer"
    optimization_time_ms: float = 0.0
    constraints_satisfied: bool = True
    pareto_optimal: bool = False

class TeamObjective(ABC):
    """Abstract base class for optimization objectives"""
    
    @abstractmethod
    def evaluate(self, team: List[Dict[str, Any]], captain: Dict[str, Any], 
                 vice_captain: Dict[str, Any]) -> float:
        """Evaluate team against this objective"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get objective name"""
        pass

class ExpectedPointsObjective(TeamObjective):
    """Maximize expected fantasy points"""
    
    def evaluate(self, team: List[Dict[str, Any]], captain: Dict[str, Any], 
                 vice_captain: Dict[str, Any]) -> float:
        total_points = 0.0
        
        for player in team:
            points = player.get('expected_points', 50.0)
            
            if player == captain:
                points *= 2.0  # Captain bonus
            elif player == vice_captain:
                points *= 1.5  # Vice-captain bonus
            
            total_points += points
        
        return total_points
    
    def get_name(self) -> str:
        return "Expected Points"

class VarianceMinimizationObjective(TeamObjective):
    """Minimize team variance (risk)"""
    
    def evaluate(self, team: List[Dict[str, Any]], captain: Dict[str, Any], 
                 vice_captain: Dict[str, Any]) -> float:
        variances = []
        
        for player in team:
            uncertainty = player.get('uncertainty', 0.3)
            expected_points = player.get('expected_points', 50.0)
            variance = (uncertainty * expected_points) ** 2
            
            if player == captain:
                variance *= 4.0  # Captain variance multiplied
            elif player == vice_captain:
                variance *= 2.25  # Vice-captain variance multiplied
            
            variances.append(variance)
        
        total_variance = sum(variances)
        return -total_variance  # Negative because we want to minimize

class UniquenessObjective(TeamObjective):
    """Maximize team uniqueness (differentiation)"""
    
    def __init__(self, existing_teams: List[List[Dict[str, Any]]] = None):
        self.existing_teams = existing_teams or []
    
    def evaluate(self, team: List[Dict[str, Any]], captain: Dict[str, Any], 
                 vice_captain: Dict[str, Any]) -> float:
        if not self.existing_teams:
            return 100.0  # Maximum uniqueness if no existing teams
        
        team_player_ids = {p.get('player_id', 0) for p in team}
        
        min_overlap = float('inf')
        for existing_team in self.existing_teams:
            existing_ids = {p.get('player_id', 0) for p in existing_team}
            overlap = len(team_player_ids.intersection(existing_ids))
            min_overlap = min(min_overlap, overlap)
        
        # Convert to uniqueness score (0-100)
        uniqueness = max(0, 100 - (min_overlap * 10))
        return uniqueness

class BalanceObjective(TeamObjective):
    """Maximize team balance across roles"""
    
    def evaluate(self, team: List[Dict[str, Any]], captain: Dict[str, Any], 
                 vice_captain: Dict[str, Any]) -> float:
        role_counts = defaultdict(int)
        role_quality = defaultdict(list)
        
        for player in team:
            role = self._normalize_role(player.get('role', 'Unknown'))
            expected_points = player.get('expected_points', 50.0)
            
            role_counts[role] += 1
            role_quality[role].append(expected_points)
        
        # Calculate balance score
        balance_score = 0.0
        
        # Role distribution balance
        ideal_distribution = {'batsman': 4, 'bowler': 4, 'allrounder': 2, 'wicketkeeper': 1}
        for role, ideal_count in ideal_distribution.items():
            actual_count = role_counts.get(role, 0)
            deviation = abs(actual_count - ideal_count)
            balance_score += max(0, 20 - deviation * 5)  # Penalty for deviation
        
        # Role quality balance
        for role, qualities in role_quality.items():
            if qualities:
                avg_quality = sum(qualities) / len(qualities)
                balance_score += min(avg_quality, 20)  # Cap at 20 points
        
        return balance_score
    
    def _normalize_role(self, role: str) -> str:
        """Normalize role names"""
        role_lower = role.lower()
        if 'wicket' in role_lower or 'wk' in role_lower:
            return 'wicketkeeper'
        elif 'allrounder' in role_lower or 'all-rounder' in role_lower:
            return 'allrounder'
        elif 'bowl' in role_lower:
            return 'bowler'
        elif 'bat' in role_lower:
            return 'batsman'
        else:
            return 'batsman'

class ParetoOptimizer:
    """
    Multi-objective optimization using Pareto efficiency
    Based on NSGA-III algorithm principles
    """
    
    def __init__(self, objectives: List[TeamObjective]):
        self.objectives = objectives
        self.pareto_fronts = []
    
    def optimize(self, population: List[OptimizedTeam]) -> List[OptimizedTeam]:
        """Find Pareto optimal teams"""
        logger.info(f"🎯 Finding Pareto optimal teams from {len(population)} candidates...")
        
        # Evaluate all teams on all objectives
        for team in population:
            team.objective_scores = []
            for objective in self.objectives:
                score = objective.evaluate(team.players, team.captain, team.vice_captain)
                team.objective_scores.append(score)
        
        # Non-dominated sorting
        fronts = self._non_dominated_sort(population)
        
        # Update Pareto ranks
        for rank, front in enumerate(fronts):
            for team in front:
                team.metrics.pareto_rank = rank
                team.pareto_optimal = (rank == 0)
        
        self.pareto_fronts = fronts
        logger.info(f"✅ Found {len(fronts[0]) if fronts else 0} Pareto optimal teams")
        
        return fronts[0] if fronts else []
    
    def _non_dominated_sort(self, population: List[OptimizedTeam]) -> List[List[OptimizedTeam]]:
        """Non-dominated sorting algorithm"""
        fronts = []
        
        # Calculate dominance relationships
        for i, team_i in enumerate(population):
            team_i.dominated_by = []
            team_i.dominates_count = 0
            
            for j, team_j in enumerate(population):
                if i != j:
                    if self._dominates(team_i, team_j):
                        team_i.dominated_by.append(j)
                    elif self._dominates(team_j, team_i):
                        team_i.dominates_count += 1
        
        # Build fronts
        current_front = []
        for i, team in enumerate(population):
            if team.dominates_count == 0:
                current_front.append(team)
        
        fronts.append(current_front)
        
        # Build subsequent fronts
        while current_front:
            next_front = []
            for team_i in current_front:
                for j in team_i.dominated_by:
                    population[j].dominates_count -= 1
                    if population[j].dominates_count == 0:
                        next_front.append(population[j])
            
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts
    
    def _dominates(self, team_a: OptimizedTeam, team_b: OptimizedTeam) -> bool:
        """Check if team_a dominates team_b"""
        better_in_at_least_one = False
        
        for i in range(len(self.objectives)):
            if team_a.objective_scores[i] < team_b.objective_scores[i]:
                return False  # team_a is worse in this objective
            elif team_a.objective_scores[i] > team_b.objective_scores[i]:
                better_in_at_least_one = True
        
        return better_in_at_least_one

class ReinforcementLearningOptimizer:
    """
    RL-based team selection inspired by Deep Q-Networks
    Simplified implementation for sequential player selection
    """
    
    def __init__(self, state_size: int = 50, action_size: int = 100):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.01
        
        # Simple Q-table (would be neural network in full implementation)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 1000
    
    def select_team_rl(self, available_players: List[Dict[str, Any]], 
                       constraints: OptimizationConstraints) -> OptimizedTeam:
        """Select team using RL-inspired approach"""
        logger.info("🤖 Selecting team using RL approach...")
        start_time = time.time()
        
        selected_players = []
        budget_remaining = constraints.budget
        state = self._encode_state(selected_players, available_players, budget_remaining)
        
        # Sequential player selection
        for position in range(11):
            # Get valid actions (players that can be selected)
            valid_actions = self._get_valid_actions(
                available_players, selected_players, budget_remaining, constraints
            )
            
            if not valid_actions:
                break
            
            # Select action (player) using epsilon-greedy
            if random.random() < self.epsilon:
                # Exploration: random selection
                selected_player = random.choice(valid_actions)
            else:
                # Exploitation: select best Q-value
                best_player = None
                best_q_value = float('-inf')
                
                for player in valid_actions:
                    action_key = self._encode_player_action(player)
                    q_value = self.q_table[state][action_key]
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_player = player
                
                selected_player = best_player or random.choice(valid_actions)
            
            # Add player to team
            selected_players.append(selected_player)
            budget_remaining -= selected_player.get('cost', 9.0)
            
            # Update state
            state = self._encode_state(selected_players, available_players, budget_remaining)
        
        # Select captain and vice-captain
        captain, vice_captain = self._select_captains_rl(selected_players)
        
        # Calculate cost and metrics
        total_cost = sum(p.get('cost', 9.0) for p in selected_players)
        metrics = self._calculate_team_metrics(selected_players, captain, vice_captain)
        
        optimization_time = (time.time() - start_time) * 1000
        
        return OptimizedTeam(
            players=selected_players,
            captain=captain,
            vice_captain=vice_captain,
            total_cost=total_cost,
            metrics=metrics,
            generation_method="reinforcement_learning",
            optimization_time_ms=optimization_time,
            constraints_satisfied=self._validate_constraints(selected_players, constraints)
        )
    
    def _encode_state(self, selected_players: List[Dict], available_players: List[Dict], 
                     budget: float) -> str:
        """Encode current state for Q-table lookup"""
        # Simplified state encoding
        num_selected = len(selected_players)
        budget_tier = int(budget / 10)  # Budget in tiers of 10
        
        role_counts = defaultdict(int)
        for player in selected_players:
            role = player.get('role', 'Unknown').lower()
            if 'bat' in role:
                role_counts['bat'] += 1
            elif 'bowl' in role:
                role_counts['bowl'] += 1
            elif 'allrounder' in role:
                role_counts['all'] += 1
            elif 'wicket' in role:
                role_counts['wk'] += 1
        
        return f"{num_selected}_{budget_tier}_{role_counts['bat']}_{role_counts['bowl']}_{role_counts['all']}_{role_counts['wk']}"
    
    def _encode_player_action(self, player: Dict[str, Any]) -> str:
        """Encode player selection as action"""
        return f"player_{player.get('player_id', 0)}"
    
    def _get_valid_actions(self, available_players: List[Dict], selected_players: List[Dict], 
                          budget: float, constraints: OptimizationConstraints) -> List[Dict]:
        """Get valid player selections"""
        selected_ids = {p.get('player_id', 0) for p in selected_players}
        
        valid_players = []
        for player in available_players:
            # Check if already selected
            if player.get('player_id', 0) in selected_ids:
                continue
            
            # Check budget constraint
            if player.get('cost', 9.0) > budget:
                continue
            
            # Check team composition constraints
            if self._violates_constraints(selected_players + [player], constraints):
                continue
            
            valid_players.append(player)
        
        return valid_players
    
    def _violates_constraints(self, team: List[Dict], constraints: OptimizationConstraints) -> bool:
        """Check if team violates constraints"""
        role_counts = defaultdict(int)
        team_counts = defaultdict(int)
        
        for player in team:
            role = player.get('role', 'Unknown').lower()
            team_id = player.get('team_id', 0)
            
            if 'bat' in role:
                role_counts['batsman'] += 1
            elif 'bowl' in role:
                role_counts['bowler'] += 1
            elif 'allrounder' in role:
                role_counts['allrounder'] += 1
            elif 'wicket' in role:
                role_counts['wicketkeeper'] += 1
            
            team_counts[team_id] += 1
        
        # Check role constraints
        if role_counts['batsman'] > constraints.max_batsmen:
            return True
        if role_counts['bowler'] > constraints.max_bowlers:
            return True
        if role_counts['allrounder'] > constraints.max_all_rounders:
            return True
        if role_counts['wicketkeeper'] > constraints.max_wicket_keepers:
            return True
        
        # Check team distribution constraints
        for team_id, count in team_counts.items():
            if count > constraints.max_players_per_team:
                return True
        
        return False
    
    def _select_captains_rl(self, team: List[Dict[str, Any]]) -> Tuple[Dict, Dict]:
        """Select captain and vice-captain using RL principles"""
        # Sort by expected points and captaincy suitability
        captain_scores = []
        for player in team:
            expected_points = player.get('expected_points', 50.0)
            captain_prob = player.get('captain_vice_captain_probability', 0.5)
            score = expected_points * (1 + captain_prob)
            captain_scores.append((score, player))
        
        captain_scores.sort(reverse=True)
        
        captain = captain_scores[0][1]
        vice_captain = captain_scores[1][1] if len(captain_scores) > 1 else captain
        
        return captain, vice_captain
    
    def _validate_constraints(self, team: List[Dict], constraints: OptimizationConstraints) -> bool:
        """Validate that team satisfies all constraints"""
        if len(team) != 11:
            return False
        
        total_cost = sum(p.get('cost', 9.0) for p in team)
        if total_cost > constraints.budget:
            return False
        
        return not self._violates_constraints(team, constraints)
    
    def _calculate_team_metrics(self, team: List[Dict], captain: Dict, 
                              vice_captain: Dict) -> TeamMetrics:
        """Calculate comprehensive team metrics"""
        expected_points = 0.0
        ceiling_points = 0.0
        floor_points = 0.0
        variances = []
        
        for player in team:
            base_points = player.get('expected_points', 50.0)
            uncertainty = player.get('uncertainty', 0.3)
            
            # Apply captain/vice-captain multipliers
            if player == captain:
                points = base_points * 2.0
                ceiling = (base_points + uncertainty * 20) * 2.0
                floor = max(0, base_points - uncertainty * 20) * 2.0
            elif player == vice_captain:
                points = base_points * 1.5
                ceiling = (base_points + uncertainty * 20) * 1.5
                floor = max(0, base_points - uncertainty * 20) * 1.5
            else:
                points = base_points
                ceiling = base_points + uncertainty * 20
                floor = max(0, base_points - uncertainty * 20)
            
            expected_points += points
            ceiling_points += ceiling
            floor_points += floor
            variances.append((uncertainty * base_points) ** 2)
        
        total_variance = sum(variances)
        
        return TeamMetrics(
            expected_points=expected_points,
            ceiling_score=ceiling_points,
            floor_score=floor_points,
            variance=total_variance,
            risk_score=total_variance / max(expected_points, 1),
            consistency_score=max(0, 100 - total_variance),
            balance_score=self._calculate_balance_score(team)
        )
    
    def _calculate_balance_score(self, team: List[Dict]) -> float:
        """Calculate team balance score"""
        role_counts = defaultdict(int)
        for player in team:
            role = player.get('role', 'Unknown').lower()
            if 'bat' in role:
                role_counts['batsman'] += 1
            elif 'bowl' in role:
                role_counts['bowler'] += 1
            elif 'allrounder' in role:
                role_counts['allrounder'] += 1
            elif 'wicket' in role:
                role_counts['wicketkeeper'] += 1
        
        # Ideal distribution
        ideal = {'batsman': 4, 'bowler': 4, 'allrounder': 2, 'wicketkeeper': 1}
        
        balance_score = 0.0
        for role, ideal_count in ideal.items():
            actual_count = role_counts.get(role, 0)
            deviation = abs(actual_count - ideal_count)
            balance_score += max(0, 25 - deviation * 5)
        
        return balance_score

class AdvancedTeamOptimizer:
    """
    Main advanced team optimizer with multiple optimization strategies
    """
    
    def __init__(self, optimization_mode: str = "multi_objective"):
        self.optimization_mode = optimization_mode
        self.constraints = OptimizationConstraints()
        self.existing_teams = []
        
        # Initialize optimizers
        self.objectives = [
            ExpectedPointsObjective(),
            VarianceMinimizationObjective(),
            UniquenessObjective(self.existing_teams),
            BalanceObjective()
        ]
        
        self.pareto_optimizer = ParetoOptimizer(self.objectives)
        self.rl_optimizer = ReinforcementLearningOptimizer()
        
        logger.info(f"🚀 Advanced Team Optimizer initialized (Mode: {optimization_mode})")
    
    def optimize_teams(self, player_features: List[Dict[str, Any]], 
                      match_format: str, match_context: Dict[str, Any], 
                      num_teams: int = 5) -> List[OptimizedTeam]:
        """
        Generate optimized teams using advanced algorithms
        """
        logger.info(f"🎯 Optimizing {num_teams} teams using {self.optimization_mode} approach...")
        start_time = time.time()
        
        optimized_teams = []
        
        try:
            if self.optimization_mode == "multi_objective":
                optimized_teams = self._optimize_multi_objective(
                    player_features, match_format, match_context, num_teams
                )
            elif self.optimization_mode == "reinforcement_learning":
                optimized_teams = self._optimize_rl(
                    player_features, match_format, match_context, num_teams
                )
            elif self.optimization_mode == "hybrid":
                optimized_teams = self._optimize_hybrid(
                    player_features, match_format, match_context, num_teams
                )
            else:
                # Fallback to base team generation
                logger.info("🔄 Using fallback team generation...")
                base_teams = generate_world_class_ai_teams(player_features, match_format, match_context)
                optimized_teams = self._convert_to_optimized_format(base_teams)
            
            # Post-processing
            optimized_teams = self._post_process_teams(optimized_teams)
            
            optimization_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Team optimization complete in {optimization_time:.0f}ms")
            
            return optimized_teams[:num_teams]
        
        except Exception as e:
            logger.error(f"❌ Advanced optimization failed: {e}")
            # Fallback to base generation
            base_teams = generate_world_class_ai_teams(player_features, match_format, match_context)
            return self._convert_to_optimized_format(base_teams)[:num_teams]
    
    def _optimize_multi_objective(self, player_features: List[Dict], match_format: str, 
                                match_context: Dict, num_teams: int) -> List[OptimizedTeam]:
        """Multi-objective optimization using Pareto efficiency"""
        logger.info("🎯 Running multi-objective optimization...")
        
        # Generate diverse population
        population = self._generate_diverse_population(player_features, num_teams * 3)
        
        # Find Pareto optimal solutions
        pareto_optimal = self.pareto_optimizer.optimize(population)
        
        # Select diverse teams from Pareto front
        selected_teams = self._select_diverse_teams(pareto_optimal, num_teams)
        
        return selected_teams
    
    def _optimize_rl(self, player_features: List[Dict], match_format: str, 
                    match_context: Dict, num_teams: int) -> List[OptimizedTeam]:
        """RL-based optimization"""
        logger.info("🤖 Running RL-based optimization...")
        
        teams = []
        for i in range(num_teams):
            team = self.rl_optimizer.select_team_rl(player_features, self.constraints)
            teams.append(team)
            
            # Update existing teams for uniqueness
            self.existing_teams.append(team.players)
        
        return teams
    
    def _optimize_hybrid(self, player_features: List[Dict], match_format: str, 
                        match_context: Dict, num_teams: int) -> List[OptimizedTeam]:
        """Hybrid optimization combining multiple approaches"""
        logger.info("🔀 Running hybrid optimization...")
        
        teams = []
        
        # 50% multi-objective
        mo_teams = self._optimize_multi_objective(
            player_features, match_format, match_context, num_teams // 2
        )
        teams.extend(mo_teams)
        
        # 30% RL-based
        rl_teams = self._optimize_rl(
            player_features, match_format, match_context, max(1, num_teams * 3 // 10)
        )
        teams.extend(rl_teams)
        
        # 20% fallback to proven methods
        remaining = num_teams - len(teams)
        if remaining > 0:
            base_teams = generate_world_class_ai_teams(player_features, match_format, match_context)
            fallback_teams = self._convert_to_optimized_format(base_teams)
            teams.extend(fallback_teams[:remaining])
        
        return teams[:num_teams]
    
    def _generate_diverse_population(self, player_features: List[Dict], 
                                   population_size: int) -> List[OptimizedTeam]:
        """Generate diverse population for optimization"""
        population = []
        
        for _ in range(population_size):
            # Use existing team generation with randomization
            base_teams = generate_world_class_ai_teams(
                player_features, "T20", {"randomize": True}
            )
            
            if base_teams:
                optimized_team = self._convert_single_team_to_optimized(base_teams[0])
                population.append(optimized_team)
        
        return population
    
    def _select_diverse_teams(self, pareto_teams: List[OptimizedTeam], 
                            num_teams: int) -> List[OptimizedTeam]:
        """Select diverse teams from Pareto optimal set"""
        if len(pareto_teams) <= num_teams:
            return pareto_teams
        
        selected = []
        remaining = pareto_teams.copy()
        
        # Select first team (highest expected points)
        best_team = max(remaining, key=lambda t: t.metrics.expected_points)
        selected.append(best_team)
        remaining.remove(best_team)
        
        # Select remaining teams based on diversity
        while len(selected) < num_teams and remaining:
            best_diversity_score = -1
            best_team = None
            
            for candidate in remaining:
                diversity_score = self._calculate_diversity_score(candidate, selected)
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_team = candidate
            
            if best_team:
                selected.append(best_team)
                remaining.remove(best_team)
        
        return selected
    
    def _calculate_diversity_score(self, candidate: OptimizedTeam, 
                                 selected: List[OptimizedTeam]) -> float:
        """Calculate diversity score for team selection"""
        if not selected:
            return 1.0
        
        candidate_ids = {p.get('player_id', 0) for p in candidate.players}
        
        min_overlap = float('inf')
        for selected_team in selected:
            selected_ids = {p.get('player_id', 0) for p in selected_team.players}
            overlap = len(candidate_ids.intersection(selected_ids))
            min_overlap = min(min_overlap, overlap)
        
        # Higher score for lower overlap
        diversity_score = max(0, 11 - min_overlap) / 11
        return diversity_score
    
    def _convert_to_optimized_format(self, base_teams: List[Dict]) -> List[OptimizedTeam]:
        """Convert base teams to optimized format"""
        optimized_teams = []
        
        for team_dict in base_teams:
            optimized_team = self._convert_single_team_to_optimized(team_dict)
            optimized_teams.append(optimized_team)
        
        return optimized_teams
    
    def _convert_single_team_to_optimized(self, team_dict: Dict) -> OptimizedTeam:
        """Convert single team to optimized format"""
        players = team_dict.get('players', [])
        captain = team_dict.get('captain', players[0] if players else {})
        vice_captain = team_dict.get('vice_captain', players[1] if len(players) > 1 else captain)
        
        total_cost = sum(p.get('cost', 9.0) for p in players)
        metrics = self.rl_optimizer._calculate_team_metrics(players, captain, vice_captain)
        
        return OptimizedTeam(
            players=players,
            captain=captain,
            vice_captain=vice_captain,
            total_cost=total_cost,
            metrics=metrics,
            generation_method="converted_from_base",
            constraints_satisfied=True
        )
    
    def _post_process_teams(self, teams: List[OptimizedTeam]) -> List[OptimizedTeam]:
        """Post-process teams for final optimization"""
        # Sort by a composite score
        for team in teams:
            composite_score = (
                team.metrics.expected_points * 0.4 +
                team.metrics.balance_score * 0.2 +
                team.metrics.uniqueness_score * 0.2 +
                (100 - team.metrics.risk_score) * 0.2
            )
            team.metrics.composite_score = composite_score
        
        # Sort by composite score
        teams.sort(key=lambda t: t.metrics.composite_score, reverse=True)
        
        return teams

# Convenience functions
def optimize_teams_advanced(player_features: List[Dict[str, Any]], 
                          match_format: str, match_context: Dict[str, Any],
                          optimization_mode: str = "multi_objective",
                          num_teams: int = 5) -> List[OptimizedTeam]:
    """
    High-level function for advanced team optimization
    
    Args:
        player_features: List of player feature dictionaries
        match_format: Format of the match (T20, ODI, TEST)
        match_context: Additional match context
        optimization_mode: "multi_objective", "reinforcement_learning", or "hybrid"
        num_teams: Number of teams to generate
    
    Returns:
        List of optimized teams with detailed analytics
    """
    optimizer = AdvancedTeamOptimizer(optimization_mode)
    return optimizer.optimize_teams(player_features, match_format, match_context, num_teams)

if __name__ == "__main__":
    # Test the advanced optimizer
    def test_advanced_optimizer():
        print("🧪 Testing Advanced Team Optimizer...")
        
        # Create sample player data
        sample_players = []
        for i in range(30):
            player = {
                'player_id': i,
                'name': f'Player {i}',
                'role': ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper'][i % 4],
                'team_id': i % 2,
                'cost': 8.0 + random.random() * 4.0,
                'expected_points': 40.0 + random.random() * 40.0,
                'uncertainty': 0.1 + random.random() * 0.3,
                'captain_vice_captain_probability': random.random()
            }
            sample_players.append(player)
        
        # Test different optimization modes
        for mode in ["multi_objective", "reinforcement_learning", "hybrid"]:
            print(f"\n🎯 Testing {mode} optimization...")
            
            teams = optimize_teams_advanced(
                sample_players, "T20", {}, mode, 3
            )
            
            for i, team in enumerate(teams):
                print(f"  Team {i+1}: {team.metrics.expected_points:.1f} points "
                     f"(Risk: {team.metrics.risk_score:.2f}, "
                     f"Balance: {team.metrics.balance_score:.1f})")
        
        print("✅ Advanced optimizer test complete!")
    
    test_advanced_optimizer()