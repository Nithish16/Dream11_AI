#!/usr/bin/env python3
"""
Multi-Objective Evolutionary Optimizer - NSGA-III Implementation
Advanced evolutionary algorithms for optimal team selection with multiple objectives
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
from collections import defaultdict

@dataclass
class Individual:
    """Individual solution in the evolutionary algorithm"""
    team: List[int]  # Player indices
    objectives: List[float] = field(default_factory=list)
    constraints_violation: float = 0.0
    fitness: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0
    feasible: bool = True

@dataclass
class OptimizationObjectives:
    """Multiple objectives for team optimization"""
    maximize_expected_points: Callable[[List[int], List[Dict[str, Any]]], float]
    minimize_risk: Callable[[List[int], List[Dict[str, Any]]], float]
    maximize_ceiling: Callable[[List[int], List[Dict[str, Any]]], float]
    minimize_ownership: Callable[[List[int], List[Dict[str, Any]]], float]
    maximize_floor: Callable[[List[int], List[Dict[str, Any]]], float]

@dataclass
class OptimizationConstraints:
    """Constraints for team formation"""
    team_size: int = 11
    max_credits: float = 100.0
    min_batsmen: int = 3
    max_batsmen: int = 6
    min_bowlers: int = 3
    max_bowlers: int = 6
    min_allrounders: int = 1
    max_allrounders: int = 4
    min_wicketkeepers: int = 1
    max_wicketkeepers: int = 2
    max_players_per_team: int = 7

class NSGA3:
    """NSGA-III Multi-Objective Evolutionary Algorithm"""
    
    def __init__(self, population_size: int = 100, num_objectives: int = 5, 
                 num_generations: int = 500, crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1):
        self.population_size = population_size
        self.num_objectives = num_objectives
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Generate reference points for NSGA-III
        self.reference_points = self._generate_reference_points()
        
        # Statistics
        self.convergence_history = []
        self.diversity_history = []
        
    def _generate_reference_points(self, num_divisions: int = 6) -> np.ndarray:
        """Generate structured reference points for NSGA-III"""
        
        def generate_unit_simplex_points(n_dim, n_divisions):
            """Generate points on unit simplex"""
            if n_dim == 1:
                return np.array([[1.0]])
            
            points = []
            for i in range(n_divisions + 1):
                weight = i / n_divisions
                if n_dim == 2:
                    points.append([weight, 1.0 - weight])
                else:
                    sub_points = generate_unit_simplex_points(n_dim - 1, n_divisions - i)
                    for sub_point in sub_points:
                        point = [weight] + [x * (1.0 - weight) for x in sub_point]
                        points.append(point)
            
            return np.array(points)
        
        # Generate reference points on unit simplex
        reference_points = generate_unit_simplex_points(self.num_objectives, num_divisions)
        
        print(f"Generated {len(reference_points)} reference points for {self.num_objectives} objectives")
        
        return reference_points
    
    def optimize(self, players: List[Dict[str, Any]], 
                 objectives: OptimizationObjectives,
                 constraints: OptimizationConstraints) -> List[Individual]:
        """Main optimization loop"""
        
        print(f"🧬 Starting NSGA-III optimization with {self.population_size} individuals for {self.num_generations} generations")
        
        # Initialize population
        population = self._initialize_population(players, constraints)
        
        # Evaluate initial population
        self._evaluate_population(population, players, objectives, constraints)
        
        best_solutions = []
        
        for generation in range(self.num_generations):
            # Selection, crossover, and mutation
            offspring = self._create_offspring(population, players, constraints)
            
            # Evaluate offspring
            self._evaluate_population(offspring, players, objectives, constraints)
            
            # Environmental selection using NSGA-III
            combined_population = population + offspring
            population = self._environmental_selection(combined_population)
            
            # Track convergence
            self._update_statistics(population, generation)
            
            # Store best solutions periodically
            if generation % 50 == 0:
                pareto_front = self._get_pareto_front(population)
                best_solutions.extend(pareto_front)
                
                print(f"Generation {generation}: Pareto front size: {len(pareto_front)}")
                self._print_objective_stats(pareto_front)
            
            # Adaptive parameter adjustment
            self._adapt_parameters(generation)
        
        # Final Pareto-optimal solutions
        final_pareto_front = self._get_pareto_front(population)
        
        print(f"✅ Optimization complete. Final Pareto front: {len(final_pareto_front)} solutions")
        
        return final_pareto_front
    
    def _initialize_population(self, players: List[Dict[str, Any]], 
                             constraints: OptimizationConstraints) -> List[Individual]:
        """Initialize population with feasible solutions"""
        
        population = []
        attempts = 0
        max_attempts = self.population_size * 10
        
        while len(population) < self.population_size and attempts < max_attempts:
            # Generate random team
            team = self._generate_random_team(players, constraints)
            
            if team and self._is_feasible(team, players, constraints):
                individual = Individual(team=team)
                population.append(individual)
            
            attempts += 1
        
        # Fill remaining with less strict constraints if needed
        while len(population) < self.population_size:
            team = self._generate_random_team_relaxed(players, constraints)
            individual = Individual(team=team)
            population.append(individual)
        
        print(f"Initialized population with {len(population)} individuals")
        
        return population
    
    def _generate_random_team(self, players: List[Dict[str, Any]], 
                            constraints: OptimizationConstraints) -> List[int]:
        """Generate a random feasible team"""
        
        # Categorize players by role
        batsmen = [i for i, p in enumerate(players) if self._is_batsman(p.get('role', ''))]
        bowlers = [i for i, p in enumerate(players) if self._is_bowler(p.get('role', ''))]
        allrounders = [i for i, p in enumerate(players) if self._is_allrounder(p.get('role', ''))]
        wicketkeepers = [i for i, p in enumerate(players) if self._is_wicketkeeper(p.get('role', ''))]
        
        team = []
        
        try:
            # Select minimum required players from each role
            team.extend(random.sample(wicketkeepers, min(constraints.min_wicketkeepers, len(wicketkeepers))))
            team.extend(random.sample(allrounders, min(constraints.min_allrounders, len(allrounders))))
            
            remaining_slots = constraints.team_size - len(team)
            
            # Fill remaining slots with batsmen and bowlers
            min_bat_needed = max(0, constraints.min_batsmen)
            min_bowl_needed = max(0, constraints.min_bowlers)
            
            if min_bat_needed > 0:
                available_bat = [b for b in batsmen if b not in team]
                team.extend(random.sample(available_bat, min(min_bat_needed, len(available_bat))))
            
            if min_bowl_needed > 0:
                available_bowl = [b for b in bowlers if b not in team]
                team.extend(random.sample(available_bowl, min(min_bowl_needed, len(available_bowl))))
            
            # Fill remaining slots randomly
            while len(team) < constraints.team_size:
                available_players = [i for i in range(len(players)) if i not in team]
                if not available_players:
                    break
                
                player_idx = random.choice(available_players)
                
                # Check role constraints
                if self._can_add_player(team, player_idx, players, constraints):
                    team.append(player_idx)
                else:
                    # Try multiple times to find suitable player
                    for _ in range(10):
                        player_idx = random.choice(available_players)
                        if self._can_add_player(team, player_idx, players, constraints):
                            team.append(player_idx)
                            break
                    else:
                        # Force add if no suitable player found
                        team.append(random.choice(available_players[:5]))
            
            return team[:constraints.team_size]
            
        except Exception as e:
            return []
    
    def _generate_random_team_relaxed(self, players: List[Dict[str, Any]], 
                                    constraints: OptimizationConstraints) -> List[int]:
        """Generate random team with relaxed constraints"""
        
        # Simple random selection
        available_players = list(range(len(players)))
        team = random.sample(available_players, min(constraints.team_size, len(available_players)))
        
        return team
    
    def _can_add_player(self, current_team: List[int], player_idx: int, 
                       players: List[Dict[str, Any]], constraints: OptimizationConstraints) -> bool:
        """Check if player can be added to current team"""
        
        if player_idx in current_team:
            return False
        
        # Check role constraints
        role_counts = self._count_roles(current_team + [player_idx], players)
        
        if role_counts['batsmen'] > constraints.max_batsmen:
            return False
        if role_counts['bowlers'] > constraints.max_bowlers:
            return False
        if role_counts['allrounders'] > constraints.max_allrounders:
            return False
        if role_counts['wicketkeepers'] > constraints.max_wicketkeepers:
            return False
        
        # Check credit constraints
        total_credits = sum(players[i].get('credits', 8.5) for i in current_team + [player_idx])
        if total_credits > constraints.max_credits:
            return False
        
        return True
    
    def _is_batsman(self, role: str) -> bool:
        """Check if player is a batsman"""
        role_lower = role.lower()
        return 'bat' in role_lower and 'allrounder' not in role_lower and 'wk' not in role_lower
    
    def _is_bowler(self, role: str) -> bool:
        """Check if player is a bowler"""
        role_lower = role.lower()
        return 'bowl' in role_lower and 'allrounder' not in role_lower
    
    def _is_allrounder(self, role: str) -> bool:
        """Check if player is an all-rounder"""
        role_lower = role.lower()
        return 'allrounder' in role_lower or 'all-rounder' in role_lower
    
    def _is_wicketkeeper(self, role: str) -> bool:
        """Check if player is a wicket-keeper"""
        role_lower = role.lower()
        return 'wk' in role_lower or 'wicket' in role_lower or 'keeper' in role_lower
    
    def _count_roles(self, team: List[int], players: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count players by role in team"""
        
        counts = {'batsmen': 0, 'bowlers': 0, 'allrounders': 0, 'wicketkeepers': 0}
        
        for player_idx in team:
            role = players[player_idx].get('role', '')
            
            if self._is_wicketkeeper(role):
                counts['wicketkeepers'] += 1
            elif self._is_allrounder(role):
                counts['allrounders'] += 1
            elif self._is_bowler(role):
                counts['bowlers'] += 1
            else:
                counts['batsmen'] += 1
        
        return counts
    
    def _is_feasible(self, team: List[int], players: List[Dict[str, Any]], 
                    constraints: OptimizationConstraints) -> bool:
        """Check if team satisfies all constraints"""
        
        if len(team) != constraints.team_size:
            return False
        
        # Check role constraints
        role_counts = self._count_roles(team, players)
        
        if not (constraints.min_batsmen <= role_counts['batsmen'] <= constraints.max_batsmen):
            return False
        if not (constraints.min_bowlers <= role_counts['bowlers'] <= constraints.max_bowlers):
            return False
        if not (constraints.min_allrounders <= role_counts['allrounders'] <= constraints.max_allrounders):
            return False
        if not (constraints.min_wicketkeepers <= role_counts['wicketkeepers'] <= constraints.max_wicketkeepers):
            return False
        
        # Check credit constraints
        total_credits = sum(players[i].get('credits', 8.5) for i in team)
        if total_credits > constraints.max_credits:
            return False
        
        return True
    
    def _evaluate_population(self, population: List[Individual], 
                           players: List[Dict[str, Any]], 
                           objectives: OptimizationObjectives,
                           constraints: OptimizationConstraints):
        """Evaluate all individuals in population"""
        
        for individual in population:
            # Calculate objectives
            individual.objectives = [
                objectives.maximize_expected_points(individual.team, players),
                -objectives.minimize_risk(individual.team, players),  # Negative for maximization
                objectives.maximize_ceiling(individual.team, players),
                -objectives.minimize_ownership(individual.team, players),  # Negative for maximization
                objectives.maximize_floor(individual.team, players)
            ]
            
            # Calculate constraint violations
            individual.constraints_violation = self._calculate_constraint_violation(
                individual.team, players, constraints
            )
            
            individual.feasible = individual.constraints_violation == 0
    
    def _calculate_constraint_violation(self, team: List[int], 
                                      players: List[Dict[str, Any]], 
                                      constraints: OptimizationConstraints) -> float:
        """Calculate total constraint violation"""
        
        violation = 0.0
        
        # Team size violation
        if len(team) != constraints.team_size:
            violation += abs(len(team) - constraints.team_size) * 1000
        
        # Role violations
        role_counts = self._count_roles(team, players)
        
        if role_counts['batsmen'] < constraints.min_batsmen:
            violation += (constraints.min_batsmen - role_counts['batsmen']) * 100
        if role_counts['batsmen'] > constraints.max_batsmen:
            violation += (role_counts['batsmen'] - constraints.max_batsmen) * 100
        
        if role_counts['bowlers'] < constraints.min_bowlers:
            violation += (constraints.min_bowlers - role_counts['bowlers']) * 100
        if role_counts['bowlers'] > constraints.max_bowlers:
            violation += (role_counts['bowlers'] - constraints.max_bowlers) * 100
        
        if role_counts['allrounders'] < constraints.min_allrounders:
            violation += (constraints.min_allrounders - role_counts['allrounders']) * 100
        if role_counts['allrounders'] > constraints.max_allrounders:
            violation += (role_counts['allrounders'] - constraints.max_allrounders) * 100
        
        if role_counts['wicketkeepers'] < constraints.min_wicketkeepers:
            violation += (constraints.min_wicketkeepers - role_counts['wicketkeepers']) * 100
        if role_counts['wicketkeepers'] > constraints.max_wicketkeepers:
            violation += (role_counts['wicketkeepers'] - constraints.max_wicketkeepers) * 100
        
        # Credit violation
        total_credits = sum(players[i].get('credits', 8.5) for i in team)
        if total_credits > constraints.max_credits:
            violation += (total_credits - constraints.max_credits) * 10
        
        return violation
    
    def _create_offspring(self, population: List[Individual], 
                         players: List[Dict[str, Any]], 
                         constraints: OptimizationConstraints) -> List[Individual]:
        """Create offspring through selection, crossover, and mutation"""
        
        offspring = []
        
        while len(offspring) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, 3)
            parent2 = self._tournament_selection(population, 3)
            
            # Crossover
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2, players, constraints)
            else:
                child1, child2 = Individual(team=parent1.team.copy()), Individual(team=parent2.team.copy())
            
            # Mutation
            if random.random() < self.mutation_prob:
                child1 = self._mutate(child1, players, constraints)
            if random.random() < self.mutation_prob:
                child2 = self._mutate(child2, players, constraints)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int) -> Individual:
        """Tournament selection"""
        
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Select feasible individuals first
        feasible_individuals = [ind for ind in tournament if ind.feasible]
        
        if feasible_individuals:
            # Among feasible, select based on rank and crowding distance
            tournament = feasible_individuals
        
        # Sort by rank, then by crowding distance
        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        return tournament[0]
    
    def _crossover(self, parent1: Individual, parent2: Individual, 
                  players: List[Dict[str, Any]], constraints: OptimizationConstraints) -> Tuple[Individual, Individual]:
        """Uniform crossover for team formation"""
        
        # Create masks for crossover
        mask1 = [random.random() < 0.5 for _ in range(len(parent1.team))]
        mask2 = [not m for m in mask1]
        
        # Initialize children
        child1_team = []
        child2_team = []
        
        # Apply crossover with repair mechanism
        for i, (m1, m2) in enumerate(zip(mask1, mask2)):
            if m1 and i < len(parent1.team):
                child1_team.append(parent1.team[i])
            if m2 and i < len(parent2.team):
                child2_team.append(parent2.team[i])
        
        # Repair children to ensure proper team size and constraints
        child1_team = self._repair_team(child1_team, players, constraints)
        child2_team = self._repair_team(child2_team, players, constraints)
        
        return Individual(team=child1_team), Individual(team=child2_team)
    
    def _repair_team(self, team: List[int], players: List[Dict[str, Any]], 
                    constraints: OptimizationConstraints) -> List[int]:
        """Repair team to satisfy constraints"""
        
        # Remove duplicates
        team = list(dict.fromkeys(team))
        
        # Add missing players if team is too small
        while len(team) < constraints.team_size:
            available_players = [i for i in range(len(players)) if i not in team]
            if not available_players:
                break
            
            # Try to add player that satisfies role constraints
            added = False
            for player_idx in random.sample(available_players, min(10, len(available_players))):
                if self._can_add_player(team, player_idx, players, constraints):
                    team.append(player_idx)
                    added = True
                    break
            
            if not added:
                # Force add random player if no suitable found
                team.append(random.choice(available_players))
        
        # Remove excess players if team is too large
        while len(team) > constraints.team_size:
            team.pop()
        
        return team
    
    def _mutate(self, individual: Individual, players: List[Dict[str, Any]], 
               constraints: OptimizationConstraints) -> Individual:
        """Mutate individual by replacing random players"""
        
        mutated_team = individual.team.copy()
        
        # Replace 1-3 random players
        num_mutations = random.randint(1, min(3, len(mutated_team)))
        
        for _ in range(num_mutations):
            if not mutated_team:
                break
            
            # Remove random player
            remove_idx = random.randint(0, len(mutated_team) - 1)
            removed_player = mutated_team.pop(remove_idx)
            
            # Add random replacement
            available_players = [i for i in range(len(players)) if i not in mutated_team]
            
            if available_players:
                # Try to find suitable replacement
                for _ in range(10):
                    replacement = random.choice(available_players)
                    if self._can_add_player(mutated_team, replacement, players, constraints):
                        mutated_team.append(replacement)
                        break
                else:
                    # Add random replacement if no suitable found
                    mutated_team.append(random.choice(available_players))
        
        # Repair if needed
        mutated_team = self._repair_team(mutated_team, players, constraints)
        
        return Individual(team=mutated_team)
    
    def _environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """NSGA-III environmental selection"""
        
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(population)
        
        # Select individuals for next generation
        selected = []
        front_idx = 0
        
        while len(selected) + len(fronts[front_idx]) <= self.population_size:
            # Calculate crowding distance for current front
            self._calculate_crowding_distance(fronts[front_idx])
            selected.extend(fronts[front_idx])
            front_idx += 1
            
            if front_idx >= len(fronts):
                break
        
        # Fill remaining slots from the next front using reference point association
        if len(selected) < self.population_size and front_idx < len(fronts):
            remaining_slots = self.population_size - len(selected)
            last_front = fronts[front_idx]
            
            # Associate with reference points and select
            selected_from_last_front = self._reference_point_selection(last_front, remaining_slots)
            selected.extend(selected_from_last_front)
        
        return selected
    
    def _non_dominated_sorting(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting"""
        
        fronts = []
        
        # Calculate domination relationships
        for i, ind1 in enumerate(population):
            ind1.dominated_solutions = []
            ind1.domination_count = 0
            
            for j, ind2 in enumerate(population):
                if i != j:
                    if self._dominates(ind1, ind2):
                        ind1.dominated_solutions.append(j)
                    elif self._dominates(ind2, ind1):
                        ind1.domination_count += 1
        
        # First front
        current_front = []
        for i, individual in enumerate(population):
            if individual.domination_count == 0:
                individual.rank = 0
                current_front.append(individual)
        
        fronts.append(current_front)
        
        # Subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for individual in fronts[front_idx]:
                for j in individual.dominated_solutions:
                    population[j].domination_count -= 1
                    if population[j].domination_count == 0:
                        population[j].rank = front_idx + 1
                        next_front.append(population[j])
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1] if fronts and not fronts[-1] else fronts
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2"""
        
        # Handle constraint violation first
        if ind1.constraints_violation != ind2.constraints_violation:
            return ind1.constraints_violation < ind2.constraints_violation
        
        # For feasible solutions, use Pareto dominance
        if ind1.feasible and ind2.feasible:
            at_least_one_better = False
            for obj1, obj2 in zip(ind1.objectives, ind2.objectives):
                if obj1 < obj2:
                    return False
                elif obj1 > obj2:
                    at_least_one_better = True
            
            return at_least_one_better
        
        return False
    
    def _calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front"""
        
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0
        
        # Calculate for each objective
        for obj_idx in range(self.num_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Set boundary solutions to infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distance for intermediate solutions
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance
    
    def _reference_point_selection(self, front: List[Individual], num_select: int) -> List[Individual]:
        """Select individuals from front using reference point association"""
        
        if len(front) <= num_select:
            return front
        
        # Normalize objectives
        normalized_front = self._normalize_objectives(front)
        
        # Associate individuals with reference points
        associations = self._associate_with_reference_points(normalized_front)
        
        # Select individuals based on niche count
        selected = []
        niche_count = defaultdict(int)
        
        # Count individuals already selected in previous fronts
        # (In a complete implementation, this would track across all previous fronts)
        
        while len(selected) < num_select:
            # Find reference point with minimum niche count
            min_niche = min(niche_count.values()) if niche_count else 0
            candidate_refs = [ref_idx for ref_idx, count in niche_count.items() if count == min_niche]
            
            if not candidate_refs:
                candidate_refs = list(range(len(self.reference_points)))
            
            # Select from least crowded reference point
            selected_ref = random.choice(candidate_refs)
            
            # Find individuals associated with this reference point
            candidates = [ind for ind, ref_idx in associations.items() if ref_idx == selected_ref and ind not in selected]
            
            if candidates:
                # Select individual with minimum distance to reference point
                best_candidate = min(candidates, key=lambda ind: self._distance_to_reference_point(
                    self._normalize_individual_objectives(ind), selected_ref
                ))
                selected.append(best_candidate)
                niche_count[selected_ref] += 1
            else:
                # No candidates for this reference point, select randomly from remaining
                remaining = [ind for ind in front if ind not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
        
        return selected
    
    def _normalize_objectives(self, front: List[Individual]) -> List[Individual]:
        """Normalize objectives to [0, 1] range"""
        
        if not front:
            return front
        
        # Find min and max for each objective
        min_obj = [min(ind.objectives[i] for ind in front) for i in range(self.num_objectives)]
        max_obj = [max(ind.objectives[i] for ind in front) for i in range(self.num_objectives)]
        
        # Normalize
        normalized_front = []
        for individual in front:
            normalized_ind = Individual(team=individual.team.copy())
            normalized_ind.objectives = []
            
            for i in range(self.num_objectives):
                if max_obj[i] - min_obj[i] > 0:
                    normalized_obj = (individual.objectives[i] - min_obj[i]) / (max_obj[i] - min_obj[i])
                else:
                    normalized_obj = 0.0
                normalized_ind.objectives.append(normalized_obj)
            
            normalized_front.append(normalized_ind)
        
        return normalized_front
    
    def _normalize_individual_objectives(self, individual: Individual) -> List[float]:
        """Normalize individual objectives (simplified)"""
        return individual.objectives  # In practice, use proper normalization
    
    def _associate_with_reference_points(self, normalized_front: List[Individual]) -> Dict[Individual, int]:
        """Associate individuals with reference points"""
        
        associations = {}
        
        for individual in normalized_front:
            best_ref_idx = 0
            min_distance = float('inf')
            
            for ref_idx, ref_point in enumerate(self.reference_points):
                distance = self._distance_to_reference_point(individual.objectives, ref_idx)
                if distance < min_distance:
                    min_distance = distance
                    best_ref_idx = ref_idx
            
            associations[individual] = best_ref_idx
        
        return associations
    
    def _distance_to_reference_point(self, objectives: List[float], ref_idx: int) -> float:
        """Calculate distance from objectives to reference point"""
        
        ref_point = self.reference_points[ref_idx]
        
        # Perpendicular distance to reference line
        # Simplified calculation
        return np.linalg.norm(np.array(objectives) - np.array(ref_point))
    
    def _get_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """Get the Pareto-optimal front from population"""
        
        fronts = self._non_dominated_sorting(population)
        return fronts[0] if fronts else []
    
    def _update_statistics(self, population: List[Individual], generation: int):
        """Update convergence and diversity statistics"""
        
        pareto_front = self._get_pareto_front(population)
        
        if pareto_front:
            # Convergence metric (average objective values)
            avg_objectives = [np.mean([ind.objectives[i] for ind in pareto_front]) 
                            for i in range(self.num_objectives)]
            self.convergence_history.append(avg_objectives)
            
            # Diversity metric (average crowding distance)
            self._calculate_crowding_distance(pareto_front)
            avg_crowding = np.mean([ind.crowding_distance for ind in pareto_front 
                                  if ind.crowding_distance != float('inf')])
            self.diversity_history.append(avg_crowding)
    
    def _adapt_parameters(self, generation: int):
        """Adapt algorithm parameters during evolution"""
        
        # Increase mutation probability in later generations for diversity
        if generation > self.num_generations * 0.7:
            self.mutation_prob = min(0.3, self.mutation_prob * 1.05)
        
        # Decrease crossover probability in later generations for exploitation
        if generation > self.num_generations * 0.8:
            self.crossover_prob = max(0.6, self.crossover_prob * 0.98)
    
    def _print_objective_stats(self, pareto_front: List[Individual]):
        """Print statistics about the Pareto front"""
        
        if not pareto_front:
            return
        
        obj_names = ['Expected Points', 'Risk (neg)', 'Ceiling', 'Ownership (neg)', 'Floor']
        
        for i, name in enumerate(obj_names):
            values = [ind.objectives[i] for ind in pareto_front]
            print(f"  {name}: min={min(values):.2f}, max={max(values):.2f}, avg={np.mean(values):.2f}")

# Objective functions
def maximize_expected_points(team: List[int], players: List[Dict[str, Any]]) -> float:
    """Calculate expected points for the team"""
    total_points = sum(players[i].get('final_score', 50.0) for i in team)
    
    # Add captain and vice-captain bonuses (simplified)
    if len(team) >= 2:
        captain_idx = max(team, key=lambda i: players[i].get('final_score', 0))
        vice_captain_idx = max([i for i in team if i != captain_idx], 
                              key=lambda i: players[i].get('final_score', 0))
        
        total_points += players[captain_idx].get('final_score', 0)  # 2x captain
        total_points += players[vice_captain_idx].get('final_score', 0) * 0.5  # 1.5x VC
    
    return total_points

def minimize_risk(team: List[int], players: List[Dict[str, Any]]) -> float:
    """Calculate risk (variance) of the team"""
    scores = [players[i].get('final_score', 50.0) for i in team]
    
    if len(scores) > 1:
        return np.var(scores)
    else:
        return 0.0

def maximize_ceiling(team: List[int], players: List[Dict[str, Any]]) -> float:
    """Calculate ceiling potential of the team"""
    # Ceiling = expected + 2 * std for each player
    ceiling_total = 0
    
    for i in team:
        expected = players[i].get('final_score', 50.0)
        consistency = players[i].get('consistency_score', 50.0)
        
        # Higher consistency = lower variance
        std_dev = max(5, 30 - consistency * 0.3)
        ceiling = expected + 2 * std_dev
        ceiling_total += ceiling
    
    return ceiling_total

def minimize_ownership(team: List[int], players: List[Dict[str, Any]]) -> float:
    """Calculate average ownership of the team"""
    return np.mean([players[i].get('ownership_prediction', 50.0) for i in team])

def maximize_floor(team: List[int], players: List[Dict[str, Any]]) -> float:
    """Calculate floor potential of the team"""
    # Floor = expected - 2 * std for each player, but minimum 0
    floor_total = 0
    
    for i in team:
        expected = players[i].get('final_score', 50.0)
        consistency = players[i].get('consistency_score', 50.0)
        
        # Higher consistency = lower variance
        std_dev = max(5, 30 - consistency * 0.3)
        floor = max(0, expected - 2 * std_dev)
        floor_total += floor
    
    return floor_total

# Export
__all__ = ['NSGA3', 'Individual', 'OptimizationObjectives', 'OptimizationConstraints',
           'maximize_expected_points', 'minimize_risk', 'maximize_ceiling', 
           'minimize_ownership', 'maximize_floor']