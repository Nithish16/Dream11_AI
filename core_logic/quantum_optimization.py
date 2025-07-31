#!/usr/bin/env python3
"""
Quantum-Inspired Optimization Engine - Advanced Team Selection
Quantum computing principles applied to fantasy cricket optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import random
from collections import defaultdict
import cmath
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QuantumState:
    """Quantum state representation for team selection"""
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    entanglement_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    phase_factors: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class QuantumGate:
    """Quantum gate for state manipulation"""
    name: str
    matrix: np.ndarray
    target_qubits: List[int] = field(default_factory=list)
    control_qubits: List[int] = field(default_factory=list)

@dataclass
class QuantumTeamSolution:
    """Quantum-optimized team solution"""
    team_indices: List[int]
    quantum_score: float
    classical_score: float
    entanglement_score: float
    coherence_factor: float
    measurement_confidence: float

class QuantumRegister:
    """Quantum register for representing player selection states"""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        
        # Initialize in superposition state
        self.state_vector = np.ones(self.num_states, dtype=complex) / np.sqrt(self.num_states)
        
        # Phase tracking
        self.phases = np.zeros(self.num_states)
        
        # Entanglement tracking
        self.entanglement_map = defaultdict(list)
    
    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate to create superposition"""
        h_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_single_qubit_gate(h_gate, qubit)
    
    def apply_rotation(self, qubit: int, theta: float, phi: float = 0):
        """Apply rotation gate with performance-based angle"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        exp_phi = np.exp(1j * phi)
        
        rotation_gate = np.array([
            [cos_half, -sin_half * exp_phi],
            [sin_half, cos_half * exp_phi]
        ], dtype=complex)
        
        self._apply_single_qubit_gate(rotation_gate, qubit)
    
    def apply_controlled_not(self, control: int, target: int):
        """Apply CNOT gate for player correlation"""
        cnot_matrix = self._get_cnot_matrix(control, target)
        self.state_vector = cnot_matrix @ self.state_vector
        
        # Update entanglement
        self.entanglement_map[control].append(target)
        self.entanglement_map[target].append(control)
    
    def apply_toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli gate for complex constraints"""
        toffoli_matrix = self._get_toffoli_matrix(control1, control2, target)
        self.state_vector = toffoli_matrix @ self.state_vector
    
    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply single-qubit gate to the register"""
        for i in range(self.num_states):
            if self._get_bit(i, qubit) == 0:
                j = self._flip_bit(i, qubit)
                if j < self.num_states:
                    # Apply gate matrix
                    state_0 = self.state_vector[i]
                    state_1 = self.state_vector[j]
                    
                    self.state_vector[i] = gate[0, 0] * state_0 + gate[0, 1] * state_1
                    self.state_vector[j] = gate[1, 0] * state_0 + gate[1, 1] * state_1
    
    def _get_cnot_matrix(self, control: int, target: int) -> np.ndarray:
        """Generate CNOT gate matrix for register"""
        matrix = np.eye(self.num_states, dtype=complex)
        
        for i in range(self.num_states):
            if self._get_bit(i, control) == 1:
                j = self._flip_bit(i, target)
                if j != i and j < self.num_states:
                    # Swap rows i and j
                    matrix[i, i] = 0
                    matrix[j, j] = 0
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        
        return matrix
    
    def _get_toffoli_matrix(self, control1: int, control2: int, target: int) -> np.ndarray:
        """Generate Toffoli gate matrix"""
        matrix = np.eye(self.num_states, dtype=complex)
        
        for i in range(self.num_states):
            if self._get_bit(i, control1) == 1 and self._get_bit(i, control2) == 1:
                j = self._flip_bit(i, target)
                if j != i and j < self.num_states:
                    matrix[i, i] = 0
                    matrix[j, j] = 0
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        
        return matrix
    
    def _get_bit(self, number: int, position: int) -> int:
        """Get bit at position"""
        return (number >> position) & 1
    
    def _flip_bit(self, number: int, position: int) -> int:
        """Flip bit at position"""
        return number ^ (1 << position)
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.state_vector) ** 2
    
    def measure(self, num_measurements: int = 1) -> List[int]:
        """Measure the quantum state"""
        probabilities = self.get_probabilities()
        measurements = np.random.choice(
            self.num_states, 
            size=num_measurements, 
            p=probabilities
        )
        return measurements.tolist()
    
    def get_entanglement_entropy(self) -> float:
        """Calculate entanglement entropy"""
        probabilities = self.get_probabilities()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return entropy / self.num_qubits  # Normalized

class QuantumAnnealingOptimizer:
    """Quantum annealing-inspired optimizer for team selection"""
    
    def __init__(self, num_players: int, team_size: int = 11):
        self.num_players = num_players
        self.team_size = team_size
        self.num_qubits = min(num_players, 20)  # Limit for computational efficiency
        
        # Annealing parameters
        self.initial_temperature = 10.0
        self.final_temperature = 0.01
        self.annealing_steps = 1000
        
        # Quantum parameters
        self.transverse_field_strength = 1.0
        self.longitudinal_field_strength = 0.0
        
        # Optimization history
        self.energy_history = []
        self.temperature_schedule = []
    
    def optimize_team_selection(self, players: List[Dict[str, Any]], 
                              constraints: Dict[str, Any],
                              objectives: Dict[str, float]) -> QuantumTeamSolution:
        """Optimize team selection using quantum annealing"""
        
        print("🔮 Starting Quantum-Inspired Team Optimization...")
        
        # Initialize quantum register
        quantum_register = QuantumRegister(self.num_qubits)
        
        # Create Hamiltonian (energy function)
        hamiltonian = self._create_hamiltonian(players, constraints, objectives)
        
        # Apply quantum annealing
        best_solution = self._quantum_annealing(quantum_register, hamiltonian, players)
        
        print(f"✅ Quantum optimization complete. Score: {best_solution.quantum_score:.2f}")
        
        return best_solution
    
    def _create_hamiltonian(self, players: List[Dict[str, Any]], 
                          constraints: Dict[str, Any],
                          objectives: Dict[str, float]) -> np.ndarray:
        """Create Hamiltonian matrix representing the optimization problem"""
        
        # Problem size
        problem_size = min(self.num_players, self.num_qubits)
        hamiltonian = np.zeros((2**problem_size, 2**problem_size))
        
        # Add objective terms
        for i in range(2**problem_size):
            state_energy = self._calculate_state_energy(i, players[:problem_size], constraints, objectives)
            hamiltonian[i, i] = state_energy
        
        # Add interaction terms (player correlations)
        for i in range(problem_size):
            for j in range(i + 1, problem_size):
                if i < len(players) and j < len(players):
                    correlation = self._calculate_player_correlation(players[i], players[j])
                    
                    # Add correlation to Hamiltonian
                    for state in range(2**problem_size):
                        bit_i = self._get_bit(state, i)
                        bit_j = self._get_bit(state, j)
                        
                        if bit_i == 1 and bit_j == 1:
                            hamiltonian[state, state] += correlation * 0.1
        
        return hamiltonian
    
    def _calculate_state_energy(self, state: int, players: List[Dict[str, Any]], 
                              constraints: Dict[str, Any], objectives: Dict[str, float]) -> float:
        """Calculate energy for a quantum state"""
        
        # Extract team from binary state
        team_indices = []
        for i in range(len(players)):
            if self._get_bit(state, i) == 1:
                team_indices.append(i)
        
        # Base energy from team quality
        team_score = sum(players[i].get('final_score', 0) for i in team_indices)
        base_energy = -team_score  # Negative because we minimize energy
        
        # Constraint penalties
        constraint_penalty = self._calculate_constraint_penalties(team_indices, players, constraints)
        
        # Size penalty
        size_penalty = abs(len(team_indices) - self.team_size) * 100
        
        total_energy = base_energy + constraint_penalty + size_penalty
        
        return total_energy
    
    def _calculate_constraint_penalties(self, team_indices: List[int], 
                                      players: List[Dict[str, Any]], 
                                      constraints: Dict[str, Any]) -> float:
        """Calculate constraint violation penalties"""
        penalty = 0.0
        
        if not team_indices:
            return 1000.0  # High penalty for empty team
        
        # Role constraints
        role_counts = defaultdict(int)
        total_credits = 0.0
        
        for idx in team_indices:
            if idx < len(players):
                player = players[idx]
                role = player.get('role', '').lower()
                
                if 'bat' in role and 'allrounder' not in role:
                    role_counts['batsman'] += 1
                elif 'bowl' in role and 'allrounder' not in role:
                    role_counts['bowler'] += 1
                elif 'allrounder' in role:
                    role_counts['allrounder'] += 1
                elif 'wk' in role or 'wicket' in role:
                    role_counts['wicketkeeper'] += 1
                
                total_credits += player.get('credits', 8.5)
        
        # Role constraint penalties
        penalty += max(0, 3 - role_counts['batsman']) * 50  # Min 3 batsmen
        penalty += max(0, role_counts['batsman'] - 6) * 50  # Max 6 batsmen
        penalty += max(0, 3 - role_counts['bowler']) * 50   # Min 3 bowlers
        penalty += max(0, role_counts['bowler'] - 6) * 50   # Max 6 bowlers
        penalty += max(0, 1 - role_counts['wicketkeeper']) * 100  # Min 1 WK
        penalty += max(0, role_counts['wicketkeeper'] - 2) * 100  # Max 2 WK
        
        # Credit constraint
        penalty += max(0, total_credits - constraints.get('max_credits', 100)) * 10
        
        return penalty
    
    def _calculate_player_correlation(self, player1: Dict[str, Any], 
                                    player2: Dict[str, Any]) -> float:
        """Calculate correlation between two players"""
        
        # Team synergy
        if player1.get('team_name', '') == player2.get('team_name', ''):
            team_synergy = 0.1
        else:
            team_synergy = -0.05  # Small penalty for opposing teams
        
        # Role complementarity
        role1 = player1.get('role', '').lower()
        role2 = player2.get('role', '').lower()
        
        role_synergy = 0.0
        if ('bat' in role1 and 'bowl' in role2) or ('bowl' in role1 and 'bat' in role2):
            role_synergy = 0.05  # Complementary roles
        
        # Performance correlation
        score1 = player1.get('final_score', 50)
        score2 = player2.get('final_score', 50)
        performance_correlation = abs(score1 - score2) / 100  # Normalized difference
        
        return team_synergy + role_synergy - performance_correlation
    
    def _quantum_annealing(self, quantum_register: QuantumRegister, 
                          hamiltonian: np.ndarray,
                          players: List[Dict[str, Any]]) -> QuantumTeamSolution:
        """Perform quantum annealing optimization"""
        
        best_energy = float('inf')
        best_state = 0
        best_measurement_history = []
        
        # Annealing schedule
        temperatures = np.logspace(
            np.log10(self.initial_temperature),
            np.log10(self.final_temperature),
            self.annealing_steps
        )
        
        # Initialize in equal superposition
        for qubit in range(quantum_register.num_qubits):
            quantum_register.apply_hadamard(qubit)
        
        for step, temperature in enumerate(temperatures):
            # Calculate annealing parameter
            s = step / (self.annealing_steps - 1)  # 0 to 1
            
            # Transverse field strength (decreases with annealing)
            gamma = self.transverse_field_strength * (1 - s)
            
            # Problem Hamiltonian strength (increases with annealing)
            delta = self.longitudinal_field_strength * s
            
            # Apply quantum evolution
            self._apply_quantum_evolution(quantum_register, hamiltonian, gamma, delta, temperature)
            
            # Measure system periodically
            if step % 100 == 0:
                measurements = quantum_register.measure(num_measurements=10)
                
                for measurement in measurements:
                    energy = self._calculate_measurement_energy(measurement, hamiltonian)
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_state = measurement
                        best_measurement_history = measurements
            
            # Track progress
            self.energy_history.append(best_energy)
            self.temperature_schedule.append(temperature)
        
        # Final measurement and solution extraction
        final_measurements = quantum_register.measure(num_measurements=100)
        best_team_indices = self._extract_team_from_state(best_state, players)
        
        # Calculate solution metrics
        quantum_score = -best_energy  # Convert back to maximization
        classical_score = sum(players[i].get('final_score', 0) for i in best_team_indices if i < len(players))
        entanglement_score = quantum_register.get_entanglement_entropy()
        coherence_factor = self._calculate_coherence_factor(quantum_register)
        measurement_confidence = self._calculate_measurement_confidence(final_measurements, best_state)
        
        return QuantumTeamSolution(
            team_indices=best_team_indices,
            quantum_score=quantum_score,
            classical_score=classical_score,
            entanglement_score=entanglement_score,
            coherence_factor=coherence_factor,
            measurement_confidence=measurement_confidence
        )
    
    def _apply_quantum_evolution(self, quantum_register: QuantumRegister, 
                               hamiltonian: np.ndarray, gamma: float, delta: float, temperature: float):
        """Apply quantum evolution step"""
        
        # Apply transverse field (maintains superposition)
        if gamma > 0:
            for qubit in range(quantum_register.num_qubits):
                # Small random rotation to maintain quantum coherence
                theta = gamma * np.random.normal(0, 0.1)
                quantum_register.apply_rotation(qubit, theta)
        
        # Apply problem Hamiltonian evolution
        if delta > 0:
            # Simplified evolution - in practice would use matrix exponentiation
            evolution_strength = delta / temperature if temperature > 0 else delta
            
            # Apply controlled operations based on problem structure
            for i in range(min(3, quantum_register.num_qubits - 1)):
                for j in range(i + 1, min(i + 3, quantum_register.num_qubits)):
                    if np.random.random() < evolution_strength * 0.1:
                        quantum_register.apply_controlled_not(i, j)
        
        # Add thermal noise
        if temperature > 0:
            noise_strength = temperature * 0.01
            for qubit in range(quantum_register.num_qubits):
                if np.random.random() < noise_strength:
                    noise_angle = np.random.uniform(0, 2 * np.pi)
                    quantum_register.apply_rotation(qubit, noise_angle * 0.1)
    
    def _calculate_measurement_energy(self, state: int, hamiltonian: np.ndarray) -> float:
        """Calculate energy of measured state"""
        if state < hamiltonian.shape[0]:
            return hamiltonian[state, state].real
        return float('inf')
    
    def _extract_team_from_state(self, state: int, players: List[Dict[str, Any]]) -> List[int]:
        """Extract team indices from quantum state"""
        team_indices = []
        
        for i in range(min(len(players), self.num_qubits)):
            if self._get_bit(state, i) == 1:
                team_indices.append(i)
        
        # Ensure team size constraints
        if len(team_indices) > self.team_size:
            # Keep top players by score
            team_indices.sort(key=lambda i: players[i].get('final_score', 0) if i < len(players) else 0, reverse=True)
            team_indices = team_indices[:self.team_size]
        elif len(team_indices) < self.team_size:
            # Add remaining best players
            available_players = [i for i in range(len(players)) if i not in team_indices]
            available_players.sort(key=lambda i: players[i].get('final_score', 0), reverse=True)
            
            needed = self.team_size - len(team_indices)
            team_indices.extend(available_players[:needed])
        
        return team_indices
    
    def _calculate_coherence_factor(self, quantum_register: QuantumRegister) -> float:
        """Calculate quantum coherence factor"""
        state_vector = quantum_register.state_vector
        coherence = np.sum(np.abs(state_vector[state_vector != 0])) / len(state_vector)
        return min(1.0, coherence)
    
    def _calculate_measurement_confidence(self, measurements: List[int], best_state: int) -> float:
        """Calculate confidence in measurement results"""
        if not measurements:
            return 0.0
        
        # Count occurrences of best state
        best_state_count = measurements.count(best_state)
        confidence = best_state_count / len(measurements)
        
        return confidence
    
    def _get_bit(self, number: int, position: int) -> int:
        """Get bit at position"""
        return (number >> position) & 1

class QuantumInspiredGeneticAlgorithm:
    """Quantum-inspired genetic algorithm for team optimization"""
    
    def __init__(self, population_size: int = 50, num_generations: int = 100):
        self.population_size = population_size
        self.num_generations = num_generations
        
        # Quantum-inspired parameters
        self.rotation_angle = np.pi / 6  # 30 degrees
        self.mutation_probability = 0.1
        self.quantum_crossover_probability = 0.8
        
        # Population tracking
        self.quantum_population = []
        self.classical_population = []
        self.fitness_history = []
    
    def optimize(self, players: List[Dict[str, Any]], 
                constraints: Dict[str, Any]) -> List[QuantumTeamSolution]:
        """Optimize using quantum-inspired genetic algorithm"""
        
        print("🧬 Starting Quantum-Inspired Genetic Algorithm...")
        
        # Initialize quantum population
        self._initialize_quantum_population(len(players))
        
        best_solutions = []
        
        for generation in range(self.num_generations):
            # Measure quantum population to get classical candidates
            classical_candidates = self._measure_quantum_population()
            
            # Evaluate fitness
            fitness_scores = self._evaluate_population_fitness(classical_candidates, players, constraints)
            
            # Update quantum population based on fitness
            self._update_quantum_population(fitness_scores)
            
            # Apply quantum operations
            self._apply_quantum_crossover()
            self._apply_quantum_mutation()
            
            # Track best solution
            best_idx = np.argmax(fitness_scores)
            best_solution = self._create_quantum_solution(
                classical_candidates[best_idx], players, fitness_scores[best_idx]
            )
            best_solutions.append(best_solution)
            
            # Progress tracking
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {fitness_scores[best_idx]:.2f}")
            
            self.fitness_history.append(max(fitness_scores))
        
        print("✅ Quantum-Inspired GA optimization complete")
        
        # Return top solutions
        best_solutions.sort(key=lambda x: x.quantum_score, reverse=True)
        return best_solutions[:5]
    
    def _initialize_quantum_population(self, num_players: int):
        """Initialize quantum population in superposition"""
        self.quantum_population = []
        
        for _ in range(self.population_size):
            # Each individual is represented as quantum amplitudes
            num_qubits = min(num_players, 16)  # Limit for efficiency
            
            # Initialize in equal superposition
            amplitudes = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
            
            # Add small random phases
            phases = np.random.uniform(0, 2*np.pi, 2**num_qubits)
            amplitudes *= np.exp(1j * phases)
            
            quantum_individual = QuantumState(
                amplitudes=amplitudes,
                probabilities=np.abs(amplitudes)**2,
                phase_factors=phases
            )
            
            self.quantum_population.append(quantum_individual)
    
    def _measure_quantum_population(self) -> List[List[int]]:
        """Measure quantum population to get classical bit strings"""
        classical_candidates = []
        
        for quantum_individual in self.quantum_population:
            # Measure quantum state
            probabilities = quantum_individual.probabilities
            measured_state = np.random.choice(len(probabilities), p=probabilities)
            
            # Convert to bit string (team selection)
            num_qubits = int(np.log2(len(probabilities)))
            bit_string = []
            
            for i in range(num_qubits):
                bit = (measured_state >> i) & 1
                bit_string.append(bit)
            
            classical_candidates.append(bit_string)
        
        return classical_candidates
    
    def _evaluate_population_fitness(self, classical_candidates: List[List[int]], 
                                   players: List[Dict[str, Any]], 
                                   constraints: Dict[str, Any]) -> List[float]:
        """Evaluate fitness of classical candidates"""
        fitness_scores = []
        
        for candidate in classical_candidates:
            # Extract team from bit string
            team_indices = [i for i, bit in enumerate(candidate) if bit == 1 and i < len(players)]
            
            # Calculate fitness
            fitness = self._calculate_team_fitness(team_indices, players, constraints)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _calculate_team_fitness(self, team_indices: List[int], 
                              players: List[Dict[str, Any]], 
                              constraints: Dict[str, Any]) -> float:
        """Calculate fitness score for a team"""
        
        if not team_indices:
            return 0.0
        
        # Base score from player performance
        base_score = sum(players[i].get('final_score', 0) for i in team_indices if i < len(players))
        
        # Constraint penalties
        constraint_penalty = 0.0
        
        # Team size penalty
        size_diff = abs(len(team_indices) - 11)
        constraint_penalty += size_diff * 10
        
        # Role constraints
        role_counts = defaultdict(int)
        total_credits = 0.0
        
        for idx in team_indices:
            if idx < len(players):
                player = players[idx]
                role = player.get('role', '').lower()
                
                if 'bat' in role and 'allrounder' not in role:
                    role_counts['batsman'] += 1
                elif 'bowl' in role and 'allrounder' not in role:
                    role_counts['bowler'] += 1
                elif 'allrounder' in role:
                    role_counts['allrounder'] += 1
                elif 'wk' in role or 'wicket' in role:
                    role_counts['wicketkeeper'] += 1
                
                total_credits += player.get('credits', 8.5)
        
        # Role penalties
        constraint_penalty += max(0, 3 - role_counts['batsman']) * 5
        constraint_penalty += max(0, role_counts['batsman'] - 6) * 5
        constraint_penalty += max(0, 3 - role_counts['bowler']) * 5
        constraint_penalty += max(0, role_counts['bowler'] - 6) * 5
        constraint_penalty += max(0, 1 - role_counts['wicketkeeper']) * 10
        constraint_penalty += max(0, role_counts['wicketkeeper'] - 2) * 10
        
        # Credit penalty
        constraint_penalty += max(0, total_credits - constraints.get('max_credits', 100)) * 2
        
        fitness = base_score - constraint_penalty
        
        return max(0, fitness)
    
    def _update_quantum_population(self, fitness_scores: List[float]):
        """Update quantum population based on fitness"""
        
        # Normalize fitness scores
        max_fitness = max(fitness_scores) if fitness_scores else 1.0
        normalized_fitness = [f / max_fitness for f in fitness_scores]
        
        # Update quantum amplitudes based on fitness
        for i, (quantum_individual, fitness) in enumerate(zip(self.quantum_population, normalized_fitness)):
            # Rotation angle based on fitness
            theta = self.rotation_angle * fitness
            
            # Apply rotation to amplitudes
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=complex)
            
            # Update amplitudes (simplified rotation)
            for j in range(0, len(quantum_individual.amplitudes), 2):
                if j + 1 < len(quantum_individual.amplitudes):
                    amp_pair = np.array([
                        quantum_individual.amplitudes[j],
                        quantum_individual.amplitudes[j + 1]
                    ])
                    
                    rotated_pair = rotation_matrix @ amp_pair
                    quantum_individual.amplitudes[j] = rotated_pair[0]
                    quantum_individual.amplitudes[j + 1] = rotated_pair[1]
            
            # Renormalize and update probabilities
            norm = np.linalg.norm(quantum_individual.amplitudes)
            if norm > 0:
                quantum_individual.amplitudes /= norm
                quantum_individual.probabilities = np.abs(quantum_individual.amplitudes)**2
    
    def _apply_quantum_crossover(self):
        """Apply quantum crossover operations"""
        
        for i in range(0, len(self.quantum_population) - 1, 2):
            if np.random.random() < self.quantum_crossover_probability:
                parent1 = self.quantum_population[i]
                parent2 = self.quantum_population[i + 1]
                
                # Quantum crossover: interference between amplitudes
                alpha = np.random.uniform(0.3, 0.7)
                
                new_amplitudes1 = alpha * parent1.amplitudes + (1 - alpha) * parent2.amplitudes
                new_amplitudes2 = (1 - alpha) * parent1.amplitudes + alpha * parent2.amplitudes
                
                # Normalize
                new_amplitudes1 /= np.linalg.norm(new_amplitudes1)
                new_amplitudes2 /= np.linalg.norm(new_amplitudes2)
                
                # Update quantum states
                parent1.amplitudes = new_amplitudes1
                parent1.probabilities = np.abs(new_amplitudes1)**2
                
                parent2.amplitudes = new_amplitudes2
                parent2.probabilities = np.abs(new_amplitudes2)**2
    
    def _apply_quantum_mutation(self):
        """Apply quantum mutation operations"""
        
        for quantum_individual in self.quantum_population:
            if np.random.random() < self.mutation_probability:
                # Phase mutation
                phase_noise = np.random.uniform(-np.pi/4, np.pi/4, len(quantum_individual.amplitudes))
                quantum_individual.amplitudes *= np.exp(1j * phase_noise)
                
                # Amplitude perturbation
                amplitude_noise = np.random.normal(0, 0.1, len(quantum_individual.amplitudes))
                quantum_individual.amplitudes += amplitude_noise
                
                # Renormalize
                norm = np.linalg.norm(quantum_individual.amplitudes)
                if norm > 0:
                    quantum_individual.amplitudes /= norm
                    quantum_individual.probabilities = np.abs(quantum_individual.amplitudes)**2
    
    def _create_quantum_solution(self, bit_string: List[int], 
                               players: List[Dict[str, Any]], 
                               fitness: float) -> QuantumTeamSolution:
        """Create quantum solution from bit string"""
        
        team_indices = [i for i, bit in enumerate(bit_string) if bit == 1 and i < len(players)]
        
        # Ensure proper team size
        if len(team_indices) > 11:
            team_indices.sort(key=lambda i: players[i].get('final_score', 0), reverse=True)
            team_indices = team_indices[:11]
        elif len(team_indices) < 11:
            available = [i for i in range(len(players)) if i not in team_indices]
            available.sort(key=lambda i: players[i].get('final_score', 0), reverse=True)
            team_indices.extend(available[:11-len(team_indices)])
        
        classical_score = sum(players[i].get('final_score', 0) for i in team_indices if i < len(players))
        
        return QuantumTeamSolution(
            team_indices=team_indices,
            quantum_score=fitness,
            classical_score=classical_score,
            entanglement_score=0.0,  # Not calculated in GA
            coherence_factor=0.8,    # Assumed coherence
            measurement_confidence=0.9  # High confidence for GA
        )

# Utility functions
def optimize_team_with_quantum_annealing(players: List[Dict[str, Any]], 
                                       constraints: Dict[str, Any],
                                       objectives: Dict[str, float]) -> QuantumTeamSolution:
    """Optimize team using quantum annealing"""
    optimizer = QuantumAnnealingOptimizer(len(players))
    return optimizer.optimize_team_selection(players, constraints, objectives)

def optimize_team_with_quantum_ga(players: List[Dict[str, Any]], 
                                constraints: Dict[str, Any]) -> List[QuantumTeamSolution]:
    """Optimize team using quantum-inspired genetic algorithm"""
    optimizer = QuantumInspiredGeneticAlgorithm()
    return optimizer.optimize(players, constraints)

# Export
__all__ = ['QuantumAnnealingOptimizer', 'QuantumInspiredGeneticAlgorithm', 'QuantumTeamSolution',
           'QuantumState', 'QuantumRegister', 'optimize_team_with_quantum_annealing',
           'optimize_team_with_quantum_ga']