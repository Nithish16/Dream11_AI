#!/usr/bin/env python3
"""
Format-Specific Analysis Engine - Advanced Cricket Format Intelligence
Implements Phase 1, 2, and 3 format-specific enhancements for T20, ODI, and TEST cricket
"""

import numpy as np
import pandas as pd
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import math
from abc import ABC, abstractmethod

# Try advanced ML imports
try:
    import tensorflow as tf
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, transpile, assemble
    from qiskit.providers.aer import QasmSimulator
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

@dataclass
class FormatSpecificWeights:
    """Enhanced format-specific weight configurations"""
    # Core weights
    ema_score: float
    consistency: float
    opportunity_index: float
    form_momentum: float
    role_bonus: float
    
    # Advanced weights
    boundary_percentage: float = 0.0
    strike_rate_weight: float = 0.0
    average_weight: float = 0.0
    pressure_handling: float = 0.0
    partnership_building: float = 0.0
    death_overs_skill: float = 0.0
    power_play_skill: float = 0.0
    middle_overs_skill: float = 0.0
    temperament_score: float = 0.0
    session_adaptability: float = 0.0

@dataclass 
class FormatContext:
    """Context-specific information for each format"""
    format_type: str
    typical_duration: int  # minutes
    overs_per_innings: int
    max_overs_per_bowler: int
    powerplay_overs: int
    death_overs_start: int
    key_phases: List[str]
    critical_skills: List[str]
    environment_sensitivity: float  # 0-1 how much environment matters

# =============================================================================
# PHASE 1: ENHANCED FORMAT-SPECIFIC ANALYSIS
# =============================================================================

class FormatSpecificAnalyzer:
    """Enhanced format-specific analysis with advanced environmental intelligence"""
    
    def __init__(self):
        self.format_contexts = self._initialize_format_contexts()
        self.format_weights = self._initialize_format_weights()
        self.environmental_multipliers = self._initialize_environmental_multipliers()
    
    def _initialize_format_contexts(self) -> Dict[str, FormatContext]:
        """Initialize comprehensive format contexts"""
        return {
            'T20': FormatContext(
                format_type='T20',
                typical_duration=200,  # 3.5 hours
                overs_per_innings=20,
                max_overs_per_bowler=4,
                powerplay_overs=6,
                death_overs_start=16,
                key_phases=['powerplay', 'middle_overs', 'death_overs'],
                critical_skills=['explosive_batting', 'death_bowling', 'fielding_agility'],
                environment_sensitivity=0.8  # High impact of dew, conditions
            ),
            'ODI': FormatContext(
                format_type='ODI',
                typical_duration=480,  # 8 hours
                overs_per_innings=50,
                max_overs_per_bowler=10,
                powerplay_overs=10,
                death_overs_start=41,
                key_phases=['powerplay', 'middle_overs', 'death_overs', 'partnerships'],
                critical_skills=['sustained_batting', 'middle_overs_bowling', 'partnership_building'],
                environment_sensitivity=0.6  # Moderate impact
            ),
            'TEST': FormatContext(
                format_type='TEST',
                typical_duration=2400,  # 5 days
                overs_per_innings=0,  # Unlimited
                max_overs_per_bowler=0,  # Unlimited
                powerplay_overs=0,
                death_overs_start=0,
                key_phases=['session1', 'session2', 'session3', 'day1-5'],
                critical_skills=['patience', 'technique', 'reverse_swing', 'spin_bowling'],
                environment_sensitivity=1.0  # Maximum impact over 5 days
            )
        }
    
    def _initialize_format_weights(self) -> Dict[str, FormatSpecificWeights]:
        """Initialize enhanced format-specific weights"""
        return {
            'T20': FormatSpecificWeights(
                # Core weights (explosive focus)
                ema_score=0.20,
                consistency=0.10,  # Lower for T20
                opportunity_index=0.35,  # Highest - risk appetite
                form_momentum=0.25,
                role_bonus=0.10,
                
                # Advanced weights (T20 specific)
                boundary_percentage=0.25,  # Critical in T20
                strike_rate_weight=0.30,   # Most important
                average_weight=0.05,       # Less important
                pressure_handling=0.20,    # Death overs pressure
                death_overs_skill=0.30,    # Specialist skill
                power_play_skill=0.25,     # Powerplay importance
                middle_overs_skill=0.15,   # Least important phase
                temperament_score=0.05     # Less critical
            ),
            
            'ODI': FormatSpecificWeights(
                # Core weights (balanced approach)
                ema_score=0.30,
                consistency=0.25,
                opportunity_index=0.20,
                form_momentum=0.15,
                role_bonus=0.10,
                
                # Advanced weights (ODI specific)
                boundary_percentage=0.15,  # Moderate importance
                strike_rate_weight=0.20,   # Balanced importance
                average_weight=0.25,       # More important than T20
                pressure_handling=0.25,    # Partnership pressure
                partnership_building=0.30, # Critical for ODI
                death_overs_skill=0.20,    # Important but not critical
                power_play_skill=0.15,     # Foundation setting
                middle_overs_skill=0.30,   # Most critical phase
                temperament_score=0.15     # Building partnerships
            ),
            
            'TEST': FormatSpecificWeights(
                # Core weights (consistency focus)
                ema_score=0.15,
                consistency=0.40,  # Highest weight
                opportunity_index=0.20,
                form_momentum=0.05,  # Lowest - long-term view
                role_bonus=0.20,  # Higher for specialist roles
                
                # Advanced weights (TEST specific)
                boundary_percentage=0.05,  # Least important
                strike_rate_weight=0.10,   # Patience valued more
                average_weight=0.35,       # Most critical
                pressure_handling=0.15,    # Session pressure
                partnership_building=0.35, # Most critical
                session_adaptability=0.30, # Adapting to conditions
                temperament_score=0.40     # Most important for TEST
            )
        }
    
    def _initialize_environmental_multipliers(self) -> Dict[str, Dict[str, float]]:
        """Initialize environmental impact multipliers per format"""
        return {
            'T20': {
                'dew_factor': 1.3,      # High impact on T20 evening games
                'pitch_pace': 1.2,      # Fast pitches favor T20
                'boundary_size': 1.4,   # Critical for T20
                'crowd_pressure': 1.2,  # High energy format
                'temperature': 1.1      # Affects player stamina
            },
            'ODI': {
                'dew_factor': 1.1,      # Moderate impact
                'pitch_deterioration': 1.3,  # 50 overs wear
                'weather_changes': 1.2,      # Day-long games
                'pitch_pace': 1.1,           # Moderate impact
                'partnership_conditions': 1.2 # Building partnerships
            },
            'TEST': {
                'pitch_evolution': 1.5,      # Maximum impact over 5 days
                'weather_interruptions': 1.4, # Long format most affected
                'session_conditions': 1.3,    # Morning/afternoon/evening
                'reverse_swing_conditions': 1.2, # Old ball conditions
                'spin_conditions': 1.3       # Day 4-5 spin
            }
        }
    
    def analyze_format_specific_performance(self, player_features, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Comprehensive format-specific performance analysis"""
        format_type = match_context.get('match_format', 'T20').upper()
        
        # Get format-specific weights and context
        weights = self.format_weights.get(format_type, self.format_weights['T20'])
        context = self.format_contexts.get(format_type, self.format_contexts['T20'])
        
        # Calculate base format score
        base_score = self._calculate_base_format_score(player_features, weights)
        
        # Apply environmental intelligence
        environmental_score = self._calculate_environmental_impact(
            player_features, match_context, format_type
        )
        
        # Calculate advanced format-specific metrics
        advanced_metrics = self._calculate_advanced_format_metrics(
            player_features, format_type, match_context
        )
        
        # Phase-specific analysis
        phase_analysis = self._analyze_format_phases(
            player_features, format_type, match_context
        )
        
        return {
            'base_score': base_score,
            'environmental_score': environmental_score,
            'advanced_metrics': advanced_metrics,
            'phase_analysis': phase_analysis,
            'final_format_score': (base_score * 0.4 + environmental_score * 0.3 + 
                                 advanced_metrics * 0.2 + phase_analysis * 0.1)
        }
    
    def _calculate_base_format_score(self, player_features, weights: FormatSpecificWeights) -> float:
        """Calculate base format-specific score with enhanced weights"""
        # Core components
        ema_component = getattr(player_features, 'ema_score', 0) * weights.ema_score
        consistency_component = getattr(player_features, 'consistency_score', 0) * weights.consistency
        opportunity_component = getattr(player_features, 'dynamic_opportunity_index', 1) * 20 * weights.opportunity_index
        momentum_component = (getattr(player_features, 'form_momentum', 0) + 1) * 10 * weights.form_momentum
        
        # Advanced components (with fallbacks for missing attributes)
        boundary_component = getattr(player_features, 'boundary_percentage', 0.15) * 100 * weights.boundary_percentage
        strike_rate_component = getattr(player_features, 'strike_rate_factor', 1.0) * 50 * weights.strike_rate_weight
        average_component = getattr(player_features, 'batting_average_factor', 1.0) * 30 * weights.average_weight
        pressure_component = getattr(player_features, 'pressure_handling_score', 0.5) * 50 * weights.pressure_handling
        
        # Role-specific bonus
        role_bonus = self._calculate_format_role_bonus(player_features.role, weights)
        
        base_score = (ema_component + consistency_component + opportunity_component + 
                     momentum_component + boundary_component + strike_rate_component + 
                     average_component + pressure_component + role_bonus)
        
        return round(base_score, 2)
    
    def _calculate_environmental_impact(self, player_features, match_context: Dict[str, Any], format_type: str) -> float:
        """Calculate environmental impact on player performance"""
        multipliers = self.environmental_multipliers.get(format_type, {})
        context = self.format_contexts[format_type]
        
        environmental_score = getattr(player_features, 'ema_score', 50)  # Base score
        
        # Apply format-specific environmental factors
        if format_type == 'T20':
            # Dew factor (critical for T20 evening games)
            if match_context.get('match_time', 'evening') == 'evening':
                dew_impact = multipliers.get('dew_factor', 1.0)
                if 'bowl' in player_features.role.lower():
                    environmental_score *= (2.0 - dew_impact)  # Bowlers affected negatively
                else:
                    environmental_score *= dew_impact  # Batsmen benefit
            
            # Boundary size impact
            boundary_size = match_context.get('boundary_size', 'medium')
            boundary_multiplier = multipliers.get('boundary_size', 1.0)
            if boundary_size == 'small' and 'bat' in player_features.role.lower():
                environmental_score *= boundary_multiplier
        
        elif format_type == 'ODI':
            # Pitch deterioration over 50 overs
            pitch_condition = match_context.get('pitch_condition', 'good')
            if pitch_condition in ['wearing', 'deteriorating']:
                deterioration_multiplier = multipliers.get('pitch_deterioration', 1.0)
                if 'spin' in getattr(player_features, 'bowling_style', '').lower():
                    environmental_score *= deterioration_multiplier
        
        elif format_type == 'TEST':
            # 5-day pitch evolution
            match_day = match_context.get('match_day', 1)
            pitch_evolution_multiplier = multipliers.get('pitch_evolution', 1.0)
            
            if match_day >= 4:  # Day 4-5 spin conditions
                if 'spin' in getattr(player_features, 'bowling_style', '').lower():
                    environmental_score *= pitch_evolution_multiplier
            
            # Session-specific conditions
            session = match_context.get('session', 'afternoon')
            session_multiplier = multipliers.get('session_conditions', 1.0)
            
            if session == 'morning' and 'bowl' in player_features.role.lower():
                environmental_score *= session_multiplier  # Morning conditions favor bowlers
        
        return round(environmental_score * context.environment_sensitivity, 2)
    
    def _calculate_advanced_format_metrics(self, player_features, format_type: str, match_context: Dict[str, Any]) -> float:
        """Calculate advanced format-specific metrics"""
        weights = self.format_weights[format_type]
        advanced_score = 0
        
        if format_type == 'T20':
            # Death overs specialists
            death_overs_skill = getattr(player_features, 'death_overs_average', 0.5) * 100
            advanced_score += death_overs_skill * weights.death_overs_skill
            
            # Power play specialists
            power_play_skill = getattr(player_features, 'powerplay_strike_rate', 1.0) * 50
            advanced_score += power_play_skill * weights.power_play_skill
        
        elif format_type == 'ODI':
            # Partnership building
            partnership_skill = getattr(player_features, 'partnership_contribution', 0.5) * 100
            advanced_score += partnership_skill * weights.partnership_building
            
            # Middle overs mastery
            middle_overs_skill = getattr(player_features, 'middle_overs_performance', 0.5) * 100
            advanced_score += middle_overs_skill * weights.middle_overs_skill
        
        elif format_type == 'TEST':
            # Session adaptability
            session_adaptability = getattr(player_features, 'session_average_variance', 0.8) * 100
            advanced_score += session_adaptability * weights.session_adaptability
            
            # Temperament score (most critical for TEST)
            temperament = getattr(player_features, 'temperament_rating', 0.6) * 100
            advanced_score += temperament * weights.temperament_score
        
        return round(advanced_score, 2)
    
    def _analyze_format_phases(self, player_features, format_type: str, match_context: Dict[str, Any]) -> float:
        """Analyze player performance in different phases of the format"""
        context = self.format_contexts[format_type]
        phase_score = 0
        
        for phase in context.key_phases:
            phase_performance = getattr(player_features, f'{phase}_Performance', 0.5)
            phase_score += phase_performance * 20  # Normalize to 0-100 scale
        
        return round(phase_score / len(context.key_phases), 2)
    
    def _calculate_format_role_bonus(self, role: str, weights: FormatSpecificWeights) -> float:
        """Calculate format-specific role bonuses"""
        role_lower = role.lower()
        base_bonus = weights.role_bonus
        
        # Format-specific role preferences built into weights
        if 'allrounder' in role_lower:
            return base_bonus * 5  # All-rounders valuable in all formats
        elif 'wk' in role_lower or 'wicket' in role_lower:
            return base_bonus * 3  # Wicket-keepers provide flexibility
        elif 'bowl' in role_lower:
            return base_bonus * 4  # Bowlers critical for taking wickets
        else:  # Batsmen
            return base_bonus * 3  # Consistent run scoring
    
    def get_format_recommendation(self, player_features, all_formats: List[str] = ['T20', 'ODI', 'TEST']) -> Dict[str, Any]:
        """Recommend best format for a player based on their skills"""
        format_scores = {}
        
        for format_type in all_formats:
            match_context = {'match_format': format_type}
            analysis = self.analyze_format_specific_performance(player_features, match_context)
            format_scores[format_type] = analysis['final_format_score']
        
        best_format = max(format_scores, key=format_scores.get)
        
        return {
            'best_format': best_format,
            'format_scores': format_scores,
            'specialization_level': max(format_scores.values()) - min(format_scores.values()),
            'is_specialist': max(format_scores.values()) - min(format_scores.values()) > 20
        }

# =============================================================================
# PHASE 2: FORMAT-SPECIFIC NEURAL NETWORKS & MONTE CARLO
# =============================================================================

class FormatSpecificNeuralEngine:
    """Phase 2: Advanced neural networks for each format with Monte Carlo simulations"""
    
    def __init__(self):
        self.models = {}
        self.simulation_engines = {}
        self._initialize_neural_models()
        self._initialize_simulation_engines()
    
    def _initialize_neural_models(self):
        """Initialize format-specific neural network models"""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available, using simplified models")
            self.models = {
                'T20': self._create_simple_model('explosive'),
                'ODI': self._create_simple_model('balanced'),
                'TEST': self._create_simple_model('consistent')
            }
        
        # Create format-specific neural architectures
        self.models = {
            'T20': self._create_t20_transformer_model(),
            'ODI': self._create_odi_lstm_model(),
            'TEST': self._create_test_cnn_model()
        }
    
    def _create_simple_model(self, model_type: str):
        """Fallback simple models when TensorFlow not available"""
        class SimpleModel:
            def __init__(self, model_type):
                self.model_type = model_type
                self.weights = self._get_simple_weights()
            
            def _get_simple_weights(self):
                if self.model_type == 'explosive':
                    return {'strike_rate': 0.4, 'boundaries': 0.3, 'recent_form': 0.3}
                elif self.model_type == 'balanced':
                    return {'average': 0.3, 'consistency': 0.3, 'partnerships': 0.4}
                else:  # consistent
                    return {'average': 0.5, 'consistency': 0.4, 'temperament': 0.1}
            
            def predict(self, features):
                # Simple weighted prediction
                prediction = 0
                for feature, weight in self.weights.items():
                    feature_value = getattr(features, feature, 50)  # Default value
                    prediction += feature_value * weight
                return prediction
        
        return SimpleModel(model_type)
    
    def _create_t20_transformer_model(self):
        """T20-specific Transformer model focusing on explosive performance"""
        if not TF_AVAILABLE:
            return self._create_simple_model('explosive')
        
        # Transformer architecture for T20 (attention on power-play & death overs)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),  # 20 features
            tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # 0-1 explosive potential
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_odi_lstm_model(self):
        """ODI-specific LSTM model for sustained performance patterns"""
        if not TF_AVAILABLE:
            return self._create_simple_model('balanced')
        
        # LSTM architecture for ODI (sequential 50-over patterns)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(50, 15)),  # 50 overs, 15 features
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # 0-1 sustained performance
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_test_cnn_model(self):
        """TEST-specific CNN model for session-wise pattern recognition"""
        if not TF_AVAILABLE:
            return self._create_simple_model('consistent')
        
        # CNN architecture for TEST (convolutional patterns over sessions)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(15, 10)),  # 15 sessions, 10 features
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # 0-1 consistency score
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _initialize_simulation_engines(self):
        """Initialize Monte Carlo simulation engines for each format"""
        self.simulation_engines = {
            'T20': MonteCarloT20Simulator(),
            'ODI': MonteCarloODISimulator(),
            'TEST': MonteCarloTESTSimulator()
        }
    
    def predict_format_performance(self, player_features, format_type: str, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance using format-specific neural networks"""
        model = self.models.get(format_type, self.models['T20'])
        
        # Prepare features for the model
        feature_vector = self._prepare_neural_features(player_features, format_type)
        
        # Get neural prediction
        if hasattr(model, 'predict'):
            neural_prediction = model.predict(feature_vector)
        else:
            neural_prediction = model.predict(player_features)
        
        # Run Monte Carlo simulation
        simulation_results = self.simulation_engines[format_type].simulate_performance(
            player_features, match_context, iterations=1000
        )
        
        return {
            'neural_prediction': float(neural_prediction) if isinstance(neural_prediction, (int, float)) else float(neural_prediction[0]),
            'simulation_mean': simulation_results['mean_performance'],
            'simulation_std': simulation_results['std_performance'],
            'confidence_interval': simulation_results['confidence_interval'],
            'upside_potential': simulation_results['upside_potential'],
            'downside_risk': simulation_results['downside_risk']
        }
    
    def _prepare_neural_features(self, player_features, format_type: str) -> np.ndarray:
        """Prepare features for neural network input"""
        # Extract relevant features based on format
        features = []
        
        # Common features
        features.extend([
            getattr(player_features, 'ema_score', 50),
            getattr(player_features, 'consistency_score', 50),
            getattr(player_features, 'form_momentum', 0),
            getattr(player_features, 'dynamic_opportunity_index', 1),
        ])
        
        # Format-specific features
        if format_type == 'T20':
            features.extend([
                getattr(player_features, 'strike_rate_factor', 1.0) * 100,
                getattr(player_features, 'boundary_percentage', 0.15) * 100,
                getattr(player_features, 'death_overs_average', 0.5) * 100,
                getattr(player_features, 'powerplay_strike_rate', 1.0) * 100,
            ])
        elif format_type == 'ODI':
            features.extend([
                getattr(player_features, 'batting_average_factor', 1.0) * 50,
                getattr(player_features, 'partnership_contribution', 0.5) * 100,
                getattr(player_features, 'middle_overs_performance', 0.5) * 100,
                getattr(player_features, 'pressure_handling_score', 0.5) * 100,
            ])
        else:  # TEST
            features.extend([
                getattr(player_features, 'batting_average_factor', 1.0) * 50,
                getattr(player_features, 'temperament_rating', 0.6) * 100,
                getattr(player_features, 'session_average_variance', 0.8) * 100,
                getattr(player_features, 'partnership_contribution', 0.5) * 100,
            ])
        
        # Pad or truncate to expected size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20]).reshape(1, -1)

class MonteCarloSimulator(ABC):
    """Base class for Monte Carlo simulations"""
    
    @abstractmethod
    def simulate_performance(self, player_features, match_context: Dict[str, Any], iterations: int = 1000) -> Dict[str, float]:
        pass

class MonteCarloT20Simulator(MonteCarloSimulator):
    """Monte Carlo simulator for T20 format"""
    
    def simulate_performance(self, player_features, match_context: Dict[str, Any], iterations: int = 1000) -> Dict[str, float]:
        """Simulate T20 performance scenarios"""
        performances = []
        
        for _ in range(iterations):
            # Simulate T20 match scenario
            base_performance = getattr(player_features, 'ema_score', 50)
            
            # Random factors affecting T20 performance
            dew_factor = random.uniform(0.9, 1.3) if match_context.get('match_time') == 'evening' else 1.0
            pressure_factor = random.uniform(0.8, 1.2)
            form_factor = random.uniform(0.9, 1.1 + getattr(player_features, 'form_momentum', 0) * 0.1)
            
            # T20-specific simulation
            powerplay_performance = self._simulate_powerplay(player_features) * 0.3
            middle_overs_performance = self._simulate_middle_overs(player_features) * 0.4  
            death_overs_performance = self._simulate_death_overs(player_features) * 0.3
            
            total_performance = (powerplay_performance + middle_overs_performance + death_overs_performance)
            total_performance *= dew_factor * pressure_factor * form_factor
            
            performances.append(total_performance)
        
        performances = np.array(performances)
        
        return {
            'mean_performance': float(np.mean(performances)),
            'std_performance': float(np.std(performances)),
            'confidence_interval': (float(np.percentile(performances, 5)), float(np.percentile(performances, 95))),
            'upside_potential': float(np.percentile(performances, 90)),
            'downside_risk': float(np.percentile(performances, 10))
        }
    
    def _simulate_powerplay(self, player_features) -> float:
        """Simulate powerplay performance"""
        base_score = getattr(player_features, 'ema_score', 50)
        powerplay_skill = getattr(player_features, 'powerplay_strike_rate', 1.0)
        return base_score * powerplay_skill * random.uniform(0.8, 1.4)
    
    def _simulate_middle_overs(self, player_features) -> float:
        """Simulate middle overs performance"""
        base_score = getattr(player_features, 'ema_score', 50)
        consistency = getattr(player_features, 'consistency_score', 50) / 100
        return base_score * (0.8 + consistency * 0.4) * random.uniform(0.9, 1.2)
    
    def _simulate_death_overs(self, player_features) -> float:
        """Simulate death overs performance"""
        base_score = getattr(player_features, 'ema_score', 50)
        death_skill = getattr(player_features, 'death_overs_average', 0.5)
        pressure_handling = getattr(player_features, 'pressure_handling_score', 0.5)
        return base_score * death_skill * pressure_handling * random.uniform(0.6, 1.6)

class MonteCarloODISimulator(MonteCarloSimulator):
    """Monte Carlo simulator for ODI format"""
    
    def simulate_performance(self, player_features, match_context: Dict[str, Any], iterations: int = 1000) -> Dict[str, float]:
        """Simulate ODI performance scenarios"""
        performances = []
        
        for _ in range(iterations):
            base_performance = getattr(player_features, 'ema_score', 50)
            
            # ODI-specific factors
            pitch_deterioration = random.uniform(0.95, 1.2)  # Pitch wears over 50 overs
            partnership_factor = random.uniform(0.9, 1.3)
            pressure_factor = random.uniform(0.85, 1.15)
            
            # ODI phase simulation
            powerplay_perf = self._simulate_odi_powerplay(player_features) * 0.2
            middle_overs_perf = self._simulate_odi_middle_overs(player_features) * 0.5
            death_overs_perf = self._simulate_odi_death_overs(player_features) * 0.3
            
            total_performance = (powerplay_perf + middle_overs_perf + death_overs_perf)
            total_performance *= pitch_deterioration * partnership_factor * pressure_factor
            
            performances.append(total_performance)
        
        performances = np.array(performances)
        
        return {
            'mean_performance': float(np.mean(performances)),
            'std_performance': float(np.std(performances)),
            'confidence_interval': (float(np.percentile(performances, 5)), float(np.percentile(performances, 95))),
            'upside_potential': float(np.percentile(performances, 85)),
            'downside_risk': float(np.percentile(performances, 15))
        }
    
    def _simulate_odi_powerplay(self, player_features) -> float:
        """Simulate ODI powerplay (10 overs)"""
        base_score = getattr(player_features, 'ema_score', 50)
        return base_score * random.uniform(0.9, 1.3)
    
    def _simulate_odi_middle_overs(self, player_features) -> float:
        """Simulate ODI middle overs (30 overs) - most critical"""
        base_score = getattr(player_features, 'ema_score', 50)
        partnership_skill = getattr(player_features, 'partnership_contribution', 0.5)
        middle_overs_skill = getattr(player_features, 'middle_overs_performance', 0.5)
        return base_score * partnership_skill * middle_overs_skill * random.uniform(0.8, 1.4)
    
    def _simulate_odi_death_overs(self, player_features) -> float:
        """Simulate ODI death overs (10 overs)"""
        base_score = getattr(player_features, 'ema_score', 50)
        pressure_handling = getattr(player_features, 'pressure_handling_score', 0.5)
        return base_score * pressure_handling * random.uniform(0.7, 1.5)

class MonteCarloTESTSimulator(MonteCarloSimulator):
    """Monte Carlo simulator for TEST format"""
    
    def simulate_performance(self, player_features, match_context: Dict[str, Any], iterations: int = 1000) -> Dict[str, float]:
        """Simulate TEST performance scenarios"""
        performances = []
        
        for _ in range(iterations):
            base_performance = getattr(player_features, 'ema_score', 50)
            
            # TEST-specific factors
            pitch_evolution = self._simulate_5_day_pitch_evolution()
            weather_interruptions = random.uniform(0.9, 1.1)
            session_variations = random.uniform(0.85, 1.15)
            temperament_factor = getattr(player_features, 'temperament_rating', 0.6)
            
            # Simulate 5-day performance
            total_performance = 0
            for day in range(1, 6):
                day_performance = self._simulate_test_day_performance(player_features, day, pitch_evolution)
                total_performance += day_performance
            
            total_performance /= 5  # Average over 5 days
            total_performance *= weather_interruptions * session_variations * (0.5 + temperament_factor)
            
            performances.append(total_performance)
        
        performances = np.array(performances)
        
        return {
            'mean_performance': float(np.mean(performances)),
            'std_performance': float(np.std(performances)),
            'confidence_interval': (float(np.percentile(performances, 10)), float(np.percentile(performances, 90))),
            'upside_potential': float(np.percentile(performances, 80)),
            'downside_risk': float(np.percentile(performances, 20))
        }
    
    def _simulate_5_day_pitch_evolution(self) -> Dict[int, float]:
        """Simulate how pitch changes over 5 days"""
        return {
            1: random.uniform(0.9, 1.1),   # Day 1: Fresh pitch
            2: random.uniform(0.95, 1.05), # Day 2: Slight wear
            3: random.uniform(1.0, 1.1),   # Day 3: Some deterioration
            4: random.uniform(1.1, 1.3),   # Day 4: Significant wear, spin
            5: random.uniform(1.2, 1.4)    # Day 5: Maximum deterioration
        }
    
    def _simulate_test_day_performance(self, player_features, day: int, pitch_evolution: Dict[int, float]) -> float:
        """Simulate performance for a specific day"""
        base_score = getattr(player_features, 'ema_score', 50)
        consistency = getattr(player_features, 'consistency_score', 50) / 100
        session_adaptability = getattr(player_features, 'session_average_variance', 0.8)
        
        # Day-specific factors
        pitch_factor = pitch_evolution[day]
        consistency_factor = 0.7 + consistency * 0.6  # More consistent = less variation
        
        # Session simulation (3 sessions per day)
        session_performances = []
        for session in range(3):
            session_perf = base_score * consistency_factor * session_adaptability * pitch_factor
            session_perf *= random.uniform(0.8, 1.2)  # Session variation
            session_performances.append(session_perf)
        
        return sum(session_performances) / 3

# =============================================================================
# PHASE 3: QUANTUM OPTIMIZATION & ADAPTIVE LEARNING
# =============================================================================

class QuantumFormatOptimizer:
    """Phase 3: Quantum-enhanced optimization for different formats"""
    
    def __init__(self):
        self.quantum_available = QUANTUM_AVAILABLE
        self.adaptive_learners = self._initialize_adaptive_learners()
        self.cross_format_intelligence = CrossFormatIntelligence()
    
    def _initialize_adaptive_learners(self) -> Dict[str, 'AdaptiveFormatLearner']:
        """Initialize adaptive learning systems for each format"""
        return {
            'T20': AdaptiveFormatLearner('T20'),
            'ODI': AdaptiveFormatLearner('ODI'),
            'TEST': AdaptiveFormatLearner('TEST')
        }
    
    def quantum_optimize_team(self, players: List, format_type: str, match_context: Dict[str, Any]) -> List:
        """Quantum-enhanced team optimization per format"""
        if not self.quantum_available:
            return self._classical_format_optimization(players, format_type)
        
        # Format-specific quantum algorithms
        if format_type == 'T20':
            return self._quantum_explosive_optimization(players)
        elif format_type == 'ODI':
            return self._quantum_balanced_optimization(players)
        else:  # TEST
            return self._quantum_stable_optimization(players)
    
    def _quantum_explosive_optimization(self, players: List) -> List:
        """Quantum optimization for T20 (maximize variance/upside)"""
        # Simulate quantum annealing for explosive combinations
        n_players = len(players)
        n_qubits = min(n_players, 20)  # Limit for classical simulation
        
        # Create quantum circuit for T20 optimization
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Add superposition
        qc.h(range(n_qubits))
        
        # Add entanglement for team synergy
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Add phase rotation based on explosive potential
        for i, player in enumerate(players[:n_qubits]):
            explosive_factor = getattr(player, 'boundary_percentage', 0.15) * 2 * np.pi
            qc.rz(explosive_factor, i)
        
        # Measure
        qc.measure_all()
        
        # Simulate quantum computation
        simulator = QasmSimulator()
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Select team based on quantum measurement
        best_bitstring = max(counts, key=counts.get)
        selected_indices = [i for i, bit in enumerate(best_bitstring) if bit == '1']
        
        # Ensure we have exactly 11 players
        if len(selected_indices) != 11:
            return self._classical_format_optimization(players, 'T20')
        
        return [players[i] for i in selected_indices if i < len(players)]
    
    def _quantum_balanced_optimization(self, players: List) -> List:
        """Quantum optimization for ODI (balance multiple objectives)"""
        # Use Variational Quantum Eigensolver (VQE) approach for balanced solutions
        if not self.quantum_available:
            return self._classical_format_optimization(players, 'ODI')
        
        # For now, use classical approximation of quantum balanced approach
        return self._classical_balanced_selection(players)
    
    def _quantum_stable_optimization(self, players: List) -> List:
        """Quantum optimization for TEST (minimize variance/maximize stability)"""
        # Use Grover's algorithm concept for finding stable combinations
        if not self.quantum_available:
            return self._classical_format_optimization(players, 'TEST')
        
        # Classical approximation focusing on consistency
        return self._classical_stable_selection(players)
    
    def _classical_format_optimization(self, players: List, format_type: str) -> List:
        """Classical fallback optimization"""
        if format_type == 'T20':
            # Sort by explosive potential
            sorted_players = sorted(players, key=lambda p: getattr(p, 'boundary_percentage', 0.15), reverse=True)
        elif format_type == 'ODI':
            # Sort by balanced performance
            sorted_players = sorted(players, key=lambda p: getattr(p, 'ema_score', 50) * getattr(p, 'consistency_score', 50), reverse=True)
        else:  # TEST
            # Sort by consistency
            sorted_players = sorted(players, key=lambda p: getattr(p, 'consistency_score', 50), reverse=True)
        
        return sorted_players[:11] if len(sorted_players) >= 11 else sorted_players
    
    def _classical_balanced_selection(self, players: List) -> List:
        """Classical balanced selection for ODI"""
        # Multi-objective optimization simulation
        scores = []
        for player in players:
            balance_score = (
                getattr(player, 'ema_score', 50) * 0.3 +
                getattr(player, 'consistency_score', 50) * 0.3 +
                getattr(player, 'partnership_contribution', 0.5) * 100 * 0.4
            )
            scores.append((balance_score, player))
        
        scores.sort(key=lambda x: x[0], reverse=True) 
        return [player for _, player in scores[:11]]
    
    def _classical_stable_selection(self, players: List) -> List:
        """Classical stable selection for TEST"""
        scores = []
        for player in players:
            stability_score = (
                getattr(player, 'consistency_score', 50) * 0.5 +
                getattr(player, 'temperament_rating', 0.6) * 100 * 0.3 +
                getattr(player, 'session_average_variance', 0.8) * 100 * 0.2
            )
            scores.append((stability_score, player))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [player for _, player in scores[:11]]

class AdaptiveFormatLearner:
    """Adaptive learning system for format-specific improvements"""
    
    def __init__(self, format_type: str):
        self.format_type = format_type
        self.performance_history = []
        self.weight_adjustments = {}
        self.meta_parameters = self._initialize_meta_parameters()
    
    def _initialize_meta_parameters(self) -> Dict[str, float]:
        """Initialize meta-learning parameters"""
        return {
            'learning_rate': 0.01,
            'adaptation_threshold': 0.05,
            'confidence_threshold': 0.7,
            'forgetting_factor': 0.95
        }
    
    def learn_from_results(self, predictions: List[float], actual_results: List[float], player_features: List):
        """Learn from match results and adapt"""
        if len(predictions) != len(actual_results):
            return
        
        # Calculate prediction errors
        errors = [abs(pred - actual) for pred, actual in zip(predictions, actual_results)]
        mean_error = sum(errors) / len(errors)
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'mean_error': mean_error,
            'predictions': predictions,
            'actual_results': actual_results,
            'features': player_features
        })
        
        # Adapt if error exceeds threshold
        if mean_error > self.meta_parameters['adaptation_threshold']:
            self._adapt_parameters(errors, player_features)
    
    def _adapt_parameters(self, errors: List[float], player_features: List):
        """Adapt model parameters based on errors"""
        learning_rate = self.meta_parameters['learning_rate']
        
        # Analyze which types of players had higher errors
        for i, (error, features) in enumerate(zip(errors, player_features)):
            # Identify player characteristics with high errors
            if error > sum(errors) / len(errors):  # Above average error
                # Adjust weights for this type of player
                role = getattr(features, 'role', 'unknown').lower()
                if role not in self.weight_adjustments:
                    self.weight_adjustments[role] = 1.0
                
                # Decrease weight for consistently over-predicted roles
                self.weight_adjustments[role] *= (1 - learning_rate)
    
    def get_adapted_weights(self) -> Dict[str, float]:
        """Get current adapted weights"""
        return self.weight_adjustments
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained"""
        if len(self.performance_history) < 10:
            return False
        
        # Check if recent performance is degrading
        recent_errors = [entry['mean_error'] for entry in self.performance_history[-5:]]
        older_errors = [entry['mean_error'] for entry in self.performance_history[-10:-5]]
        
        recent_mean = sum(recent_errors) / len(recent_errors)
        older_mean = sum(older_errors) / len(older_errors)
        
        return recent_mean > older_mean * 1.2  # 20% degradation

class CrossFormatIntelligence:
    """Cross-format learning and intelligence transfer"""
    
    def __init__(self):
        self.format_correlations = self._initialize_format_correlations()
        self.specialist_indicators = {}
        self.all_format_performers = []
    
    def _initialize_format_correlations(self) -> Dict[str, Dict[str, float]]:
        """Initialize correlations between formats"""
        return {
            'T20': {'ODI': 0.6, 'TEST': 0.3},  # T20 skills partially transfer
            'ODI': {'T20': 0.6, 'TEST': 0.7},  # ODI is middle ground
            'TEST': {'T20': 0.3, 'ODI': 0.7}   # TEST skills better transfer to ODI
        }
    
    def analyze_cross_format_performance(self, player_id: str, format_performances: Dict[str, float]) -> Dict[str, Any]:
        """Analyze player performance across formats"""
        formats = ['T20', 'ODI', 'TEST']
        performance_variance = np.var(list(format_performances.values()))
        performance_mean = np.mean(list(format_performances.values()))
        
        # Determine specialization
        best_format = max(format_performances, key=format_performances.get)
        specialization_level = format_performances[best_format] - performance_mean
        
        is_specialist = specialization_level > 15  # Significant advantage in one format
        is_all_format = performance_variance < 10  # Consistent across formats
        
        return {
            'specialization': {
                'is_specialist': is_specialist,
                'best_format': best_format if is_specialist else None,
                'specialization_level': specialization_level
            },
            'versatility': {
                'is_all_format_performer': is_all_format,
                'performance_variance': performance_variance,
                'consistency_rating': max(0, 100 - performance_variance * 2)
            },
            'transfer_potential': self._calculate_transfer_potential(format_performances),
            'format_recommendations': self._get_format_recommendations(format_performances)
        }
    
    def _calculate_transfer_potential(self, format_performances: Dict[str, float]) -> Dict[str, float]:
        """Calculate how performance in one format predicts others"""
        transfer_potential = {}
        
        for source_format, source_perf in format_performances.items():
            transfer_potential[source_format] = {}
            for target_format in format_performances:
                if source_format != target_format:
                    correlation = self.format_correlations[source_format][target_format]
                    expected_performance = source_perf * correlation
                    transfer_potential[source_format][target_format] = expected_performance
        
        return transfer_potential
    
    def _get_format_recommendations(self, format_performances: Dict[str, float]) -> Dict[str, str]:
        """Get recommendations for each format"""
        recommendations = {}
        
        for format_type, performance in format_performances.items():
            if performance > 80:
                recommendations[format_type] = "Highly Recommended - Elite Performance"
            elif performance > 65:
                recommendations[format_type] = "Recommended - Strong Performance"
            elif performance > 50:
                recommendations[format_type] = "Consider - Average Performance" 
            else:
                recommendations[format_type] = "Avoid - Below Par Performance"
        
        return recommendations
    
    def learn_format_patterns(self, all_player_data: List[Dict[str, Any]]):
        """Learn patterns across all players and formats"""
        # Analyze which skills transfer between formats
        for player_data in all_player_data:
            player_id = player_data.get('player_id')
            format_performances = player_data.get('format_performances', {})
            
            if len(format_performances) >= 2:  # Has multi-format data
                analysis = self.analyze_cross_format_performance(player_id, format_performances)
                
                # Update specialist indicators
                if analysis['specialization']['is_specialist']:
                    best_format = analysis['specialization']['best_format']
                    if best_format not in self.specialist_indicators:
                        self.specialist_indicators[best_format] = []
                    self.specialist_indicators[best_format].append(player_id)
                
                # Track all-format performers
                if analysis['versatility']['is_all_format_performer']:
                    self.all_format_performers.append(player_id)

# =============================================================================
# MAIN FORMAT-SPECIFIC COORDINATOR
# =============================================================================

class AdvancedFormatEngine:
    """Main coordinator for all format-specific enhancements"""
    
    def __init__(self):
        self.phase1_analyzer = FormatSpecificAnalyzer()
        self.phase2_neural = FormatSpecificNeuralEngine()
        self.phase3_quantum = QuantumFormatOptimizer()
        
        print("🚀 Advanced Format-Specific Engine Initialized")
        print("✅ Phase 1: Enhanced Format Analysis - ACTIVE")
        print("✅ Phase 2: Neural Networks & Monte Carlo - ACTIVE")  
        print("✅ Phase 3: Quantum Optimization & Adaptive Learning - ACTIVE")
    
    def comprehensive_format_analysis(self, player_features, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Complete format-specific analysis using all 3 phases"""
        format_type = match_context.get('match_format', 'T20').upper()
        
        # Phase 1: Enhanced format-specific analysis
        phase1_results = self.phase1_analyzer.analyze_format_specific_performance(
            player_features, match_context
        )
        
        # Phase 2: Neural network prediction + Monte Carlo simulation
        phase2_results = self.phase2_neural.predict_format_performance(
            player_features, format_type, match_context
        )
        
        # Phase 3: Get adaptive learning insights
        adaptive_learner = self.phase3_quantum.adaptive_learners[format_type]
        adaptive_weights = adaptive_learner.get_adapted_weights()
        
        # Cross-format intelligence
        cross_format_analysis = self.phase3_quantum.cross_format_intelligence.analyze_cross_format_performance(
            str(getattr(player_features, 'player_id', 'unknown')),
            {format_type: phase1_results['final_format_score']}
        )
        
        return {
            'format_type': format_type,
            'phase1_analysis': phase1_results,
            'phase2_neural_simulation': phase2_results,
            'phase3_adaptive_insights': {
                'adaptive_weights': adaptive_weights,
                'should_retrain': adaptive_learner.should_retrain()
            },
            'cross_format_intelligence': cross_format_analysis,
            'final_comprehensive_score': self._calculate_comprehensive_score(
                phase1_results, phase2_results, adaptive_weights
            ),
            'confidence_level': self._calculate_confidence_level(phase1_results, phase2_results),
            'format_recommendation': self._get_final_format_recommendation(
                phase1_results, phase2_results, cross_format_analysis
            )
        }
    
    def _calculate_comprehensive_score(self, phase1: Dict, phase2: Dict, adaptive_weights: Dict) -> float:
        """Calculate final comprehensive score combining all phases"""
        base_score = phase1['final_format_score']
        neural_score = phase2['neural_prediction'] * 100  # Scale to 0-100
        simulation_mean = phase2['simulation_mean']
        
        # Apply adaptive weights
        role_weight = adaptive_weights.get('role', 1.0)
        
        comprehensive_score = (
            base_score * 0.4 +           # Phase 1 weight
            neural_score * 0.3 +         # Phase 2 neural weight  
            simulation_mean * 0.3        # Phase 2 simulation weight
        ) * role_weight
        
        return round(comprehensive_score, 2)
    
    def _calculate_confidence_level(self, phase1: Dict, phase2: Dict) -> float:
        """Calculate confidence in the prediction"""
        # Lower variance in simulation = higher confidence
        simulation_std = phase2['simulation_std']
        confidence = max(0, min(100, 100 - simulation_std * 2))
        
        return round(confidence, 2)
    
    def _get_final_format_recommendation(self, phase1: Dict, phase2: Dict, cross_format: Dict) -> str:
        """Get final recommendation for this format"""
        comprehensive_score = self._calculate_comprehensive_score(phase1, phase2, {})
        
        if comprehensive_score > 80:
            return "Highly Recommended - Elite Performance Expected"
        elif comprehensive_score > 65:
            return "Recommended - Strong Performance Expected"
        elif comprehensive_score > 50:
            return "Consider - Average Performance Expected"
        else:
            return "Avoid - Below Par Performance Expected"
    
    def optimize_team_for_format(self, players: List, format_type: str, match_context: Dict[str, Any]) -> List:
        """Optimize team selection using quantum algorithms"""
        return self.phase3_quantum.quantum_optimize_team(players, format_type, match_context)
    
    def learn_from_match_results(self, format_type: str, predictions: List[float], actual_results: List[float], player_features: List):
        """Update adaptive learning systems with match results"""
        adaptive_learner = self.phase3_quantum.adaptive_learners[format_type]
        adaptive_learner.learn_from_results(predictions, actual_results, player_features)
        
        print(f"📊 Updated {format_type} adaptive learning with {len(predictions)} results")
        if adaptive_learner.should_retrain():
            print(f"🔄 {format_type} model recommended for retraining")