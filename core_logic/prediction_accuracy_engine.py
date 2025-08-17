#!/usr/bin/env python3
"""
Advanced Prediction Accuracy Engine
Enhanced prediction algorithms with machine learning and statistical analysis
"""

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class MockNumPy:
        def array(self, x):
            return x
        def mean(self, x):
            return sum(x) / len(x) if x else 0
    np = MockNumPy()
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerPerformanceMetrics:
    """Enhanced player performance analysis"""
    player_id: int
    player_name: str
    role: str
    team: str
    
    # Core statistics
    recent_scores: List[float] = field(default_factory=list)
    career_average: float = 0.0
    format_average: Dict[str, float] = field(default_factory=dict)
    venue_performance: Dict[str, float] = field(default_factory=dict)
    
    # Advanced metrics
    consistency_index: float = 0.0  # Lower variance = higher consistency
    form_momentum: float = 0.0  # Recent performance trend
    pressure_performance: float = 0.0  # Performance in crucial matches
    matchup_advantage: float = 0.0  # Performance vs specific opposition
    conditions_suitability: float = 0.0  # Performance in similar conditions
    
    # Predictive factors
    expected_points: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    selection_probability: float = 0.0
    captain_suitability: float = 0.0
    differential_factor: float = 0.0  # How unique this pick is
    
    # Historical context
    last_updated: datetime = field(default_factory=datetime.now)
    sample_size: int = 0
    data_reliability: float = 0.0

@dataclass
class PredictionModel:
    """Individual prediction model configuration"""
    model_name: str
    weight: float
    accuracy_score: float
    prediction_function: Callable
    confidence_threshold: float = 0.5
    is_enabled: bool = True

class PredictionAccuracyEngine:
    """
    Advanced prediction accuracy engine with multiple models and validation
    """
    
    def __init__(self, db_path: str = "prediction_accuracy.db"):
        self.db_path = db_path
        
        # Initialize prediction models
        self.prediction_models = {
            'ema_model': PredictionModel(
                model_name='Exponential Moving Average',
                weight=0.25,
                accuracy_score=0.72,
                prediction_function=self._ema_prediction
            ),
            'consistency_model': PredictionModel(
                model_name='Consistency-Based Prediction',
                weight=0.20,
                accuracy_score=0.68,
                prediction_function=self._consistency_prediction
            ),
            'form_momentum_model': PredictionModel(
                model_name='Form Momentum Analysis',
                weight=0.20,
                accuracy_score=0.69,
                prediction_function=self._form_momentum_prediction
            ),
            'venue_model': PredictionModel(
                model_name='Venue-Specific Analysis',
                weight=0.15,
                accuracy_score=0.66,
                prediction_function=self._venue_prediction
            ),
            'matchup_model': PredictionModel(
                model_name='Opposition Matchup Analysis',
                weight=0.10,
                accuracy_score=0.63,
                prediction_function=self._matchup_prediction
            ),
            'conditions_model': PredictionModel(
                model_name='Playing Conditions Model',
                weight=0.10,
                accuracy_score=0.61,
                prediction_function=self._conditions_prediction
            )
        }
        
        # Performance tracking
        self.model_performance_history = defaultdict(list)
        self.prediction_accuracy_log = []
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize prediction accuracy database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Player performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performance_metrics (
                player_id INTEGER,
                player_name TEXT,
                role TEXT,
                team TEXT,
                match_format TEXT,
                venue TEXT,
                recent_scores TEXT,
                career_average REAL,
                consistency_index REAL,
                form_momentum REAL,
                pressure_performance REAL,
                expected_points REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                selection_probability REAL,
                captain_suitability REAL,
                last_updated TIMESTAMP,
                PRIMARY KEY (player_id, match_format, venue)
            )
        ''')
        
        # Prediction model performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT,
                prediction_date TIMESTAMP,
                predicted_score REAL,
                actual_score REAL,
                accuracy_score REAL,
                confidence_level REAL,
                match_id TEXT,
                player_id INTEGER
            )
        ''')
        
        # Ensemble prediction results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_predictions (
                prediction_id TEXT PRIMARY KEY,
                match_id TEXT,
                player_id INTEGER,
                predicted_score REAL,
                confidence_score REAL,
                model_contributions TEXT,
                actual_score REAL,
                accuracy_achieved REAL,
                prediction_date TIMESTAMP
            )
        ''')
        
        # Feature importance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                feature_name TEXT PRIMARY KEY,
                importance_score REAL,
                correlation_with_performance REAL,
                stability_score REAL,
                last_calculated TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Prediction accuracy database initialized: {self.db_path}")
    
    def analyze_player_performance(self, player_data: Dict[str, Any], 
                                 match_context: Dict[str, Any]) -> PlayerPerformanceMetrics:
        """
        Comprehensive player performance analysis
        """
        player_id = player_data.get('player_id', 0)
        player_name = player_data.get('name', 'Unknown')
        role = player_data.get('role', 'Unknown')
        team = player_data.get('team', 'Unknown')
        
        # Initialize metrics
        metrics = PlayerPerformanceMetrics(
            player_id=player_id,
            player_name=player_name,
            role=role,
            team=team
        )
        
        # Extract recent scores
        recent_scores = player_data.get('recent_scores', [])
        if recent_scores:
            metrics.recent_scores = recent_scores
            metrics.career_average = statistics.mean(recent_scores)
            metrics.sample_size = len(recent_scores)
        
        # Calculate advanced metrics
        metrics.consistency_index = self._calculate_consistency_index(recent_scores)
        metrics.form_momentum = self._calculate_form_momentum(recent_scores)
        metrics.pressure_performance = self._calculate_pressure_performance(player_data, match_context)
        metrics.conditions_suitability = self._calculate_conditions_suitability(player_data, match_context)
        
        # Generate predictions using ensemble
        predictions = self._generate_ensemble_prediction(metrics, match_context)
        metrics.expected_points = predictions['expected_points']
        metrics.confidence_interval = predictions['confidence_interval']
        metrics.selection_probability = predictions['selection_probability']
        metrics.captain_suitability = predictions['captain_suitability']
        
        # Calculate reliability score
        metrics.data_reliability = self._calculate_data_reliability(metrics)
        
        return metrics
    
    def _calculate_consistency_index(self, scores: List[float]) -> float:
        """Calculate consistency index (lower variance = higher consistency)"""
        if len(scores) < 3:
            return 0.5  # Default for insufficient data
        
        try:
            mean_score = statistics.mean(scores)
            variance = statistics.variance(scores)
            
            if mean_score <= 0:
                return 0.0
            
            # Coefficient of variation (inverted for consistency)
            cv = math.sqrt(variance) / mean_score
            consistency = max(0.0, 1.0 - min(cv, 1.0))
            
            return consistency
            
        except Exception as e:
            logger.error(f"❌ Error calculating consistency index: {e}")
            return 0.5
    
    def _calculate_form_momentum(self, scores: List[float]) -> float:
        """Calculate form momentum using weighted recent performance"""
        if len(scores) < 2:
            return 0.0
        
        try:
            # Use exponential weights (recent games matter more)
            weights = [0.4, 0.3, 0.2, 0.1] if len(scores) >= 4 else [0.5, 0.3, 0.2][:len(scores)]
            
            recent_scores = scores[-len(weights):]
            weighted_recent = sum(score * weight for score, weight in zip(recent_scores, weights))
            
            # Compare with overall average
            overall_avg = statistics.mean(scores)
            
            if overall_avg <= 0:
                return 0.0
            
            momentum = (weighted_recent - overall_avg) / overall_avg
            return max(-1.0, min(1.0, momentum))  # Clamp between -1 and 1
            
        except Exception as e:
            logger.error(f"❌ Error calculating form momentum: {e}")
            return 0.0
    
    def _calculate_pressure_performance(self, player_data: Dict[str, Any], 
                                      match_context: Dict[str, Any]) -> float:
        """Calculate performance in high-pressure situations"""
        # This would analyze performance in crucial matches, finals, etc.
        # For now, using a simplified approach based on recent performance consistency
        
        recent_scores = player_data.get('recent_scores', [])
        if len(recent_scores) < 3:
            return 0.5
        
        # High-pressure matches typically show higher variance
        # Players who maintain consistency under pressure score higher
        consistency = self._calculate_consistency_index(recent_scores)
        
        # Adjust based on match importance (if available)
        match_importance = match_context.get('importance', 0.5)  # 0-1 scale
        
        pressure_score = consistency * (1 + match_importance * 0.2)
        return min(1.0, pressure_score)
    
    def _calculate_conditions_suitability(self, player_data: Dict[str, Any], 
                                        match_context: Dict[str, Any]) -> float:
        """Calculate suitability for current match conditions"""
        # Analyze historical performance in similar conditions
        venue = match_context.get('venue', '')
        format_type = match_context.get('format', 'Unknown')
        
        # Venue performance
        venue_performance = player_data.get('venue_performance', {})
        venue_score = venue_performance.get(venue, 0.5)
        
        # Format performance
        format_performance = player_data.get('format_performance', {})
        format_score = format_performance.get(format_type, 0.5)
        
        # Combine scores
        conditions_score = (venue_score * 0.6 + format_score * 0.4)
        return max(0.0, min(1.0, conditions_score))
    
    def _calculate_data_reliability(self, metrics: PlayerPerformanceMetrics) -> float:
        """Calculate reliability of the data for predictions"""
        sample_size = metrics.sample_size
        
        # Reliability increases with sample size
        size_reliability = min(1.0, sample_size / 20)  # Max reliability at 20+ matches
        
        # Recency factor (newer data is more reliable)
        days_since_update = (datetime.now() - metrics.last_updated).days
        recency_reliability = max(0.1, 1.0 - days_since_update / 30)  # Decays over 30 days
        
        # Combined reliability
        overall_reliability = (size_reliability * 0.7 + recency_reliability * 0.3)
        return overall_reliability
    
    def _generate_ensemble_prediction(self, metrics: PlayerPerformanceMetrics, 
                                    match_context: Dict[str, Any]) -> Dict[str, float]:
        """Generate ensemble prediction using multiple models"""
        
        predictions = {}
        total_weight = 0
        weighted_sum = 0
        confidence_scores = []
        
        for model_name, model in self.prediction_models.items():
            if not model.is_enabled:
                continue
            
            try:
                # Get prediction from individual model
                model_prediction = model.prediction_function(metrics, match_context)
                
                # Weight by model accuracy and confidence
                effective_weight = model.weight * model.accuracy_score
                
                weighted_sum += model_prediction['score'] * effective_weight
                total_weight += effective_weight
                confidence_scores.append(model_prediction['confidence'])
                
                predictions[model_name] = model_prediction
                
            except Exception as e:
                logger.error(f"❌ Error in {model_name}: {e}")
                continue
        
        if total_weight == 0:
            return {
                'expected_points': 50.0,  # Default expectation
                'confidence_interval': (30.0, 70.0),
                'selection_probability': 0.5,
                'captain_suitability': 0.3
            }
        
        # Calculate ensemble results
        expected_points = weighted_sum / total_weight
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5
        
        # Calculate confidence interval
        point_variance = sum(
            (pred['score'] - expected_points) ** 2 * model.weight 
            for model_name, model in self.prediction_models.items() 
            for pred in [predictions.get(model_name, {'score': expected_points})]
            if model.is_enabled
        ) / total_weight if total_weight > 0 else 100
        
        std_dev = math.sqrt(point_variance)
        confidence_interval = (
            max(0, expected_points - 1.96 * std_dev),
            expected_points + 1.96 * std_dev
        )
        
        # Calculate selection probability (sigmoid function)
        selection_probability = 1 / (1 + math.exp(-(expected_points - 45) / 10))
        
        # Calculate captain suitability
        captain_suitability = self._calculate_captain_suitability(expected_points, avg_confidence, metrics)
        
        return {
            'expected_points': expected_points,
            'confidence_interval': confidence_interval,
            'selection_probability': selection_probability,
            'captain_suitability': captain_suitability,
            'model_predictions': predictions,
            'ensemble_confidence': avg_confidence
        }
    
    def _calculate_captain_suitability(self, expected_points: float, confidence: float, 
                                     metrics: PlayerPerformanceMetrics) -> float:
        """Calculate suitability as captain"""
        # High expected points
        points_factor = min(1.0, expected_points / 80)  # Normalize to 80 points max
        
        # High consistency (low risk)
        consistency_factor = metrics.consistency_index
        
        # High confidence in prediction
        confidence_factor = confidence
        
        # Role-based adjustment
        role_multipliers = {
            'batsman': 1.2,
            'all-rounder': 1.1,
            'wicket-keeper': 1.0,
            'bowler': 0.8
        }
        role_factor = role_multipliers.get(metrics.role.lower(), 1.0)
        
        captain_score = (
            points_factor * 0.4 +
            consistency_factor * 0.3 +
            confidence_factor * 0.2 +
            role_factor * 0.1
        )
        
        return min(1.0, captain_score)
    
    # Individual prediction model functions
    def _ema_prediction(self, metrics: PlayerPerformanceMetrics, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Exponential Moving Average based prediction"""
        if not metrics.recent_scores:
            return {'score': 45.0, 'confidence': 0.3}
        
        # Calculate EMA with alpha = 0.3
        ema = metrics.recent_scores[0]
        alpha = 0.3
        
        for score in metrics.recent_scores[1:]:
            ema = alpha * score + (1 - alpha) * ema
        
        # Adjust for sample size
        confidence = min(0.9, len(metrics.recent_scores) / 15)
        
        return {'score': ema, 'confidence': confidence}
    
    def _consistency_prediction(self, metrics: PlayerPerformanceMetrics, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Consistency-based prediction"""
        if metrics.career_average <= 0:
            return {'score': 40.0, 'confidence': 0.2}
        
        # Use career average adjusted by consistency
        base_score = metrics.career_average
        consistency_adjustment = (metrics.consistency_index - 0.5) * 10  # -5 to +5 adjustment
        
        predicted_score = base_score + consistency_adjustment
        confidence = metrics.consistency_index * 0.8
        
        return {'score': max(0, predicted_score), 'confidence': confidence}
    
    def _form_momentum_prediction(self, metrics: PlayerPerformanceMetrics, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Form momentum based prediction"""
        if not metrics.recent_scores:
            return {'score': 42.0, 'confidence': 0.2}
        
        base_score = metrics.career_average if metrics.career_average > 0 else 45
        momentum_adjustment = metrics.form_momentum * 15  # -15 to +15 adjustment
        
        predicted_score = base_score + momentum_adjustment
        confidence = abs(metrics.form_momentum) * 0.7  # Higher momentum = higher confidence
        
        return {'score': max(0, predicted_score), 'confidence': min(0.9, confidence)}
    
    def _venue_prediction(self, metrics: PlayerPerformanceMetrics, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Venue-specific prediction"""
        venue = match_context.get('venue', '')
        
        if venue and venue in metrics.venue_performance:
            venue_avg = metrics.venue_performance[venue]
            confidence = 0.6  # Moderate confidence for venue-specific data
        else:
            venue_avg = metrics.career_average if metrics.career_average > 0 else 45
            confidence = 0.3  # Lower confidence without venue data
        
        return {'score': venue_avg, 'confidence': confidence}
    
    def _matchup_prediction(self, metrics: PlayerPerformanceMetrics, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Opposition matchup based prediction"""
        opposition = match_context.get('opposition', '')
        
        # This would analyze historical performance against specific teams
        # For now, using a simplified approach
        base_score = metrics.career_average if metrics.career_average > 0 else 45
        
        # Slight adjustment based on conditions suitability
        matchup_adjustment = (metrics.conditions_suitability - 0.5) * 8
        
        predicted_score = base_score + matchup_adjustment
        confidence = 0.5  # Moderate confidence for matchup analysis
        
        return {'score': max(0, predicted_score), 'confidence': confidence}
    
    def _conditions_prediction(self, metrics: PlayerPerformanceMetrics, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Playing conditions based prediction"""
        base_score = metrics.career_average if metrics.career_average > 0 else 45
        
        # Adjust based on conditions suitability
        conditions_adjustment = (metrics.conditions_suitability - 0.5) * 10
        
        predicted_score = base_score + conditions_adjustment
        confidence = metrics.conditions_suitability * 0.6
        
        return {'score': max(0, predicted_score), 'confidence': confidence}
    
    def validate_predictions(self, predictions: List[Dict[str, Any]], actual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate prediction accuracy against actual results"""
        if not predictions or not actual_results:
            return {'accuracy': 0.0, 'mae': 999.0, 'rmse': 999.0}
        
        errors = []
        absolute_errors = []
        
        for pred, actual in zip(predictions, actual_results):
            if 'expected_points' in pred and 'actual_points' in actual:
                error = pred['expected_points'] - actual['actual_points']
                errors.append(error)
                absolute_errors.append(abs(error))
        
        if not errors:
            return {'accuracy': 0.0, 'mae': 999.0, 'rmse': 999.0}
        
        # Calculate metrics
        mae = statistics.mean(absolute_errors)  # Mean Absolute Error
        rmse = math.sqrt(statistics.mean([e**2 for e in errors]))  # Root Mean Square Error
        
        # Accuracy as percentage (closer predictions = higher accuracy)
        max_possible_error = 100  # Assuming max points around 100
        accuracy = max(0, (1 - mae / max_possible_error)) * 100
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'prediction_count': len(errors)
        }
    
    def update_model_weights(self, validation_results: Dict[str, float]):
        """Update model weights based on validation results"""
        accuracy = validation_results.get('accuracy', 0) / 100
        
        # Update weights for better-performing models
        for model_name, model in self.prediction_models.items():
            # This is a simplified weight update - in practice, you'd track individual model performance
            if accuracy > model.accuracy_score:
                model.weight = min(1.0, model.weight * 1.05)  # Increase weight by 5%
                model.accuracy_score = accuracy * 0.1 + model.accuracy_score * 0.9  # EMA update
            else:
                model.weight = max(0.05, model.weight * 0.98)  # Decrease weight slightly
        
        # Normalize weights
        total_weight = sum(model.weight for model in self.prediction_models.values())
        if total_weight > 0:
            for model in self.prediction_models.values():
                model.weight /= total_weight
        
        logger.info(f"📊 Updated model weights based on {accuracy:.1%} accuracy")
    
    def get_prediction_insights(self) -> Dict[str, Any]:
        """Get insights about prediction performance"""
        return {
            'active_models': len([m for m in self.prediction_models.values() if m.is_enabled]),
            'model_weights': {name: f"{model.weight:.3f}" for name, model in self.prediction_models.items()},
            'model_accuracy': {name: f"{model.accuracy_score:.1%}" for name, model in self.prediction_models.items()},
            'best_performing_model': max(self.prediction_models.items(), key=lambda x: x[1].accuracy_score)[0],
            'ensemble_confidence': 'High' if all(m.accuracy_score > 0.6 for m in self.prediction_models.values()) else 'Medium'
        }

# Global prediction engine instance
prediction_engine = PredictionAccuracyEngine()

def get_prediction_engine() -> PredictionAccuracyEngine:
    """Get global prediction engine instance"""
    return prediction_engine

def analyze_player_for_prediction(player_data: Dict[str, Any], match_context: Dict[str, Any]) -> PlayerPerformanceMetrics:
    """Analyze player for prediction using global engine"""
    return prediction_engine.analyze_player_performance(player_data, match_context)