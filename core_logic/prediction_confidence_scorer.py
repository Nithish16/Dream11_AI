#!/usr/bin/env python3
"""
Advanced Prediction Confidence Scoring System
Multi-dimensional confidence analysis for Dream11 predictions
"""

import numpy as np
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import math

try:
    from .prediction_accuracy_engine import PlayerPerformanceMetrics, get_prediction_engine
except ImportError:
    from prediction_accuracy_engine import PlayerPerformanceMetrics, get_prediction_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfidenceFactors:
    """Individual confidence factors for prediction scoring"""
    data_quality_score: float = 0.0      # Quality and recency of data
    sample_size_score: float = 0.0       # Adequacy of historical data
    model_agreement_score: float = 0.0   # Agreement between different models
    consistency_score: float = 0.0       # Player's performance consistency
    conditions_match_score: float = 0.0  # Match conditions similarity
    recent_form_score: float = 0.0       # Recent performance trend
    opponent_factor_score: float = 0.0   # Historical vs opponent performance
    
    # Meta-factors
    prediction_stability: float = 0.0     # How stable prediction is over time
    external_validation: float = 0.0      # Validation from external sources
    domain_expertise: float = 0.0         # Expert knowledge integration

@dataclass
class ConfidenceBreakdown:
    """Detailed confidence breakdown for transparency"""
    overall_confidence: float
    confidence_level: str  # "Very High", "High", "Medium", "Low", "Very Low"
    factors: ConfidenceFactors
    risk_assessment: Dict[str, float]
    reliability_indicators: Dict[str, Any]
    recommendation_strength: str
    uncertainty_range: Tuple[float, float]

class PredictionConfidenceScorer:
    """
    Advanced confidence scoring for predictions with multi-dimensional analysis
    """
    
    def __init__(self, db_path: str = "confidence_scoring.db"):
        self.db_path = db_path
        self.prediction_engine = get_prediction_engine()
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.55,
            'low': 0.40,
            'very_low': 0.0
        }
        
        # Factor weights for overall confidence calculation
        self.factor_weights = {
            'data_quality_score': 0.20,
            'sample_size_score': 0.15,
            'model_agreement_score': 0.15,
            'consistency_score': 0.12,
            'conditions_match_score': 0.10,
            'recent_form_score': 0.10,
            'opponent_factor_score': 0.08,
            'prediction_stability': 0.05,
            'external_validation': 0.03,
            'domain_expertise': 0.02
        }
        
        # Risk factors
        self.risk_factors = {
            'injury_risk': 0.15,
            'weather_impact': 0.10,
            'rotation_risk': 0.12,
            'form_volatility': 0.18,
            'venue_uncertainty': 0.08,
            'opposition_strength': 0.12,
            'recent_performance_drop': 0.15,
            'external_factors': 0.10
        }
        
        # Initialize database
        self._init_database()
        
        # Confidence history for learning
        self.confidence_history = defaultdict(list)
    
    def _init_database(self):
        """Initialize confidence scoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Confidence scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS confidence_scores (
                prediction_id TEXT PRIMARY KEY,
                player_id INTEGER,
                match_id TEXT,
                overall_confidence REAL,
                confidence_level TEXT,
                confidence_factors TEXT,
                risk_assessment TEXT,
                predicted_points REAL,
                actual_points REAL,
                confidence_accuracy REAL,
                prediction_date TIMESTAMP,
                validation_date TIMESTAMP
            )
        ''')
        
        # Factor performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_performance (
                factor_name TEXT,
                prediction_date TIMESTAMP,
                factor_score REAL,
                actual_performance REAL,
                factor_accuracy REAL,
                sample_size INTEGER,
                PRIMARY KEY (factor_name, prediction_date)
            )
        ''')
        
        # Confidence calibration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                confidence_bucket TEXT,
                predicted_success_rate REAL,
                actual_success_rate REAL,
                sample_size INTEGER,
                last_updated TIMESTAMP,
                PRIMARY KEY (confidence_bucket)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Confidence scoring database initialized: {self.db_path}")
    
    def calculate_confidence_score(self, player_metrics: PlayerPerformanceMetrics, 
                                 match_context: Dict[str, Any],
                                 prediction_data: Dict[str, Any]) -> ConfidenceBreakdown:
        """
        Calculate comprehensive confidence score for a player prediction
        """
        factors = ConfidenceFactors()
        
        # 1. Data Quality Score
        factors.data_quality_score = self._calculate_data_quality_score(player_metrics)
        
        # 2. Sample Size Score
        factors.sample_size_score = self._calculate_sample_size_score(player_metrics)
        
        # 3. Model Agreement Score
        factors.model_agreement_score = self._calculate_model_agreement_score(prediction_data)
        
        # 4. Consistency Score
        factors.consistency_score = player_metrics.consistency_index
        
        # 5. Conditions Match Score
        factors.conditions_match_score = self._calculate_conditions_match_score(
            player_metrics, match_context
        )
        
        # 6. Recent Form Score
        factors.recent_form_score = self._calculate_recent_form_score(player_metrics)
        
        # 7. Opponent Factor Score
        factors.opponent_factor_score = self._calculate_opponent_factor_score(
            player_metrics, match_context
        )
        
        # 8. Prediction Stability
        factors.prediction_stability = self._calculate_prediction_stability(
            player_metrics, prediction_data
        )
        
        # 9. External Validation (placeholder for now)
        factors.external_validation = 0.5
        
        # 10. Domain Expertise (placeholder for now)
        factors.domain_expertise = 0.6
        
        # Calculate overall confidence
        overall_confidence = self._calculate_weighted_confidence(factors)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(player_metrics, match_context)
        
        # Generate reliability indicators
        reliability_indicators = self._generate_reliability_indicators(factors, player_metrics)
        
        # Determine recommendation strength
        recommendation_strength = self._determine_recommendation_strength(
            overall_confidence, risk_assessment
        )
        
        # Calculate uncertainty range
        uncertainty_range = self._calculate_uncertainty_range(factors, prediction_data)
        
        return ConfidenceBreakdown(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            factors=factors,
            risk_assessment=risk_assessment,
            reliability_indicators=reliability_indicators,
            recommendation_strength=recommendation_strength,
            uncertainty_range=uncertainty_range
        )
    
    def _calculate_data_quality_score(self, metrics: PlayerPerformanceMetrics) -> float:
        """Calculate data quality score based on recency and completeness"""
        # Recency score (newer data is better)
        days_since_update = (datetime.now() - metrics.last_updated).days
        recency_score = max(0.1, 1.0 - (days_since_update / 30.0))  # Decays over 30 days
        
        # Completeness score (more data points are better)
        completeness_score = min(1.0, metrics.sample_size / 15.0)  # Full score at 15+ matches
        
        # Data reliability from metrics
        reliability_score = metrics.data_reliability
        
        return (recency_score * 0.4 + completeness_score * 0.4 + reliability_score * 0.2)
    
    def _calculate_sample_size_score(self, metrics: PlayerPerformanceMetrics) -> float:
        """Calculate score based on sample size adequacy"""
        if metrics.sample_size >= 20:
            return 1.0
        elif metrics.sample_size >= 15:
            return 0.9
        elif metrics.sample_size >= 10:
            return 0.7
        elif metrics.sample_size >= 5:
            return 0.5
        elif metrics.sample_size >= 3:
            return 0.3
        else:
            return 0.1
    
    def _calculate_model_agreement_score(self, prediction_data: Dict[str, Any]) -> float:
        """Calculate agreement between different prediction models"""
        model_predictions = prediction_data.get('model_predictions', {})
        
        if len(model_predictions) < 2:
            return 0.5  # Default for insufficient models
        
        # Extract predicted scores from different models
        scores = []
        for model_name, prediction in model_predictions.items():
            if 'score' in prediction:
                scores.append(prediction['score'])
        
        if len(scores) < 2:
            return 0.5
        
        # Calculate coefficient of variation (lower = better agreement)
        mean_score = statistics.mean(scores)
        if mean_score <= 0:
            return 0.5
        
        try:
            std_dev = statistics.stdev(scores)
            cv = std_dev / mean_score
            
            # Convert to agreement score (0-1, higher = better agreement)
            agreement_score = max(0.0, 1.0 - min(cv, 1.0))
            return agreement_score
            
        except Exception:
            return 0.5
    
    def _calculate_conditions_match_score(self, metrics: PlayerPerformanceMetrics, 
                                        match_context: Dict[str, Any]) -> float:
        """Calculate how well current conditions match historical performance"""
        return metrics.conditions_suitability
    
    def _calculate_recent_form_score(self, metrics: PlayerPerformanceMetrics) -> float:
        """Calculate recent form score with momentum consideration"""
        # Base form momentum
        form_score = (metrics.form_momentum + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        
        # Adjust for consistency
        consistency_adjustment = metrics.consistency_index * 0.2
        
        # Combine scores
        return min(1.0, form_score + consistency_adjustment)
    
    def _calculate_opponent_factor_score(self, metrics: PlayerPerformanceMetrics, 
                                       match_context: Dict[str, Any]) -> float:
        """Calculate opponent-specific performance factor"""
        # This would analyze historical performance against specific opponents
        # For now, using matchup advantage from metrics
        return (metrics.matchup_advantage + 1.0) / 2.0  # Convert to 0-1 range
    
    def _calculate_prediction_stability(self, metrics: PlayerPerformanceMetrics, 
                                      prediction_data: Dict[str, Any]) -> float:
        """Calculate how stable the prediction is over time"""
        # Check if we have confidence interval data
        confidence_interval = prediction_data.get('confidence_interval', (0, 100))
        predicted_score = prediction_data.get('expected_points', 50)
        
        if predicted_score <= 0:
            return 0.3
        
        # Calculate interval width relative to prediction
        interval_width = confidence_interval[1] - confidence_interval[0]
        relative_width = interval_width / predicted_score
        
        # Lower relative width = higher stability
        stability_score = max(0.0, 1.0 - min(relative_width / 2.0, 1.0))
        
        return stability_score
    
    def _calculate_weighted_confidence(self, factors: ConfidenceFactors) -> float:
        """Calculate weighted overall confidence score"""
        total_score = 0.0
        
        for factor_name, weight in self.factor_weights.items():
            factor_value = getattr(factors, factor_name, 0.0)
            total_score += factor_value * weight
        
        return min(1.0, max(0.0, total_score))
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level category"""
        if confidence_score >= self.confidence_thresholds['very_high']:
            return "Very High"
        elif confidence_score >= self.confidence_thresholds['high']:
            return "High"
        elif confidence_score >= self.confidence_thresholds['medium']:
            return "Medium"
        elif confidence_score >= self.confidence_thresholds['low']:
            return "Low"
        else:
            return "Very Low"
    
    def _calculate_risk_assessment(self, metrics: PlayerPerformanceMetrics, 
                                 match_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various risk factors"""
        risk_scores = {}
        
        # Form volatility risk (higher volatility = higher risk)
        if metrics.recent_scores and len(metrics.recent_scores) > 1:
            try:
                volatility = statistics.stdev(metrics.recent_scores) / statistics.mean(metrics.recent_scores)
                risk_scores['form_volatility'] = min(1.0, volatility)
            except:
                risk_scores['form_volatility'] = 0.5
        else:
            risk_scores['form_volatility'] = 0.5
        
        # Injury/rotation risk (placeholder)
        risk_scores['injury_risk'] = 0.2  # Default low risk
        
        # Weather impact (based on venue and conditions)
        venue = match_context.get('venue', '')
        risk_scores['weather_impact'] = 0.3 if 'outdoor' in venue.lower() else 0.1
        
        # Rotation risk (based on recent matches)
        risk_scores['rotation_risk'] = 0.15  # Default
        
        # Venue uncertainty
        venue_performance = getattr(metrics, 'venue_performance', {})
        current_venue = match_context.get('venue', '')
        if current_venue not in venue_performance:
            risk_scores['venue_uncertainty'] = 0.6
        else:
            risk_scores['venue_uncertainty'] = 0.2
        
        # Opposition strength
        risk_scores['opposition_strength'] = 0.5  # Default
        
        # Recent performance drop
        if metrics.form_momentum < -0.2:
            risk_scores['recent_performance_drop'] = 0.7
        else:
            risk_scores['recent_performance_drop'] = 0.2
        
        # External factors
        risk_scores['external_factors'] = 0.3  # Default
        
        return risk_scores
    
    def _generate_reliability_indicators(self, factors: ConfidenceFactors, 
                                       metrics: PlayerPerformanceMetrics) -> Dict[str, Any]:
        """Generate reliability indicators for transparency"""
        return {
            'data_points_available': metrics.sample_size,
            'data_recency_days': (datetime.now() - metrics.last_updated).days,
            'consistency_rating': self._get_rating(metrics.consistency_index),
            'form_trend': 'positive' if metrics.form_momentum > 0.1 else 'negative' if metrics.form_momentum < -0.1 else 'stable',
            'conditions_familiarity': self._get_rating(factors.conditions_match_score),
            'model_consensus': self._get_rating(factors.model_agreement_score),
            'prediction_precision': self._get_rating(factors.prediction_stability)
        }
    
    def _get_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Average"
        elif score >= 0.2:
            return "Below Average"
        else:
            return "Poor"
    
    def _determine_recommendation_strength(self, confidence: float, 
                                         risk_assessment: Dict[str, float]) -> str:
        """Determine recommendation strength based on confidence and risk"""
        # Calculate overall risk
        weighted_risk = sum(
            risk_score * self.risk_factors.get(risk_name, 0.1)
            for risk_name, risk_score in risk_assessment.items()
        )
        
        # Adjust confidence for risk
        risk_adjusted_confidence = confidence * (1 - weighted_risk * 0.3)
        
        if risk_adjusted_confidence >= 0.8:
            return "Strong Buy"
        elif risk_adjusted_confidence >= 0.65:
            return "Buy"
        elif risk_adjusted_confidence >= 0.5:
            return "Hold/Consider"
        elif risk_adjusted_confidence >= 0.35:
            return "Weak"
        else:
            return "Avoid"
    
    def _calculate_uncertainty_range(self, factors: ConfidenceFactors, 
                                   prediction_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate uncertainty range for the prediction"""
        base_prediction = prediction_data.get('expected_points', 50)
        confidence_interval = prediction_data.get('confidence_interval', (base_prediction * 0.7, base_prediction * 1.3))
        
        # Adjust range based on confidence factors
        uncertainty_multiplier = 1.0 - (factors.model_agreement_score * 0.3 + factors.prediction_stability * 0.2)
        
        range_width = (confidence_interval[1] - confidence_interval[0]) * uncertainty_multiplier
        center = (confidence_interval[1] + confidence_interval[0]) / 2
        
        return (
            max(0, center - range_width / 2),
            center + range_width / 2
        )
    
    def validate_confidence_accuracy(self, prediction_id: str, actual_performance: float):
        """Validate confidence accuracy against actual performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get confidence prediction
            cursor.execute('''
                SELECT overall_confidence, predicted_points, confidence_factors
                FROM confidence_scores
                WHERE prediction_id = ?
            ''', (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                return
            
            confidence, predicted_points, factors_json = result
            
            # Calculate confidence accuracy
            prediction_error = abs(predicted_points - actual_performance)
            max_possible_error = max(predicted_points, actual_performance, 50)  # Reasonable max
            
            accuracy = max(0, 1 - prediction_error / max_possible_error)
            
            # Update record
            cursor.execute('''
                UPDATE confidence_scores
                SET actual_points = ?, confidence_accuracy = ?, validation_date = ?
                WHERE prediction_id = ?
            ''', (actual_performance, accuracy, datetime.now(), prediction_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📊 Validated confidence for {prediction_id}: {accuracy:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"❌ Error validating confidence: {e}")
    
    def get_confidence_insights(self) -> Dict[str, Any]:
        """Get insights about confidence scoring performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall accuracy by confidence level
            cursor.execute('''
                SELECT confidence_level, 
                       AVG(confidence_accuracy) as avg_accuracy,
                       COUNT(*) as sample_size
                FROM confidence_scores
                WHERE actual_points IS NOT NULL
                GROUP BY confidence_level
            ''')
            
            accuracy_by_level = cursor.fetchall()
            
            # Factor performance
            cursor.execute('''
                SELECT factor_name, AVG(factor_accuracy) as avg_accuracy
                FROM factor_performance
                GROUP BY factor_name
                ORDER BY avg_accuracy DESC
            ''')
            
            factor_performance = cursor.fetchall()
            
            conn.close()
            
            return {
                'confidence_calibration': [
                    {
                        'level': row[0],
                        'accuracy': f"{row[1]:.1%}" if row[1] else "N/A",
                        'sample_size': row[2]
                    }
                    for row in accuracy_by_level
                ],
                'factor_performance': [
                    {
                        'factor': row[0],
                        'accuracy': f"{row[1]:.1%}" if row[1] else "N/A"
                    }
                    for row in factor_performance
                ],
                'total_predictions': len(self.confidence_history),
                'avg_confidence': f"{statistics.mean([c for cs in self.confidence_history.values() for c in cs]):.1%}" if self.confidence_history else "N/A"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting confidence insights: {e}")
            return {'error': str(e)}

# Global confidence scorer instance
confidence_scorer = PredictionConfidenceScorer()

def get_confidence_scorer() -> PredictionConfidenceScorer:
    """Get global confidence scorer instance"""
    return confidence_scorer

def score_prediction_confidence(player_metrics: PlayerPerformanceMetrics,
                              match_context: Dict[str, Any],
                              prediction_data: Dict[str, Any]) -> ConfidenceBreakdown:
    """Score prediction confidence using global scorer"""
    return confidence_scorer.calculate_confidence_score(player_metrics, match_context, prediction_data)