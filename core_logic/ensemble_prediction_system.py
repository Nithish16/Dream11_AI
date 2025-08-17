#!/usr/bin/env python3
"""
Advanced Ensemble Prediction System
Combines multiple prediction models with confidence scoring and A/B testing
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
import threading
import hashlib

try:
    from .prediction_accuracy_engine import PlayerPerformanceMetrics, get_prediction_engine
    from .prediction_confidence_scorer import ConfidenceBreakdown, get_confidence_scorer
    from .ab_testing_framework import get_ab_testing_framework
except ImportError:
    from prediction_accuracy_engine import PlayerPerformanceMetrics, get_prediction_engine
    from prediction_confidence_scorer import ConfidenceBreakdown, get_confidence_scorer
    from ab_testing_framework import get_ab_testing_framework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Complete ensemble prediction with all components"""
    player_id: int
    player_name: str
    predicted_points: float
    confidence_score: float
    confidence_level: str
    confidence_breakdown: ConfidenceBreakdown
    model_contributions: Dict[str, float]
    risk_factors: Dict[str, float]
    recommendation: str
    captain_suitability: float
    selection_probability: float
    uncertainty_range: Tuple[float, float]
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    
    # A/B testing info
    ab_test_variant: Optional[str] = None
    ab_test_id: Optional[str] = None

@dataclass
class TeamPrediction:
    """Complete team prediction with optimization"""
    team_composition: List[EnsemblePrediction]
    total_predicted_points: float
    team_confidence: float
    captain_recommendation: EnsemblePrediction
    vice_captain_recommendation: EnsemblePrediction
    bench_players: List[EnsemblePrediction]
    budget_utilization: float
    risk_assessment: Dict[str, float]
    diversification_score: float
    expected_rank: int
    optimization_strategy: str

class EnsemblePredictionSystem:
    """
    Advanced ensemble system combining all prediction components
    """
    
    def __init__(self, db_path: str = "ensemble_predictions.db"):
        self.db_path = db_path
        
        # Component integrations
        self.prediction_engine = get_prediction_engine()
        self.confidence_scorer = get_confidence_scorer()
        self.ab_testing = get_ab_testing_framework()
        
        # Ensemble configuration
        self.ensemble_weights = {
            'prediction_accuracy': 0.35,
            'confidence_score': 0.25,
            'recent_form': 0.20,
            'consistency': 0.15,
            'conditions_match': 0.05
        }
        
        # Team optimization parameters
        self.team_constraints = {
            'max_budget': 100.0,
            'min_players_per_team': 3,
            'max_players_per_team': 7,
            'wicket_keepers': (1, 1),
            'batsmen': (3, 5),
            'all_rounders': (1, 3),
            'bowlers': (3, 5)
        }
        
        # Risk management
        self.risk_thresholds = {
            'max_players_same_team': 7,
            'max_budget_per_player': 15.0,
            'min_confidence_threshold': 0.3,
            'diversification_target': 0.7
        }
        
        # Active experiments tracking
        self.active_experiments = {}
        
        # Performance tracking
        self.prediction_history = defaultdict(list)
        self.team_performance_history = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize ensemble prediction database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensemble predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_predictions (
                prediction_id TEXT PRIMARY KEY,
                player_id INTEGER,
                player_name TEXT,
                predicted_points REAL,
                confidence_score REAL,
                confidence_level TEXT,
                confidence_breakdown TEXT,
                model_contributions TEXT,
                risk_factors TEXT,
                recommendation TEXT,
                captain_suitability REAL,
                selection_probability REAL,
                uncertainty_range TEXT,
                ab_test_variant TEXT,
                ab_test_id TEXT,
                prediction_timestamp TIMESTAMP,
                actual_points REAL,
                prediction_accuracy REAL
            )
        ''')
        
        # Team predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_predictions (
                team_id TEXT PRIMARY KEY,
                match_id TEXT,
                team_composition TEXT,
                total_predicted_points REAL,
                team_confidence REAL,
                captain_id INTEGER,
                vice_captain_id INTEGER,
                budget_utilization REAL,
                risk_assessment TEXT,
                diversification_score REAL,
                expected_rank INTEGER,
                optimization_strategy TEXT,
                prediction_timestamp TIMESTAMP,
                actual_points REAL,
                actual_rank INTEGER,
                team_accuracy REAL
            )
        ''')
        
        # Ensemble performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_performance (
                metric_name TEXT,
                metric_value REAL,
                calculation_date TIMESTAMP,
                sample_size INTEGER,
                confidence_interval TEXT,
                PRIMARY KEY (metric_name, calculation_date)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Ensemble prediction database initialized: {self.db_path}")
    
    def predict_player_performance(self, player_data: Dict[str, Any], 
                                 match_context: Dict[str, Any],
                                 ab_test_enabled: bool = True) -> EnsemblePrediction:
        """
        Generate comprehensive player prediction with all components
        """
        # 1. Get base player performance metrics
        player_metrics = self.prediction_engine.analyze_player_performance(
            player_data, match_context
        )
        
        # 2. Generate prediction data from ensemble
        prediction_data = self.prediction_engine._generate_ensemble_prediction(
            player_metrics, match_context
        )
        
        # 3. A/B testing variant assignment
        ab_test_variant = None
        ab_test_id = None
        
        if ab_test_enabled and self.active_experiments:
            # Get active experiment for player predictions
            for test_id, experiment in self.active_experiments.items():
                ab_test_variant = self.ab_testing.assign_variant(
                    test_id, user_id=str(player_data.get('player_id', 0))
                )
                if ab_test_variant:
                    ab_test_id = test_id
                    # Apply variant-specific adjustments
                    prediction_data = self._apply_ab_variant_adjustments(
                        prediction_data, ab_test_variant
                    )
                    break
        
        # 4. Calculate confidence score
        confidence_breakdown = self.confidence_scorer.calculate_confidence_score(
            player_metrics, match_context, prediction_data
        )
        
        # 5. Apply ensemble weighting
        final_prediction = self._apply_ensemble_weighting(
            prediction_data, confidence_breakdown, player_metrics
        )
        
        # 6. Generate risk factors
        risk_factors = self._calculate_comprehensive_risk_factors(
            player_metrics, match_context, confidence_breakdown
        )
        
        # 7. Create ensemble prediction
        ensemble_prediction = EnsemblePrediction(
            player_id=player_data.get('player_id', 0),
            player_name=player_data.get('name', 'Unknown'),
            predicted_points=final_prediction['points'],
            confidence_score=confidence_breakdown.overall_confidence,
            confidence_level=confidence_breakdown.confidence_level,
            confidence_breakdown=confidence_breakdown,
            model_contributions=prediction_data.get('model_predictions', {}),
            risk_factors=risk_factors,
            recommendation=confidence_breakdown.recommendation_strength,
            captain_suitability=final_prediction['captain_suitability'],
            selection_probability=final_prediction['selection_probability'],
            uncertainty_range=confidence_breakdown.uncertainty_range,
            ab_test_variant=ab_test_variant,
            ab_test_id=ab_test_id
        )
        
        # 8. Save prediction
        self._save_ensemble_prediction(ensemble_prediction)
        
        # 9. Record A/B test result if applicable
        if ab_test_id and ab_test_variant:
            self.ab_testing.record_prediction_result(
                ab_test_id, ab_test_variant, final_prediction['points']
            )
        
        return ensemble_prediction
    
    def _apply_ab_variant_adjustments(self, prediction_data: Dict[str, Any], 
                                    variant_id: str) -> Dict[str, Any]:
        """Apply A/B testing variant adjustments to prediction"""
        # This would contain variant-specific logic
        # For now, applying slight modifications based on variant
        if 'aggressive' in variant_id.lower():
            prediction_data['expected_points'] *= 1.05
        elif 'conservative' in variant_id.lower():
            prediction_data['expected_points'] *= 0.95
        
        return prediction_data
    
    def _apply_ensemble_weighting(self, prediction_data: Dict[str, Any],
                                confidence_breakdown: ConfidenceBreakdown,
                                player_metrics: PlayerPerformanceMetrics) -> Dict[str, float]:
        """Apply ensemble weighting to final prediction"""
        
        base_points = prediction_data.get('expected_points', 50)
        
        # Weight factors
        weights = self.ensemble_weights
        
        # Calculate weighted adjustments
        prediction_weight = weights['prediction_accuracy']
        confidence_weight = weights['confidence_score'] * confidence_breakdown.overall_confidence
        form_weight = weights['recent_form'] * ((player_metrics.form_momentum + 1) / 2)
        consistency_weight = weights['consistency'] * player_metrics.consistency_index
        conditions_weight = weights['conditions_match'] * player_metrics.conditions_suitability
        
        # Combine weights
        total_weight = prediction_weight + confidence_weight + form_weight + consistency_weight + conditions_weight
        
        # Apply weighted adjustments
        if total_weight > 0:
            adjustment_factor = (
                prediction_weight * 1.0 +
                confidence_weight * 1.1 +
                form_weight * (1.2 if player_metrics.form_momentum > 0 else 0.9) +
                consistency_weight * 1.05 +
                conditions_weight * 1.1
            ) / total_weight
        else:
            adjustment_factor = 1.0
        
        final_points = base_points * adjustment_factor
        
        return {
            'points': final_points,
            'captain_suitability': prediction_data.get('captain_suitability', 0.5) * adjustment_factor,
            'selection_probability': prediction_data.get('selection_probability', 0.5) * confidence_breakdown.overall_confidence
        }
    
    def _calculate_comprehensive_risk_factors(self, player_metrics: PlayerPerformanceMetrics,
                                            match_context: Dict[str, Any],
                                            confidence_breakdown: ConfidenceBreakdown) -> Dict[str, float]:
        """Calculate comprehensive risk factors"""
        risk_factors = confidence_breakdown.risk_assessment.copy()
        
        # Add ensemble-specific risks
        risk_factors['prediction_uncertainty'] = 1 - confidence_breakdown.overall_confidence
        risk_factors['data_reliability_risk'] = 1 - player_metrics.data_reliability
        risk_factors['form_volatility'] = 1 - player_metrics.consistency_index
        
        # Captain-specific risks
        if player_metrics.role.lower() in ['batsman', 'all-rounder']:
            risk_factors['captain_risk'] = 0.2
        else:
            risk_factors['captain_risk'] = 0.6
        
        return risk_factors
    
    def predict_optimal_team(self, available_players: List[Dict[str, Any]],
                           match_context: Dict[str, Any],
                           optimization_strategy: str = 'balanced') -> TeamPrediction:
        """
        Generate optimal team using ensemble predictions
        """
        logger.info(f"🎯 Generating optimal team with {len(available_players)} players")
        
        # 1. Generate predictions for all players
        player_predictions = []
        for player_data in available_players:
            try:
                prediction = self.predict_player_performance(player_data, match_context)
                player_predictions.append(prediction)
            except Exception as e:
                logger.error(f"❌ Error predicting {player_data.get('name', 'Unknown')}: {e}")
                continue
        
        # 2. Apply optimization strategy
        if optimization_strategy == 'aggressive':
            selected_team = self._optimize_aggressive_team(player_predictions)
        elif optimization_strategy == 'conservative':
            selected_team = self._optimize_conservative_team(player_predictions)
        elif optimization_strategy == 'high_ceiling':
            selected_team = self._optimize_high_ceiling_team(player_predictions)
        else:  # balanced
            selected_team = self._optimize_balanced_team(player_predictions)
        
        # 3. Select captain and vice-captain
        captain, vice_captain = self._select_captain_vice_captain(selected_team)
        
        # 4. Calculate team metrics
        team_metrics = self._calculate_team_metrics(selected_team, captain, vice_captain, optimization_strategy)
        
        # 5. Create team prediction
        team_prediction = TeamPrediction(
            team_composition=selected_team,
            total_predicted_points=team_metrics['total_points'],
            team_confidence=team_metrics['team_confidence'],
            captain_recommendation=captain,
            vice_captain_recommendation=vice_captain,
            bench_players=team_metrics['bench_players'],
            budget_utilization=team_metrics['budget_utilization'],
            risk_assessment=team_metrics['risk_assessment'],
            diversification_score=team_metrics['diversification_score'],
            expected_rank=team_metrics['expected_rank'],
            optimization_strategy=optimization_strategy
        )
        
        # 6. Save team prediction
        self._save_team_prediction(team_prediction, match_context)
        
        return team_prediction
    
    def _optimize_balanced_team(self, player_predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Optimize team for balanced risk/reward"""
        # Sort by value score (predicted points * confidence * (1 - avg_risk))
        def value_score(p: EnsemblePrediction) -> float:
            avg_risk = statistics.mean(p.risk_factors.values()) if p.risk_factors else 0.5
            return p.predicted_points * p.confidence_score * (1 - avg_risk * 0.3)
        
        sorted_players = sorted(player_predictions, key=value_score, reverse=True)
        
        # Select top 11 players meeting constraints
        selected_team = []
        role_counts = defaultdict(int)
        
        for player in sorted_players:
            if len(selected_team) >= 11:
                break
            
            # Check role constraints (simplified)
            if self._can_add_player_to_team(player, selected_team, role_counts):
                selected_team.append(player)
                role_counts[self._get_player_role(player)] += 1
        
        return selected_team
    
    def _optimize_aggressive_team(self, player_predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Optimize team for maximum points (higher risk)"""
        # Sort by predicted points with high ceiling bias
        def aggressive_score(p: EnsemblePrediction) -> float:
            return p.predicted_points * 1.2 if p.uncertainty_range[1] > p.predicted_points * 1.3 else p.predicted_points
        
        sorted_players = sorted(player_predictions, key=aggressive_score, reverse=True)
        return self._select_team_with_constraints(sorted_players)
    
    def _optimize_conservative_team(self, player_predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Optimize team for consistency (lower risk)"""
        # Sort by floor score and confidence
        def conservative_score(p: EnsemblePrediction) -> float:
            floor_points = p.uncertainty_range[0]
            return floor_points * p.confidence_score * 1.5
        
        sorted_players = sorted(player_predictions, key=conservative_score, reverse=True)
        return self._select_team_with_constraints(sorted_players)
    
    def _optimize_high_ceiling_team(self, player_predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Optimize team for tournament/GPP play"""
        # Sort by ceiling potential
        def ceiling_score(p: EnsemblePrediction) -> float:
            ceiling_points = p.uncertainty_range[1]
            return ceiling_points * (2 - p.selection_probability)  # Favor low-owned high ceiling
        
        sorted_players = sorted(player_predictions, key=ceiling_score, reverse=True)
        return self._select_team_with_constraints(sorted_players)
    
    def _select_team_with_constraints(self, sorted_players: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """Select team respecting role and budget constraints"""
        selected_team = []
        role_counts = defaultdict(int)
        
        for player in sorted_players:
            if len(selected_team) >= 11:
                break
            
            if self._can_add_player_to_team(player, selected_team, role_counts):
                selected_team.append(player)
                role_counts[self._get_player_role(player)] += 1
        
        return selected_team
    
    def _can_add_player_to_team(self, player: EnsemblePrediction, 
                              current_team: List[EnsemblePrediction],
                              role_counts: Dict[str, int]) -> bool:
        """Check if player can be added respecting constraints"""
        # Simplified constraint checking
        role = self._get_player_role(player)
        
        # Role limits (simplified)
        role_limits = {'batsman': 5, 'bowler': 5, 'all-rounder': 3, 'wicket-keeper': 1}
        
        if role_counts[role] >= role_limits.get(role, 2):
            return False
        
        # Budget constraint (placeholder)
        # Would need actual player prices here
        
        return True
    
    def _get_player_role(self, player: EnsemblePrediction) -> str:
        """Extract player role from player data"""
        # Extract from player data if available
        if hasattr(player, 'player_data') and player.player_data:
            return player.player_data.get('role', 'batsman')
        
        # Default role distribution to ensure team completion
        player_id = getattr(player, 'player_id', 0)
        roles = ['batsman', 'bowler', 'all-rounder', 'wicket-keeper']
        return roles[player_id % len(roles)]
    
    def _select_captain_vice_captain(self, team: List[EnsemblePrediction]) -> Tuple[EnsemblePrediction, EnsemblePrediction]:
        """Select optimal captain and vice-captain"""
        # Sort by captain suitability
        sorted_for_captain = sorted(team, key=lambda p: p.captain_suitability, reverse=True)
        
        captain = sorted_for_captain[0]
        vice_captain = sorted_for_captain[1] if len(sorted_for_captain) > 1 else sorted_for_captain[0]
        
        return captain, vice_captain
    
    def _calculate_team_metrics(self, team: List[EnsemblePrediction], 
                              captain: EnsemblePrediction,
                              vice_captain: EnsemblePrediction,
                              strategy: str) -> Dict[str, Any]:
        """Calculate comprehensive team metrics"""
        
        # Total predicted points (captain gets 2x, vice-captain gets 1.5x)
        total_points = sum(p.predicted_points for p in team)
        total_points += captain.predicted_points  # Captain bonus
        total_points += vice_captain.predicted_points * 0.5  # Vice-captain bonus
        
        # Team confidence (weighted average)
        team_confidence = statistics.mean([p.confidence_score for p in team])
        
        # Risk assessment
        risk_assessment = self._calculate_team_risk_assessment(team)
        
        # Diversification score
        diversification_score = self._calculate_diversification_score(team)
        
        # Expected rank (simplified)
        expected_rank = max(1, int(10 - (team_confidence * total_points / 100)))
        
        return {
            'total_points': total_points,
            'team_confidence': team_confidence,
            'bench_players': [],  # Would be calculated if applicable
            'budget_utilization': 0.95,  # Placeholder
            'risk_assessment': risk_assessment,
            'diversification_score': diversification_score,
            'expected_rank': expected_rank
        }
    
    def _calculate_team_risk_assessment(self, team: List[EnsemblePrediction]) -> Dict[str, float]:
        """Calculate team-level risk factors"""
        risk_scores = defaultdict(list)
        
        for player in team:
            for risk_name, risk_value in player.risk_factors.items():
                risk_scores[risk_name].append(risk_value)
        
        # Average risk scores across team
        team_risks = {
            risk_name: statistics.mean(scores)
            for risk_name, scores in risk_scores.items()
        }
        
        # Add team-specific risks
        team_risks['concentration_risk'] = 0.3  # Placeholder
        team_risks['captain_dependency'] = 0.4   # Placeholder
        
        return team_risks
    
    def _calculate_diversification_score(self, team: List[EnsemblePrediction]) -> float:
        """Calculate team diversification score"""
        # Simplified diversification based on prediction variance
        predicted_points = [p.predicted_points for p in team]
        
        if len(predicted_points) < 2:
            return 0.5
        
        mean_points = statistics.mean(predicted_points)
        if mean_points <= 0:
            return 0.5
        
        cv = statistics.stdev(predicted_points) / mean_points
        
        # Higher coefficient of variation = better diversification
        diversification_score = min(1.0, cv / 0.5)  # Normalize to 0-1
        
        return diversification_score
    
    def _save_ensemble_prediction(self, prediction: EnsemblePrediction):
        """Save ensemble prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            prediction_id = hashlib.sha256(
                f"{prediction.player_id}_{prediction.prediction_timestamp}".encode()
            ).hexdigest()[:16]
            
            cursor.execute('''
                INSERT OR REPLACE INTO ensemble_predictions
                (prediction_id, player_id, player_name, predicted_points, confidence_score,
                 confidence_level, confidence_breakdown, model_contributions, risk_factors,
                 recommendation, captain_suitability, selection_probability, uncertainty_range,
                 ab_test_variant, ab_test_id, prediction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id, prediction.player_id, prediction.player_name,
                prediction.predicted_points, prediction.confidence_score,
                prediction.confidence_level, json.dumps(prediction.confidence_breakdown.__dict__),
                json.dumps(prediction.model_contributions), json.dumps(prediction.risk_factors),
                prediction.recommendation, prediction.captain_suitability,
                prediction.selection_probability, json.dumps(prediction.uncertainty_range),
                prediction.ab_test_variant, prediction.ab_test_id, prediction.prediction_timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving ensemble prediction: {e}")
    
    def _save_team_prediction(self, team_prediction: TeamPrediction, match_context: Dict[str, Any]):
        """Save team prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            team_id = hashlib.sha256(
                f"{match_context.get('match_id', 'unknown')}_{datetime.now()}".encode()
            ).hexdigest()[:16]
            
            team_composition_data = [
                {
                    'player_id': p.player_id,
                    'player_name': p.player_name,
                    'predicted_points': p.predicted_points,
                    'confidence': p.confidence_score
                }
                for p in team_prediction.team_composition
            ]
            
            cursor.execute('''
                INSERT OR REPLACE INTO team_predictions
                (team_id, match_id, team_composition, total_predicted_points, team_confidence,
                 captain_id, vice_captain_id, budget_utilization, risk_assessment,
                 diversification_score, expected_rank, optimization_strategy, prediction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                team_id, match_context.get('match_id', 'unknown'),
                json.dumps(team_composition_data), team_prediction.total_predicted_points,
                team_prediction.team_confidence, team_prediction.captain_recommendation.player_id,
                team_prediction.vice_captain_recommendation.player_id, team_prediction.budget_utilization,
                json.dumps(team_prediction.risk_assessment), team_prediction.diversification_score,
                team_prediction.expected_rank, team_prediction.optimization_strategy, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Saved team prediction: {team_id}")
            
        except Exception as e:
            logger.error(f"❌ Error saving team prediction: {e}")
    
    def get_ensemble_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prediction accuracy by confidence level
            cursor.execute('''
                SELECT confidence_level, 
                       AVG(prediction_accuracy) as avg_accuracy,
                       COUNT(*) as sample_size
                FROM ensemble_predictions
                WHERE actual_points IS NOT NULL
                GROUP BY confidence_level
            ''')
            
            accuracy_by_confidence = cursor.fetchall()
            
            # Team prediction performance
            cursor.execute('''
                SELECT optimization_strategy,
                       AVG(team_accuracy) as avg_accuracy,
                       AVG(expected_rank - actual_rank) as rank_difference,
                       COUNT(*) as sample_size
                FROM team_predictions
                WHERE actual_points IS NOT NULL
                GROUP BY optimization_strategy
            ''')
            
            team_performance = cursor.fetchall()
            
            conn.close()
            
            return {
                'ensemble_accuracy': {
                    'by_confidence_level': [
                        {
                            'confidence_level': row[0],
                            'accuracy': f"{row[1]:.1%}" if row[1] else "N/A",
                            'sample_size': row[2]
                        }
                        for row in accuracy_by_confidence
                    ]
                },
                'team_performance': {
                    'by_strategy': [
                        {
                            'strategy': row[0],
                            'accuracy': f"{row[1]:.1%}" if row[1] else "N/A",
                            'avg_rank_difference': row[2] if row[2] else 0,
                            'sample_size': row[3]
                        }
                        for row in team_performance
                    ]
                },
                'system_metrics': {
                    'total_predictions': len(self.prediction_history),
                    'active_experiments': len(self.active_experiments),
                    'avg_team_confidence': f"{statistics.mean([t.team_confidence for t in self.team_performance_history]):.1%}" if self.team_performance_history else "N/A"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance insights: {e}")
            return {'error': str(e)}

# Global ensemble system instance
ensemble_system = EnsemblePredictionSystem()

def get_ensemble_system() -> EnsemblePredictionSystem:
    """Get global ensemble system instance"""
    return ensemble_system

def predict_player_with_ensemble(player_data: Dict[str, Any], 
                                match_context: Dict[str, Any]) -> EnsemblePrediction:
    """Generate ensemble prediction for player"""
    return ensemble_system.predict_player_performance(player_data, match_context)

def generate_optimal_team(available_players: List[Dict[str, Any]],
                         match_context: Dict[str, Any],
                         strategy: str = 'balanced') -> TeamPrediction:
    """Generate optimal team using ensemble system"""
    return ensemble_system.predict_optimal_team(available_players, match_context, strategy)