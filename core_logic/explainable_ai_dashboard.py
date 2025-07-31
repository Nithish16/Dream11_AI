#!/usr/bin/env python3
"""
Explainable AI Dashboard - Transparent Decision Making
Advanced interpretability and explanation system for AI decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import math
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_name: str
    importance_score: float
    confidence_interval: Tuple[float, float]
    feature_type: str  # 'numerical', 'categorical', 'derived'
    impact_direction: str  # 'positive', 'negative', 'neutral'
    stability_score: float = 0.0

@dataclass
class DecisionPath:
    """Decision path for individual predictions"""
    player_name: str
    final_decision: str
    decision_confidence: float
    key_factors: List[Dict[str, Any]] = field(default_factory=list)
    alternative_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class ModelExplanation:
    """Comprehensive model explanation"""
    model_name: str
    prediction_type: str
    global_importance: List[FeatureImportance] = field(default_factory=list)
    local_explanations: List[DecisionPath] = field(default_factory=list)
    model_performance: Dict[str, float] = field(default_factory=dict)
    uncertainty_analysis: Dict[str, Any] = field(default_factory=dict)
    bias_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TeamExplanation:
    """Team selection explanation"""
    team_composition: List[Dict[str, Any]] = field(default_factory=list)
    selection_rationale: Dict[str, Any] = field(default_factory=dict)
    trade_off_analysis: List[Dict[str, Any]] = field(default_factory=list)
    constraint_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_reward_profile: Dict[str, float] = field(default_factory=dict)

class SHAPExplainer:
    """SHAP-like explainer for model interpretability"""
    
    def __init__(self):
        self.baseline_values = {}
        self.feature_interactions = defaultdict(dict)
        self.explanation_cache = {}
    
    def explain_prediction(self, model_output: float, 
                         feature_values: Dict[str, float],
                         feature_names: List[str],
                         baseline: Optional[float] = None) -> Dict[str, float]:
        """Generate SHAP-like explanations for predictions"""
        
        if baseline is None:
            baseline = self.baseline_values.get('default', 50.0)
        
        # Calculate Shapley values (simplified approximation)
        shapley_values = {}
        total_contribution = model_output - baseline
        
        # Normalize feature values
        normalized_features = self._normalize_features(feature_values, feature_names)
        
        # Calculate individual contributions
        raw_contributions = {}
        for feature_name, value in normalized_features.items():
            # Simplified contribution calculation
            raw_contributions[feature_name] = value * self._get_feature_coefficient(feature_name)
        
        # Normalize contributions to sum to total_contribution
        total_raw = sum(raw_contributions.values())
        if total_raw != 0:
            normalization_factor = total_contribution / total_raw
            for feature_name in raw_contributions:
                shapley_values[feature_name] = raw_contributions[feature_name] * normalization_factor
        else:
            # Equal distribution if no meaningful contributions
            equal_contribution = total_contribution / len(feature_names)
            shapley_values = {name: equal_contribution for name in feature_names}
        
        return shapley_values
    
    def _normalize_features(self, feature_values: Dict[str, float], 
                          feature_names: List[str]) -> Dict[str, float]:
        """Normalize feature values for consistent interpretation"""
        normalized = {}
        
        for name in feature_names:
            if name in feature_values:
                value = feature_values[name]
                
                # Apply normalization based on feature type
                if 'score' in name.lower():
                    normalized[name] = (value - 50) / 50  # Assuming 50 is average score
                elif 'credits' in name.lower():
                    normalized[name] = (value - 8.5) / 3.0  # Assuming 8.5 average, 3.0 std
                elif 'percentage' in name.lower() or 'rate' in name.lower():
                    normalized[name] = (value - 50) / 50  # Percentage features
                else:
                    # Generic z-score normalization
                    normalized[name] = (value - self._get_feature_mean(name)) / self._get_feature_std(name)
            else:
                normalized[name] = 0.0
        
        return normalized
    
    def _get_feature_coefficient(self, feature_name: str) -> float:
        """Get feature importance coefficient"""
        # Predefined coefficients based on domain knowledge
        coefficients = {
            'final_score': 0.4,
            'ema_score': 0.3,
            'consistency_score': 0.2,
            'form_momentum': 0.15,
            'credits': -0.1,
            'ownership_prediction': -0.05,
            'injury_risk': -0.2,
            'venue_suitability': 0.1,
            'opposition_matchup': 0.15,
            'recent_form_avg': 0.25
        }
        
        return coefficients.get(feature_name, 0.1)
    
    def _get_feature_mean(self, feature_name: str) -> float:
        """Get feature mean for normalization"""
        means = {
            'final_score': 50.0,
            'ema_score': 50.0,
            'consistency_score': 50.0,
            'form_momentum': 0.0,
            'credits': 8.5,
            'ownership_prediction': 50.0,
            'injury_risk': 0.1,
            'venue_suitability': 0.5,
            'opposition_matchup': 0.5,
            'recent_form_avg': 40.0
        }
        return means.get(feature_name, 0.0)
    
    def _get_feature_std(self, feature_name: str) -> float:
        """Get feature standard deviation for normalization"""
        stds = {
            'final_score': 20.0,
            'ema_score': 15.0,
            'consistency_score': 25.0,
            'form_momentum': 0.5,
            'credits': 2.5,
            'ownership_prediction': 30.0,
            'injury_risk': 0.2,
            'venue_suitability': 0.3,
            'opposition_matchup': 0.3,
            'recent_form_avg': 15.0
        }
        return max(stds.get(feature_name, 1.0), 0.01)  # Avoid division by zero

class LIMEExplainer:
    """Local Interpretable Model-agnostic Explanations"""
    
    def __init__(self, num_perturbations: int = 1000):
        self.num_perturbations = num_perturbations
        self.local_models = {}
    
    def explain_instance(self, instance: Dict[str, float], 
                        prediction_function: callable,
                        feature_names: List[str],
                        num_features: int = 10) -> Dict[str, float]:
        """Generate local explanation for a single instance"""
        
        # Generate perturbations around the instance
        perturbations = self._generate_perturbations(instance, feature_names)
        
        # Get predictions for perturbations
        predictions = []
        for perturbation in perturbations:
            try:
                pred = prediction_function(perturbation)
                predictions.append(pred)
            except:
                predictions.append(0.0)
        
        # Fit local linear model
        X = np.array([[pert.get(fname, 0) for fname in feature_names] for pert in perturbations])
        y = np.array(predictions)
        
        # Calculate distances for weighting
        original_vector = np.array([instance.get(fname, 0) for fname in feature_names])
        distances = np.array([np.linalg.norm(row - original_vector) for row in X])
        weights = np.exp(-distances / np.std(distances)) if np.std(distances) > 0 else np.ones(len(distances))
        
        # Fit weighted linear regression
        coefficients = self._fit_weighted_linear_regression(X, y, weights)
        
        # Return top feature contributions
        feature_contributions = dict(zip(feature_names, coefficients))
        
        # Sort by absolute importance and return top features
        sorted_features = sorted(feature_contributions.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        return dict(sorted_features[:num_features])
    
    def _generate_perturbations(self, instance: Dict[str, float], 
                              feature_names: List[str]) -> List[Dict[str, float]]:
        """Generate perturbations around instance"""
        perturbations = []
        
        for _ in range(self.num_perturbations):
            perturbation = instance.copy()
            
            # Perturb each feature
            for feature_name in feature_names:
                if feature_name in instance:
                    original_value = instance[feature_name]
                    
                    # Add noise based on feature type
                    if 'credits' in feature_name.lower():
                        noise = np.random.normal(0, 1.0)  # Credit noise
                    elif 'score' in feature_name.lower():
                        noise = np.random.normal(0, 10.0)  # Score noise
                    else:
                        noise = np.random.normal(0, original_value * 0.1)  # 10% relative noise
                    
                    perturbation[feature_name] = max(0, original_value + noise)
            
            perturbations.append(perturbation)
        
        return perturbations
    
    def _fit_weighted_linear_regression(self, X: np.ndarray, y: np.ndarray, 
                                      weights: np.ndarray) -> np.ndarray:
        """Fit weighted linear regression"""
        try:
            # Weighted least squares
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            
            # Add regularization for stability
            regularization = 1e-6 * np.eye(XtWX.shape[0])
            coefficients = np.linalg.solve(XtWX + regularization, XtWy)
            
            return coefficients
        except:
            # Fallback to simple correlation
            correlations = []
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            
            return np.array(correlations)

class ExplainableAIDashboard:
    """Main explainable AI dashboard for comprehensive interpretability"""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        
        # Dashboard components
        self.explanations_history = []
        self.feature_importance_tracker = defaultdict(list)
        self.model_performance_tracker = defaultdict(list)
        
        # Visualization settings
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
    def explain_player_prediction(self, player_data: Dict[str, Any], 
                                prediction: float,
                                model_name: str = "Enhanced Predictor") -> DecisionPath:
        """Generate comprehensive explanation for player prediction"""
        
        player_name = player_data.get('name', 'Unknown Player')
        
        # Extract relevant features
        feature_values = self._extract_prediction_features(player_data)
        feature_names = list(feature_values.keys())
        
        # Generate SHAP explanations
        shap_values = self.shap_explainer.explain_prediction(
            prediction, feature_values, feature_names
        )
        
        # Generate LIME explanations
        def prediction_function(features):
            return self._simulate_prediction(features)
        
        lime_values = self.lime_explainer.explain_instance(
            feature_values, prediction_function, feature_names
        )
        
        # Combine explanations
        key_factors = self._combine_explanations(shap_values, lime_values, feature_values)
        
        # Generate alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(
            player_data, feature_values, prediction
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(player_data, shap_values)
        
        # Determine decision confidence
        decision_confidence = self._calculate_decision_confidence(
            prediction, shap_values, feature_values
        )
        
        decision_path = DecisionPath(
            player_name=player_name,
            final_decision=f"Predicted Score: {prediction:.1f}",
            decision_confidence=decision_confidence,
            key_factors=key_factors,
            alternative_scenarios=alternative_scenarios,
            risk_factors=risk_factors
        )
        
        return decision_path
    
    def explain_team_selection(self, selected_team: List[Dict[str, Any]], 
                             all_players: List[Dict[str, Any]],
                             selection_algorithm: str = "Enhanced Optimizer") -> TeamExplanation:
        """Generate comprehensive explanation for team selection"""
        
        # Analyze team composition
        team_composition = self._analyze_team_composition(selected_team)
        
        # Generate selection rationale
        selection_rationale = self._generate_selection_rationale(selected_team, all_players)
        
        # Analyze trade-offs
        trade_off_analysis = self._analyze_selection_tradeoffs(selected_team, all_players)
        
        # Analyze constraints
        constraint_analysis = self._analyze_constraint_satisfaction(selected_team)
        
        # Calculate risk-reward profile
        risk_reward_profile = self._calculate_team_risk_reward(selected_team)
        
        team_explanation = TeamExplanation(
            team_composition=team_composition,
            selection_rationale=selection_rationale,
            trade_off_analysis=trade_off_analysis,
            constraint_analysis=constraint_analysis,
            risk_reward_profile=risk_reward_profile
        )
        
        return team_explanation
    
    def generate_model_explanation(self, model_name: str, 
                                 training_data: List[Dict[str, Any]],
                                 predictions: List[float]) -> ModelExplanation:
        """Generate global model explanation"""
        
        # Calculate global feature importance
        global_importance = self._calculate_global_feature_importance(training_data, predictions)
        
        # Generate local explanations for sample instances
        local_explanations = []
        sample_size = min(10, len(training_data))
        sample_indices = np.random.choice(len(training_data), sample_size, replace=False)
        
        for idx in sample_indices:
            player_data = training_data[idx]
            prediction = predictions[idx]
            local_explanation = self.explain_player_prediction(player_data, prediction, model_name)
            local_explanations.append(local_explanation)
        
        # Calculate model performance metrics
        model_performance = self._calculate_model_performance_metrics(predictions, training_data)
        
        # Analyze uncertainty
        uncertainty_analysis = self._analyze_model_uncertainty(predictions, training_data)
        
        # Analyze bias
        bias_analysis = self._analyze_model_bias(training_data, predictions)
        
        model_explanation = ModelExplanation(
            model_name=model_name,
            prediction_type="Player Performance Score",
            global_importance=global_importance,
            local_explanations=local_explanations,
            model_performance=model_performance,
            uncertainty_analysis=uncertainty_analysis,
            bias_analysis=bias_analysis
        )
        
        return model_explanation
    
    def _extract_prediction_features(self, player_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features for prediction explanation"""
        features = {}
        
        # Core performance features
        features['final_score'] = player_data.get('final_score', 0.0)
        features['ema_score'] = player_data.get('ema_score', 0.0)
        features['consistency_score'] = player_data.get('consistency_score', 0.0)
        features['form_momentum'] = player_data.get('form_momentum', 0.0)
        
        # Economic features
        features['credits'] = player_data.get('credits', 8.5)
        features['ownership_prediction'] = player_data.get('ownership_prediction', 50.0)
        
        # Risk features
        features['injury_risk'] = player_data.get('injury_risk', 0.1)
        features['form_volatility'] = player_data.get('form_volatility', 0.0)
        
        # Contextual features
        features['venue_suitability'] = player_data.get('venue_suitability', 0.5)
        features['opposition_matchup'] = player_data.get('opposition_matchup', 0.5)
        
        # Recent performance
        recent_form = player_data.get('recent_form', [])
        if recent_form:
            recent_scores = [match.get('fantasy_points', 0) for match in recent_form[:5]]
            features['recent_form_avg'] = np.mean(recent_scores) if recent_scores else 0.0
        else:
            features['recent_form_avg'] = 0.0
        
        return features
    
    def _simulate_prediction(self, features: Dict[str, float]) -> float:
        """Simulate prediction for LIME explanation"""
        # Simplified prediction function
        prediction = (
            features.get('final_score', 0) * 0.4 +
            features.get('ema_score', 0) * 0.3 +
            features.get('consistency_score', 0) * 0.2 +
            features.get('recent_form_avg', 0) * 0.1
        )
        
        return max(0, prediction)
    
    def _combine_explanations(self, shap_values: Dict[str, float], 
                            lime_values: Dict[str, float],
                            feature_values: Dict[str, float]) -> List[Dict[str, Any]]:
        """Combine SHAP and LIME explanations"""
        key_factors = []
        
        # Get all features mentioned in either explanation
        all_features = set(shap_values.keys()) | set(lime_values.keys())
        
        for feature in all_features:
            shap_contribution = shap_values.get(feature, 0.0)
            lime_contribution = lime_values.get(feature, 0.0)
            feature_value = feature_values.get(feature, 0.0)
            
            # Average the contributions
            avg_contribution = (shap_contribution + lime_contribution) / 2
            
            # Determine impact direction
            if avg_contribution > 0.1:
                impact = "Positive"
            elif avg_contribution < -0.1:
                impact = "Negative"
            else:
                impact = "Neutral"
            
            # Generate human-readable explanation
            explanation = self._generate_feature_explanation(feature, feature_value, avg_contribution)
            
            key_factors.append({
                'feature_name': feature,
                'feature_value': feature_value,
                'contribution': avg_contribution,
                'impact': impact,
                'explanation': explanation,
                'confidence': min(abs(avg_contribution) * 2, 1.0)
            })
        
        # Sort by absolute contribution
        key_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return key_factors[:8]  # Top 8 factors
    
    def _generate_feature_explanation(self, feature_name: str, 
                                    feature_value: float, 
                                    contribution: float) -> str:
        """Generate human-readable explanation for feature"""
        
        explanations = {
            'final_score': f"Player's overall performance score of {feature_value:.1f}",
            'ema_score': f"Exponential moving average score of {feature_value:.1f}",
            'consistency_score': f"Performance consistency rating of {feature_value:.1f}%",
            'form_momentum': f"Current form trend: {'improving' if feature_value > 0 else 'declining'}",
            'credits': f"Player cost of {feature_value:.1f} credits",
            'ownership_prediction': f"Expected ownership of {feature_value:.1f}%",
            'injury_risk': f"Injury risk level of {feature_value*100:.1f}%",
            'venue_suitability': f"Venue suitability score of {feature_value:.2f}",
            'opposition_matchup': f"Opposition matchup advantage of {feature_value:.2f}",
            'recent_form_avg': f"Recent form average of {feature_value:.1f} points"
        }
        
        base_explanation = explanations.get(feature_name, f"{feature_name}: {feature_value:.2f}")
        
        if contribution > 0:
            impact_text = "positively influences the prediction"
        elif contribution < 0:
            impact_text = "negatively influences the prediction"
        else:
            impact_text = "has minimal impact on the prediction"
        
        return f"{base_explanation} {impact_text}"
    
    def _generate_alternative_scenarios(self, player_data: Dict[str, Any], 
                                      feature_values: Dict[str, float],
                                      current_prediction: float) -> List[Dict[str, Any]]:
        """Generate alternative scenarios for what-if analysis"""
        scenarios = []
        
        # Scenario 1: If player had perfect recent form
        scenario_1 = feature_values.copy()
        scenario_1['recent_form_avg'] = 80.0
        scenario_1['form_momentum'] = 0.5
        new_prediction_1 = self._simulate_prediction(scenario_1)
        
        scenarios.append({
            'scenario': "Perfect Recent Form",
            'description': "If player had exceptional recent performances",
            'predicted_score': new_prediction_1,
            'change': new_prediction_1 - current_prediction,
            'likelihood': "Low"
        })
        
        # Scenario 2: If player had injury concerns
        scenario_2 = feature_values.copy()
        scenario_2['injury_risk'] = 0.8
        scenario_2['final_score'] *= 0.7
        new_prediction_2 = self._simulate_prediction(scenario_2)
        
        scenarios.append({
            'scenario': "Injury Concerns",
            'description': "If player had significant injury risk",
            'predicted_score': new_prediction_2,
            'change': new_prediction_2 - current_prediction,
            'likelihood': "Medium"
        })
        
        # Scenario 3: If player had ideal venue conditions
        scenario_3 = feature_values.copy()
        scenario_3['venue_suitability'] = 1.0
        scenario_3['opposition_matchup'] = 0.8
        new_prediction_3 = self._simulate_prediction(scenario_3)
        
        scenarios.append({
            'scenario': "Favorable Conditions",
            'description': "If venue and opposition matchup were ideal",
            'predicted_score': new_prediction_3,
            'change': new_prediction_3 - current_prediction,
            'likelihood': "Medium"
        })
        
        return scenarios
    
    def _identify_risk_factors(self, player_data: Dict[str, Any], 
                             shap_values: Dict[str, float]) -> List[str]:
        """Identify risk factors for player selection"""
        risks = []
        
        # High injury risk
        if player_data.get('injury_risk', 0) > 0.3:
            risks.append(f"High injury risk ({player_data.get('injury_risk', 0)*100:.1f}%)")
        
        # Poor recent form
        if shap_values.get('recent_form_avg', 0) < -5:
            risks.append("Below-average recent form")
        
        # High volatility
        if player_data.get('form_volatility', 0) > 20:
            risks.append("Inconsistent performance history")
        
        # High ownership
        if player_data.get('ownership_prediction', 0) > 80:
            risks.append("Very high expected ownership")
        
        # Unfavorable matchup
        if player_data.get('opposition_matchup', 0.5) < 0.3:
            risks.append("Challenging opposition matchup")
        
        return risks
    
    def _calculate_decision_confidence(self, prediction: float, 
                                     shap_values: Dict[str, float],
                                     feature_values: Dict[str, float]) -> float:
        """Calculate confidence in the prediction decision"""
        
        # Factors that increase confidence
        positive_factors = 0
        negative_factors = 0
        
        # Strong positive contributors increase confidence
        strong_positive = sum(1 for v in shap_values.values() if v > 5)
        positive_factors += strong_positive * 0.1
        
        # Consistent features (low volatility) increase confidence
        if feature_values.get('consistency_score', 0) > 70:
            positive_factors += 0.2
        
        # Recent good form increases confidence
        if feature_values.get('recent_form_avg', 0) > 60:
            positive_factors += 0.15
        
        # Risk factors decrease confidence
        if feature_values.get('injury_risk', 0) > 0.3:
            negative_factors += 0.2
        
        if feature_values.get('form_volatility', 0) > 25:
            negative_factors += 0.15
        
        # Calculate base confidence from prediction magnitude
        base_confidence = min(prediction / 100, 0.8)
        
        # Adjust confidence
        final_confidence = base_confidence + positive_factors - negative_factors
        
        return max(0.1, min(0.95, final_confidence))
    
    def _analyze_team_composition(self, selected_team: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze team composition breakdown"""
        composition = []
        
        # Role analysis
        role_counts = defaultdict(int)
        for player in selected_team:
            role = player.get('role', 'Unknown').lower()
            if 'bat' in role and 'allrounder' not in role:
                role_counts['Batsman'] += 1
            elif 'bowl' in role and 'allrounder' not in role:
                role_counts['Bowler'] += 1
            elif 'allrounder' in role:
                role_counts['All-rounder'] += 1
            elif 'wk' in role or 'wicket' in role:
                role_counts['Wicket-keeper'] += 1
            else:
                role_counts['Other'] += 1
        
        for role, count in role_counts.items():
            composition.append({
                'category': 'Role Distribution',
                'subcategory': role,
                'count': count,
                'percentage': (count / len(selected_team)) * 100
            })
        
        # Team analysis
        team_counts = defaultdict(int)
        for player in selected_team:
            team = player.get('team_name', 'Unknown')
            team_counts[team] += 1
        
        for team, count in team_counts.items():
            composition.append({
                'category': 'Team Distribution',
                'subcategory': team,
                'count': count,
                'percentage': (count / len(selected_team)) * 100
            })
        
        return composition
    
    def _generate_selection_rationale(self, selected_team: List[Dict[str, Any]], 
                                    all_players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate rationale for team selection"""
        
        # Calculate team stats
        total_score = sum(player.get('final_score', 0) for player in selected_team)
        total_credits = sum(player.get('credits', 8.5) for player in selected_team)
        avg_ownership = np.mean([player.get('ownership_prediction', 50) for player in selected_team])
        
        # Top performers
        top_performers = sorted(selected_team, key=lambda p: p.get('final_score', 0), reverse=True)[:3]
        
        # Budget efficiency
        budget_utilization = (total_credits / 100) * 100
        
        rationale = {
            'total_expected_score': total_score,
            'budget_utilization': budget_utilization,
            'average_ownership': avg_ownership,
            'top_performers': [p.get('name', 'Unknown') for p in top_performers],
            'selection_strategy': self._identify_selection_strategy(selected_team),
            'key_strengths': self._identify_team_strengths(selected_team),
            'potential_concerns': self._identify_team_concerns(selected_team)
        }
        
        return rationale
    
    def _identify_selection_strategy(self, selected_team: List[Dict[str, Any]]) -> str:
        """Identify the selection strategy used"""
        
        avg_ownership = np.mean([player.get('ownership_prediction', 50) for player in selected_team])
        avg_credits = np.mean([player.get('credits', 8.5) for player in selected_team])
        score_variance = np.var([player.get('final_score', 50) for player in selected_team])
        
        if avg_ownership < 40:
            return "Differential/Contrarian Strategy"
        elif avg_credits > 9.0:
            return "Premium Player Strategy"
        elif score_variance < 100:
            return "Balanced/Conservative Strategy"
        else:
            return "High-Risk/High-Reward Strategy"
    
    def _identify_team_strengths(self, selected_team: List[Dict[str, Any]]) -> List[str]:
        """Identify team strengths"""
        strengths = []
        
        # High total score
        total_score = sum(player.get('final_score', 0) for player in selected_team)
        if total_score > 600:
            strengths.append("High expected team score")
        
        # Good budget utilization
        total_credits = sum(player.get('credits', 8.5) for player in selected_team)
        if 95 <= total_credits <= 100:
            strengths.append("Optimal budget utilization")
        
        # Low average ownership
        avg_ownership = np.mean([player.get('ownership_prediction', 50) for player in selected_team])
        if avg_ownership < 45:
            strengths.append("Good differential potential")
        
        # Consistent performers
        consistent_players = sum(1 for p in selected_team if p.get('consistency_score', 0) > 70)
        if consistent_players >= 6:
            strengths.append("High number of consistent performers")
        
        return strengths
    
    def _identify_team_concerns(self, selected_team: List[Dict[str, Any]]) -> List[str]:
        """Identify potential team concerns"""
        concerns = []
        
        # High injury risk players
        risky_players = sum(1 for p in selected_team if p.get('injury_risk', 0) > 0.3)
        if risky_players > 2:
            concerns.append(f"{risky_players} players with elevated injury risk")
        
        # Budget concerns
        total_credits = sum(player.get('credits', 8.5) for player in selected_team)
        if total_credits < 90:
            concerns.append("Underutilized budget - could upgrade players")
        
        # High ownership concentration
        high_ownership = sum(1 for p in selected_team if p.get('ownership_prediction', 0) > 70)
        if high_ownership > 5:
            concerns.append("High ownership in multiple players reduces differential upside")
        
        # Form concerns
        poor_form = sum(1 for p in selected_team if p.get('form_momentum', 0) < -0.2)
        if poor_form > 3:
            concerns.append(f"{poor_form} players showing declining form")
        
        return concerns
    
    def _analyze_selection_tradeoffs(self, selected_team: List[Dict[str, Any]], 
                                   all_players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze selection trade-offs"""
        tradeoffs = []
        
        # Find players just outside the team
        selected_ids = {p.get('player_id', 0) for p in selected_team}
        alternatives = [p for p in all_players if p.get('player_id', 0) not in selected_ids]
        alternatives.sort(key=lambda p: p.get('final_score', 0), reverse=True)
        
        # Analyze top alternatives
        for alt_player in alternatives[:5]:
            # Find most similar selected player for comparison
            most_similar = min(selected_team, 
                             key=lambda p: abs(p.get('credits', 8.5) - alt_player.get('credits', 8.5)))
            
            tradeoffs.append({
                'alternative_player': alt_player.get('name', 'Unknown'),
                'alternative_score': alt_player.get('final_score', 0),
                'selected_player': most_similar.get('name', 'Unknown'),
                'selected_score': most_similar.get('final_score', 0),
                'score_difference': alt_player.get('final_score', 0) - most_similar.get('final_score', 0),
                'credit_difference': alt_player.get('credits', 8.5) - most_similar.get('credits', 8.5),
                'ownership_difference': alt_player.get('ownership_prediction', 50) - most_similar.get('ownership_prediction', 50)
            })
        
        return tradeoffs
    
    def _analyze_constraint_satisfaction(self, selected_team: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the team satisfies constraints"""
        
        # Role constraints
        role_counts = defaultdict(int)
        for player in selected_team:
            role = player.get('role', 'Unknown').lower()
            if 'bat' in role and 'allrounder' not in role:
                role_counts['batsman'] += 1
            elif 'bowl' in role and 'allrounder' not in role:
                role_counts['bowler'] += 1
            elif 'allrounder' in role:
                role_counts['allrounder'] += 1
            elif 'wk' in role or 'wicket' in role:
                role_counts['wicketkeeper'] += 1
        
        # Budget constraint
        total_credits = sum(player.get('credits', 8.5) for player in selected_team)
        
        constraint_analysis = {
            'team_size': {
                'required': 11,
                'actual': len(selected_team),
                'satisfied': len(selected_team) == 11
            },
            'batsmen': {
                'min_required': 3,
                'max_allowed': 6,
                'actual': role_counts['batsman'],
                'satisfied': 3 <= role_counts['batsman'] <= 6
            },
            'bowlers': {
                'min_required': 3,
                'max_allowed': 6,
                'actual': role_counts['bowler'],
                'satisfied': 3 <= role_counts['bowler'] <= 6
            },
            'allrounders': {
                'min_required': 1,
                'max_allowed': 4,
                'actual': role_counts['allrounder'],
                'satisfied': 1 <= role_counts['allrounder'] <= 4
            },
            'wicketkeepers': {
                'min_required': 1,
                'max_allowed': 2,
                'actual': role_counts['wicketkeeper'],
                'satisfied': 1 <= role_counts['wicketkeeper'] <= 2
            },
            'budget': {
                'max_allowed': 100.0,
                'actual': total_credits,
                'satisfied': total_credits <= 100.0,
                'utilization': (total_credits / 100.0) * 100
            }
        }
        
        return constraint_analysis
    
    def _calculate_team_risk_reward(self, selected_team: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate team risk-reward profile"""
        
        scores = [player.get('final_score', 0) for player in selected_team]
        consistencies = [player.get('consistency_score', 50) for player in selected_team]
        injury_risks = [player.get('injury_risk', 0.1) for player in selected_team]
        
        # Reward metrics
        expected_return = sum(scores)
        upside_potential = sum(score * 1.5 for score in scores if score > 60)
        
        # Risk metrics
        downside_risk = sum(max(0, 50 - score) for score in scores)
        injury_risk = np.mean(injury_risks)
        consistency_risk = 100 - np.mean(consistencies)
        
        # Overall metrics
        risk_score = (downside_risk + injury_risk * 100 + consistency_risk) / 3
        reward_score = (expected_return + upside_potential) / 2
        
        risk_reward_ratio = reward_score / max(risk_score, 1)
        
        return {
            'expected_return': expected_return,
            'upside_potential': upside_potential,
            'downside_risk': downside_risk,
            'injury_risk': injury_risk,
            'consistency_risk': consistency_risk,
            'overall_risk_score': risk_score,
            'overall_reward_score': reward_score,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _calculate_global_feature_importance(self, training_data: List[Dict[str, Any]], 
                                           predictions: List[float]) -> List[FeatureImportance]:
        """Calculate global feature importance across all predictions"""
        
        # Extract features from all training data
        all_features = defaultdict(list)
        for player_data in training_data:
            feature_values = self._extract_prediction_features(player_data)
            for feature_name, value in feature_values.items():
                all_features[feature_name].append(value)
        
        # Calculate importance for each feature
        feature_importances = []
        
        for feature_name, values in all_features.items():
            if len(values) > 1:
                # Calculate correlation with predictions
                correlation = np.corrcoef(values, predictions[:len(values)])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Calculate stability (inverse of coefficient of variation)
                cv = np.std(values) / (np.mean(values) + 1e-8)
                stability = 1.0 / (1.0 + cv)
                
                # Determine impact direction
                if correlation > 0.1:
                    impact_direction = 'positive'
                elif correlation < -0.1:
                    impact_direction = 'negative'
                else:
                    impact_direction = 'neutral'
                
                # Confidence interval (simplified)
                importance_score = abs(correlation)
                margin = 0.1 * importance_score
                confidence_interval = (importance_score - margin, importance_score + margin)
                
                feature_importance = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=importance_score,
                    confidence_interval=confidence_interval,
                    feature_type='numerical',  # Simplified
                    impact_direction=impact_direction,
                    stability_score=stability
                )
                
                feature_importances.append(feature_importance)
        
        # Sort by importance score
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        
        return feature_importances
    
    def _calculate_model_performance_metrics(self, predictions: List[float], 
                                           training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate model performance metrics"""
        
        # Simulate actual scores for evaluation (in practice, use real results)
        actual_scores = [player.get('final_score', 50) + np.random.normal(0, 5) 
                        for player in training_data]
        
        # Calculate metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual_scores)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actual_scores))**2))
        
        # Correlation
        correlation = np.corrcoef(predictions, actual_scores)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # R-squared
        ss_res = np.sum((np.array(actual_scores) - np.array(predictions))**2)
        ss_tot = np.sum((np.array(actual_scores) - np.mean(actual_scores))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'correlation': correlation,
            'r_squared': r_squared,
            'prediction_range': max(predictions) - min(predictions),
            'prediction_mean': np.mean(predictions),
            'prediction_std': np.std(predictions)
        }
    
    def _analyze_model_uncertainty(self, predictions: List[float], 
                                 training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model uncertainty"""
        
        # Prediction uncertainty
        prediction_std = np.std(predictions)
        prediction_range = max(predictions) - min(predictions)
        
        # Feature uncertainty (based on missing values, outliers)
        feature_completeness = []
        for player_data in training_data:
            features = self._extract_prediction_features(player_data)
            completeness = sum(1 for v in features.values() if v != 0) / len(features)
            feature_completeness.append(completeness)
        
        avg_completeness = np.mean(feature_completeness)
        
        # Confidence intervals for predictions
        confidence_intervals = []
        for pred in predictions:
            margin = prediction_std * 1.96  # 95% confidence interval
            confidence_intervals.append((pred - margin, pred + margin))
        
        return {
            'prediction_uncertainty': prediction_std,
            'prediction_range': prediction_range,
            'feature_completeness': avg_completeness,
            'confidence_intervals': confidence_intervals[:5],  # Sample
            'uncertainty_sources': [
                'Feature measurement noise',
                'Model approximation error',
                'Data incompleteness',
                'Temporal changes in player performance'
            ]
        }
    
    def _analyze_model_bias(self, training_data: List[Dict[str, Any]], 
                          predictions: List[float]) -> Dict[str, Any]:
        """Analyze potential model bias"""
        
        # Team bias analysis
        team_performance = defaultdict(list)
        for i, player_data in enumerate(training_data):
            team = player_data.get('team_name', 'Unknown')
            if i < len(predictions):
                team_performance[team].append(predictions[i])
        
        team_bias = {}
        overall_mean = np.mean(predictions)
        for team, preds in team_performance.items():
            if len(preds) > 1:
                team_mean = np.mean(preds)
                bias = team_mean - overall_mean
                team_bias[team] = bias
        
        # Role bias analysis
        role_performance = defaultdict(list)
        for i, player_data in enumerate(training_data):
            role = player_data.get('role', 'Unknown')
            if i < len(predictions):
                role_performance[role].append(predictions[i])
        
        role_bias = {}
        for role, preds in role_performance.items():
            if len(preds) > 1:
                role_mean = np.mean(preds)
                bias = role_mean - overall_mean
                role_bias[role] = bias
        
        return {
            'team_bias': dict(sorted(team_bias.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
            'role_bias': dict(sorted(role_bias.items(), key=lambda x: abs(x[1]), reverse=True)),
            'overall_bias': np.mean(predictions) - 50.0,  # Assuming 50 is neutral
            'bias_mitigation_suggestions': [
                'Include more balanced training data across teams',
                'Apply fairness constraints during optimization',
                'Regular bias monitoring and correction',
                'Cross-validation across different team compositions'
            ]
        }
    
    def generate_dashboard_report(self, team_explanation: TeamExplanation, 
                                model_explanation: ModelExplanation) -> str:
        """Generate comprehensive dashboard report"""
        
        report = []
        report.append("🔍 EXPLAINABLE AI DASHBOARD REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Team Selection Summary
        report.append("📊 TEAM SELECTION ANALYSIS")
        report.append("-" * 30)
        rationale = team_explanation.selection_rationale
        report.append(f"Expected Score: {rationale['total_expected_score']:.1f}")
        report.append(f"Budget Utilization: {rationale['budget_utilization']:.1f}%")
        report.append(f"Selection Strategy: {rationale['selection_strategy']}")
        report.append(f"Top Performers: {', '.join(rationale['top_performers'])}")
        report.append("")
        
        # Key Strengths
        if rationale['key_strengths']:
            report.append("✅ KEY STRENGTHS:")
            for strength in rationale['key_strengths']:
                report.append(f"  • {strength}")
            report.append("")
        
        # Potential Concerns
        if rationale['potential_concerns']:
            report.append("⚠️ POTENTIAL CONCERNS:")
            for concern in rationale['potential_concerns']:
                report.append(f"  • {concern}")
            report.append("")
        
        # Model Performance
        report.append("🤖 MODEL PERFORMANCE")
        report.append("-" * 30)
        perf = model_explanation.model_performance
        report.append(f"Model Accuracy (R²): {perf['r_squared']:.3f}")
        report.append(f"Mean Absolute Error: {perf['mean_absolute_error']:.2f}")
        report.append(f"Prediction Correlation: {perf['correlation']:.3f}")
        report.append("")
        
        # Top Feature Importance
        report.append("🎯 TOP INFLUENTIAL FACTORS")
        report.append("-" * 30)
        for i, feature in enumerate(model_explanation.global_importance[:5]):
            direction = "↗️" if feature.impact_direction == 'positive' else "↘️" if feature.impact_direction == 'negative' else "↔️"
            report.append(f"{i+1}. {feature.feature_name} {direction} (Impact: {feature.importance_score:.3f})")
        report.append("")
        
        # Risk Analysis
        risk_reward = team_explanation.risk_reward_profile
        report.append("⚖️ RISK-REWARD ANALYSIS")
        report.append("-" * 30)
        report.append(f"Risk-Reward Ratio: {risk_reward['risk_reward_ratio']:.2f}")
        report.append(f"Injury Risk: {risk_reward['injury_risk']*100:.1f}%")
        report.append(f"Consistency Risk: {risk_reward['consistency_risk']:.1f}%")
        report.append("")
        
        return "\n".join(report)

# Export
__all__ = ['ExplainableAIDashboard', 'SHAPExplainer', 'LIMEExplainer', 
           'ModelExplanation', 'TeamExplanation', 'DecisionPath', 'FeatureImportance']