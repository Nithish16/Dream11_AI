#!/usr/bin/env python3
"""
Dynamic Credit Prediction Engine - Real-time Credit Valuation System
Advanced ML-based credit prediction replacing static 8.5 credit system
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import XGBRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime, timedelta

@dataclass
class CreditFeatures:
    """Features used for credit prediction"""
    # Performance features
    recent_form_avg: float = 0.0
    consistency_score: float = 0.0
    ema_score: float = 0.0
    performance_rating: float = 0.0
    
    # Role and scarcity features
    role_scarcity_index: float = 0.0
    position_value: float = 0.0
    versatility_score: float = 0.0
    
    # Market features
    ownership_demand: float = 0.0
    expert_rating: float = 0.0
    fantasy_reputation: float = 0.0
    
    # Contextual features
    matchup_favorability: float = 0.0
    venue_suitability: float = 0.0
    format_suitability: float = 0.0
    
    # Risk features
    injury_risk: float = 0.0
    form_volatility: float = 0.0
    selection_certainty: float = 1.0
    
    # Meta features
    tournament_importance: float = 0.5
    captain_potential: float = 0.0
    differential_value: float = 0.0

class DynamicCreditPredictor:
    """Advanced credit prediction system using ensemble ML models"""
    
    def __init__(self):
        self.models = {
            'xgboost': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_weights = {'xgboost': 0.7, 'random_forest': 0.3}
        
        # Credit constraints
        self.min_credit = 5.0
        self.max_credit = 15.0
        self.default_credit = 8.5
        
        # Load pre-trained models if available
        self._load_models()
    
    def extract_credit_features(self, player_data: Dict[str, Any], 
                               match_context: Dict[str, Any],
                               market_data: Dict[str, Any] = None) -> CreditFeatures:
        """Extract comprehensive features for credit prediction"""
        
        features = CreditFeatures()
        
        # Performance features
        recent_matches = player_data.get('recent_form', [])
        if recent_matches:
            recent_scores = [match.get('fantasy_points', 0) for match in recent_matches[:5]]
            features.recent_form_avg = np.mean(recent_scores) if recent_scores else 0.0
            features.consistency_score = self._calculate_consistency(recent_scores)
            features.form_volatility = np.std(recent_scores) if len(recent_scores) > 1 else 0.0
        
        features.ema_score = player_data.get('ema_score', 0.0)
        features.performance_rating = player_data.get('performance_rating', 50.0)
        
        # Role and scarcity features
        role = player_data.get('role', 'Unknown')
        features.role_scarcity_index = self._calculate_role_scarcity(role, match_context)
        features.position_value = self._calculate_position_value(role, player_data)
        features.versatility_score = self._calculate_versatility(role, player_data)
        
        # Market features
        if market_data:
            features.ownership_demand = market_data.get('player_ownership_percentage', 50.0) / 100
            features.expert_rating = max(0, market_data.get('expert_consensus', 0.0))
            features.fantasy_reputation = self._calculate_fantasy_reputation(player_data)
        
        # Contextual features
        features.matchup_favorability = player_data.get('matchup_score', 1.0)
        features.venue_suitability = self._calculate_venue_suitability(player_data, match_context)
        features.format_suitability = self._calculate_format_suitability(role, match_context)
        
        # Risk features
        features.injury_risk = player_data.get('injury_risk', 0.1)
        features.selection_certainty = self._calculate_selection_certainty(player_data)
        
        # Meta features
        features.tournament_importance = match_context.get('tournament_importance', 0.5)
        features.captain_potential = player_data.get('captain_vice_captain_probability', 0.0) / 100
        features.differential_value = self._calculate_differential_value(features.ownership_demand)
        
        return features
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate consistency score from recent performances"""
        if len(scores) < 2:
            return 0.0
        
        mean_score = np.mean(scores)
        if mean_score == 0:
            return 0.0
        
        cv = np.std(scores) / mean_score
        consistency = max(0, 100 - (cv * 50))
        return consistency
    
    def _calculate_role_scarcity(self, role: str, match_context: Dict[str, Any]) -> float:
        """Calculate how scarce/valuable the role is"""
        role_lower = role.lower()
        
        # Base scarcity values
        scarcity_map = {
            'wicket-keeper': 0.9,  # High scarcity - only 1-2 per team
            'wk': 0.9,
            'allrounder': 0.7,    # Medium-high scarcity - valuable versatility
            'all-rounder': 0.7,
            'bowler': 0.5,        # Medium scarcity
            'batsman': 0.4,       # Lower scarcity - more available
            'batter': 0.4
        }
        
        base_scarcity = 0.5  # Default
        for role_key, scarcity in scarcity_map.items():
            if role_key in role_lower:
                base_scarcity = scarcity
                break
        
        # Adjust for match format
        match_format = match_context.get('match_format', 'T20').lower()
        if match_format == 't20':
            if 'allrounder' in role_lower:
                base_scarcity *= 1.2  # All-rounders more valuable in T20
        elif match_format == 'test':
            if 'bowler' in role_lower:
                base_scarcity *= 1.1  # Bowlers more valuable in Tests
        
        return min(1.0, base_scarcity)
    
    def _calculate_position_value(self, role: str, player_data: Dict[str, Any]) -> float:
        """Calculate positional value based on batting order and role"""
        batting_position = player_data.get('batting_position', 7)
        role_lower = role.lower()
        
        # Position value mapping
        if 'wicket' in role_lower or 'wk' in role_lower:
            return 0.8  # High value regardless of position
        elif batting_position <= 2:
            return 0.9  # Openers - high value
        elif batting_position <= 4:
            return 0.8  # Top order - high value
        elif batting_position <= 6:
            return 0.6  # Middle order - medium value
        else:
            return 0.4  # Lower order - lower value
    
    def _calculate_versatility(self, role: str, player_data: Dict[str, Any]) -> float:
        """Calculate player versatility score"""
        role_lower = role.lower()
        
        # All-rounders are most versatile
        if 'allrounder' in role_lower or 'all-rounder' in role_lower:
            batting_avg = player_data.get('batting_stats', {}).get('average', 0)
            bowling_avg = player_data.get('bowling_stats', {}).get('average', 100)
            
            # Both skills good = high versatility
            if batting_avg > 30 and bowling_avg < 35:
                return 0.9
            elif batting_avg > 25 or bowling_avg < 40:
                return 0.7
            else:
                return 0.5
        
        # Wicket-keeper batsmen are versatile
        elif 'wicket' in role_lower or 'wk' in role_lower:
            return 0.7
        
        # Specialist roles
        else:
            return 0.3
    
    def _calculate_fantasy_reputation(self, player_data: Dict[str, Any]) -> float:
        """Calculate fantasy cricket reputation"""
        # Based on historical performance consistency
        performance_rating = player_data.get('performance_rating', 50.0)
        consistency = player_data.get('consistency_score', 50.0)
        
        # High performers with consistency get high reputation
        reputation = (performance_rating * 0.6 + consistency * 0.4) / 100
        
        return max(0, min(1, reputation))
    
    def _calculate_venue_suitability(self, player_data: Dict[str, Any], 
                                   match_context: Dict[str, Any]) -> float:
        """Calculate how suitable the venue is for the player"""
        # Placeholder - would analyze historical venue performance
        venue_stats = player_data.get('venue_specific', {})
        
        if venue_stats:
            home_avg = venue_stats.get('home_average', 40.0)
            away_avg = venue_stats.get('away_average', 35.0)
            neutral_avg = venue_stats.get('neutral_average', 37.0)
            
            # Determine venue type (simplified)
            venue_familiarity = venue_stats.get('venue_familiarity', 0.5)
            
            if venue_familiarity > 0.7:
                return min(1.0, home_avg / 50.0)
            else:
                return min(1.0, away_avg / 50.0)
        
        return 0.5  # Neutral if no data
    
    def _calculate_format_suitability(self, role: str, match_context: Dict[str, Any]) -> float:
        """Calculate format suitability for the player"""
        match_format = match_context.get('match_format', 'T20').lower()
        role_lower = role.lower()
        
        format_suitability = {
            't20': {
                'allrounder': 0.9,
                'wicket-keeper': 0.8,
                'batsman': 0.7,
                'bowler': 0.7
            },
            'odi': {
                'batsman': 0.9,
                'allrounder': 0.8,
                'bowler': 0.7,
                'wicket-keeper': 0.7
            },
            'test': {
                'bowler': 0.9,
                'batsman': 0.8,
                'allrounder': 0.7,
                'wicket-keeper': 0.6
            }
        }
        
        base_suitability = 0.7  # Default
        
        if match_format in format_suitability:
            for role_key, suitability in format_suitability[match_format].items():
                if role_key in role_lower:
                    base_suitability = suitability
                    break
        
        return base_suitability
    
    def _calculate_selection_certainty(self, player_data: Dict[str, Any]) -> float:
        """Calculate certainty of player being selected"""
        # Based on recent playing time and team position
        recent_matches = player_data.get('recent_form', [])
        
        if len(recent_matches) >= 5:
            # High certainty if played most recent matches
            return 0.95
        elif len(recent_matches) >= 3:
            return 0.8
        elif len(recent_matches) >= 1:
            return 0.6
        else:
            return 0.3  # Low certainty if not playing recently
    
    def _calculate_differential_value(self, ownership: float) -> float:
        """Calculate differential value (lower ownership = higher differential value)"""
        # Inverse relationship with ownership
        differential = 1.0 - ownership
        
        # Boost very low ownership players
        if ownership < 0.1:
            differential *= 1.5
        elif ownership < 0.2:
            differential *= 1.2
        
        return min(1.0, differential)
    
    def prepare_training_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical player performances"""
        X = []
        y = []
        
        for data_point in historical_data:
            features = self.extract_credit_features(
                data_point['player_data'],
                data_point['match_context'],
                data_point.get('market_data', {})
            )
            
            # Convert features to array
            feature_array = [
                features.recent_form_avg, features.consistency_score, features.ema_score,
                features.performance_rating, features.role_scarcity_index, features.position_value,
                features.versatility_score, features.ownership_demand, features.expert_rating,
                features.fantasy_reputation, features.matchup_favorability, features.venue_suitability,
                features.format_suitability, features.injury_risk, features.form_volatility,
                features.selection_certainty, features.tournament_importance, features.captain_potential,
                features.differential_value
            ]
            
            X.append(feature_array)
            y.append(data_point['actual_credit'])
        
        return np.array(X), np.array(y)
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train ensemble models and return performance metrics"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            results[model_name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'cv_mae': cv_mae
            }
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            
            print(f"{model_name} - Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        
        # Save models
        self._save_models()
        
        return results
    
    def predict_credit(self, player_data: Dict[str, Any], 
                      match_context: Dict[str, Any],
                      market_data: Dict[str, Any] = None) -> float:
        """Predict dynamic credit for a player"""
        
        # Extract features
        features = self.extract_credit_features(player_data, match_context, market_data)
        
        # Convert to array
        feature_array = np.array([[
            features.recent_form_avg, features.consistency_score, features.ema_score,
            features.performance_rating, features.role_scarcity_index, features.position_value,
            features.versatility_score, features.ownership_demand, features.expert_rating,
            features.fantasy_reputation, features.matchup_favorability, features.venue_suitability,
            features.format_suitability, features.injury_risk, features.form_volatility,
            features.selection_certainty, features.tournament_importance, features.captain_potential,
            features.differential_value
        ]])
        
        # Scale features
        try:
            feature_array_scaled = self.scaler.transform(feature_array)
        except:
            # If scaler not fitted, return default
            return self.default_credit
        
        # Ensemble prediction
        predictions = []
        total_weight = 0
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(feature_array_scaled)[0]
                weight = self.model_weights.get(model_name, 1.0)
                predictions.append(pred * weight)
                total_weight += weight
            except:
                continue
        
        if predictions and total_weight > 0:
            ensemble_pred = sum(predictions) / total_weight
            
            # Apply constraints
            final_credit = max(self.min_credit, min(self.max_credit, ensemble_pred))
            
            # Apply adjustments
            final_credit = self._apply_credit_adjustments(final_credit, features)
            
            return round(final_credit, 1)
        else:
            return self.default_credit
    
    def _apply_credit_adjustments(self, base_credit: float, features: CreditFeatures) -> float:
        """Apply final adjustments to credit prediction"""
        adjusted_credit = base_credit
        
        # High injury risk reduces credit
        if features.injury_risk > 0.3:
            adjusted_credit *= (1 - features.injury_risk * 0.2)
        
        # Low selection certainty reduces credit
        if features.selection_certainty < 0.5:
            adjusted_credit *= (0.8 + features.selection_certainty * 0.4)
        
        # Captain potential increases credit
        if features.captain_potential > 0.7:
            adjusted_credit *= 1.1
        elif features.captain_potential > 0.5:
            adjusted_credit *= 1.05
        
        # High differential value for low ownership
        if features.ownership_demand < 0.2 and features.performance_rating > 60:
            adjusted_credit *= 0.95  # Slightly reduce for differential value
        
        return max(self.min_credit, min(self.max_credit, adjusted_credit))
    
    def batch_predict_credits(self, players_data: List[Dict[str, Any]], 
                            match_context: Dict[str, Any],
                            market_data: Dict[str, Any] = None) -> List[float]:
        """Predict credits for multiple players"""
        credits = []
        
        for player_data in players_data:
            player_market_data = None
            if market_data:
                player_id = player_data.get('player_id', 0)
                player_market_data = market_data.get(str(player_id), {})
            
            credit = self.predict_credit(player_data, match_context, player_market_data)
            credits.append(credit)
        
        return credits
    
    def _save_models(self):
        """Save trained models and scalers"""
        model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(model_dir, f'credit_model_{name}.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'credit_scaler.pkl'))
        
        # Save feature importance
        joblib.dump(self.feature_importance, os.path.join(model_dir, 'credit_feature_importance.pkl'))
    
    def _load_models(self):
        """Load pre-trained models if available"""
        model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        
        if not os.path.exists(model_dir):
            return
        
        try:
            # Load models
            for name in self.models.keys():
                model_path = os.path.join(model_dir, f'credit_model_{name}.pkl')
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'credit_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load feature importance
            importance_path = os.path.join(model_dir, 'credit_feature_importance.pkl')
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)
                
            print("✅ Loaded pre-trained credit prediction models")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from trained models"""
        return self.feature_importance.copy()
    
    def generate_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for initial model training"""
        X = []
        y = []
        
        np.random.seed(42)
        
        for _ in range(n_samples):
            # Generate synthetic features
            recent_form_avg = np.random.gamma(2, 20)  # Realistic cricket scores
            consistency_score = np.random.beta(2, 2) * 100
            ema_score = recent_form_avg + np.random.normal(0, 5)
            performance_rating = np.random.beta(2, 3) * 100
            
            role_scarcity = np.random.choice([0.4, 0.5, 0.7, 0.9], p=[0.4, 0.3, 0.2, 0.1])
            position_value = np.random.beta(2, 2)
            versatility_score = np.random.beta(1.5, 2)
            
            ownership_demand = np.random.beta(2, 3)
            expert_rating = np.random.beta(2, 2)
            fantasy_reputation = np.random.beta(2, 2)
            
            matchup_favorability = np.random.beta(3, 3)
            venue_suitability = np.random.beta(2, 2)
            format_suitability = np.random.beta(3, 2)
            
            injury_risk = np.random.beta(1, 9)  # Low injury risk usually
            form_volatility = np.random.gamma(1, 5)
            selection_certainty = np.random.beta(4, 1)  # Usually high
            
            tournament_importance = np.random.beta(2, 2)
            captain_potential = np.random.beta(1, 3)
            differential_value = 1 - ownership_demand
            
            features = [
                recent_form_avg, consistency_score, ema_score, performance_rating,
                role_scarcity, position_value, versatility_score, ownership_demand,
                expert_rating, fantasy_reputation, matchup_favorability, venue_suitability,
                format_suitability, injury_risk, form_volatility, selection_certainty,
                tournament_importance, captain_potential, differential_value
            ]
            
            # Generate realistic credit based on features
            base_credit = 7.0 + (performance_rating / 100) * 4  # 7-11 base range
            
            # Adjustments
            if role_scarcity > 0.8:
                base_credit += 1.5  # WK premium
            elif role_scarcity > 0.6:
                base_credit += 1.0  # All-rounder premium
            
            if captain_potential > 0.7:
                base_credit += 0.8
            
            if form_volatility > 15:
                base_credit -= 0.5  # Volatile players get discount
            
            if injury_risk > 0.3:
                base_credit -= 1.0
            
            # Add some noise
            base_credit += np.random.normal(0, 0.3)
            
            # Constrain
            final_credit = max(5.0, min(15.0, base_credit))
            
            X.append(features)
            y.append(final_credit)
        
        return np.array(X), np.array(y)

# Utility function to replace static credit assignment
def assign_dynamic_credits(players_data: List[Dict[str, Any]], 
                         match_context: Dict[str, Any],
                         market_data: Dict[str, Any] = None) -> List[float]:
    """
    Replace the static assign_player_credits function with dynamic prediction
    """
    predictor = DynamicCreditPredictor()
    
    # Generate and train on synthetic data if no models exist
    if not hasattr(predictor.scaler, 'mean_'):
        print("🔧 Training credit prediction models with synthetic data...")
        X_synthetic, y_synthetic = predictor.generate_synthetic_training_data(2000)
        predictor.train_models(X_synthetic, y_synthetic)
    
    return predictor.batch_predict_credits(players_data, match_context, market_data)

# Export
__all__ = ['DynamicCreditPredictor', 'assign_dynamic_credits', 'CreditFeatures']