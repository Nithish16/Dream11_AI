#!/usr/bin/env python3
"""
Reinforcement Learning Strategy Engine - Adaptive Team Selection
Advanced RL-based strategy learning for optimal team composition
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RLState:
    """Reinforcement Learning State representation"""
    # Match context
    venue_type: str = "neutral"
    pitch_condition: str = "balanced"
    weather_condition: str = "clear"
    match_importance: float = 0.5
    
    # Team composition
    batsmen_count: int = 0
    bowlers_count: int = 0
    allrounders_count: int = 0
    wicketkeepers_count: int = 0
    
    # Budget utilization
    credits_used: float = 0.0
    credits_remaining: float = 100.0
    
    # Performance metrics
    avg_form_score: float = 50.0
    total_ownership: float = 0.0
    risk_level: float = 0.5
    
    # Historical performance
    similar_context_success: float = 0.5

@dataclass
class RLAction:
    """Reinforcement Learning Action representation"""
    action_type: str  # 'select_player', 'change_captain', 'adjust_risk'
    player_id: Optional[int] = None
    target_role: Optional[str] = None
    risk_adjustment: float = 0.0
    confidence: float = 0.5

@dataclass
class RLExperience:
    """Experience tuple for RL learning"""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)

class QNetworkAgent:
    """Deep Q-Network Agent for team selection"""
    
    def __init__(self, state_size: int = 15, action_size: int = 10, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Q-table approximation (in production, use neural network)
        self.q_table = defaultdict(lambda: np.random.uniform(-1, 1, action_size))
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Performance tracking
        self.training_rewards = []
        self.success_rate = 0.0
    
    def state_to_key(self, state: RLState) -> str:
        """Convert state to hashable key"""
        return f"{state.venue_type}_{state.pitch_condition}_{state.batsmen_count}_{state.bowlers_count}_{int(state.credits_used)}_{int(state.avg_form_score)}"
    
    def get_action(self, state: RLState, valid_actions: List[int]) -> int:
        """Choose action using epsilon-greedy policy"""
        
        state_key = self.state_to_key(state)
        
        if random.random() <= self.epsilon:
            # Exploration: random action from valid actions
            return random.choice(valid_actions) if valid_actions else 0
        else:
            # Exploitation: best action from Q-table
            q_values = self.q_table[state_key]
            
            # Mask invalid actions
            masked_q_values = np.full(self.action_size, -np.inf)
            for action in valid_actions:
                if action < len(q_values):
                    masked_q_values[action] = q_values[action]
            
            return np.argmax(masked_q_values)
    
    def remember(self, experience: RLExperience):
        """Store experience in memory"""
        self.memory.append(experience)
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        for experience in batch:
            state_key = self.state_to_key(experience.state)
            next_state_key = self.state_to_key(experience.next_state)
            
            # Current Q-values
            current_q = self.q_table[state_key].copy()
            
            # Target Q-value
            if experience.done:
                target = experience.reward
            else:
                next_q_max = np.max(self.q_table[next_state_key])
                target = experience.reward + self.gamma * next_q_max
            
            # Update Q-value
            action_idx = self._action_to_index(experience.action)
            if 0 <= action_idx < self.action_size:
                current_q[action_idx] = target
                self.q_table[state_key] = current_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _action_to_index(self, action: RLAction) -> int:
        """Convert action to index"""
        action_mapping = {
            'select_batsman': 0,
            'select_bowler': 1,
            'select_allrounder': 2,
            'select_wicketkeeper': 3,
            'change_captain': 4,
            'increase_risk': 5,
            'decrease_risk': 6,
            'optimize_ownership': 7,
            'balance_team': 8,
            'finalize_team': 9
        }
        return action_mapping.get(action.action_type, 0)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'success_rate': self.success_rate
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.q_table = defaultdict(lambda: np.random.uniform(-1, 1, self.action_size))
                self.q_table.update(model_data.get('q_table', {}))
                self.epsilon = model_data.get('epsilon', self.epsilon_min)
                self.training_rewards = model_data.get('training_rewards', [])
                self.success_rate = model_data.get('success_rate', 0.0)
                
                print(f"✅ Loaded RL model from {filepath}")
            except Exception as e:
                print(f"⚠️ Error loading RL model: {e}")

class PolicyGradientAgent:
    """Policy Gradient Agent for strategic decisions"""
    
    def __init__(self, state_size: int = 15, action_size: int = 10, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Policy parameters (simplified linear policy)
        self.policy_weights = np.random.normal(0, 0.1, (state_size, action_size))
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities from policy"""
        logits = np.dot(state, self.policy_weights)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action based on policy"""
        probabilities = self.get_action_probabilities(state)
        action = np.random.choice(self.action_size, p=probabilities)
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float):
        """Store experience for episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        if not self.episode_states:
            return
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards()
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Update policy
        for state, action, reward in zip(self.episode_states, self.episode_actions, discounted_rewards):
            # Calculate gradients
            probabilities = self.get_action_probabilities(state)
            
            # Policy gradient update
            grad = np.outer(state, np.zeros(self.action_size))
            grad[:, action] = state * (1 - probabilities[action])
            
            # Update weights
            self.policy_weights += self.learning_rate * reward * grad
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _calculate_discounted_rewards(self, gamma: float = 0.99) -> np.ndarray:
        """Calculate discounted cumulative rewards"""
        discounted_rewards = np.zeros_like(self.episode_rewards, dtype=np.float32)
        cumulative_reward = 0
        
        for i in reversed(range(len(self.episode_rewards))):
            cumulative_reward = self.episode_rewards[i] + gamma * cumulative_reward
            discounted_rewards[i] = cumulative_reward
        
        return discounted_rewards

class ReinforcementLearningStrategy:
    """Main RL Strategy Engine for team optimization"""
    
    def __init__(self):
        self.q_agent = QNetworkAgent()
        self.policy_agent = PolicyGradientAgent()
        
        # Load pre-trained models if available
        self.model_dir = os.path.join(os.path.dirname(__file__), 'rl_models')
        self._load_models()
        
        # Strategy components
        self.strategy_memory = defaultdict(list)
        self.performance_history = deque(maxlen=1000)
        
        # Reward function weights
        self.reward_weights = {
            'team_score': 0.4,
            'constraint_satisfaction': 0.2,
            'risk_balance': 0.15,
            'ownership_optimization': 0.15,
            'budget_efficiency': 0.1
        }
    
    def _load_models(self):
        """Load pre-trained RL models"""
        q_model_path = os.path.join(self.model_dir, 'q_network_model.pkl')
        policy_model_path = os.path.join(self.model_dir, 'policy_model.pkl')
        
        self.q_agent.load_model(q_model_path)
        
        if os.path.exists(policy_model_path):
            try:
                with open(policy_model_path, 'rb') as f:
                    policy_data = pickle.load(f)
                self.policy_agent.policy_weights = policy_data.get('weights', self.policy_agent.policy_weights)
                print(f"✅ Loaded Policy Gradient model")
            except Exception as e:
                print(f"⚠️ Error loading Policy model: {e}")
    
    def create_state(self, players: List[Dict[str, Any]], 
                    current_team: List[int],
                    match_context: Dict[str, Any]) -> RLState:
        """Create RL state from current situation"""
        
        state = RLState()
        
        # Match context
        state.venue_type = self._classify_venue_type(match_context.get('venue', ''))
        state.pitch_condition = match_context.get('pitch_type', 'balanced')
        state.weather_condition = self._classify_weather(match_context.get('weather', {}))
        state.match_importance = match_context.get('importance', 0.5)
        
        # Team composition
        role_counts = self._count_roles(current_team, players)
        state.batsmen_count = role_counts.get('batsman', 0)
        state.bowlers_count = role_counts.get('bowler', 0)
        state.allrounders_count = role_counts.get('allrounder', 0)
        state.wicketkeepers_count = role_counts.get('wicketkeeper', 0)
        
        # Budget analysis
        total_credits = sum(players[i].get('credits', 8.5) for i in current_team)
        state.credits_used = total_credits
        state.credits_remaining = 100.0 - total_credits
        
        # Performance metrics
        if current_team:
            form_scores = [players[i].get('final_score', 50.0) for i in current_team]
            state.avg_form_score = np.mean(form_scores)
            
            ownership_scores = [players[i].get('ownership_prediction', 50.0) for i in current_team]
            state.total_ownership = np.mean(ownership_scores)
            
            # Risk assessment
            score_variance = np.var(form_scores) if len(form_scores) > 1 else 0
            state.risk_level = min(1.0, score_variance / 1000)
        
        # Historical context
        state.similar_context_success = self._get_historical_success_rate(state)
        
        return state
    
    def _classify_venue_type(self, venue: str) -> str:
        """Classify venue type"""
        venue_lower = venue.lower()
        
        if any(word in venue_lower for word in ['lords', 'oval', 'old trafford']):
            return 'seaming'
        elif any(word in venue_lower for word in ['wankhede', 'chinnaswamy', 'melbourne']):
            return 'batting'
        elif any(word in venue_lower for word in ['eden', 'chepauk', 'dharamshala']):
            return 'spinning'
        else:
            return 'neutral'
    
    def _classify_weather(self, weather: Dict[str, Any]) -> str:
        """Classify weather conditions"""
        if not weather:
            return 'clear'
        
        humidity = weather.get('humidity', 60)
        cloud_cover = weather.get('cloud_cover', 30)
        
        if cloud_cover > 70:
            return 'overcast'
        elif humidity > 80:
            return 'humid'
        else:
            return 'clear'
    
    def _count_roles(self, team_indices: List[int], players: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count players by role"""
        role_counts = defaultdict(int)
        
        for idx in team_indices:
            if idx < len(players):
                role = players[idx].get('role', '').lower()
                
                if 'bat' in role and 'allrounder' not in role:
                    role_counts['batsman'] += 1
                elif 'bowl' in role and 'allrounder' not in role:
                    role_counts['bowler'] += 1
                elif 'allrounder' in role or 'all-rounder' in role:
                    role_counts['allrounder'] += 1
                elif 'wk' in role or 'wicket' in role:
                    role_counts['wicketkeeper'] += 1
        
        return role_counts
    
    def _get_historical_success_rate(self, state: RLState) -> float:
        """Get historical success rate for similar contexts"""
        # Simplified lookup based on venue and conditions
        context_key = f"{state.venue_type}_{state.pitch_condition}_{state.weather_condition}"
        
        if context_key in self.strategy_memory:
            recent_results = self.strategy_memory[context_key][-10:]  # Last 10 results
            if recent_results:
                return np.mean(recent_results)
        
        return 0.5  # Default neutral success rate
    
    def get_strategic_recommendation(self, players: List[Dict[str, Any]], 
                                   current_team: List[int],
                                   match_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get RL-based strategic recommendation"""
        
        # Create current state
        state = self.create_state(players, current_team, match_context)
        
        # Get valid actions
        valid_actions = self._get_valid_actions(state, players, current_team)
        
        # Get recommendation from Q-agent
        q_action_idx = self.q_agent.get_action(state, valid_actions)
        q_recommendation = self._index_to_action(q_action_idx, state, players, current_team)
        
        # Get recommendation from Policy agent
        state_vector = self._state_to_vector(state)
        policy_action_idx = self.policy_agent.select_action(state_vector)
        policy_recommendation = self._index_to_action(policy_action_idx, state, players, current_team)
        
        # Combine recommendations
        combined_recommendation = self._combine_recommendations(q_recommendation, policy_recommendation)
        
        # Add strategic insights
        insights = self._generate_strategic_insights(state, players, current_team)
        
        return {
            'primary_recommendation': combined_recommendation,
            'q_agent_suggestion': q_recommendation,
            'policy_agent_suggestion': policy_recommendation,
            'strategic_insights': insights,
            'confidence_score': self._calculate_confidence(state),
            'expected_improvement': self._predict_improvement(state, combined_recommendation)
        }
    
    def _get_valid_actions(self, state: RLState, players: List[Dict[str, Any]], 
                          current_team: List[int]) -> List[int]:
        """Get list of valid action indices"""
        valid_actions = []
        
        # Can always try to balance team or change captain
        valid_actions.extend([4, 8])  # change_captain, balance_team
        
        # Check if we can add more players
        if len(current_team) < 11 and state.credits_remaining > 5:
            if state.batsmen_count < 6:
                valid_actions.append(0)  # select_batsman
            if state.bowlers_count < 6:
                valid_actions.append(1)  # select_bowler
            if state.allrounders_count < 4:
                valid_actions.append(2)  # select_allrounder
            if state.wicketkeepers_count < 2:
                valid_actions.append(3)  # select_wicketkeeper
        
        # Risk adjustment actions
        if state.risk_level < 0.8:
            valid_actions.append(5)  # increase_risk
        if state.risk_level > 0.2:
            valid_actions.append(6)  # decrease_risk
        
        # Ownership optimization
        if state.total_ownership > 60:
            valid_actions.append(7)  # optimize_ownership
        
        # Finalize if team is complete
        if len(current_team) == 11:
            valid_actions.append(9)  # finalize_team
        
        return valid_actions if valid_actions else [8]  # Default to balance_team
    
    def _index_to_action(self, action_idx: int, state: RLState, 
                        players: List[Dict[str, Any]], current_team: List[int]) -> RLAction:
        """Convert action index to RLAction"""
        
        action_types = [
            'select_batsman', 'select_bowler', 'select_allrounder', 'select_wicketkeeper',
            'change_captain', 'increase_risk', 'decrease_risk', 'optimize_ownership',
            'balance_team', 'finalize_team'
        ]
        
        if 0 <= action_idx < len(action_types):
            action_type = action_types[action_idx]
        else:
            action_type = 'balance_team'
        
        # Create action with appropriate parameters
        action = RLAction(action_type=action_type)
        
        if action_type.startswith('select_'):
            role = action_type.split('_')[1]
            action.target_role = role
            # Find best available player for this role
            action.player_id = self._find_best_available_player(players, current_team, role, state)
        
        elif action_type in ['increase_risk', 'decrease_risk']:
            action.risk_adjustment = 0.1 if 'increase' in action_type else -0.1
        
        # Set confidence based on state
        action.confidence = min(1.0, 0.5 + state.similar_context_success)
        
        return action
    
    def _find_best_available_player(self, players: List[Dict[str, Any]], 
                                  current_team: List[int], role: str,
                                  state: RLState) -> Optional[int]:
        """Find best available player for given role"""
        
        available_players = []
        
        for i, player in enumerate(players):
            if i not in current_team:
                player_role = player.get('role', '').lower()
                player_credits = player.get('credits', 8.5)
                
                # Check if player fits role and budget
                if (self._role_matches(player_role, role) and 
                    player_credits <= state.credits_remaining):
                    
                    score = player.get('final_score', 0)
                    available_players.append((i, score))
        
        if available_players:
            # Sort by score and return best player
            available_players.sort(key=lambda x: x[1], reverse=True)
            return available_players[0][0]
        
        return None
    
    def _role_matches(self, player_role: str, target_role: str) -> bool:
        """Check if player role matches target role"""
        role_mapping = {
            'batsman': ['bat'],
            'bowler': ['bowl', 'pace', 'spin'],
            'allrounder': ['allrounder', 'all-rounder'],
            'wicketkeeper': ['wk', 'wicket', 'keeper']
        }
        
        target_keywords = role_mapping.get(target_role, [])
        return any(keyword in player_role for keyword in target_keywords)
    
    def _state_to_vector(self, state: RLState) -> np.ndarray:
        """Convert state to vector for policy agent"""
        return np.array([
            hash(state.venue_type) % 100 / 100,  # Normalized venue type
            hash(state.pitch_condition) % 100 / 100,  # Normalized pitch condition
            hash(state.weather_condition) % 100 / 100,  # Normalized weather
            state.match_importance,
            state.batsmen_count / 11,
            state.bowlers_count / 11,
            state.allrounders_count / 11,
            state.wicketkeepers_count / 11,
            state.credits_used / 100,
            state.credits_remaining / 100,
            state.avg_form_score / 100,
            state.total_ownership / 100,
            state.risk_level,
            state.similar_context_success,
            len(state.__dict__) / 20  # Additional normalization factor
        ])
    
    def _combine_recommendations(self, q_rec: RLAction, policy_rec: RLAction) -> RLAction:
        """Combine recommendations from both agents"""
        
        # Weight recommendations based on confidence and recent performance
        q_weight = 0.6 if self.q_agent.success_rate > 0.5 else 0.4
        policy_weight = 1.0 - q_weight
        
        # Choose primary recommendation based on weights
        if q_weight > policy_weight:
            primary = q_rec
            primary.confidence = (q_rec.confidence * q_weight + policy_rec.confidence * policy_weight)
        else:
            primary = policy_rec
            primary.confidence = (policy_rec.confidence * policy_weight + q_rec.confidence * q_weight)
        
        return primary
    
    def _generate_strategic_insights(self, state: RLState, players: List[Dict[str, Any]], 
                                   current_team: List[int]) -> List[str]:
        """Generate strategic insights"""
        insights = []
        
        # Team composition insights
        if state.batsmen_count < 3:
            insights.append("Consider adding more batting strength")
        elif state.batsmen_count > 6:
            insights.append("Team may be too batting-heavy")
        
        if state.bowlers_count < 3:
            insights.append("Bowling attack needs strengthening")
        elif state.bowlers_count > 6:
            insights.append("Consider balancing with more batting options")
        
        # Budget insights
        if state.credits_remaining < 5 and len(current_team) < 11:
            insights.append("Budget constraints may limit player options")
        elif state.credits_remaining > 20:
            insights.append("Consider upgrading to premium players")
        
        # Risk insights
        if state.risk_level > 0.7:
            insights.append("Current team selection is high-risk, high-reward")
        elif state.risk_level < 0.3:
            insights.append("Conservative selection - consider some differential picks")
        
        # Ownership insights
        if state.total_ownership > 70:
            insights.append("High ownership team - consider some unique selections")
        elif state.total_ownership < 30:
            insights.append("Very differential team - high risk but potentially high reward")
        
        return insights
    
    def _calculate_confidence(self, state: RLState) -> float:
        """Calculate confidence in recommendations"""
        
        factors = [
            state.similar_context_success,  # Historical performance
            min(1.0, state.avg_form_score / 60),  # Current form quality
            1.0 - abs(state.credits_remaining - 10) / 30,  # Budget optimization
            1.0 - abs(state.risk_level - 0.5),  # Risk balance
        ]
        
        return np.mean(factors)
    
    def _predict_improvement(self, state: RLState, action: RLAction) -> float:
        """Predict expected improvement from action"""
        
        base_improvement = 0.05  # 5% base improvement
        
        # Adjust based on action type
        if action.action_type == 'change_captain':
            base_improvement += 0.03
        elif action.action_type.startswith('select_'):
            base_improvement += 0.02
        elif 'risk' in action.action_type:
            base_improvement += 0.01
        
        # Adjust based on confidence
        return base_improvement * action.confidence
    
    def update_performance(self, state: RLState, action: RLAction, 
                          actual_score: float, expected_score: float):
        """Update RL agents based on performance"""
        
        # Calculate reward
        reward = self._calculate_reward(actual_score, expected_score, state)
        
        # Create experience for Q-agent
        next_state = state  # Simplified - in practice, would be actual next state
        experience = RLExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=True
        )
        
        self.q_agent.remember(experience)
        self.q_agent.replay()
        
        # Update policy agent
        state_vector = self._state_to_vector(state)
        action_idx = self._action_to_index(action)
        self.policy_agent.store_experience(state_vector, action_idx, reward)
        self.policy_agent.update_policy()
        
        # Update performance history
        self.performance_history.append({
            'reward': reward,
            'actual_score': actual_score,
            'expected_score': expected_score,
            'timestamp': datetime.now()
        })
        
        # Update success rate
        recent_rewards = [p['reward'] for p in list(self.performance_history)[-100:]]
        self.q_agent.success_rate = np.mean([r > 0 for r in recent_rewards]) if recent_rewards else 0.5
        
        # Store strategy result
        context_key = f"{state.venue_type}_{state.pitch_condition}_{state.weather_condition}"
        success = 1.0 if reward > 0 else 0.0
        self.strategy_memory[context_key].append(success)
    
    def _calculate_reward(self, actual_score: float, expected_score: float, state: RLState) -> float:
        """Calculate reward for RL training"""
        
        # Performance reward
        score_diff = actual_score - expected_score
        performance_reward = np.tanh(score_diff / 50)  # Normalize between -1 and 1
        
        # Constraint satisfaction reward
        constraint_reward = 1.0 if (state.batsmen_count >= 3 and state.bowlers_count >= 3 and 
                                   state.wicketkeepers_count >= 1) else -0.5
        
        # Budget efficiency reward
        budget_efficiency = 1.0 - abs(state.credits_remaining - 5) / 20  # Optimal remaining ~5 credits
        
        # Risk balance reward
        risk_balance = 1.0 - abs(state.risk_level - 0.5)  # Optimal risk ~0.5
        
        # Ownership optimization reward
        ownership_reward = 1.0 - abs(state.total_ownership - 50) / 50  # Optimal ownership ~50%
        
        # Combine rewards
        total_reward = (
            performance_reward * self.reward_weights['team_score'] +
            constraint_reward * self.reward_weights['constraint_satisfaction'] +
            budget_efficiency * self.reward_weights['budget_efficiency'] +
            risk_balance * self.reward_weights['risk_balance'] +
            ownership_reward * self.reward_weights['ownership_optimization']
        )
        
        return total_reward
    
    def _action_to_index(self, action: RLAction) -> int:
        """Convert RLAction to index for policy agent"""
        action_mapping = {
            'select_batsman': 0,
            'select_bowler': 1,
            'select_allrounder': 2,
            'select_wicketkeeper': 3,
            'change_captain': 4,
            'increase_risk': 5,
            'decrease_risk': 6,
            'optimize_ownership': 7,
            'balance_team': 8,
            'finalize_team': 9
        }
        return action_mapping.get(action.action_type, 8)
    
    def save_models(self):
        """Save trained RL models"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save Q-agent
        q_model_path = os.path.join(self.model_dir, 'q_network_model.pkl')
        self.q_agent.save_model(q_model_path)
        
        # Save Policy agent
        policy_model_path = os.path.join(self.model_dir, 'policy_model.pkl')
        policy_data = {
            'weights': self.policy_agent.policy_weights,
            'performance_history': list(self.performance_history)
        }
        
        with open(policy_model_path, 'wb') as f:
            pickle.dump(policy_data, f)
        
        print("✅ RL models saved successfully")

# Export
__all__ = ['ReinforcementLearningStrategy', 'RLState', 'RLAction', 'QNetworkAgent', 'PolicyGradientAgent']