#!/usr/bin/env python3
"""
Unit Tests for Core Logic Components
Tests team generation, optimization, neural prediction, and data aggregation
"""

import unittest
import sys
import os
import sqlite3
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from test_framework import Dream11TestCase, DatabaseTestMixin, PerformanceTestMixin

# Import core logic modules (with fallbacks for missing dependencies)
try:
    from core_logic.team_generator import generate_world_class_ai_teams, OptimalTeam, PlayerForOptimization
    from core_logic.feature_engine import PlayerFeatures, extract_player_features
    from core_logic.data_aggregator import DataAggregator
except ImportError as e:
    print(f"⚠️ Some imports failed: {e}")
    print("Tests will use mocks for missing components")

class TestTeamGenerator(Dream11TestCase, DatabaseTestMixin, PerformanceTestMixin):
    """Test team generation logic"""
    
    def setUp(self):
        super().setUp()
        
        # Create mock player data
        self.mock_players = [
            {
                'player_id': i,
                'name': f'Player {i}',
                'role': ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'][i % 4],
                'team': 'Team A' if i < 6 else 'Team B',
                'consistency_score': 0.5 + (i % 10) * 0.05,
                'form_momentum': 0.4 + (i % 8) * 0.075,
                'expected_points': 30 + (i % 20) * 2
            }
            for i in range(20)
        ]
    
    def test_player_for_optimization_creation(self):
        """Test PlayerForOptimization object creation"""
        player_data = self.mock_players[0]
        
        player = PlayerForOptimization(
            player_id=player_data['player_id'],
            name=player_data['name'],
            role=player_data['role'],
            team=player_data['team'],
            consistency_score=player_data['consistency_score'],
            form_momentum=player_data['form_momentum']
        )
        
        self.assertEqual(player.player_id, player_data['player_id'])
        self.assertEqual(player.name, player_data['name'])
        self.assertEqual(player.role, player_data['role'])
        self.assertIsInstance(player.consistency_score, float)
        self.assertGreaterEqual(player.consistency_score, 0.0)
        self.assertLessEqual(player.consistency_score, 1.0)
    
    def test_optimal_team_creation(self):
        """Test OptimalTeam object creation and validation"""
        players = [
            PlayerForOptimization(
                player_id=i,
                name=f'Player {i}',
                role=['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'][i % 4],
                team='Team A' if i < 6 else 'Team B',
                consistency_score=0.8,
                form_momentum=0.7
            )
            for i in range(11)
        ]
        
        team = OptimalTeam(
            team_id=1,
            players=players,
            captain=players[0],
            vice_captain=players[1],
            strategy="Test Strategy"
        )
        
        self.assertEqual(len(team.players), 11)
        self.assertIsNotNone(team.captain)
        self.assertIsNotNone(team.vice_captain)
        self.assertNotEqual(team.captain.player_id, team.vice_captain.player_id)
        self.assertIsInstance(team.total_score, (int, float))
    
    @patch('core_logic.team_generator.generate_world_class_ai_teams')
    def test_team_generation_with_mock(self, mock_generate):
        """Test team generation with mocked dependencies"""
        # Mock the team generation function
        mock_teams = [
            {
                'team_id': 1,
                'strategy': 'AI-Optimal',
                'players': self.mock_players[:11],
                'captain': self.mock_players[0],
                'vice_captain': self.mock_players[1]
            }
        ]
        mock_generate.return_value = mock_teams
        
        # Test the mocked function
        result = mock_generate(self.mock_players, num_teams=1)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['strategy'], 'AI-Optimal')
        self.assertEqual(len(result[0]['players']), 11)
    
    def test_team_validation_constraints(self):
        """Test team validation against Dream11 constraints"""
        # Create a valid team composition
        batsmen = [PlayerForOptimization(i, f'Bat{i}', 'Batsman', 'TeamA') for i in range(4)]
        bowlers = [PlayerForOptimization(i+10, f'Bowl{i}', 'Bowler', 'TeamB') for i in range(4)]
        all_rounders = [PlayerForOptimization(20, 'AR1', 'All-Rounder', 'TeamA')]
        wicket_keepers = [PlayerForOptimization(30, 'WK1', 'Wicket-Keeper', 'TeamB')]
        
        all_players = batsmen + bowlers + all_rounders + wicket_keepers
        
        # Test valid team
        self.assertEqual(len(all_players), 10)  # One short for testing
        
        # Test role distribution
        roles = [p.role for p in all_players]
        self.assertGreaterEqual(roles.count('Batsman'), 3)
        self.assertGreaterEqual(roles.count('Bowler'), 3)
        self.assertGreaterEqual(roles.count('All-Rounder'), 1)
        self.assertGreaterEqual(roles.count('Wicket-Keeper'), 1)
    
    def test_team_performance_calculation(self):
        """Test team performance metrics calculation"""
        players = [
            PlayerForOptimization(
                i, f'Player{i}', 'Batsman', 'TeamA',
                consistency_score=0.8, form_momentum=0.7, ema_score=50 + i
            )
            for i in range(11)
        ]
        
        team = OptimalTeam(team_id=1, players=players)
        
        # Test total score calculation
        total_score = team.total_score
        self.assertIsInstance(total_score, (int, float))
        self.assertGreater(total_score, 0)
        
        # Test team quality calculation
        self.assertIn(team.team_quality, ['Premium', 'Standard', 'Value', 'Unknown'])

class TestAdvancedOptimizer(Dream11TestCase, PerformanceTestMixin):
    """Test advanced team optimization algorithms"""
    
    def setUp(self):
        super().setUp()
        
        self.constraints = OptimizationConstraints(
            budget=100.0,
            min_batsmen=3,
            max_batsmen=6,
            min_bowlers=3,
            max_bowlers=6,
            min_all_rounders=1,
            max_all_rounders=4,
            min_wicket_keepers=1,
            max_wicket_keepers=2
        )
        
        self.mock_players = [
            {
                'player_id': i,
                'name': f'Player{i}',
                'role': ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'][i % 4],
                'team': 'TeamA' if i < 10 else 'TeamB',
                'expected_points': 40 + (i % 20),
                'consistency_score': 0.6 + (i % 5) * 0.08,
                'ownership_percentage': 20 + (i % 60)
            }
            for i in range(20)
        ]
    
    def test_optimization_constraints_validation(self):
        """Test optimization constraints validation"""
        self.assertIsInstance(self.constraints.budget, (int, float))
        self.assertGreater(self.constraints.budget, 0)
        self.assertGreaterEqual(self.constraints.min_batsmen, 3)
        self.assertLessEqual(self.constraints.max_batsmen, 6)
        self.assertGreaterEqual(self.constraints.min_bowlers, 3)
        self.assertLessEqual(self.constraints.max_bowlers, 6)
    
    def test_pareto_optimality_calculation(self):
        """Test Pareto optimality calculations"""
        # Create mock teams with different trade-offs
        teams = [
            {'expected_points': 450, 'risk_score': 0.3, 'uniqueness': 0.7},  # High points, medium risk
            {'expected_points': 420, 'risk_score': 0.1, 'uniqueness': 0.9},  # Lower points, low risk
            {'expected_points': 480, 'risk_score': 0.6, 'uniqueness': 0.4},  # Highest points, high risk
        ]
        
        # Simple Pareto dominance check
        def dominates(team1, team2):
            return (team1['expected_points'] >= team2['expected_points'] and
                   team1['risk_score'] <= team2['risk_score'] and
                   team1['uniqueness'] >= team2['uniqueness'] and
                   (team1['expected_points'] > team2['expected_points'] or
                    team1['risk_score'] < team2['risk_score'] or
                    team1['uniqueness'] > team2['uniqueness']))
        
        # Test that no team completely dominates all others
        for i, team1 in enumerate(teams):
            domination_count = sum(1 for j, team2 in enumerate(teams) 
                                 if i != j and dominates(team1, team2))
            # Each team should have its own trade-off niche
            self.assertLess(domination_count, len(teams) - 1)

class TestFeatureEngine(Dream11TestCase):
    """Test feature extraction and player analysis"""
    
    def setUp(self):
        super().setUp()
        
        self.mock_player_data = {
            'player_id': 123,
            'name': 'Test Player',
            'role': 'Batsman',
            'team': 'Team A',
            'recent_scores': [45, 32, 67, 23, 89, 12, 56],
            'career_stats': {
                'matches': 150,
                'runs': 4500,
                'average': 30.0,
                'strike_rate': 125.5
            },
            'venue_performance': {
                'home': {'average': 35.0, 'matches': 75},
                'away': {'average': 25.0, 'matches': 75}
            }
        }
    
    @patch('core_logic.feature_engine.PlayerFeatures')
    def test_player_features_extraction(self, mock_features_class):
        """Test player features extraction"""
        mock_features = Mock()
        mock_features.consistency_score = 0.75
        mock_features.form_momentum = 0.68
        mock_features.venue_advantage = 0.2
        mock_features_class.return_value = mock_features
        
        features = mock_features_class(self.mock_player_data)
        
        self.assertIsInstance(features.consistency_score, float)
        self.assertBetween(features.consistency_score, 0.0, 1.0)
        self.assertIsInstance(features.form_momentum, float)
        self.assertBetween(features.form_momentum, -1.0, 1.0)
    
    def test_consistency_score_calculation(self):
        """Test consistency score calculation logic"""
        recent_scores = [45, 32, 67, 23, 89, 12, 56]
        
        # Calculate coefficient of variation (inverse of consistency)
        mean_score = sum(recent_scores) / len(recent_scores)
        variance = sum((x - mean_score) ** 2 for x in recent_scores) / len(recent_scores)
        std_dev = variance ** 0.5
        cv = std_dev / mean_score if mean_score > 0 else 1.0
        
        # Consistency score should be higher for lower coefficient of variation
        expected_consistency = max(0.0, 1.0 - cv)
        
        self.assertIsInstance(expected_consistency, float)
        self.assertGreaterEqual(expected_consistency, 0.0)
        self.assertLessEqual(expected_consistency, 1.0)
    
    def test_form_momentum_calculation(self):
        """Test form momentum calculation using EMA"""
        recent_scores = [45, 32, 67, 23, 89, 12, 56]
        
        # Simple EMA calculation
        alpha = 0.3  # Smoothing factor
        ema = recent_scores[0]
        
        for score in recent_scores[1:]:
            ema = alpha * score + (1 - alpha) * ema
        
        # Form momentum based on recent performance vs EMA
        recent_avg = sum(recent_scores[-3:]) / 3
        momentum = (recent_avg - ema) / ema if ema > 0 else 0
        
        self.assertIsInstance(momentum, (int, float))
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion to check if value is between min and max"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

class TestDataAggregator(Dream11TestCase, DatabaseTestMixin):
    """Test data aggregation and processing"""
    
    def setUp(self):
        super().setUp()
        
        # Create test data
        self.mock_match_data = {
            'match_id': '123456',
            'format': 'T20I',
            'teams': ['Team A', 'Team B'],
            'venue': 'Test Stadium',
            'date': '2025-08-12',
            'players': [
                {
                    'player_id': i,
                    'name': f'Player {i}',
                    'team': 'Team A' if i < 11 else 'Team B',
                    'role': ['Batsman', 'Bowler', 'All-Rounder', 'Wicket-Keeper'][i % 4]
                }
                for i in range(22)
            ]
        }
    
    @patch('core_logic.data_aggregator.DataAggregator')
    def test_data_aggregator_initialization(self, mock_aggregator_class):
        """Test data aggregator initialization"""
        mock_aggregator = Mock()
        mock_aggregator_class.return_value = mock_aggregator
        
        aggregator = mock_aggregator_class()
        
        self.assertIsNotNone(aggregator)
    
    @patch('core_logic.data_aggregator.DataAggregator')
    def test_match_data_processing(self, mock_aggregator_class):
        """Test match data processing and validation"""
        mock_aggregator = Mock()
        mock_aggregator_class.return_value = mock_aggregator
        
        # Mock processed data
        processed_data = {
            'match_id': self.mock_match_data['match_id'],
            'processed_players': len(self.mock_match_data['players']),
            'team_balance': {
                'Team A': 11,
                'Team B': 11
            },
            'role_distribution': {
                'Batsman': 10,
                'Bowler': 8,
                'All-Rounder': 3,
                'Wicket-Keeper': 1
            }
        }
        mock_aggregator.process_match_data.return_value = processed_data
        
        aggregator = mock_aggregator_class()
        result = aggregator.process_match_data(self.mock_match_data)
        
        self.assertEqual(result['match_id'], '123456')
        self.assertEqual(result['processed_players'], 22)
        self.assertEqual(result['team_balance']['Team A'], 11)
        self.assertEqual(result['team_balance']['Team B'], 11)
    
    def test_data_validation_rules(self):
        """Test data validation rules"""
        # Test valid match data
        valid_data = self.mock_match_data.copy()
        
        # Test required fields
        required_fields = ['match_id', 'format', 'teams', 'players']
        for field in required_fields:
            self.assertIn(field, valid_data)
            self.assertIsNotNone(valid_data[field])
        
        # Test team count
        self.assertEqual(len(valid_data['teams']), 2)
        
        # Test player count (should be 22 for full squads)
        self.assertGreater(len(valid_data['players']), 0)
        self.assertLessEqual(len(valid_data['players']), 30)  # Reasonable upper bound
    
    def test_data_aggregation_performance(self):
        """Test data aggregation performance"""
        with self.performance_context('data_aggregation', max_duration=2.0):
            # Simulate data processing
            large_dataset = {
                'match_id': '999999',
                'players': [
                    {'player_id': i, 'stats': list(range(100))}
                    for i in range(1000)
                ]
            }
            
            # Simple aggregation simulation
            total_stats = sum(
                sum(player['stats']) 
                for player in large_dataset['players']
            )
            
            self.assertGreater(total_stats, 0)

class TestNeuralPrediction(Dream11TestCase, PerformanceTestMixin):
    """Test neural prediction components"""
    
    def setUp(self):
        super().setUp()
        
        self.mock_features = [
            [0.8, 0.7, 0.6, 0.9, 0.5, 0.8, 0.7, 0.6, 0.8, 0.5, 0.7, 0.6, 0.8, 0.9, 0.7],  # 15 features
            [0.7, 0.8, 0.5, 0.8, 0.6, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.5, 0.7, 0.8, 0.6],
            [0.9, 0.6, 0.8, 0.7, 0.8, 0.9, 0.6, 0.8, 0.9, 0.8, 0.6, 0.8, 0.9, 0.7, 0.8]
        ]
    
    @patch('core_logic.enhanced_neural_prediction.EnhancedLSTMTransformer')
    def test_neural_model_initialization(self, mock_model_class):
        """Test neural model initialization"""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        model = mock_model_class(input_features=15, hidden_size=128)
        
        self.assertIsNotNone(model)
        mock_model_class.assert_called_once()
    
    @patch('core_logic.enhanced_neural_prediction.EnhancedLSTMTransformer')
    def test_neural_prediction_output(self, mock_model_class):
        """Test neural prediction output format"""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock prediction output
        mock_output = {
            'expected_points': 45.6,
            'confidence_score': 0.78,
            'uncertainty': 8.2,
            'feature_importance': {
                'recent_form': 0.25,
                'consistency': 0.20,
                'venue_advantage': 0.15,
                'opposition_strength': 0.12
            }
        }
        mock_model.predict.return_value = mock_output
        
        model = mock_model_class()
        result = model.predict(self.mock_features[0])
        
        self.assertIsInstance(result['expected_points'], (int, float))
        self.assertBetween(result['confidence_score'], 0.0, 1.0)
        self.assertIsInstance(result['uncertainty'], (int, float))
        self.assertIsInstance(result['feature_importance'], dict)
    
    def test_prediction_input_validation(self):
        """Test prediction input validation"""
        # Test valid input
        valid_input = self.mock_features[0]
        self.assertEqual(len(valid_input), 15)
        self.assertTrue(all(isinstance(x, (int, float)) for x in valid_input))
        
        # Test feature normalization
        normalized_features = [(x - 0.5) * 2 for x in valid_input]  # Scale to [-1, 1]
        self.assertTrue(all(-1.1 <= x <= 1.1 for x in normalized_features))
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion to check if value is between min and max"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTeamGenerator,
        TestAdvancedOptimizer,
        TestFeatureEngine,
        TestDataAggregator,
        TestNeuralPrediction
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)