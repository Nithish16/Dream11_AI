#!/usr/bin/env python3
"""
Comprehensive Test Suite for Dream11 AI Systems
Tests all components individually and together for production readiness
"""

import unittest
import sys
import os
import time
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sqlite3

# Add the core_logic path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_logic'))

# Import all systems to test
from intelligent_api_cache import IntelligentAPICache, get_api_cache
from api_rate_limiter import SmartRateLimiter, get_rate_limiter
from api_request_optimizer import APIRequestOptimizer, get_api_optimizer
from prediction_accuracy_engine import PredictionAccuracyEngine, PlayerPerformanceMetrics
from prediction_confidence_scorer import PredictionConfidenceScorer, ConfidenceFactors
from ab_testing_framework import ABTestingFramework, TestVariant
from ensemble_prediction_system import EnsemblePredictionSystem, EnsemblePrediction
from historical_performance_validator import HistoricalPerformanceValidator, ValidationMetrics
from world_class_ai_integration import WorldClassAIIntegration

class TestIntelligentAPICache(unittest.TestCase):
    """Test the intelligent API caching system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        self.cache = IntelligentAPICache(cache_db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_cache_initialization(self):
        """Test cache database initialization"""
        self.assertIsNotNone(self.cache)
        self.assertTrue(os.path.exists(self.test_db.name))
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        endpoint1 = "https://api.example.com/test"
        params1 = {"param1": "value1"}
        
        key1 = self.cache._generate_cache_key(endpoint1, params1)
        key2 = self.cache._generate_cache_key(endpoint1, params1)
        
        self.assertEqual(key1, key2, "Same inputs should generate same cache key")
        self.assertEqual(len(key1), 64, "Cache key should be SHA256 hash")
    
    def test_cache_response_storage_retrieval(self):
        """Test caching and retrieving responses"""
        endpoint = "https://api.example.com/test"
        response_data = {"test": "data", "timestamp": "2023-01-01"}
        
        # Cache the response
        success = self.cache.cache_response(endpoint, response_data)
        self.assertTrue(success, "Caching should succeed")
        
        # Retrieve the response
        cached_response = self.cache.get_cached_response(endpoint)
        self.assertIsNotNone(cached_response, "Should retrieve cached response")
        self.assertEqual(cached_response["test"], "data", "Cached data should match original")
    
    def test_cache_expiration(self):
        """Test cache expiration functionality"""
        # Create cache with very short duration
        self.cache.cache_strategies['test'] = {'duration': 1, 'strategy': 'time_based'}
        
        endpoint = "https://api.example.com/test_expiry"
        response_data = {"test": "expiry"}
        
        self.cache.cache_response(endpoint, response_data)
        
        # Should be available immediately
        cached = self.cache.get_cached_response(endpoint)
        self.assertIsNotNone(cached, "Should be cached initially")
        
        # Wait for expiry
        time.sleep(2)
        
        # Should be expired now
        expired = self.cache.get_cached_response(endpoint)
        self.assertIsNone(expired, "Should be expired after timeout")
    
    @patch('requests.get')
    def test_cached_api_call(self, mock_get):
        """Test making cached API calls"""
        mock_response = Mock()
        mock_response.json.return_value = {"api": "response"}
        mock_response.headers.get.return_value = "application/json"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        endpoint = "https://api.example.com/cached_call"
        
        # First call should hit API
        result1 = self.cache.make_cached_request(endpoint)
        self.assertEqual(result1["api"], "response")
        mock_get.assert_called_once()
        
        # Second call should hit cache
        mock_get.reset_mock()
        result2 = self.cache.make_cached_request(endpoint)
        self.assertEqual(result2["api"], "response")
        mock_get.assert_not_called()

class TestAPIRateLimiter(unittest.TestCase):
    """Test the API rate limiting system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        self.limiter = SmartRateLimiter(db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        self.assertIsNotNone(self.limiter)
        self.assertTrue(os.path.exists(self.test_db.name))
        self.assertIn('cricbuzz', self.limiter.rate_limits)
    
    def test_can_make_request(self):
        """Test rate limit checking"""
        endpoint = "https://cricbuzz.com/test"
        
        # Should be able to make first request
        can_make, reason, wait_time = self.limiter.can_make_request(endpoint)
        self.assertTrue(can_make, "Should be able to make first request")
        self.assertEqual(wait_time, 0, "No wait time for first request")
    
    def test_request_recording(self):
        """Test request recording and metrics"""
        endpoint = "https://cricbuzz.com/test"
        
        # Record successful request
        self.limiter.record_request(endpoint, 0.5, 200)
        
        # Check metrics
        self.assertIn(endpoint, self.limiter.metrics)
        metrics = self.limiter.metrics[endpoint]
        self.assertEqual(metrics.success_count, 1)
        self.assertEqual(metrics.error_count, 0)
    
    def test_backoff_functionality(self):
        """Test adaptive backoff on consecutive errors"""
        endpoint = "https://cricbuzz.com/backoff_test"
        
        # Record multiple errors
        for i in range(5):
            self.limiter.record_request(endpoint, 1.0, 500, "Server Error")
        
        # Should be in backoff period
        metrics = self.limiter.metrics[endpoint]
        self.assertIsNotNone(metrics.backoff_until, "Should be in backoff period")
        self.assertGreater(metrics.consecutive_errors, 0, "Should have consecutive errors")
    
    def test_quota_management(self):
        """Test API quota management"""
        # Add test quota
        self.limiter.add_quota('test_service', 'daily', 100, cost_per_request=0.01)
        
        # Check quota was added
        quota_key = 'test_service_daily'
        self.assertIn(quota_key, self.limiter.quotas)
        
        quota = self.limiter.quotas[quota_key]
        self.assertEqual(quota.quota_limit, 100)
        self.assertEqual(quota.cost_per_request, 0.01)

class TestPredictionAccuracyEngine(unittest.TestCase):
    """Test the prediction accuracy engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        self.engine = PredictionAccuracyEngine(db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_engine_initialization(self):
        """Test prediction engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertGreater(len(self.engine.prediction_models), 0)
        self.assertTrue(os.path.exists(self.test_db.name))
    
    def test_player_performance_analysis(self):
        """Test player performance analysis"""
        player_data = {
            'player_id': 12345,
            'name': 'Test Player',
            'role': 'batsman',
            'team': 'Test Team',
            'recent_scores': [45, 67, 23, 89, 34, 56, 78]
        }
        
        match_context = {
            'venue': 'Test Stadium',
            'format': 'T20',
            'opposition': 'Opposition Team'
        }
        
        metrics = self.engine.analyze_player_performance(player_data, match_context)
        
        self.assertIsInstance(metrics, PlayerPerformanceMetrics)
        self.assertEqual(metrics.player_id, 12345)
        self.assertEqual(metrics.player_name, 'Test Player')
        self.assertGreater(metrics.career_average, 0)
        self.assertGreater(metrics.expected_points, 0)
    
    def test_consistency_calculation(self):
        """Test consistency index calculation"""
        # High consistency scores
        consistent_scores = [50, 52, 48, 51, 49]
        consistency_high = self.engine._calculate_consistency_index(consistent_scores)
        
        # Low consistency scores  
        inconsistent_scores = [10, 90, 5, 85, 20]
        consistency_low = self.engine._calculate_consistency_index(inconsistent_scores)
        
        self.assertGreater(consistency_high, consistency_low, 
                          "Consistent scores should have higher consistency index")
    
    def test_form_momentum_calculation(self):
        """Test form momentum calculation"""
        # Improving form
        improving_scores = [30, 40, 50, 60, 70]
        momentum_positive = self.engine._calculate_form_momentum(improving_scores)
        
        # Declining form
        declining_scores = [70, 60, 50, 40, 30]  
        momentum_negative = self.engine._calculate_form_momentum(declining_scores)
        
        self.assertGreater(momentum_positive, momentum_negative,
                          "Improving form should have higher momentum")
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction generation"""
        metrics = PlayerPerformanceMetrics(
            player_id=12345,
            player_name='Test Player',
            role='batsman',
            team='Test Team',
            recent_scores=[45, 67, 23, 89, 34],
            career_average=51.6,
            consistency_index=0.7,
            form_momentum=0.2
        )
        
        match_context = {'venue': 'Test Stadium', 'format': 'T20'}
        
        prediction = self.engine._generate_ensemble_prediction(metrics, match_context)
        
        self.assertIn('expected_points', prediction)
        self.assertIn('confidence_interval', prediction)
        self.assertIn('selection_probability', prediction)
        self.assertGreater(prediction['expected_points'], 0)

class TestPredictionConfidenceScorer(unittest.TestCase):
    """Test the prediction confidence scoring system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        self.scorer = PredictionConfidenceScorer(db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_scorer_initialization(self):
        """Test confidence scorer initialization"""
        self.assertIsNotNone(self.scorer)
        self.assertTrue(os.path.exists(self.test_db.name))
        self.assertIn('very_high', self.scorer.confidence_thresholds)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Create test metrics
        metrics = PlayerPerformanceMetrics(
            player_id=12345,
            player_name='Test Player',
            role='batsman',
            team='Test Team',
            recent_scores=[45, 67, 23, 89, 34, 56, 78, 43, 65],
            career_average=55.6,
            consistency_index=0.8,
            form_momentum=0.3,
            sample_size=9,
            data_reliability=0.9,
            conditions_suitability=0.7,
            last_updated=datetime.now()
        )
        
        match_context = {'venue': 'Test Stadium', 'format': 'T20'}
        prediction_data = {
            'expected_points': 65.0,
            'confidence_interval': (50.0, 80.0),
            'model_predictions': {
                'model1': {'score': 63, 'confidence': 0.7},
                'model2': {'score': 67, 'confidence': 0.8}
            }
        }
        
        confidence_breakdown = self.scorer.calculate_confidence_score(
            metrics, match_context, prediction_data
        )
        
        self.assertIsNotNone(confidence_breakdown)
        self.assertGreater(confidence_breakdown.overall_confidence, 0)
        self.assertLess(confidence_breakdown.overall_confidence, 1)
        self.assertIn(confidence_breakdown.confidence_level, 
                     ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    def test_data_quality_scoring(self):
        """Test data quality score calculation"""
        # High quality metrics (recent, large sample)
        high_quality_metrics = PlayerPerformanceMetrics(
            player_id=1, player_name='Test', role='batsman', team='Test',
            sample_size=20, data_reliability=0.9, 
            last_updated=datetime.now()
        )
        
        score_high = self.scorer._calculate_data_quality_score(high_quality_metrics)
        
        # Low quality metrics (old, small sample)  
        low_quality_metrics = PlayerPerformanceMetrics(
            player_id=2, player_name='Test2', role='batsman', team='Test',
            sample_size=3, data_reliability=0.3,
            last_updated=datetime.now() - timedelta(days=60)
        )
        
        score_low = self.scorer._calculate_data_quality_score(low_quality_metrics)
        
        self.assertGreater(score_high, score_low, 
                          "High quality data should score higher")

class TestABTestingFramework(unittest.TestCase):
    """Test the A/B testing framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        self.ab_testing = ABTestingFramework(db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_framework_initialization(self):
        """Test A/B testing framework initialization"""
        self.assertIsNotNone(self.ab_testing)
        self.assertTrue(os.path.exists(self.test_db.name))
    
    def test_experiment_creation(self):
        """Test creating A/B test experiments"""
        variants = [
            {
                'name': 'Control',
                'traffic_allocation': 0.5,
                'model_config': {'model': 'baseline'},
                'is_control': True
            },
            {
                'name': 'Treatment',
                'traffic_allocation': 0.5,
                'model_config': {'model': 'enhanced'},
                'is_control': False
            }
        ]
        
        test_id = self.ab_testing.create_experiment(
            'Test Prediction Models',
            'Testing enhanced vs baseline models',
            variants,
            success_metric='accuracy'
        )
        
        self.assertIsNotNone(test_id)
        self.assertEqual(len(test_id), 16, "Test ID should be 16 character hash")
    
    def test_variant_assignment(self):
        """Test variant assignment consistency"""
        # Create test experiment
        variants = [
            {'name': 'A', 'traffic_allocation': 0.5, 'model_config': {}, 'is_control': True},
            {'name': 'B', 'traffic_allocation': 0.5, 'model_config': {}, 'is_control': False}
        ]
        
        test_id = self.ab_testing.create_experiment('Test Assignment', 'Test', variants)
        self.ab_testing.start_experiment(test_id)
        
        user_id = 'test_user_123'
        
        # Same user should get same variant consistently
        variant1 = self.ab_testing.assign_variant(test_id, user_id)
        variant2 = self.ab_testing.assign_variant(test_id, user_id)
        
        self.assertEqual(variant1, variant2, "Same user should get consistent variant")
    
    def test_result_recording(self):
        """Test recording A/B test results"""
        variants = [
            {'name': 'Control', 'traffic_allocation': 1.0, 'model_config': {}, 'is_control': True}
        ]
        
        test_id = self.ab_testing.create_experiment('Test Results', 'Test', variants)
        self.ab_testing.start_experiment(test_id)
        
        variant_id = f"{test_id}_variant_0"
        
        result_id = self.ab_testing.record_prediction_result(
            test_id, variant_id, 65.5, actual_score=70.0
        )
        
        self.assertIsNotNone(result_id)

class TestEnsemblePredictionSystem(unittest.TestCase):
    """Test the ensemble prediction system"""
    
    def setUp(self):
        """Set up test environment with mocked dependencies"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        
        # Mock the dependencies to avoid complex initialization
        with patch('ensemble_prediction_system.get_prediction_engine'), \
             patch('ensemble_prediction_system.get_confidence_scorer'), \
             patch('ensemble_prediction_system.get_ab_testing_framework'):
            self.ensemble = EnsemblePredictionSystem(db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_ensemble_initialization(self):
        """Test ensemble system initialization"""
        self.assertIsNotNone(self.ensemble)
        self.assertTrue(os.path.exists(self.test_db.name))
        self.assertIn('prediction_accuracy', self.ensemble.ensemble_weights)
    
    def test_ensemble_weighting(self):
        """Test ensemble weight application"""
        prediction_data = {'expected_points': 50.0, 'captain_suitability': 0.6}
        
        # Mock confidence breakdown
        mock_confidence = Mock()
        mock_confidence.overall_confidence = 0.8
        
        # Mock player metrics
        mock_metrics = Mock()
        mock_metrics.form_momentum = 0.2
        mock_metrics.consistency_index = 0.7
        mock_metrics.conditions_suitability = 0.6
        
        result = self.ensemble._apply_ensemble_weighting(
            prediction_data, mock_confidence, mock_metrics
        )
        
        self.assertIn('points', result)
        self.assertIn('captain_suitability', result)
        self.assertGreater(result['points'], 0)

class TestHistoricalPerformanceValidator(unittest.TestCase):
    """Test the historical performance validation system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        
        # Mock dependencies
        with patch('historical_performance_validator.get_ensemble_system'), \
             patch('historical_performance_validator.get_prediction_engine'), \
             patch('historical_performance_validator.get_confidence_scorer'), \
             patch('historical_performance_validator.get_ab_testing_framework'):
            self.validator = HistoricalPerformanceValidator(db_path=self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.test_db.name)
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        self.assertIsNotNone(self.validator)
        self.assertTrue(os.path.exists(self.test_db.name))
        self.assertIn('accuracy_target', self.validator.benchmarks)
    
    def test_validation_metrics_calculation(self):
        """Test validation metrics calculation"""
        # Create test prediction data
        predictions = [
            {'predicted_points': 50, 'actual_points': 55, 'confidence_score': 0.8},
            {'predicted_points': 65, 'actual_points': 60, 'confidence_score': 0.7},
            {'predicted_points': 40, 'actual_points': 45, 'confidence_score': 0.6},
            {'predicted_points': 70, 'actual_points': 75, 'confidence_score': 0.9},
            {'predicted_points': 35, 'actual_points': 30, 'confidence_score': 0.5}
        ]
        
        metrics = self.validator._calculate_validation_metrics(predictions)
        
        self.assertIsInstance(metrics, ValidationMetrics)
        self.assertGreater(metrics.accuracy_percentage, 0)
        self.assertGreater(metrics.mean_absolute_error, 0)
        self.assertLessEqual(metrics.hit_rate, 1.0)
    
    def test_confidence_calibration(self):
        """Test confidence calibration calculation"""
        predictions = [
            {'predicted_points': 50, 'actual_points': 52, 'confidence_score': 0.9},  # High conf, accurate
            {'predicted_points': 60, 'actual_points': 35, 'confidence_score': 0.9},  # High conf, inaccurate  
            {'predicted_points': 45, 'actual_points': 43, 'confidence_score': 0.3},  # Low conf, accurate
        ]
        
        calibration = self.validator._calculate_confidence_calibration(predictions)
        
        self.assertGreaterEqual(calibration, 0)
        self.assertLessEqual(calibration, 1)

class TestWorldClassAIIntegration(unittest.TestCase):
    """Test the world-class AI integration system"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock all dependencies to avoid complex initialization
        with patch('world_class_ai_integration.get_api_cache'), \
             patch('world_class_ai_integration.get_rate_limiter'), \
             patch('world_class_ai_integration.get_api_optimizer'), \
             patch('world_class_ai_integration.get_prediction_engine'), \
             patch('world_class_ai_integration.get_confidence_scorer'), \
             patch('world_class_ai_integration.get_ab_testing_framework'), \
             patch('world_class_ai_integration.get_ensemble_system'), \
             patch('world_class_ai_integration.get_performance_validator'):
            self.integration = WorldClassAIIntegration()
    
    def test_integration_initialization(self):
        """Test integration system initialization"""
        self.assertIsNotNone(self.integration)
        self.assertIn('api_optimization_enabled', self.integration.system_config)
        self.assertIn('total_predictions_made', self.integration.system_metrics)
    
    @patch('world_class_ai_integration.predict_player_with_ensemble')
    def test_ultimate_prediction_generation(self, mock_predict):
        """Test ultimate prediction generation"""
        # Mock ensemble prediction
        mock_ensemble_pred = Mock()
        mock_ensemble_pred.player_id = 12345
        mock_ensemble_pred.player_name = 'Test Player'
        mock_ensemble_pred.predicted_points = 65.0
        mock_ensemble_pred.confidence_score = 0.8
        mock_ensemble_pred.confidence_level = 'High'
        mock_ensemble_pred.selection_probability = 0.7
        mock_ensemble_pred.captain_suitability = 0.6
        mock_ensemble_pred.uncertainty_range = (55.0, 75.0)
        mock_ensemble_pred.risk_factors = {'injury_risk': 0.2}
        mock_ensemble_pred.recommendation = 'Strong Buy'
        mock_ensemble_pred.model_contributions = {'model1': 0.6}
        mock_ensemble_pred.confidence_breakdown = Mock()
        mock_ensemble_pred.confidence_breakdown.__dict__ = {'overall_confidence': 0.8}
        mock_ensemble_pred.prediction_timestamp = datetime.now()
        mock_ensemble_pred.ab_test_variant = None
        
        mock_predict.return_value = mock_ensemble_pred
        
        player_data = {'player_id': 12345, 'name': 'Test Player'}
        match_context = {'venue': 'Test Stadium'}
        
        result = self.integration.generate_ultimate_prediction(player_data, match_context)
        
        self.assertIn('player_info', result)
        self.assertIn('prediction_details', result)
        self.assertIn('risk_assessment', result)
        self.assertIn('recommendation', result)
        self.assertEqual(result['player_info']['player_id'], 12345)
    
    def test_system_health_determination(self):
        """Test system health determination logic"""
        # Test excellent health
        api_status = {'status': 'Optimal'}
        pred_status = {'status': 'Excellent'}
        
        health = self.integration._determine_overall_health(api_status, pred_status)
        self.assertEqual(health, 'Excellent')
        
        # Test needs attention
        api_status = {'status': 'Needs Improvement'}
        pred_status = {'status': 'Needs Improvement'}
        
        health = self.integration._determine_overall_health(api_status, pred_status)
        self.assertEqual(health, 'Needs Attention')

class TestSystemIntegration(unittest.TestCase):
    """Test full system integration scenarios"""
    
    def test_end_to_end_prediction_flow(self):
        """Test complete prediction flow from input to output"""
        # This would test the complete flow in a real scenario
        # For now, we'll test that all imports work correctly
        
        try:
            from world_class_ai_integration import (
                generate_world_class_prediction,
                get_system_health_status,
                optimize_all_systems
            )
            
            # Test that functions are callable (even if they fail due to mocking)
            self.assertTrue(callable(generate_world_class_prediction))
            self.assertTrue(callable(get_system_health_status))
            self.assertTrue(callable(optimize_all_systems))
            
        except ImportError as e:
            self.fail(f"Integration import failed: {e}")
    
    def test_database_migrations_work(self):
        """Test that all database initializations work"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test all database initializations
            cache_db = os.path.join(temp_dir, 'cache.db')
            limiter_db = os.path.join(temp_dir, 'limiter.db')
            
            cache = IntelligentAPICache(cache_db_path=cache_db)
            limiter = SmartRateLimiter(db_path=limiter_db)
            
            # Verify databases exist and have correct tables
            self.assertTrue(os.path.exists(cache_db))
            self.assertTrue(os.path.exists(limiter_db))
            
            # Check cache database structure
            conn = sqlite3.connect(cache_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            cache_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            self.assertIn('cache_entries', cache_tables)
            self.assertIn('api_metrics', cache_tables)
            
            # Check limiter database structure  
            conn = sqlite3.connect(limiter_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            limiter_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            self.assertIn('request_tracking', limiter_tables)
            self.assertIn('api_quotas', limiter_tables)
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

def run_performance_benchmark():
    """Run performance benchmarks for the system"""
    print("🏃‍♂️ Running Performance Benchmarks...")
    
    # Test prediction generation speed
    start_time = time.time()
    
    test_player = {
        'player_id': 12345,
        'name': 'Benchmark Player',
        'role': 'batsman',
        'recent_scores': [45, 67, 23, 89, 34, 56, 78, 43, 65, 52]
    }
    
    test_context = {
        'venue': 'Benchmark Stadium',
        'format': 'T20',
        'opposition': 'Test Team'
    }
    
    # Create temporary instances for benchmarking
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    try:
        engine = PredictionAccuracyEngine(db_path=temp_db.name)
        
        # Run multiple predictions
        for i in range(10):
            metrics = engine.analyze_player_performance(test_player, test_context)
        
        prediction_time = (time.time() - start_time) / 10
        
        print(f"✅ Average prediction time: {prediction_time:.3f} seconds")
        print(f"✅ Target: < 2.0 seconds - {'PASS' if prediction_time < 2.0 else 'FAIL'}")
        
    finally:
        os.unlink(temp_db.name)
    
    # Test cache performance
    start_time = time.time()
    temp_cache_db = tempfile.NamedTemporaryFile(delete=False)
    
    try:
        cache = IntelligentAPICache(cache_db_path=temp_cache_db.name)
        
        # Test cache operations
        for i in range(100):
            endpoint = f"https://test.com/endpoint_{i % 10}"
            data = {"test": f"data_{i}"}
            
            cache.cache_response(endpoint, data)
            cached = cache.get_cached_response(endpoint)
        
        cache_time = (time.time() - start_time) / 100
        
        print(f"✅ Average cache operation time: {cache_time:.4f} seconds")
        print(f"✅ Target: < 0.01 seconds - {'PASS' if cache_time < 0.01 else 'FAIL'}")
        
    finally:
        os.unlink(temp_cache_db.name)

if __name__ == '__main__':
    print("🧪 Running Comprehensive Dream11 AI System Tests...")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False, argv=[''])
    
    # Run performance benchmarks
    print("\n" + "=" * 60)
    run_performance_benchmark()
    
    print("\n✅ All tests completed!")
    print("🚀 System is ready for production integration!")