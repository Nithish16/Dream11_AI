#!/usr/bin/env python3
"""
Simple Integration Test for Dream11 AI Systems
Tests core functionality without complex dependencies
"""

import sys
import os
import tempfile
import time
from datetime import datetime

# Add the core_logic path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_logic'))

def test_api_cache_integration():
    """Test API cache system integration"""
    print("🧪 Testing API Cache Integration...")
    
    try:
        from intelligent_api_cache import IntelligentAPICache
        
        # Create temporary cache
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        cache = IntelligentAPICache(cache_db_path=temp_db.name)
        
        # Test basic operations
        endpoint = "https://test.com/api"
        test_data = {"test": "data", "timestamp": str(datetime.now())}
        
        # Cache data
        success = cache.cache_response(endpoint, test_data)
        assert success, "Cache storage should succeed"
        
        # Retrieve data
        cached_data = cache.get_cached_response(endpoint)
        assert cached_data is not None, "Should retrieve cached data"
        assert cached_data["test"] == "data", "Cached data should match"
        
        # Clean up
        os.unlink(temp_db.name)
        
        print("✅ API Cache Integration: PASS")
        return True
        
    except Exception as e:
        print(f"❌ API Cache Integration: FAIL - {e}")
        return False

def test_rate_limiter_integration():
    """Test rate limiter integration"""
    print("🧪 Testing Rate Limiter Integration...")
    
    try:
        from api_rate_limiter import SmartRateLimiter
        
        # Create temporary rate limiter
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        limiter = SmartRateLimiter(db_path=temp_db.name)
        
        # Test basic operations
        endpoint = "https://cricbuzz.com/test"
        
        # Check if request can be made
        can_make, reason, wait_time = limiter.can_make_request(endpoint)
        assert can_make, "Should be able to make first request"
        assert wait_time == 0, "No wait time for first request"
        
        # Record a request
        limiter.record_request(endpoint, 0.5, 200)
        
        # Check metrics
        assert endpoint in limiter.metrics, "Should track endpoint metrics"
        metrics = limiter.metrics[endpoint]
        assert metrics.success_count == 1, "Should record successful request"
        
        # Clean up
        os.unlink(temp_db.name)
        
        print("✅ Rate Limiter Integration: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Rate Limiter Integration: FAIL - {e}")
        return False

def test_prediction_engine_integration():
    """Test prediction engine integration"""
    print("🧪 Testing Prediction Engine Integration...")
    
    try:
        from prediction_accuracy_engine import PredictionAccuracyEngine, PlayerPerformanceMetrics
        
        # Create temporary engine
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        engine = PredictionAccuracyEngine(db_path=temp_db.name)
        
        # Test player analysis
        player_data = {
            'player_id': 12345,
            'name': 'Test Player',
            'role': 'batsman',
            'team': 'Test Team',
            'recent_scores': [45, 67, 23, 89, 34]
        }
        
        match_context = {
            'venue': 'Test Stadium',
            'format': 'T20',
            'opposition': 'Opposition Team'
        }
        
        metrics = engine.analyze_player_performance(player_data, match_context)
        
        assert isinstance(metrics, PlayerPerformanceMetrics), "Should return PlayerPerformanceMetrics"
        assert metrics.player_id == 12345, "Should preserve player ID"
        assert metrics.expected_points > 0, "Should generate positive prediction"
        
        # Clean up
        os.unlink(temp_db.name)
        
        print("✅ Prediction Engine Integration: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Prediction Engine Integration: FAIL - {e}")
        return False

def test_system_integration():
    """Test basic system integration"""
    print("🧪 Testing System Integration...")
    
    try:
        # Test that we can import the world-class integration
        from world_class_ai_integration import WorldClassAIIntegration
        
        # Create integration system (with mocked dependencies)
        integration = WorldClassAIIntegration()
        
        assert integration is not None, "Should create integration system"
        assert 'api_optimization_enabled' in integration.system_config, "Should have system config"
        
        # Test basic configuration
        assert integration.system_config['api_optimization_enabled'] == True, "API optimization should be enabled"
        assert integration.system_config['prediction_validation_enabled'] == True, "Prediction validation should be enabled"
        
        print("✅ System Integration: PASS")
        return True
        
    except Exception as e:
        print(f"❌ System Integration: FAIL - {e}")
        return False

def test_database_creation():
    """Test that all databases can be created successfully"""
    print("🧪 Testing Database Creation...")
    
    try:
        import sqlite3
        
        # Test cache database
        from intelligent_api_cache import IntelligentAPICache
        temp_cache = tempfile.NamedTemporaryFile(delete=False)
        cache = IntelligentAPICache(cache_db_path=temp_cache.name)
        
        conn = sqlite3.connect(temp_cache.name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        cache_tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert 'cache_entries' in cache_tables, "Cache should have cache_entries table"
        assert 'api_metrics' in cache_tables, "Cache should have api_metrics table"
        
        # Test rate limiter database
        from api_rate_limiter import SmartRateLimiter
        temp_limiter = tempfile.NamedTemporaryFile(delete=False)
        limiter = SmartRateLimiter(db_path=temp_limiter.name)
        
        conn = sqlite3.connect(temp_limiter.name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        limiter_tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert 'request_tracking' in limiter_tables, "Limiter should have request_tracking table"
        assert 'api_quotas' in limiter_tables, "Limiter should have api_quotas table"
        
        # Clean up
        os.unlink(temp_cache.name)
        os.unlink(temp_limiter.name)
        
        print("✅ Database Creation: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Database Creation: FAIL - {e}")
        return False

def test_performance_benchmarks():
    """Test basic performance benchmarks"""
    print("🧪 Testing Performance Benchmarks...")
    
    try:
        from intelligent_api_cache import IntelligentAPICache
        
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        cache = IntelligentAPICache(cache_db_path=temp_db.name)
        
        # Benchmark cache operations
        start_time = time.time()
        
        for i in range(100):
            endpoint = f"https://test.com/{i % 10}"
            data = {"test": f"data_{i}"}
            cache.cache_response(endpoint, data)
            cached = cache.get_cached_response(endpoint)
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        print(f"   - Average cache operation time: {avg_time:.4f}s")
        
        # Performance should be reasonable
        assert avg_time < 0.1, f"Cache operations should be fast, got {avg_time:.4f}s"
        
        # Clean up
        os.unlink(temp_db.name)
        
        print("✅ Performance Benchmarks: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmarks: FAIL - {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Dream11 AI Integration Tests")
    print("=" * 50)
    
    tests = [
        test_api_cache_integration,
        test_rate_limiter_integration,
        test_prediction_engine_integration,
        test_system_integration,
        test_database_creation,
        test_performance_benchmarks
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAIL - {e}")
            failed += 1
        
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! System is ready for integration.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        return False

if __name__ == '__main__':
    success = run_integration_tests()
    
    if success:
        print("\n✅ Integration tests completed successfully!")
        print("🚀 Ready to proceed with production integration!")
    else:
        print("\n❌ Integration tests found issues.")
        print("🔧 Please fix issues before proceeding.")
    
    exit(0 if success else 1)