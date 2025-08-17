#!/usr/bin/env python3
"""
End-to-End Validation Test Suite
Tests the complete Dream11 AI system from input to final output
"""

import sys
import os
import json
import time
import tempfile
from datetime import datetime
from typing import Dict, List, Any

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_complete_prediction_workflow():
    """Test complete prediction workflow from player data to final output"""
    print("🧪 Testing Complete Prediction Workflow...")
    
    try:
        # Import the enhanced integration system
        from dream11_enhanced_integration import Dream11EnhancedSystem
        
        # Initialize system
        system = Dream11EnhancedSystem()
        
        # Test data
        test_player = {
            'player_id': 99999,
            'name': 'Test Player',
            'role': 'batsman',
            'team': 'Test Team',
            'recent_scores': [45, 67, 23, 89, 34, 56, 78]
        }
        
        test_context = {
            'venue': 'Test Stadium',
            'format': 'T20',
            'opposition': 'Opposition Team',
            'date': '2023-12-01'
        }
        
        # Generate prediction
        start_time = time.time()
        prediction = system.predict_player_performance(test_player, test_context)
        prediction_time = time.time() - start_time
        
        # Validate prediction structure
        assert isinstance(prediction, dict), "Prediction should be a dictionary"
        assert 'player_info' in prediction, "Should have player_info"
        assert 'prediction_details' in prediction, "Should have prediction_details"
        assert 'recommendation' in prediction, "Should have recommendation"
        assert 'system_info' in prediction, "Should have system_info"
        
        # Validate prediction values
        pred_details = prediction['prediction_details']
        assert pred_details['predicted_points'] > 0, "Predicted points should be positive"
        assert 0 <= pred_details['confidence_score'] <= 1, "Confidence should be between 0 and 1"
        
        print(f"   ✅ Prediction generated in {prediction_time:.3f}s")
        print(f"   ✅ Player: {prediction['player_info']['player_name']}")
        print(f"   ✅ Predicted Points: {pred_details['predicted_points']:.1f}")
        print(f"   ✅ Confidence: {pred_details['confidence_score']:.2f}")
        print(f"   ✅ System: {prediction['system_info']['prediction_system']}")
        
        print("✅ Complete Prediction Workflow: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Complete Prediction Workflow: FAIL - {e}")
        return False

def test_complete_team_generation_workflow():
    """Test complete team generation workflow"""
    print("\n🧪 Testing Complete Team Generation Workflow...")
    
    try:
        from dream11_enhanced_integration import Dream11EnhancedSystem
        
        system = Dream11EnhancedSystem()
        
        # Test players data
        test_players = []
        for i in range(20):
            test_players.append({
                'player_id': i + 1,
                'name': f'Player {i + 1}',
                'role': ['batsman', 'bowler', 'all-rounder', 'wicket-keeper'][i % 4],
                'team': 'Team A' if i < 10 else 'Team B',
                'recent_scores': [40 + (i % 30), 50 + (i % 25), 45 + (i % 20)]
            })
        
        test_context = {
            'venue': 'Test Stadium',
            'format': 'T20',
            'date': '2023-12-01'
        }
        
        # Generate team
        start_time = time.time()
        team = system.generate_optimal_team(test_players, test_context, strategy='balanced')
        team_time = time.time() - start_time
        
        # Validate team structure
        assert isinstance(team, dict), "Team should be a dictionary"
        assert 'team_composition' in team, "Should have team_composition"
        assert 'captain_recommendation' in team, "Should have captain_recommendation"
        assert 'team_metrics' in team, "Should have team_metrics"
        assert 'system_info' in team, "Should have system_info"
        
        # Validate team composition
        composition = team['team_composition']
        assert len(composition) == 11, "Team should have 11 players"
        
        # Validate team metrics
        metrics = team['team_metrics']
        assert metrics['total_predicted_points'] > 0, "Total points should be positive"
        assert 0 <= metrics['team_confidence'] <= 1, "Team confidence should be between 0 and 1"
        
        print(f"   ✅ Team generated in {team_time:.3f}s")
        print(f"   ✅ Team size: {len(composition)} players")
        print(f"   ✅ Captain: {team['captain_recommendation'].get('player_name', 'Unknown')}")
        print(f"   ✅ Total Points: {metrics['total_predicted_points']:.1f}")
        print(f"   ✅ Team Confidence: {metrics['team_confidence']:.1%}")
        print(f"   ✅ System: {team['system_info']['team_generation_system']}")
        
        print("✅ Complete Team Generation Workflow: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Complete Team Generation Workflow: FAIL - {e}")
        return False

def test_match_analysis_workflow():
    """Test complete match analysis workflow"""
    print("\n🧪 Testing Complete Match Analysis Workflow...")
    
    try:
        from dream11_enhanced_integration import Dream11EnhancedSystem
        
        system = Dream11EnhancedSystem()
        
        # Run comprehensive analysis
        start_time = time.time()
        analysis = system.run_comprehensive_analysis("TEST_MATCH_123")
        analysis_time = time.time() - start_time
        
        # Validate analysis structure
        assert isinstance(analysis, dict), "Analysis should be a dictionary"
        assert 'match_id' in analysis, "Should have match_id"
        assert 'system_status' in analysis, "Should have system_status"
        assert 'analysis_timestamp' in analysis, "Should have analysis_timestamp"
        
        # Check if analysis completed successfully
        if analysis.get('match_data_available'):
            assert 'player_predictions' in analysis, "Should have player_predictions"
            assert 'optimal_team' in analysis, "Should have optimal_team"
            
            predictions = analysis['player_predictions']
            assert len(predictions) > 0, "Should have player predictions"
            
            team = analysis['optimal_team']
            assert len(team['team_composition']) > 0, "Should have team composition"
        
        print(f"   ✅ Analysis completed in {analysis_time:.3f}s")
        print(f"   ✅ Match ID: {analysis['match_id']}")
        print(f"   ✅ Data Available: {analysis.get('match_data_available', False)}")
        print(f"   ✅ Enhanced Mode: {analysis['enhanced_mode']}")
        
        if analysis.get('player_predictions'):
            print(f"   ✅ Player Predictions: {len(analysis['player_predictions'])}")
        
        print("✅ Complete Match Analysis Workflow: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Complete Match Analysis Workflow: FAIL - {e}")
        return False

def test_system_status_and_health():
    """Test system status and health monitoring"""
    print("\n🧪 Testing System Status and Health Monitoring...")
    
    try:
        from dream11_enhanced_integration import Dream11EnhancedSystem
        
        system = Dream11EnhancedSystem()
        
        # Get system status
        start_time = time.time()
        status = system.get_system_status()
        status_time = time.time() - start_time
        
        # Validate status structure
        assert isinstance(status, dict), "Status should be a dictionary"
        assert 'timestamp' in status, "Should have timestamp"
        assert 'enhanced_mode' in status, "Should have enhanced_mode"
        assert 'systems_loaded' in status, "Should have systems_loaded"
        
        systems_loaded = status['systems_loaded']
        assert 'dream11_ultimate' in systems_loaded, "Should track dream11_ultimate"
        assert 'world_class_ai' in systems_loaded, "Should track world_class_ai"
        
        print(f"   ✅ Status retrieved in {status_time:.3f}s")
        print(f"   ✅ Enhanced Mode: {status['enhanced_mode']}")
        print(f"   ✅ Dream11 Ultimate: {'✅' if systems_loaded['dream11_ultimate'] else '❌'}")
        print(f"   ✅ World Class AI: {'✅' if systems_loaded['world_class_ai'] else '❌'}")
        
        if status.get('performance_metrics'):
            metrics = status['performance_metrics']
            print(f"   ✅ Cache Hit Rate: {metrics.get('cache_hit_rate', 'N/A')}")
            print(f"   ✅ API Requests Saved: {metrics.get('api_requests_saved', 0)}")
        
        print("✅ System Status and Health Monitoring: PASS")
        return True
        
    except Exception as e:
        print(f"❌ System Status and Health Monitoring: FAIL - {e}")
        return False

def test_data_persistence():
    """Test data persistence and retrieval"""
    print("\n🧪 Testing Data Persistence and Retrieval...")
    
    try:
        # Test that databases are created and accessible
        temp_dir = tempfile.mkdtemp()
        
        # Test core system databases
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_logic'))
        
        from intelligent_api_cache import IntelligentAPICache
        from api_rate_limiter import SmartRateLimiter
        
        # Test cache persistence
        cache_db = os.path.join(temp_dir, 'test_cache.db')
        cache = IntelligentAPICache(cache_db_path=cache_db)
        
        test_endpoint = "https://test-persistence.com/api"
        test_data = {"test": "persistence", "timestamp": datetime.now().isoformat()}
        
        # Store data
        cache.cache_response(test_endpoint, test_data)
        
        # Create new instance to test persistence
        cache2 = IntelligentAPICache(cache_db_path=cache_db)
        retrieved_data = cache2.get_cached_response(test_endpoint)
        
        assert retrieved_data is not None, "Data should persist across instances"
        assert retrieved_data["test"] == "persistence", "Retrieved data should match stored data"
        
        # Test rate limiter persistence  
        limiter_db = os.path.join(temp_dir, 'test_limiter.db')
        limiter = SmartRateLimiter(db_path=limiter_db)
        
        # Add quota and record request
        limiter.add_quota('test_service', 'daily', 1000, cost_per_request=0.01)
        limiter.record_request('https://test.com', 0.5, 200)
        
        # Create new instance to test persistence
        limiter2 = SmartRateLimiter(db_path=limiter_db)
        
        assert 'test_service_daily' in limiter2.quotas, "Quota should persist"
        assert limiter2.quotas['test_service_daily'].quota_limit == 1000, "Quota values should persist"
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        print("   ✅ Cache persistence validated")
        print("   ✅ Rate limiter persistence validated")
        print("   ✅ Database integrity maintained")
        
        print("✅ Data Persistence and Retrieval: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Data Persistence and Retrieval: FAIL - {e}")
        return False

def test_error_handling_and_fallbacks():
    """Test error handling and fallback mechanisms"""
    print("\n🧪 Testing Error Handling and Fallback Mechanisms...")
    
    try:
        from dream11_enhanced_integration import Dream11EnhancedSystem
        
        system = Dream11EnhancedSystem()
        
        # Test with invalid player data
        invalid_player = {}  # Empty player data
        test_context = {'venue': 'Test'}
        
        prediction = system.predict_player_performance(invalid_player, test_context)
        
        # Should still return a valid prediction structure
        assert isinstance(prediction, dict), "Should handle invalid input gracefully"
        assert 'player_info' in prediction, "Should have fallback player info"
        assert 'prediction_details' in prediction, "Should have fallback prediction"
        
        # Test with empty player list
        empty_team = system.generate_optimal_team([], test_context)
        assert isinstance(empty_team, dict), "Should handle empty player list"
        
        # Test with invalid match ID
        invalid_analysis = system.run_comprehensive_analysis("")
        assert isinstance(invalid_analysis, dict), "Should handle invalid match ID"
        
        print("   ✅ Invalid player data handled gracefully")
        print("   ✅ Empty player list handled gracefully") 
        print("   ✅ Invalid match ID handled gracefully")
        print("   ✅ Fallback mechanisms working")
        
        print("✅ Error Handling and Fallback Mechanisms: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Error Handling and Fallback Mechanisms: FAIL - {e}")
        return False

def test_performance_requirements():
    """Test that system meets performance requirements"""
    print("\n🧪 Testing Performance Requirements...")
    
    try:
        from dream11_enhanced_integration import Dream11EnhancedSystem
        
        system = Dream11EnhancedSystem()
        
        # Test single prediction performance
        test_player = {
            'player_id': 1,
            'name': 'Performance Test Player',
            'recent_scores': [45, 67, 23, 89, 34]
        }
        
        start_time = time.time()
        prediction = system.predict_player_performance(test_player, {})
        prediction_time = time.time() - start_time
        
        # Performance targets
        MAX_PREDICTION_TIME = 5.0  # 5 seconds max
        assert prediction_time < MAX_PREDICTION_TIME, f"Prediction took {prediction_time:.3f}s (max: {MAX_PREDICTION_TIME}s)"
        
        # Test batch prediction performance
        test_players = [
            {'player_id': i, 'name': f'Player {i}', 'recent_scores': [45, 50, 55]}
            for i in range(20)
        ]
        
        start_time = time.time()
        team = system.generate_optimal_team(test_players, {})
        team_time = time.time() - start_time
        
        MAX_TEAM_TIME = 10.0  # 10 seconds max for team generation
        assert team_time < MAX_TEAM_TIME, f"Team generation took {team_time:.3f}s (max: {MAX_TEAM_TIME}s)"
        
        print(f"   ✅ Single prediction: {prediction_time:.3f}s (< {MAX_PREDICTION_TIME}s)")
        print(f"   ✅ Team generation: {team_time:.3f}s (< {MAX_TEAM_TIME}s)")
        print("   ✅ Performance requirements met")
        
        print("✅ Performance Requirements: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Performance Requirements: FAIL - {e}")
        return False

def run_end_to_end_validation():
    """Run complete end-to-end validation suite"""
    print("🚀 Dream11 AI End-to-End Validation Suite")
    print("=" * 60)
    
    tests = [
        test_complete_prediction_workflow,
        test_complete_team_generation_workflow,
        test_match_analysis_workflow,
        test_system_status_and_health,
        test_data_persistence,
        test_error_handling_and_fallbacks,
        test_performance_requirements
    ]
    
    passed = 0
    failed = 0
    results = {}
    
    for test in tests:
        test_name = test.__name__
        try:
            start_time = time.time()
            success = test()
            test_time = time.time() - start_time
            
            results[test_name] = {
                'status': 'PASS' if success else 'FAIL',
                'duration': f"{test_time:.3f}s"
            }
            
            if success:
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            test_time = time.time() - start_time
            results[test_name] = {
                'status': 'ERROR',
                'duration': f"{test_time:.3f}s",
                'error': str(e)
            }
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("📊 End-to-End Validation Results:")
    print("=" * 60)
    
    for test_name, result in results.items():
        status_emoji = "✅" if result['status'] == 'PASS' else "❌"
        print(f"{status_emoji} {test_name}: {result['status']} ({result['duration']})")
        if 'error' in result:
            print(f"   Error: {result['error']}")
    
    print("=" * 60)
    print(f"📊 Final Results: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL END-TO-END TESTS PASSED!")
        print("🚀 System is fully validated and production-ready!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. System needs attention before production.")
        return False

if __name__ == '__main__':
    success = run_end_to_end_validation()
    
    if success:
        print("\n✅ End-to-end validation completed successfully!")
        print("🚀 Dream11 AI system is ready for production deployment!")
    else:
        print("\n❌ End-to-end validation found issues.")
        print("🔧 Please address issues before production deployment.")
    
    exit(0 if success else 1)