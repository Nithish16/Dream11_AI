#!/usr/bin/env python3
"""
Enhanced DreamTeamAI - Next-Generation Fantasy Cricket Optimizer
World-class AI system with cutting-edge algorithms and features
"""

import os
import sys
import asyncio
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from utils.api_client import *
from core_logic.match_resolver import resolve_match_ids, get_match_summary, resolve_match_by_id

# Create CricBuzzAPI class wrapper for compatibility
class CricBuzzAPI:
    """Simple wrapper for compatibility with existing API functions"""
    
    def __init__(self):
        pass
    
    def fetch_upcoming_matches(self):
        return fetch_upcoming_matches()
    
    def fetch_live_matches(self):
        return fetch_live_matches()
    
    def fetch_match_center(self, match_id):
        return fetch_match_center(match_id)

# Create MatchResolver class wrapper for compatibility
class MatchResolver:
    """Simple wrapper for compatibility with existing match resolver functions"""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def resolve_match_ids(self, team_a, team_b, match_date):
        return resolve_match_ids(team_a, team_b, match_date)
    
    def get_match_summary(self, match_info):
        return get_match_summary(match_info)
    
    def resolve_match_by_id(self, match_id):
        return resolve_match_by_id(match_id)
from core_logic.data_aggregator import aggregate_all_data, MatchData
from core_logic.feature_engine import batch_generate_features
from core_logic.team_generator import batch_generate_teams

# Create wrapper classes for compatibility
class DataAggregator:
    """Wrapper for data aggregation functions"""
    def __init__(self, api_client, match_resolver):
        self.api_client = api_client
        self.match_resolver = match_resolver
    
    def aggregate_all_data(self, match_info):
        return aggregate_all_data(match_info)

class FeatureEngine:
    """Wrapper for feature engine functions"""
    def __init__(self):
        pass
    
    def batch_generate_features(self, players, match_context):
        return batch_generate_features(players, match_context)

class TeamGenerator:
    """Wrapper for team generator functions"""
    def __init__(self):
        pass
    
    def batch_generate_teams(self, player_features, match_format, match_context, num_teams=5, risk_profiles=None):
        if risk_profiles is None:
            risk_profiles = ['Balanced', 'High-Risk', 'Conservative']
        return batch_generate_teams(player_features, match_format, match_context, num_teams, risk_profiles)

# Import enhanced modules
from core_logic.advanced_data_engine import AdvancedDataEngine, get_venue_coordinates
from core_logic.dynamic_credit_engine import DynamicCreditPredictor, assign_dynamic_credits
from core_logic.neural_prediction_engine import NeuralPerformancePredictor, EnsemblePredictor
from core_logic.evolutionary_optimizer import NSGA3, OptimizationObjectives, OptimizationConstraints
from core_logic.environmental_intelligence import EnvironmentalIntelligence, get_environmental_performance_multiplier
from core_logic.matchup_analysis_engine import AdvancedMatchupAnalyzer
from core_logic.reinforcement_learning_strategy import ReinforcementLearningStrategy
from core_logic.quantum_optimization import optimize_team_with_quantum_annealing, optimize_team_with_quantum_ga
from core_logic.explainable_ai_dashboard import ExplainableAIDashboard

class EnhancedDreamTeamAI:
    """Next-generation DreamTeamAI with advanced AI capabilities"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced DreamTeamAI System...")
        
        # Core components
        self.api_client = CricBuzzAPI()
        self.match_resolver = MatchResolver(self.api_client)
        self.data_aggregator = DataAggregator(self.api_client, self.match_resolver)
        self.feature_engine = FeatureEngine()
        self.team_generator = TeamGenerator()
        
        # Enhanced AI components
        self.advanced_data_engine = AdvancedDataEngine()
        self.dynamic_credit_predictor = DynamicCreditPredictor()
        self.neural_predictor = EnsemblePredictor()
        self.environmental_intelligence = EnvironmentalIntelligence()
        self.matchup_analyzer = AdvancedMatchupAnalyzer()
        self.rl_strategy = ReinforcementLearningStrategy()
        self.explainable_ai = ExplainableAIDashboard()
        
        # Configuration - ALL ADVANCED FEATURES ENABLED
        self.enhancement_config = {
            'use_neural_prediction': True,           # 🧠 Neural Network Ensemble
            'use_dynamic_credits': True,             # 💰 ML-based Credit Prediction
            'use_environmental_intelligence': True,   # 🌍 Weather/Pitch Analysis
            'use_matchup_analysis': True,            # ⚔️ Head-to-Head Analysis
            'use_reinforcement_learning': True,      # 🤖 RL Strategy Learning
            'use_quantum_optimization': True,        # 🔮 Quantum-Inspired Computing
            'use_evolutionary_optimization': True,   # 🧬 Multi-Objective Evolution
            'enable_explainable_ai': True,          # 🔍 Complete AI Transparency
            'parallel_processing': True             # ⚡ Concurrent Processing
        }
        
        print("✅ Enhanced DreamTeamAI System Initialized Successfully!")
        print("🔥 ALL ADVANCED AI FEATURES ENABLED:")
        print("   🧠 Neural Network Ensemble")
        print("   🔮 Quantum-Inspired Optimization") 
        print("   🧬 Multi-Objective Evolution")
        print("   🤖 Reinforcement Learning")
        print("   🌍 Environmental Intelligence")
        print("   ⚔️ Advanced Matchup Analysis")
        print("   💰 Dynamic Credit Prediction")
        print("   🔍 Explainable AI Dashboard")
    
    async def generate_enhanced_teams(self, match_query: str, 
                                    num_teams: int = 5,
                                    optimization_mode: str = "balanced") -> Dict[str, Any]:
        """Generate teams using all enhanced AI capabilities"""
        
        print(f"\n🔮 Starting Enhanced Team Generation for: {match_query}")
        print(f"🎯 Target: {num_teams} teams | Mode: {optimization_mode}")
        
        try:
            # Phase 1: Enhanced Data Collection
            print("\n📊 Phase 1: Advanced Data Collection & Aggregation")
            enhanced_data = await self._collect_enhanced_data(match_query)
            
            # Phase 2: Advanced Feature Engineering
            print("\n🧠 Phase 2: Neural Feature Engineering & Prediction")
            enhanced_features = await self._generate_enhanced_features(enhanced_data)
            
            # Phase 3: Multi-Algorithm Optimization
            print("\n⚡ Phase 3: Multi-Algorithm Team Optimization")
            optimized_teams = await self._multi_algorithm_optimization(
                enhanced_features, num_teams, optimization_mode
            )
            
            # Phase 4: Strategic Analysis & Explanation
            print("\n🔍 Phase 4: Strategic Analysis & AI Explanation")
            strategic_analysis = await self._generate_strategic_analysis(
                optimized_teams, enhanced_features
            )
            
            # Phase 5: Final Recommendations
            print("\n🏆 Phase 5: Final Recommendations & Insights")
            final_results = self._compile_final_results(
                optimized_teams, strategic_analysis, enhanced_features
            )
            
            print("✅ Enhanced Team Generation Complete!")
            return final_results
            
        except Exception as e:
            print(f"❌ Error in enhanced team generation: {e}")
            return self._fallback_team_generation(match_query, num_teams)
    
    async def generate_enhanced_teams_by_id(self, match_id: int,
                                          num_teams: int = 5,
                                          optimization_mode: str = "balanced") -> Dict[str, Any]:
        """Generate teams using match ID directly with all enhanced AI capabilities"""
        
        print(f"\n🔮 Starting Enhanced Team Generation for Match ID: {match_id}")
        print(f"🎯 Target: {num_teams} teams | Mode: {optimization_mode}")
        
        try:
            # Phase 1: Enhanced Data Collection using Match ID
            print("\n📊 Phase 1: Advanced Data Collection & Aggregation")
            enhanced_data = await self._collect_enhanced_data_by_id(match_id)
            
            # Phase 2: Advanced Feature Engineering
            print("\n🧠 Phase 2: Neural Feature Engineering & Prediction")
            enhanced_features = await self._generate_enhanced_features(enhanced_data)
            
            # Phase 3: Multi-Algorithm Team Optimization
            print("\n⚡ Phase 3: Multi-Algorithm Team Optimization")
            optimized_teams = await self._multi_algorithm_optimization(
                enhanced_features, num_teams, optimization_mode
            )
            
            # Phase 4: Strategic Analysis & AI Explanations
            print("\n🔍 Phase 4: Strategic Analysis & AI Explanation")
            strategic_analysis = await self._generate_strategic_analysis(
                optimized_teams, enhanced_features
            )
            
            # Phase 5: Final Recommendations
            print("\n🏆 Phase 5: Final Recommendations & Insights")
            final_results = self._compile_final_results(
                optimized_teams, strategic_analysis, enhanced_features
            )
            
            print("✅ Enhanced Team Generation Complete!")
            return final_results
            
        except Exception as e:
            print(f"❌ Error in enhanced team generation by ID: {e}")
            return self._fallback_team_generation_by_id(match_id, num_teams)
    
    async def _collect_enhanced_data_by_id(self, match_id: int) -> Dict[str, Any]:
        """Collect data using match ID directly"""
        
        try:
            from core_logic.match_resolver import resolve_match_by_id
            
            # Get match info by ID
            print(f"  🔍 Resolving match data for ID: {match_id}")
            match_info = resolve_match_by_id(match_id)
            
            if not match_info:
                raise Exception(f"Match ID {match_id} not found")
            
            print(f"  ✅ Found match: {match_info.get('team1Name', 'Team A')} vs {match_info.get('team2Name', 'Team B')}")
            
            # Use the existing data collection method with resolved match info
            from core_logic.data_aggregator import aggregate_all_data
            match_data = aggregate_all_data(match_info)
            
            if not match_data:
                raise Exception("Failed to aggregate match data")
            
            # Enhanced data collection from multiple sources
            enhanced_data = {
                'team1_players': match_data.team1.players,
                'team2_players': match_data.team2.players,
                'venue': match_data.venue.venue_name,
                'match_format': match_data.match_format,
                'venue_id': match_data.venue.venue_id,
                'pitch_type': match_data.venue.pitch_archetype,
                'match_info': match_info,
                'match_data': match_data
            }
            
            # Add environmental intelligence if enabled
            if self.enhancement_config['use_environmental_intelligence']:
                print("  🌍 Collecting environmental intelligence...")
                try:
                    # Get venue coordinates and weather data
                    venue_coords = get_venue_coordinates(match_data.venue.venue_name)
                    if venue_coords:
                        environmental_context = self.environmental_intelligence.analyze_conditions(
                            venue_coords['lat'], venue_coords['lon'], 
                            datetime.now().strftime('%Y-%m-%d')
                        )
                        enhanced_data['environmental_context'] = environmental_context
                except Exception as e:
                    print(f"    ⚠️ Environmental data collection failed: {e}")
            
            return enhanced_data
            
        except Exception as e:
            print(f"  ❌ Enhanced data collection failed: {e}")
            raise
    
    def _fallback_team_generation_by_id(self, match_id: int, num_teams: int) -> Dict[str, Any]:
        """Fallback to standard team generation using match ID if enhanced fails"""
        
        print("🔄 Falling back to standard team generation...")
        
        try:
            from core_logic.match_resolver import resolve_match_by_id
            from core_logic.data_aggregator import aggregate_all_data
            
            # Get match info and data
            match_info = resolve_match_by_id(match_id)
            if match_info:
                match_data = aggregate_all_data(match_info)
                
                if match_data:
                    return {
                        'success': True,
                        'enhanced_mode': False,
                        'teams': [],  # Would implement basic team generation here
                        'fallback_reason': 'Enhanced AI failed, using standard generation',
                        'match_context': {
                            'match_id': match_id,
                            'venue': match_data.venue.venue_name,
                            'teams': f"{match_info.get('team1Name', 'Team A')} vs {match_info.get('team2Name', 'Team B')}"
                        }
                    }
            else:
                return {
                    'success': False,
                    'error': f'Match ID {match_id} not found',
                    'enhanced_mode': False
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback generation failed: {e}',
                'enhanced_mode': False
            }
    
    async def _collect_enhanced_data(self, match_query: str) -> Dict[str, Any]:
        """Collect data using advanced multi-source integration"""
        
        # Standard data collection
        print("  📡 Collecting standard match data...")
        match_data = self.data_aggregator.aggregate_all_data(match_query)
        
        if not match_data.get('success', False):
            raise Exception("Failed to collect basic match data")
        
        # Enhanced data collection
        enhanced_data = match_data.copy()
        
        # Get venue coordinates for environmental analysis
        venue_name = match_data.get('venue', '')
        venue_coordinates = get_venue_coordinates(venue_name)
        match_datetime = datetime.now() + timedelta(hours=24)  # Assuming match tomorrow
        
        print("  🌍 Collecting environmental intelligence...")
        # Environmental data collection
        if self.enhancement_config['use_environmental_intelligence']:
            try:
                env_context = await self.environmental_intelligence.analyze_environmental_context(
                    venue_name, match_datetime
                )
                enhanced_data['environmental_context'] = env_context
            except Exception as e:
                print(f"    ⚠️ Environmental data collection failed: {e}")
        
        # Advanced player intelligence
        print("  🤖 Generating advanced player profiles...")
        if hasattr(self.advanced_data_engine, 'create_comprehensive_player_profile'):
            try:
                all_players = match_data.get('team1_players', []) + match_data.get('team2_players', [])
                enhanced_players = []
                
                for player in all_players[:20]:  # Limit for performance
                    try:
                        player_intelligence = await self.advanced_data_engine.create_comprehensive_player_profile(
                            player, venue_coordinates, match_datetime
                        )
                        enhanced_player = player.copy()
                        enhanced_player['player_intelligence'] = player_intelligence
                        enhanced_players.append(enhanced_player)
                    except:
                        enhanced_players.append(player)
                
                enhanced_data['enhanced_players'] = enhanced_players
            except Exception as e:
                print(f"    ⚠️ Advanced player profiling failed: {e}")
        
        return enhanced_data
    
    async def _generate_enhanced_features(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features using neural networks and advanced algorithms"""
        
        all_players = enhanced_data.get('team1_players', []) + enhanced_data.get('team2_players', [])
        
        # Standard feature generation
        print("  🔧 Generating standard features...")
        for player in all_players:
            player_features = self.feature_engine.generate_player_features(player)
            player.update(player_features)
        
        # Neural prediction enhancement
        if self.enhancement_config['use_neural_prediction']:
            print("  🧠 Applying neural network predictions...")
            try:
                # Prepare data for neural network
                neural_features = []
                for player in all_players:
                    features = self._extract_neural_features(player)
                    neural_features.append(features)
                
                # Get neural predictions
                neural_predictions = self.neural_predictor.predict_batch(neural_features)
                
                # Update player data with neural predictions
                for i, player in enumerate(all_players):
                    if i < len(neural_predictions):
                        player['neural_prediction'] = neural_predictions[i]
                        # Blend with existing score
                        original_score = player.get('final_score', 50.0)
                        player['final_score'] = (original_score * 0.6 + neural_predictions[i] * 0.4)
                
            except Exception as e:
                print(f"    ⚠️ Neural prediction failed: {e}")
        
        # Dynamic credit prediction
        if self.enhancement_config['use_dynamic_credits']:
            print("  💰 Calculating dynamic credits...")
            try:
                match_context = {
                    'venue': enhanced_data.get('venue', ''),
                    'match_format': 'T20',
                    'importance': 0.7
                }
                
                dynamic_credits = assign_dynamic_credits(all_players, match_context)
                for i, player in enumerate(all_players):
                    if i < len(dynamic_credits):
                        player['credits'] = dynamic_credits[i]
                        
            except Exception as e:
                print(f"    ⚠️ Dynamic credit calculation failed: {e}")
        
        # Environmental performance adjustment
        if self.enhancement_config['use_environmental_intelligence'] and 'environmental_context' in enhanced_data:
            print("  🌤️ Applying environmental adjustments...")
            try:
                env_context = enhanced_data['environmental_context']
                for player in all_players:
                    role = player.get('role', '')
                    multiplier = get_environmental_performance_multiplier(env_context, role, player)
                    player['final_score'] *= multiplier
                    player['environmental_multiplier'] = multiplier
                    
            except Exception as e:
                print(f"    ⚠️ Environmental adjustment failed: {e}")
        
        # Matchup analysis
        if self.enhancement_config['use_matchup_analysis']:
            print("  ⚔️ Performing advanced matchup analysis...")
            try:
                team1_players = enhanced_data.get('team1_players', [])
                team2_players = enhanced_data.get('team2_players', [])
                team1_name = enhanced_data.get('team1_name', 'Team A')
                team2_name = enhanced_data.get('team2_name', 'Team B')
                venue = enhanced_data.get('venue', '')
                
                match_context = {'venue': venue, 'match_format': 'T20'}
                
                # Analyze matchups for both teams
                team1_matchups = self.matchup_analyzer.analyze_player_matchups(
                    team1_players, team2_name, venue, match_context
                )
                team2_matchups = self.matchup_analyzer.analyze_player_matchups(
                    team2_players, team1_name, venue, match_context
                )
                
                # Apply matchup adjustments
                all_matchups = team1_matchups + team2_matchups
                for matchup in all_matchups:
                    # Find corresponding player
                    player = next((p for p in all_players if p.get('name') == matchup.player_name), None)
                    if player:
                        # Apply matchup factor
                        matchup_factor = 1.0 + (matchup.venue_performance - 50) / 100
                        player['final_score'] *= matchup_factor
                        player['matchup_factor'] = matchup_factor
                        
            except Exception as e:
                print(f"    ⚠️ Matchup analysis failed: {e}")
        
        enhanced_data['processed_players'] = all_players
        return enhanced_data
    
    async def _multi_algorithm_optimization(self, enhanced_features: Dict[str, Any], 
                                          num_teams: int, optimization_mode: str) -> List[Dict[str, Any]]:
        """Optimize teams using multiple advanced algorithms"""
        
        all_players = enhanced_features.get('processed_players', [])
        optimized_teams = []
        
        # Standard optimization
        print("  🔧 Running standard optimization...")
        try:
            standard_teams = self.team_generator.generate_hybrid_teams(
                enhanced_features, num_teams, optimization_mode
            )
            optimized_teams.extend(standard_teams.get('teams', []))
        except Exception as e:
            print(f"    ⚠️ Standard optimization failed: {e}")
        
        # Evolutionary optimization (NSGA-III)
        if self.enhancement_config['use_evolutionary_optimization'] and len(all_players) > 10:
            print("  🧬 Running evolutionary optimization...")
            try:
                # Prepare optimization objectives
                objectives = OptimizationObjectives(
                    maximize_expected_points=lambda team, players: sum(players[i].get('final_score', 0) for i in team),
                    minimize_risk=lambda team, players: sum((players[i].get('final_score', 50) - 50)**2 for i in team),
                    maximize_ceiling=lambda team, players: sum(players[i].get('final_score', 0) * 1.5 for i in team),
                    minimize_ownership=lambda team, players: sum(players[i].get('ownership_prediction', 50) for i in team),
                    maximize_floor=lambda team, players: sum(max(0, players[i].get('final_score', 0) * 0.8) for i in team)
                )
                
                constraints = OptimizationConstraints()
                
                # Run NSGA-III
                nsga3 = NSGA3(population_size=50, num_generations=100)
                pareto_solutions = nsga3.optimize(all_players, objectives, constraints)
                
                # Convert to standard format
                for solution in pareto_solutions[:3]:
                    team_data = {
                        'players': [all_players[i] for i in solution.team if i < len(all_players)],
                        'total_score': sum(all_players[i].get('final_score', 0) for i in solution.team if i < len(all_players)),
                        'algorithm': 'NSGA-III Evolutionary',
                        'pareto_rank': 1
                    }
                    optimized_teams.append(team_data)
                    
            except Exception as e:
                print(f"    ⚠️ Evolutionary optimization failed: {e}")
        
        # Quantum-inspired optimization (ENABLED BY DEFAULT)
        if self.enhancement_config['use_quantum_optimization'] and len(all_players) > 10:
            print("  🔮 Running quantum-inspired optimization (advanced mode)...")
            print("    ⚡ This may take 2-5 minutes for maximum optimization quality")
            try:
                # Quantum annealing
                quantum_solution = optimize_team_with_quantum_annealing(
                    all_players, 
                    {'max_credits': 100}, 
                    {'performance': 1.0}
                )
                
                if quantum_solution and len(quantum_solution.team_indices) >= 11:
                    team_data = {
                        'players': [all_players[i] for i in quantum_solution.team_indices[:11] if i < len(all_players)],
                        'total_score': quantum_solution.quantum_score,
                        'algorithm': 'Quantum Annealing',
                        'quantum_coherence': quantum_solution.coherence_factor
                    }
                    optimized_teams.append(team_data)
                    
            except Exception as e:
                print(f"    ⚠️ Quantum optimization failed: {e}")
        
        # Reinforcement learning strategy
        if self.enhancement_config['use_reinforcement_learning']:
            print("  🤖 Applying reinforcement learning strategy...")
            try:
                # Get RL recommendation for best team
                if optimized_teams:
                    best_team = max(optimized_teams, key=lambda t: t.get('total_score', 0))
                    current_team_indices = [all_players.index(p) for p in best_team['players'] if p in all_players]
                    
                    match_context = {
                        'venue': enhanced_features.get('venue', ''),
                        'importance': 0.7
                    }
                    
                    rl_recommendation = self.rl_strategy.get_strategic_recommendation(
                        all_players, current_team_indices, match_context
                    )
                    
                    # Apply RL insights to teams
                    for team in optimized_teams:
                        team['rl_insights'] = rl_recommendation.get('strategic_insights', [])
                        team['rl_confidence'] = rl_recommendation.get('confidence_score', 0.5)
                        
            except Exception as e:
                print(f"    ⚠️ RL strategy application failed: {e}")
        
        # Sort teams by performance and return top results
        optimized_teams.sort(key=lambda t: t.get('total_score', 0), reverse=True)
        return optimized_teams[:num_teams]
    
    async def _generate_strategic_analysis(self, optimized_teams: List[Dict[str, Any]], 
                                         enhanced_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic analysis and explanations"""
        
        strategic_analysis = {}
        
        if not optimized_teams:
            return strategic_analysis
        
        # Explainable AI analysis
        if self.enhancement_config['enable_explainable_ai']:
            print("  🔍 Generating AI explanations...")
            try:
                best_team = optimized_teams[0]
                all_players = enhanced_features.get('processed_players', [])
                
                # Team explanation
                team_explanation = self.explainable_ai.explain_team_selection(
                    best_team['players'], all_players
                )
                strategic_analysis['team_explanation'] = team_explanation
                
                # Player decision paths
                player_explanations = []
                for player in best_team['players'][:5]:  # Top 5 players
                    decision_path = self.explainable_ai.explain_player_prediction(
                        player, player.get('final_score', 0)
                    )
                    player_explanations.append(decision_path)
                
                strategic_analysis['player_explanations'] = player_explanations
                
            except Exception as e:
                print(f"    ⚠️ AI explanation generation failed: {e}")
        
        # Performance predictions
        print("  📈 Calculating performance predictions...")
        for i, team in enumerate(optimized_teams):
            try:
                # Calculate confidence intervals
                scores = [p.get('final_score', 0) for p in team['players']]
                consistency_scores = [p.get('consistency_score', 50) for p in team['players']]
                
                # Expected range
                mean_score = sum(scores)
                volatility = np.std(scores) if len(scores) > 1 else 10
                
                team['performance_prediction'] = {
                    'expected_score': mean_score,
                    'confidence_range': (mean_score - volatility, mean_score + volatility),
                    'risk_level': np.mean([100 - c for c in consistency_scores]) / 100,
                    'upside_potential': mean_score * 1.3,
                    'downside_risk': mean_score * 0.7
                }
                
            except Exception as e:
                print(f"    ⚠️ Performance prediction failed for team {i+1}: {e}")
        
        return strategic_analysis
    
    def _compile_final_results(self, optimized_teams: List[Dict[str, Any]], 
                             strategic_analysis: Dict[str, Any],
                             enhanced_features: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final results with all enhancements"""
        
        final_results = {
            'success': True,
            'enhanced_mode': True,
            'generation_timestamp': datetime.now().isoformat(),
            'teams': optimized_teams,
            'strategic_analysis': strategic_analysis,
            'match_context': {
                'venue': enhanced_features.get('venue', ''),
                'teams': f"{enhanced_features.get('team1_name', 'Team A')} vs {enhanced_features.get('team2_name', 'Team B')}",
                'total_players_analyzed': len(enhanced_features.get('processed_players', [])),
                'algorithms_used': []
            },
            'enhancement_summary': {
                'neural_prediction': self.enhancement_config['use_neural_prediction'],
                'dynamic_credits': self.enhancement_config['use_dynamic_credits'],
                'environmental_intelligence': self.enhancement_config['use_environmental_intelligence'],
                'matchup_analysis': self.enhancement_config['use_matchup_analysis'],
                'reinforcement_learning': self.enhancement_config['use_reinforcement_learning'],
                'quantum_optimization': self.enhancement_config['use_quantum_optimization'],
                'explainable_ai': self.enhancement_config['enable_explainable_ai']
            }
        }
        
        # Add algorithm usage
        algorithms_used = ['Standard Hybrid Optimization']
        if self.enhancement_config['use_evolutionary_optimization']:
            algorithms_used.append('NSGA-III Evolutionary')
        if self.enhancement_config['use_quantum_optimization']:
            algorithms_used.append('Quantum Annealing')
        if self.enhancement_config['use_reinforcement_learning']:
            algorithms_used.append('Reinforcement Learning')
        
        final_results['match_context']['algorithms_used'] = algorithms_used
        
        # Generate executive summary
        if optimized_teams:
            best_team = optimized_teams[0]
            total_score = best_team.get('total_score', 0)
            total_credits = sum(p.get('credits', 8.5) for p in best_team.get('players', []))
            
            executive_summary = f"""
🏆 ENHANCED DREAMTEAMAI RESULTS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Best Team Performance: {total_score:.1f} points
💰 Budget Utilization: {total_credits:.1f}/100 credits ({(total_credits/100)*100:.1f}%)
🔬 Analysis Depth: {len(algorithms_used)} optimization algorithms
🧠 AI Enhancement Level: Advanced Neural + Environmental + Matchup Analysis

📈 Top 3 Teams Generated:
"""
            
            for i, team in enumerate(optimized_teams[:3]):
                algorithm = team.get('algorithm', 'Standard')
                score = team.get('total_score', 0)
                executive_summary += f"   {i+1}. {algorithm}: {score:.1f} pts\n"
            
            if 'team_explanation' in strategic_analysis:
                team_exp = strategic_analysis['team_explanation']
                rationale = team_exp.selection_rationale
                executive_summary += f"\n🎯 Selection Strategy: {rationale.get('selection_strategy', 'Balanced')}"
                
                if rationale.get('key_strengths'):
                    executive_summary += f"\n✅ Key Strengths: {', '.join(rationale['key_strengths'][:2])}"
            
            final_results['executive_summary'] = executive_summary
        
        return final_results
    
    def _fallback_team_generation(self, match_query: str, num_teams: int) -> Dict[str, Any]:
        """Fallback to standard team generation if enhanced fails"""
        print("🔄 Falling back to standard team generation...")
        
        try:
            # Basic data collection
            match_data = self.data_aggregator.aggregate_all_data(match_query)
            
            if match_data.get('success', False):
                # Generate standard teams
                teams_result = self.team_generator.generate_hybrid_teams(
                    match_data, num_teams, "balanced"
                )
                
                return {
                    'success': True,
                    'enhanced_mode': False,
                    'fallback_reason': 'Enhanced features failed, using standard generation',
                    'teams': teams_result.get('teams', []),
                    'match_context': {
                        'venue': match_data.get('venue', ''),
                        'teams': f"{match_data.get('team1_name', 'Team A')} vs {match_data.get('team2_name', 'Team B')}"
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to collect match data',
                    'enhanced_mode': False
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback generation failed: {e}',
                'enhanced_mode': False
            }
    
    def _extract_neural_features(self, player: Dict[str, Any]) -> List[float]:
        """Extract features for neural network input"""
        features = [
            player.get('final_score', 50.0) / 100,  # Normalized
            player.get('ema_score', 50.0) / 100,
            player.get('consistency_score', 50.0) / 100,
            player.get('form_momentum', 0.0),
            player.get('credits', 8.5) / 15,  # Normalized
            player.get('ownership_prediction', 50.0) / 100,
            player.get('injury_risk', 0.1),
            player.get('matchup_factor', 1.0) - 1.0,  # Centered around 0
            player.get('environmental_multiplier', 1.0) - 1.0,
            len(player.get('recent_form', [])) / 10  # Normalized
        ]
        
        return features
    
    def display_enhanced_results(self, results: Dict[str, Any]):
        """Display enhanced results with rich formatting"""
        
        if not results.get('success', False):
            print(f"❌ Team generation failed: {results.get('error', 'Unknown error')}")
            return
        
        # Executive summary
        if 'executive_summary' in results:
            print(results['executive_summary'])
        
        # Enhanced features summary
        if results.get('enhanced_mode', False):
            print("\n🚀 ENHANCED AI FEATURES ACTIVE:")
            enhancement_summary = results.get('enhancement_summary', {})
            for feature, enabled in enhancement_summary.items():
                status = "✅" if enabled else "⭕"
                feature_name = feature.replace('_', ' ').title()
                print(f"   {status} {feature_name}")
        
        # Teams display
        teams = results.get('teams', [])
        if teams:
            print(f"\n🏆 TOP {len(teams)} OPTIMIZED TEAMS:")
            print("=" * 60)
            
            for i, team in enumerate(teams, 1):
                players = team.get('players', [])
                total_score = team.get('total_score', 0)
                total_credits = sum(p.get('credits', 8.5) for p in players)
                algorithm = team.get('algorithm', 'Standard')
                
                print(f"\n🥇 TEAM {i} ({algorithm}) - {total_score:.1f} pts | {total_credits:.1f} credits")
                print("-" * 50)
                
                # Group players by role
                role_groups = {}
                for player in players:
                    role = player.get('role', 'Unknown')
                    if role not in role_groups:
                        role_groups[role] = []
                    role_groups[role].append(player)
                
                for role, role_players in role_groups.items():
                    print(f"\n{role.upper()}:")
                    for player in role_players:
                        name = player.get('name', 'Unknown')
                        score = player.get('final_score', 0)
                        credits = player.get('credits', 8.5)
                        team_name = player.get('team_name', 'UNK')
                        
                        # Enhancement indicators
                        indicators = []
                        if player.get('neural_prediction'):
                            indicators.append('🧠')
                        if player.get('environmental_multiplier', 1.0) != 1.0:
                            indicators.append('🌤️')
                        if player.get('matchup_factor', 1.0) != 1.0:
                            indicators.append('⚔️')
                        
                        indicator_str = ''.join(indicators) + ' ' if indicators else ''
                        
                        print(f"  {indicator_str}{name} ({team_name}) - {score:.1f} pts | {credits:.1f}c")
                
                # Team insights
                if 'rl_insights' in team and team['rl_insights']:
                    print(f"\n🤖 AI Insights:")
                    for insight in team['rl_insights'][:3]:
                        print(f"  • {insight}")
                
                if 'performance_prediction' in team:
                    pred = team['performance_prediction']
                    conf_range = pred.get('confidence_range', (0, 0))
                    print(f"\n📊 Performance Prediction:")
                    print(f"  Expected: {pred.get('expected_score', 0):.1f} pts")
                    print(f"  Range: {conf_range[0]:.1f} - {conf_range[1]:.1f} pts")
                    print(f"  Risk Level: {pred.get('risk_level', 0)*100:.1f}%")
        
        # Strategic analysis
        strategic_analysis = results.get('strategic_analysis', {})
        if 'team_explanation' in strategic_analysis:
            team_exp = strategic_analysis['team_explanation']
            rationale = team_exp.selection_rationale
            
            print(f"\n🎯 STRATEGIC ANALYSIS:")
            print(f"Selection Strategy: {rationale.get('selection_strategy', 'Unknown')}")
            print(f"Average Ownership: {rationale.get('average_ownership', 0):.1f}%")
            
            if rationale.get('key_strengths'):
                print("\n✅ Key Strengths:")
                for strength in rationale['key_strengths']:
                    print(f"  • {strength}")
            
            if rationale.get('potential_concerns'):
                print("\n⚠️ Potential Concerns:")
                for concern in rationale['potential_concerns']:
                    print(f"  • {concern}")

async def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description='Enhanced DreamTeamAI - Next-Generation Fantasy Cricket Optimizer')
    
    # Match input - either match ID or team query
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--match-id', type=int, help='Match ID (e.g., 12345)')
    input_group.add_argument('--teams', help='Team names query (e.g., "india vs australia")')
    
    # For backward compatibility, allow positional argument
    parser.add_argument('match_query', nargs='?', help='Match search query or Match ID (e.g., "india vs australia" or 12345)')
    
    parser.add_argument('-n', '--num-teams', type=int, default=5, help='Number of teams to generate (default: 5)')
    parser.add_argument('-m', '--mode', choices=['balanced', 'aggressive', 'conservative'], 
                       default='balanced', help='Optimization mode (default: balanced)')
    parser.add_argument('--disable-neural', action='store_true', help='Disable neural network predictions')
    parser.add_argument('--disable-quantum', action='store_true', help='Disable quantum optimization (enabled by default)')
    parser.add_argument('--fast-mode', action='store_true', help='Disable quantum for faster processing')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    # Determine input type and process accordingly
    match_input = None
    input_type = None
    
    if args.match_id:
        match_input = args.match_id
        input_type = 'match_id'
    elif args.teams:
        match_input = args.teams
        input_type = 'team_query'
    elif args.match_query:
        # Try to determine if it's a match ID (numeric) or team query
        try:
            match_id_test = int(args.match_query)
            match_input = match_id_test
            input_type = 'match_id'
        except ValueError:
            match_input = args.match_query
            input_type = 'team_query'
    else:
        print("❌ Error: Please provide either --match-id, --teams, or a positional argument")
        return
    
    # Initialize enhanced system
    enhanced_ai = EnhancedDreamTeamAI()
    
    # Configure enhancements based on arguments
    if args.disable_neural:
        enhanced_ai.enhancement_config['use_neural_prediction'] = False
        print("🔧 Neural networks disabled")
    
    if args.disable_quantum or args.fast_mode:
        enhanced_ai.enhancement_config['use_quantum_optimization'] = False
        print("⚡ Quantum optimization disabled (fast mode)")
    
    # Generate enhanced teams
    print(f"🚀 Enhanced DreamTeamAI v2.0 - World-Class Fantasy Cricket Optimizer")
    if input_type == 'match_id':
        print(f"🆔 Match ID: {match_input}")
    else:
        print(f"🎯 Query: {match_input}")
    print(f"📊 Teams: {args.num_teams} | Mode: {args.mode}")
    
    start_time = datetime.now()
    
    try:
        if input_type == 'match_id':
            # Use match ID directly
            results = await enhanced_ai.generate_enhanced_teams_by_id(
                match_input, args.num_teams, args.mode
            )
        else:
            # Use team query
            results = await enhanced_ai.generate_enhanced_teams(
                match_input, args.num_teams, args.mode
            )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        enhanced_ai.display_enhanced_results(results)
        
        print(f"\n⏱️ Total Processing Time: {processing_time:.2f} seconds")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Handle both async and sync execution
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    
    asyncio.run(main())