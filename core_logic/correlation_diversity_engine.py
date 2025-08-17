#!/usr/bin/env python3
"""
Advanced Correlation-Based Diversity Engine
Uses statistical correlation analysis to create truly diverse team combinations
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback for environments without numpy
    class MockNumPy:
        def array(self, x): return x
        def corrcoef(self, x): return [[1.0 for _ in range(len(x))] for _ in range(len(x))]
        def mean(self, x): return sum(x) / len(x) if x else 0
        def std(self, x): return (sum((xi - sum(x)/len(x))**2 for xi in x) / len(x))**0.5 if x else 0
    np = MockNumPy()

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerCorrelationProfile:
    """Player correlation analysis profile"""
    player_name: str
    performance_vector: List[float] = field(default_factory=list)
    
    # Correlation metrics
    correlation_scores: Dict[str, float] = field(default_factory=dict)  # vs other players
    volatility_score: float = 0.0     # Individual performance variance
    consistency_score: float = 0.0    # Performance reliability
    
    # Diversity factors
    role_uniqueness: float = 0.0      # How unique this player's role is
    performance_uniqueness: float = 0.0  # How unique performance pattern is
    
    # Team synergy factors
    positive_correlations: List[str] = field(default_factory=list)  # Players who perform well together
    negative_correlations: List[str] = field(default_factory=list)  # Players who perform poorly together
    
    last_updated: datetime = field(default_factory=datetime.now)
    data_confidence: float = 0.0

@dataclass
class DiversityMatrix:
    """Matrix analysis for team diversity optimization"""
    players: List[str]
    correlation_matrix: List[List[float]] = field(default_factory=list)
    diversity_scores: Dict[str, float] = field(default_factory=dict)
    
    # Clustering analysis
    performance_clusters: Dict[str, List[str]] = field(default_factory=dict)
    role_clusters: Dict[str, List[str]] = field(default_factory=dict)
    
    # Optimization targets
    target_diversity_score: float = 0.7  # 0-1 scale
    max_correlation_threshold: float = 0.6  # Maximum acceptable correlation
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class CorrelationDiversityEngine:
    """Advanced diversity engine using correlation analysis"""
    
    def __init__(self, db_path: str = "correlation_diversity.db"):
        self.db_path = db_path
        self.correlation_cache = {}
        self.cache_expiry = timedelta(hours=12)
        self._init_database()
        
        # Performance simulation parameters
        self.min_performance_matches = 5  # Minimum matches needed for correlation analysis
        self.correlation_significance_threshold = 0.3  # Minimum meaningful correlation
        
    def _init_database(self):
        """Initialize correlation analysis database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Player correlation profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT NOT NULL,
                    performance_vector TEXT,  -- JSON array of recent performances
                    correlation_scores TEXT,  -- JSON dict of correlations with other players
                    volatility_score REAL DEFAULT 0.0,
                    consistency_score REAL DEFAULT 0.0,
                    role_uniqueness REAL DEFAULT 0.0,
                    performance_uniqueness REAL DEFAULT 0.0,
                    positive_correlations TEXT,  -- JSON array
                    negative_correlations TEXT,  -- JSON array
                    data_confidence REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Diversity analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diversity_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    team_combination TEXT,  -- JSON array of players
                    diversity_score REAL,
                    correlation_strength REAL,
                    predicted_variance REAL,
                    optimization_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Team performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    team_composition TEXT,  -- JSON array
                    actual_total_score REAL,
                    predicted_total_score REAL,
                    captain_performance REAL,
                    vc_performance REAL,
                    diversity_effectiveness REAL,
                    correlation_accuracy REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Correlation diversity engine database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing correlation database: {e}")
    
    def analyze_player_correlations(self, players: List[str], match_context: Dict = None) -> DiversityMatrix:
        """Analyze correlations between players for diversity optimization"""
        
        # Get or calculate correlation profiles for all players
        profiles = {}
        for player in players:
            profiles[player] = self._get_player_correlation_profile(player, match_context)
        
        # Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(players, profiles)
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(players, correlation_matrix, profiles)
        
        # Perform clustering analysis
        performance_clusters = self._cluster_by_performance(players, profiles)
        role_clusters = self._cluster_by_role(players, profiles)
        
        return DiversityMatrix(
            players=players,
            correlation_matrix=correlation_matrix,
            diversity_scores=diversity_scores,
            performance_clusters=performance_clusters,
            role_clusters=role_clusters
        )
    
    def optimize_team_diversity(self, players: List[str], team_size: int = 11, 
                              strategies: List[str] = None) -> List[Dict]:
        """Optimize team selection for maximum diversity across multiple strategies"""
        
        if len(players) < team_size:
            logger.warning(f"⚠️ Not enough players ({len(players)}) for team size ({team_size})")
            return self._generate_fallback_teams(players, strategies or ['default'])
        
        # Analyze correlations
        diversity_matrix = self.analyze_player_correlations(players)
        
        # Generate diverse teams using different optimization approaches
        diverse_teams = []
        
        strategy_methods = {
            'min_correlation': self._optimize_minimum_correlation,
            'max_diversity': self._optimize_maximum_diversity, 
            'balanced_risk': self._optimize_balanced_risk,
            'cluster_sampling': self._optimize_cluster_sampling,
            'performance_spread': self._optimize_performance_spread
        }
        
        strategies = strategies or list(strategy_methods.keys())
        
        for strategy in strategies:
            if strategy in strategy_methods:
                try:
                    team = strategy_methods[strategy](diversity_matrix, team_size)
                    if team:
                        diverse_teams.append({
                            'strategy': strategy,
                            'players': team['players'],
                            'diversity_score': team.get('diversity_score', 0.0),
                            'expected_variance': team.get('expected_variance', 0.0),
                            'correlation_strength': team.get('correlation_strength', 0.0)
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error in {strategy} optimization: {e}")
        
        # If no teams generated, create fallback
        if not diverse_teams:
            diverse_teams = self._generate_fallback_teams(players, strategies)
        
        return diverse_teams
    
    def _get_player_correlation_profile(self, player: str, match_context: Dict = None) -> PlayerCorrelationProfile:
        """Get or create correlation profile for a player"""
        
        # Check cache first
        cache_key = f"{player}_{match_context.get('match_id', 'general') if match_context else 'general'}"
        if cache_key in self.correlation_cache:
            cached_profile, cache_time = self.correlation_cache[cache_key]
            if datetime.now() - cache_time < self.cache_expiry:
                return cached_profile
        
        # Try to get from database
        profile = self._get_profile_from_database(player)
        
        # If not found or stale, generate new profile
        if not profile or self._is_profile_stale(profile):
            profile = self._generate_correlation_profile(player, match_context)
            self._save_profile_to_database(profile)
        
        # Cache result
        self.correlation_cache[cache_key] = (profile, datetime.now())
        
        return profile
    
    def _generate_correlation_profile(self, player: str, match_context: Dict = None) -> PlayerCorrelationProfile:
        """Generate correlation profile for a player"""
        
        # Simulate performance vector (in real implementation, this would come from historical data)
        performance_vector = self._simulate_player_performance_vector(player)
        
        # Calculate volatility and consistency
        volatility = self._calculate_volatility(performance_vector)
        consistency = self._calculate_consistency(performance_vector)
        
        # Calculate role and performance uniqueness
        role_uniqueness = self._calculate_role_uniqueness(player)
        performance_uniqueness = self._calculate_performance_uniqueness(performance_vector)
        
        return PlayerCorrelationProfile(
            player_name=player,
            performance_vector=performance_vector,
            volatility_score=volatility,
            consistency_score=consistency,
            role_uniqueness=role_uniqueness,
            performance_uniqueness=performance_uniqueness,
            data_confidence=0.7 if len(performance_vector) >= self.min_performance_matches else 0.3
        )
    
    def _simulate_player_performance_vector(self, player: str) -> List[float]:
        """Simulate historical performance vector for a player"""
        # In real implementation, this would fetch actual historical data
        # For now, create realistic performance patterns based on player characteristics
        
        base_performance = 30 + (hash(player) % 40)  # 30-70 base range
        num_matches = 10
        
        performances = []
        for i in range(num_matches):
            # Add some randomness and trends
            trend_factor = 1 + (i * 0.02)  # Slight upward trend
            random_factor = 1 + ((hash(f"{player}{i}") % 100 - 50) / 100)  # -50% to +50% variation
            
            performance = base_performance * trend_factor * random_factor
            performances.append(max(0, min(150, performance)))  # Cap between 0-150
        
        return performances
    
    def _build_correlation_matrix(self, players: List[str], profiles: Dict[str, PlayerCorrelationProfile]) -> List[List[float]]:
        """Build correlation matrix between all players"""
        n = len(players)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if i == j:
                    matrix[i][j] = 1.0  # Perfect correlation with self
                else:
                    correlation = self._calculate_correlation(
                        profiles[player1].performance_vector,
                        profiles[player2].performance_vector
                    )
                    matrix[i][j] = correlation
                    
                    # Update profiles with correlation data
                    if abs(correlation) > self.correlation_significance_threshold:
                        if correlation > 0:
                            profiles[player1].positive_correlations.append(player2)
                        else:
                            profiles[player1].negative_correlations.append(player2)
        
        return matrix
    
    def _calculate_correlation(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two performance vectors"""
        if not vector1 or not vector2 or len(vector1) != len(vector2):
            return 0.0
        
        if NUMPY_AVAILABLE:
            try:
                correlation_matrix = np.corrcoef(vector1, vector2)
                return float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
            except:
                pass
        
        # Fallback manual calculation
        n = len(vector1)
        if n < 2:
            return 0.0
        
        mean1 = sum(vector1) / n
        mean2 = sum(vector2) / n
        
        numerator = sum((vector1[i] - mean1) * (vector2[i] - mean2) for i in range(n))
        
        sum_sq1 = sum((vector1[i] - mean1) ** 2 for i in range(n))
        sum_sq2 = sum((vector2[i] - mean2) ** 2 for i in range(n))
        
        denominator = (sum_sq1 * sum_sq2) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_diversity_scores(self, players: List[str], correlation_matrix: List[List[float]], 
                                  profiles: Dict[str, PlayerCorrelationProfile]) -> Dict[str, float]:
        """Calculate diversity contribution score for each player"""
        diversity_scores = {}
        
        for i, player in enumerate(players):
            # Calculate average correlation with other players
            correlations = [abs(correlation_matrix[i][j]) for j in range(len(players)) if i != j]
            avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
            
            # Diversity score is inverse of correlation (lower correlation = higher diversity)
            base_diversity = 1.0 - avg_correlation
            
            # Boost for role uniqueness and performance uniqueness
            role_boost = profiles[player].role_uniqueness * 0.2
            performance_boost = profiles[player].performance_uniqueness * 0.2
            
            # Penalize for high volatility (unreliable players)
            volatility_penalty = profiles[player].volatility_score * 0.1
            
            diversity_scores[player] = max(0.0, base_diversity + role_boost + performance_boost - volatility_penalty)
        
        return diversity_scores
    
    def _optimize_minimum_correlation(self, diversity_matrix: DiversityMatrix, team_size: int) -> Dict:
        """Optimize team to minimize overall correlations"""
        players = diversity_matrix.players
        correlation_matrix = diversity_matrix.correlation_matrix
        
        if len(players) <= team_size:
            return {
                'players': players,
                'diversity_score': 1.0,
                'correlation_strength': 0.0
            }
        
        # Greedy algorithm to select players with lowest correlations
        selected_indices = []
        remaining_indices = list(range(len(players)))
        
        # Start with the player with highest diversity score
        start_idx = max(remaining_indices, key=lambda i: diversity_matrix.diversity_scores[players[i]])
        selected_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        # Iteratively add players with lowest correlation to selected team
        while len(selected_indices) < team_size and remaining_indices:
            best_candidate = None
            best_score = float('inf')
            
            for candidate_idx in remaining_indices:
                # Calculate average correlation with already selected players
                correlations = [abs(correlation_matrix[candidate_idx][selected_idx]) 
                              for selected_idx in selected_indices]
                avg_correlation = sum(correlations) / len(correlations)
                
                if avg_correlation < best_score:
                    best_score = avg_correlation
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
            else:
                break
        
        # Calculate final metrics
        selected_players = [players[i] for i in selected_indices]
        final_correlation = self._calculate_team_correlation_strength(selected_indices, correlation_matrix)
        diversity_score = 1.0 - final_correlation
        
        return {
            'players': selected_players,
            'diversity_score': diversity_score,
            'correlation_strength': final_correlation,
            'method': 'minimum_correlation'
        }
    
    def _optimize_maximum_diversity(self, diversity_matrix: DiversityMatrix, team_size: int) -> Dict:
        """Optimize team for maximum diversity scores"""
        players = diversity_matrix.players
        diversity_scores = diversity_matrix.diversity_scores
        
        # Sort players by diversity score and select top performers
        sorted_players = sorted(players, key=lambda p: diversity_scores[p], reverse=True)
        selected_players = sorted_players[:team_size]
        
        # Calculate metrics
        avg_diversity = sum(diversity_scores[p] for p in selected_players) / len(selected_players)
        
        return {
            'players': selected_players,
            'diversity_score': avg_diversity,
            'method': 'maximum_diversity'
        }
    
    def _optimize_balanced_risk(self, diversity_matrix: DiversityMatrix, team_size: int) -> Dict:
        """Optimize for balanced risk-reward profile"""
        players = diversity_matrix.players
        
        # Combine diversity score with consistency (risk management)
        risk_adjusted_scores = {}
        
        for player in players:
            diversity_score = diversity_matrix.diversity_scores.get(player, 0.5)
            # Risk adjustment would use actual volatility data
            risk_adjustment = 0.8  # Placeholder
            risk_adjusted_scores[player] = diversity_score * risk_adjustment
        
        # Select top risk-adjusted players
        selected_players = sorted(players, key=lambda p: risk_adjusted_scores[p], reverse=True)[:team_size]
        
        avg_score = sum(risk_adjusted_scores[p] for p in selected_players) / len(selected_players)
        
        return {
            'players': selected_players,
            'diversity_score': avg_score,
            'method': 'balanced_risk'
        }
    
    def _optimize_cluster_sampling(self, diversity_matrix: DiversityMatrix, team_size: int) -> Dict:
        """Sample players from different performance clusters"""
        clusters = diversity_matrix.performance_clusters
        
        if not clusters:
            # Fallback to random sampling
            selected_players = random.sample(diversity_matrix.players, min(team_size, len(diversity_matrix.players)))
            return {
                'players': selected_players,
                'diversity_score': 0.5,
                'method': 'cluster_sampling_fallback'
            }
        
        # Sample from each cluster proportionally
        selected_players = []
        cluster_list = list(clusters.values())
        players_per_cluster = max(1, team_size // len(cluster_list))
        
        for cluster in cluster_list:
            if len(selected_players) >= team_size:
                break
            
            available_players = [p for p in cluster if p not in selected_players]
            sample_size = min(players_per_cluster, len(available_players), team_size - len(selected_players))
            
            if sample_size > 0:
                sampled = random.sample(available_players, sample_size)
                selected_players.extend(sampled)
        
        # Fill remaining slots if needed
        while len(selected_players) < team_size:
            remaining_players = [p for p in diversity_matrix.players if p not in selected_players]
            if not remaining_players:
                break
            selected_players.append(random.choice(remaining_players))
        
        return {
            'players': selected_players[:team_size],
            'diversity_score': 0.7,  # Estimated
            'method': 'cluster_sampling'
        }
    
    def _optimize_performance_spread(self, diversity_matrix: DiversityMatrix, team_size: int) -> Dict:
        """Optimize for spread across performance ranges"""
        players = diversity_matrix.players
        
        # This would use actual performance data in real implementation
        # For now, create a spread based on player names (deterministic but varied)
        performance_estimates = {player: 30 + (hash(player) % 40) for player in players}
        
        # Sort by performance and select across the range
        sorted_by_performance = sorted(players, key=lambda p: performance_estimates[p])
        
        # Select players across performance spectrum
        selected_players = []
        step = max(1, len(sorted_by_performance) // team_size)
        
        for i in range(0, len(sorted_by_performance), step):
            if len(selected_players) < team_size:
                selected_players.append(sorted_by_performance[i])
        
        # Fill remaining slots with best remaining players
        while len(selected_players) < team_size:
            remaining = [p for p in sorted_by_performance if p not in selected_players]
            if not remaining:
                break
            selected_players.append(remaining[0])
        
        return {
            'players': selected_players[:team_size],
            'diversity_score': 0.6,  # Estimated
            'method': 'performance_spread'
        }
    
    def _calculate_team_correlation_strength(self, player_indices: List[int], correlation_matrix: List[List[float]]) -> float:
        """Calculate average correlation strength for a team"""
        if len(player_indices) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(player_indices)):
            for j in range(i + 1, len(player_indices)):
                idx1, idx2 = player_indices[i], player_indices[j]
                correlations.append(abs(correlation_matrix[idx1][idx2]))
        
        return sum(correlations) / len(correlations) if correlations else 0.0
    
    def _cluster_by_performance(self, players: List[str], profiles: Dict[str, PlayerCorrelationProfile]) -> Dict[str, List[str]]:
        """Cluster players by performance characteristics"""
        # Simple clustering based on performance vector averages
        clusters = {'high': [], 'medium': [], 'low': []}
        
        performance_avgs = {}
        for player in players:
            vector = profiles[player].performance_vector
            avg_perf = sum(vector) / len(vector) if vector else 50
            performance_avgs[player] = avg_perf
        
        # Define thresholds
        all_avgs = list(performance_avgs.values())
        if all_avgs:
            low_threshold = np.mean(all_avgs) - 0.5 * np.std(all_avgs) if NUMPY_AVAILABLE else sum(all_avgs) / len(all_avgs) - 10
            high_threshold = np.mean(all_avgs) + 0.5 * np.std(all_avgs) if NUMPY_AVAILABLE else sum(all_avgs) / len(all_avgs) + 10
            
            for player, avg_perf in performance_avgs.items():
                if avg_perf >= high_threshold:
                    clusters['high'].append(player)
                elif avg_perf <= low_threshold:
                    clusters['low'].append(player)
                else:
                    clusters['medium'].append(player)
        
        return clusters
    
    def _cluster_by_role(self, players: List[str], profiles: Dict[str, PlayerCorrelationProfile]) -> Dict[str, List[str]]:
        """Cluster players by role characteristics"""
        # Simple role-based clustering
        clusters = defaultdict(list)
        
        for player in players:
            # In real implementation, would use actual role data
            # For now, use role uniqueness as a proxy
            role_uniqueness = profiles[player].role_uniqueness
            
            if role_uniqueness > 0.7:
                clusters['unique'].append(player)
            elif role_uniqueness > 0.4:
                clusters['specialized'].append(player)
            else:
                clusters['common'].append(player)
        
        return dict(clusters)
    
    def _calculate_volatility(self, performance_vector: List[float]) -> float:
        """Calculate performance volatility (normalized standard deviation)"""
        if len(performance_vector) < 2:
            return 0.0
        
        mean_perf = sum(performance_vector) / len(performance_vector)
        variance = sum((x - mean_perf) ** 2 for x in performance_vector) / len(performance_vector)
        std_dev = variance ** 0.5
        
        # Normalize by mean to get coefficient of variation
        return std_dev / mean_perf if mean_perf > 0 else 0.0
    
    def _calculate_consistency(self, performance_vector: List[float]) -> float:
        """Calculate performance consistency (inverse of volatility)"""
        volatility = self._calculate_volatility(performance_vector)
        return max(0.0, 1.0 - volatility)
    
    def _calculate_role_uniqueness(self, player: str) -> float:
        """Calculate how unique a player's role is"""
        # Placeholder implementation - would use actual role data
        role_hash = hash(player) % 100
        return min(1.0, role_hash / 100.0)
    
    def _calculate_performance_uniqueness(self, performance_vector: List[float]) -> float:
        """Calculate how unique a player's performance pattern is"""
        if not performance_vector:
            return 0.5
        
        # Simple measure based on performance variance
        volatility = self._calculate_volatility(performance_vector)
        return min(1.0, volatility * 2)  # Higher volatility = more unique pattern
    
    def _generate_fallback_teams(self, players: List[str], strategies: List[str]) -> List[Dict]:
        """Generate fallback teams when optimization fails"""
        fallback_teams = []
        
        for i, strategy in enumerate(strategies):
            # Create different combinations by rotating the player list
            rotated_players = players[i:] + players[:i]
            team_players = rotated_players[:min(11, len(players))]
            
            fallback_teams.append({
                'strategy': f'{strategy}_fallback',
                'players': team_players,
                'diversity_score': 0.5,
                'correlation_strength': 0.3
            })
        
        return fallback_teams
    
    def _get_profile_from_database(self, player: str) -> Optional[PlayerCorrelationProfile]:
        """Get player correlation profile from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM player_correlations 
                WHERE player_name = ? 
                ORDER BY last_updated DESC 
                LIMIT 1
            ''', (player,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return PlayerCorrelationProfile(
                    player_name=row[1],
                    performance_vector=json.loads(row[2]) if row[2] else [],
                    correlation_scores=json.loads(row[3]) if row[3] else {},
                    volatility_score=row[4],
                    consistency_score=row[5],
                    role_uniqueness=row[6],
                    performance_uniqueness=row[7],
                    positive_correlations=json.loads(row[8]) if row[8] else [],
                    negative_correlations=json.loads(row[9]) if row[9] else [],
                    data_confidence=row[10],
                    last_updated=datetime.fromisoformat(row[11])
                )
        except Exception as e:
            logger.warning(f"⚠️ Error getting correlation profile from database: {e}")
        
        return None
    
    def _is_profile_stale(self, profile: PlayerCorrelationProfile) -> bool:
        """Check if correlation profile is stale"""
        age = datetime.now() - profile.last_updated
        return age > timedelta(days=3)  # Refresh every 3 days
    
    def _save_profile_to_database(self, profile: PlayerCorrelationProfile):
        """Save correlation profile to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO player_correlations 
                (player_name, performance_vector, correlation_scores, volatility_score,
                 consistency_score, role_uniqueness, performance_uniqueness,
                 positive_correlations, negative_correlations, data_confidence, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.player_name,
                json.dumps(profile.performance_vector),
                json.dumps(profile.correlation_scores),
                profile.volatility_score,
                profile.consistency_score,
                profile.role_uniqueness,
                profile.performance_uniqueness,
                json.dumps(profile.positive_correlations),
                json.dumps(profile.negative_correlations),
                profile.data_confidence,
                profile.last_updated.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Error saving correlation profile: {e}")

# Global instance
_correlation_engine = None

def get_correlation_diversity_engine() -> CorrelationDiversityEngine:
    """Get global correlation diversity engine instance"""
    global _correlation_engine
    if _correlation_engine is None:
        _correlation_engine = CorrelationDiversityEngine()
    return _correlation_engine

def optimize_team_diversity(players: List[str], team_size: int = 11, strategies: List[str] = None) -> List[Dict]:
    """Main interface for team diversity optimization"""
    engine = get_correlation_diversity_engine()
    return engine.optimize_team_diversity(players, team_size, strategies)