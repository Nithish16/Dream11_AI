import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from core_logic.feature_engine import PlayerFeatures

# Try to import OR-Tools, fallback to simple optimization if not available  
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    print("⚠️  OR-Tools not available, using simplified optimization")
    ORTOOLS_AVAILABLE = False

@dataclass
class PlayerForOptimization:
    """Player data structure optimized for team selection"""
    player_id: int
    name: str
    role: str
    team: str
    credits: float
    final_score: float
    
    # Feature components for risk analysis
    consistency_score: float = 0.0
    opportunity_index: float = 1.0
    ema_score: float = 0.0
    form_momentum: float = 0.0
    
    # Selection metadata
    is_captain_candidate: bool = False
    is_vice_captain_candidate: bool = False
    selection_priority: float = 0.0

@dataclass
class OptimalTeam:
    """Generated optimal team structure"""
    team_id: int
    players: List[PlayerForOptimization]
    captain: PlayerForOptimization = None
    vice_captain: PlayerForOptimization = None
    
    total_credits: float = 0.0
    total_score: float = 0.0
    risk_level: str = "Balanced"
    pack_type: str = "Pack-1"  # Pack-1 or Pack-2
    strategy: str = "Optimal"  # Optimal, Risk-Adjusted, Form-Based, Value-Picks
    
    # Team composition
    batsmen: List[PlayerForOptimization] = field(default_factory=list)
    bowlers: List[PlayerForOptimization] = field(default_factory=list)
    all_rounders: List[PlayerForOptimization] = field(default_factory=list)
    wicket_keepers: List[PlayerForOptimization] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        if self.players:
            self.total_credits = sum(p.credits for p in self.players)
            self.total_score = sum(p.final_score for p in self.players)
            
            # Add captain/VC bonuses
            if self.captain:
                self.total_score += self.captain.final_score  # 2x for captain
            if self.vice_captain:
                self.total_score += self.vice_captain.final_score * 0.5  # 1.5x for VC
            
            # Categorize players by role
            self._categorize_players()
    
    def _categorize_players(self):
        """Categorize players by their roles"""
        for player in self.players:
            role = player.role.lower()
            if 'wk' in role or 'wicket' in role or 'keeper' in role:
                self.wicket_keepers.append(player)
            elif 'allrounder' in role or 'all-rounder' in role:
                self.all_rounders.append(player)
            elif 'bowl' in role:
                self.bowlers.append(player)
            else:  # Default to batsman
                self.batsmen.append(player)

# Model A: Expert-Weighted Formula
def calculate_expert_weighted_score(player_features: PlayerFeatures, 
                                  match_format: str = "T20") -> float:
    """
    Calculate expert-weighted score based on cricket domain knowledge
    
    Args:
        player_features: Player's calculated features
        match_format: Match format (T20, ODI, Test)
    
    Returns:
        float: Expert-weighted score
    """
    format_lower = match_format.lower()
    
    # Base format-specific weights
    if format_lower == "t20":
        weights = {
            'ema_score': 0.25,
            'consistency': 0.15,
            'opportunity_index': 0.30,
            'form_momentum': 0.20,
            'role_bonus': 0.10
        }
    elif format_lower == "odi":
        weights = {
            'ema_score': 0.30,
            'consistency': 0.25,
            'opportunity_index': 0.20,
            'form_momentum': 0.15,
            'role_bonus': 0.10
        }
    else:  # Test cricket
        weights = {
            'ema_score': 0.20,
            'consistency': 0.35,
            'opportunity_index': 0.25,
            'form_momentum': 0.10,
            'role_bonus': 0.10
        }
    
    # Calculate weighted components
    ema_component = player_features.ema_score * weights['ema_score']
    consistency_component = player_features.consistency_score * weights['consistency']
    opportunity_component = player_features.dynamic_opportunity_index * 20 * weights['opportunity_index']
    momentum_component = (player_features.form_momentum + 1) * 10 * weights['form_momentum']
    
    # Role-specific bonus
    role_bonus = 0
    role = player_features.role.lower()
    if format_lower == "t20":
        if 'allrounder' in role:
            role_bonus = 5
        elif 'wk' in role:
            role_bonus = 3
    elif format_lower == "odi":
        if 'allrounder' in role:
            role_bonus = 4
        elif 'bat' in role:
            role_bonus = 2
    else:  # Test
        if 'bowl' in role:
            role_bonus = 3
        elif 'allrounder' in role:
            role_bonus = 4
    
    role_component = role_bonus * weights['role_bonus']
    
    expert_score = (ema_component + consistency_component + 
                   opportunity_component + momentum_component + role_component)
    
    return round(expert_score, 2)

# Model B: Machine Learning Predictor (Placeholder)
def calculate_ml_prediction_score(player_features: PlayerFeatures, 
                                match_context: Dict[str, Any] = None) -> float:
    """
    Machine Learning based prediction (Placeholder implementation)
    
    Args:
        player_features: Player's calculated features
        match_context: Match context for ML model
    
    Returns:
        float: ML predicted score
    """
    # Placeholder ML model - in reality would use trained models
    # like RandomForest, XGBoost, or Neural Networks
    
    if match_context is None:
        match_context = {}
    
    # Simulate ML prediction based on multiple factors
    base_prediction = player_features.performance_rating
    
    # Add some sophisticated "ML-like" adjustments
    feature_vector = [
        player_features.ema_score,
        player_features.consistency_score / 100,
        player_features.dynamic_opportunity_index,
        player_features.form_momentum,
        player_features.matchup_score
    ]
    
    # Simulate feature importance weights (would be learned from data)
    feature_weights = [0.3, 0.2, 0.25, 0.15, 0.1]
    
    ml_adjustment = sum(f * w for f, w in zip(feature_vector, feature_weights)) * 5
    
    # Add some "model uncertainty" 
    random.seed(player_features.player_id)  # Consistent randomness
    uncertainty = random.uniform(-2, 2)
    
    ml_score = base_prediction + ml_adjustment + uncertainty
    
    return round(max(0, ml_score), 2)

def get_final_player_score(player_features: PlayerFeatures, 
                          match_format: str = "T20",
                          match_context: Dict[str, Any] = None,
                          model_weights: Dict[str, float] = None) -> float:
    """
    Calculate final ensemble score combining multiple models
    
    Args:
        player_features: Player's calculated features
        match_format: Match format for context
        match_context: Additional match context
        model_weights: Custom weights for ensemble models
    
    Returns:
        float: Final ensemble score
    """
    if match_context is None:
        match_context = {}
    
    if model_weights is None:
        model_weights = {'expert': 0.7, 'ml': 0.3}
    
    # Calculate scores from both models
    expert_score = calculate_expert_weighted_score(player_features, match_format)
    ml_score = calculate_ml_prediction_score(player_features, match_context)
    
    # Ensemble combination
    final_score = (expert_score * model_weights['expert'] + 
                  ml_score * model_weights['ml'])
    
    # Apply matchup multiplier
    final_score *= player_features.matchup_score
    
    return round(final_score, 2)

def assign_player_credits(player_features: PlayerFeatures) -> float:
    """
    Assign uniform credits - prediction based on data analysis only
    
    Args:
        player_features: Player's calculated features
    
    Returns:
        float: Uniform credits (8.5 for all players)
    """
    # All players get same credits - prediction purely based on data analysis
    return 8.5

def prepare_players_for_optimization(player_features_list: List[PlayerFeatures],
                                   match_format: str = "T20",
                                   match_context: Dict[str, Any] = None,
                                   team_mapping: Dict[int, str] = None) -> List[PlayerForOptimization]:
    """
    Convert player features to optimization-ready format
    """
    if team_mapping is None:
        team_mapping = {}
    
    players_for_opt = []
    
    for features in player_features_list:
        final_score = get_final_player_score(features, match_format, match_context)
        credits = assign_player_credits(features)
        
        # Determine team name
        team_name = team_mapping.get(features.player_id, "Team A" if len(players_for_opt) < 16 else "Team B")
        
        player_opt = PlayerForOptimization(
            player_id=features.player_id,
            name=features.player_name,
            role=features.role,
            team=team_name,
            credits=credits,
            final_score=final_score,
            consistency_score=features.consistency_score,
            opportunity_index=features.dynamic_opportunity_index,
            ema_score=features.ema_score,
            form_momentum=features.form_momentum,
            is_captain_candidate=features.captain_vice_captain_probability > 30,
            is_vice_captain_candidate=features.captain_vice_captain_probability > 20
        )
        
        players_for_opt.append(player_opt)
    
    return players_for_opt

def apply_risk_profile_adjustments(players: List[PlayerForOptimization], 
                                 risk_profile: str = "Balanced") -> List[PlayerForOptimization]:
    """
    Adjust player scores based on risk profile
    
    Args:
        players: List of players for optimization
        risk_profile: 'Safe', 'Balanced', or 'High-Risk'
    
    Returns:
        List[PlayerForOptimization]: Players with adjusted scores
    """
    adjusted_players = []
    
    for player in players:
        adjusted_player = PlayerForOptimization(
            player_id=player.player_id,
            name=player.name,
            role=player.role,
            team=player.team,
            credits=player.credits,
            final_score=player.final_score,
            consistency_score=player.consistency_score,
            opportunity_index=player.opportunity_index,
            ema_score=player.ema_score,
            form_momentum=player.form_momentum,
            is_captain_candidate=player.is_captain_candidate,
            is_vice_captain_candidate=player.is_vice_captain_candidate
        )
        
        if risk_profile.lower() == "safe":
            # Safe: Maximize consistency, minimize risk
            consistency_multiplier = 1.0 + (player.consistency_score / 200)  # Up to 1.5x
            adjusted_player.final_score = player.final_score * consistency_multiplier
            
        elif risk_profile.lower() == "high-risk":
            # High-Risk: Extra weight to opportunity index and form
            opportunity_boost = (player.opportunity_index - 1) * 10
            form_boost = player.form_momentum * 5
            adjusted_player.final_score = player.final_score + opportunity_boost + form_boost
            
        # Balanced: No adjustments needed
        
        adjusted_players.append(adjusted_player)
    
    return adjusted_players

def generate_optimal_teams_ortools(player_list_with_scores: List[PlayerForOptimization],
                                 num_teams: int = 3,
                                 risk_profile: str = 'Balanced',
                                 max_credits: float = 100.0) -> List[OptimalTeam]:
    """
    Generate optimal teams using OR-Tools optimization
    """
    # Apply risk profile adjustments
    adjusted_players = apply_risk_profile_adjustments(player_list_with_scores, risk_profile)
    
    # Categorize players by role for constraints
    batsmen = [p for p in adjusted_players if is_batsman(p.role)]
    bowlers = [p for p in adjusted_players if is_bowler(p.role)]
    all_rounders = [p for p in adjusted_players if is_all_rounder(p.role)]
    wicket_keepers = [p for p in adjusted_players if is_wicket_keeper(p.role)]
    
    optimal_teams = []
    used_players = set()  # Track players used across teams for diversity
    
    for team_num in range(num_teams):
        print(f"🔧 Generating Team {team_num + 1}...")
        
        # Create OR-Tools solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("❌ OR-Tools solver not available")
            continue
        
        # Create binary variables for each player
        player_vars = {}
        for i, player in enumerate(adjusted_players):
            player_vars[player.player_id] = solver.IntVar(0, 1, f'player_{player.player_id}')
        
        # Objective: Maximize total score with diversity bonus
        objective = solver.Objective()
        for player in adjusted_players:
            score_multiplier = 1.0
            # Diversity bonus for unused players
            if player.player_id not in used_players:
                score_multiplier = 1.1
            
            objective.SetCoefficient(player_vars[player.player_id], 
                                   player.final_score * score_multiplier)
        objective.SetMaximization()
        
        # Constraint 1: Exactly 11 players
        total_players = solver.Constraint(11, 11)
        for player in adjusted_players:
            total_players.SetCoefficient(player_vars[player.player_id], 1)
        
        # Constraint 2: Credit limit
        credit_constraint = solver.Constraint(0, max_credits)
        for player in adjusted_players:
            credit_constraint.SetCoefficient(player_vars[player.player_id], player.credits)
        
        # Constraint 3: Role-based constraints
        # Batsmen: 3-6
        bat_constraint = solver.Constraint(3, 6)
        for player in batsmen:
            bat_constraint.SetCoefficient(player_vars[player.player_id], 1)
        
        # Bowlers: 3-6  
        bowl_constraint = solver.Constraint(3, 6)
        for player in bowlers:
            bowl_constraint.SetCoefficient(player_vars[player.player_id], 1)
        
        # All-rounders: 1-4
        ar_constraint = solver.Constraint(1, 4)
        for player in all_rounders:
            ar_constraint.SetCoefficient(player_vars[player.player_id], 1)
        
        # Wicket-keepers: 1-2
        wk_constraint = solver.Constraint(1, 2)
        for player in wicket_keepers:
            wk_constraint.SetCoefficient(player_vars[player.player_id], 1)
        
        # Solve the optimization problem
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            # Extract selected players
            selected_players = []
            for player in adjusted_players:
                if player_vars[player.player_id].solution_value() > 0.5:
                    selected_players.append(player)
                    used_players.add(player.player_id)  # Mark as used
            
            # Select captain and vice-captain
            captain, vice_captain = select_captain_vice_captain(selected_players)
            
            # Create optimal team
            team = OptimalTeam(
                team_id=team_num + 1,
                players=selected_players,
                captain=captain,
                vice_captain=vice_captain,
                risk_level=risk_profile
            )
            
            optimal_teams.append(team)
            
            print(f"  ✅ Team {team_num + 1} generated: {len(selected_players)} players")
        else:
            print(f"  ❌ Failed to generate Team {team_num + 1}")
    
    return optimal_teams

def generate_optimal_teams_simple(player_list_with_scores: List[PlayerForOptimization],
                                num_teams: int = 3,
                                risk_profile: str = 'Balanced',
                                max_credits: float = 100.0) -> List[OptimalTeam]:
    """
    Generate optimal teams using simplified greedy optimization (fallback) with improved error handling
    """
    # Input validation
    if not player_list_with_scores:
        print("❌ No players provided for team optimization")
        return []
    
    if len(player_list_with_scores) < 11:
        print(f"❌ Insufficient players ({len(player_list_with_scores)}) for team generation")
        return []
    
    try:
        # Apply risk profile adjustments
        adjusted_players = apply_risk_profile_adjustments(player_list_with_scores, risk_profile)
    except Exception as e:
        print(f"❌ Error applying risk profile adjustments: {e}")
        adjusted_players = player_list_with_scores
    
    # Sort players by final score (descending)
    sorted_players = sorted(adjusted_players, key=lambda x: x.final_score, reverse=True)
    
    # Categorize players by role
    batsmen = [p for p in sorted_players if is_batsman(p.role)]
    bowlers = [p for p in sorted_players if is_bowler(p.role)]
    all_rounders = [p for p in sorted_players if is_all_rounder(p.role)]
    wicket_keepers = [p for p in sorted_players if is_wicket_keeper(p.role)]
    
    optimal_teams = []
    used_players = set()
    
    for team_num in range(num_teams):
        print(f"🔧 Generating Team {team_num + 1} (Simplified)...")
        
        selected_players = []
        current_credits = 0.0
        
        # Greedy selection with more lenient role constraints for production
        role_counts = {'bat': 0, 'bowl': 0, 'ar': 0, 'wk': 0}
        role_limits = {'bat': (2, 7), 'bowl': (2, 7), 'ar': (0, 5), 'wk': (1, 3)}  # More flexible
        
        # Helper function to check if we can add a player
        def can_add_player(player, role_key):
            if player.player_id in used_players:
                return False
            # More lenient credit constraint for production
            if current_credits + player.credits > max_credits + 10:  # Allow 10 credit buffer
                return False
            if len(selected_players) >= 11:
                return False
            min_req, max_req = role_limits[role_key]
            if role_counts[role_key] >= max_req:
                return False
            return True
        
        # First, ensure minimum requirements
        for role_key, (min_req, _), player_list in [
            ('wk', role_limits['wk'], wicket_keepers),
            ('ar', role_limits['ar'], all_rounders),
            ('bat', role_limits['bat'], batsmen),
            ('bowl', role_limits['bowl'], bowlers)
        ]:
            for player in player_list:
                if role_counts[role_key] < min_req and can_add_player(player, role_key):
                    selected_players.append(player)
                    current_credits += player.credits
                    role_counts[role_key] += 1
                    used_players.add(player.player_id)
        
        # Fill remaining slots with best available players
        remaining_players = [p for p in sorted_players if p.player_id not in used_players]
        
        for player in remaining_players:
            if len(selected_players) >= 11:
                break
                
            role_key = 'bat'
            if is_bowler(player.role):
                role_key = 'bowl'
            elif is_all_rounder(player.role):
                role_key = 'ar'
            elif is_wicket_keeper(player.role):
                role_key = 'wk'
            
            if can_add_player(player, role_key):
                selected_players.append(player)
                current_credits += player.credits
                role_counts[role_key] += 1
                used_players.add(player.player_id)
        
        if len(selected_players) == 11:
            # Select captain and vice-captain
            captain, vice_captain = select_captain_vice_captain(selected_players)
            
            # Create optimal team
            team = OptimalTeam(
                team_id=team_num + 1,
                players=selected_players,
                captain=captain,
                vice_captain=vice_captain,
                risk_level=risk_profile
            )
            
            optimal_teams.append(team)
            
            print(f"  ✅ Team {team_num + 1} generated: {len(selected_players)} players")
        else:
            print(f"  ❌ Failed to generate complete Team {team_num + 1} ({len(selected_players)} players)")
    
    return optimal_teams

def generate_optimal_teams(player_list_with_scores: List[PlayerForOptimization],
                         num_teams: int = 3,
                         risk_profile: str = 'Balanced',
                         max_credits: float = 100.0) -> List[OptimalTeam]:
    """
    Generate optimal teams using best available method
    """
    if ORTOOLS_AVAILABLE:
        return generate_optimal_teams_ortools(player_list_with_scores, num_teams, risk_profile, max_credits)
    else:
        return generate_optimal_teams_simple(player_list_with_scores, num_teams, risk_profile, max_credits)

def select_captain_vice_captain(players: List[PlayerForOptimization]) -> Tuple[Optional[PlayerForOptimization], Optional[PlayerForOptimization]]:
    """
    Select captain and vice-captain using dynamic logic
    
    Args:
        players: List of selected players
    
    Returns:
        Tuple of (captain, vice_captain)
    """
    # Sort by captain candidacy and final score
    captain_candidates = [p for p in players if p.is_captain_candidate]
    if not captain_candidates:
        captain_candidates = sorted(players, key=lambda x: x.final_score, reverse=True)[:5]
    
    # Captain: Highest scoring candidate
    captain = max(captain_candidates, key=lambda x: x.final_score)
    
    # Vice-captain: Second highest, but different role if possible
    vc_candidates = [p for p in players if p != captain and p.is_vice_captain_candidate]
    if not vc_candidates:
        vc_candidates = [p for p in players if p != captain]
    
    # Prefer different role for diversity
    different_role_candidates = [p for p in vc_candidates if p.role != captain.role]
    if different_role_candidates:
        vice_captain = max(different_role_candidates, key=lambda x: x.final_score)
    else:
        vice_captain = max(vc_candidates, key=lambda x: x.final_score) if vc_candidates else None
    
    return captain, vice_captain

# Utility functions for role checking
def is_batsman(role: str) -> bool:
    """Check if player is a batsman"""
    role_lower = role.lower()
    return 'bat' in role_lower and 'allrounder' not in role_lower and 'wk' not in role_lower

def is_bowler(role: str) -> bool:
    """Check if player is a bowler"""  
    role_lower = role.lower()
    return 'bowl' in role_lower and 'allrounder' not in role_lower

def is_all_rounder(role: str) -> bool:
    """Check if player is an all-rounder"""
    role_lower = role.lower()
    return 'allrounder' in role_lower or 'all-rounder' in role_lower

def is_wicket_keeper(role: str) -> bool:
    """Check if player is a wicket-keeper"""
    role_lower = role.lower()
    return 'wk' in role_lower or 'wicket' in role_lower or 'keeper' in role_lower

def print_team_summary(team: OptimalTeam) -> None:
    """Print comprehensive team summary"""
    print(f"\n{'='*60}")
    print(f"🏆 {team.pack_type} TEAM {team.team_id} - {team.strategy.upper()}")
    print(f"{'='*60}")
    print(f"👑 Captain: {team.captain.name if team.captain else 'None'}")
    print(f"🥈 Vice Captain: {team.vice_captain.name if team.vice_captain else 'None'}")
    
    print(f"\n📋 TEAM COMPOSITION:")
    print(f"🏏 Batsmen ({len(team.batsmen)}): {', '.join(p.name for p in team.batsmen)}")
    print(f"⚡ Bowlers ({len(team.bowlers)}): {', '.join(p.name for p in team.bowlers)}")
    print(f"🔄 All-rounders ({len(team.all_rounders)}): {', '.join(p.name for p in team.all_rounders)}")
    print(f"🧤 Wicket-keepers ({len(team.wicket_keepers)}): {', '.join(p.name for p in team.wicket_keepers)}")
    
    print(f"\n📈 DETAILED PLAYER LIST:")
    for i, player in enumerate(sorted(team.players, key=lambda x: x.final_score, reverse=True), 1):
        captain_indicator = " (C)" if player == team.captain else " (VC)" if player == team.vice_captain else ""
        print(f"  {i:2d}. {player.name:20s} ({player.role:15s}){captain_indicator}")
    
    print(f"{'='*60}")

def print_hybrid_teams_summary(hybrid_teams: Dict[str, List[OptimalTeam]]) -> None:
    """Print summary of all hybrid teams"""
    print(f"\n{'🎯'*20}")
    print("🏆 HYBRID TEAM STRATEGY SUMMARY")
    print(f"{'🎯'*20}")
    
    total_teams = sum(len(teams) for teams in hybrid_teams.values())
    print(f"📊 Total Teams Generated: {total_teams}")
    
    for pack_name, teams in hybrid_teams.items():
        print(f"\n📦 {pack_name.upper()}:")
        if not teams:
            print("  ❌ No teams generated")
            continue
            
        for team in teams:
            strategy_info = f" ({team.strategy})" if team.strategy != "Optimal" else ""
            print(f"  🏆 Team {team.team_id}{strategy_info}: "
                  f"C: {team.captain.name if team.captain else 'None'} | "
                  f"VC: {team.vice_captain.name if team.vice_captain else 'None'} | "
                  f"Score: {team.total_score:.1f}")
    
    print(f"\n{'='*60}")
    print("💡 STRATEGY EXPLANATION:")
    print("📦 Pack-1: Same optimal 11 players with 3 different C/VC combinations")
    print("📦 Pack-2: Alternative teams with different selection strategies")
    print("   - Risk-Adjusted: Focus on consistent performers")
    print("   - Form-Based: Emphasizes recent form and momentum")
    print("   - Value-Picks: Best value for credits ratio")
    print(f"{'='*60}")

def generate_captain_vice_captain_variations(base_team_players: List[PlayerForOptimization], 
                                           num_variations: int = 3) -> List[Tuple[PlayerForOptimization, PlayerForOptimization]]:
    """
    Generate multiple C/VC combinations for the same 11 players
    
    Args:
        base_team_players: List of 11 selected players
        num_variations: Number of C/VC combinations to generate
    
    Returns:
        List of (captain, vice_captain) tuples
    """
    # Sort players by captain candidacy and final score
    captain_candidates = sorted(
        [p for p in base_team_players if p.is_captain_candidate or p.final_score >= 40],
        key=lambda x: x.final_score, 
        reverse=True
    )
    
    # If not enough captain candidates, use top scorers
    if len(captain_candidates) < num_variations * 2:
        captain_candidates = sorted(base_team_players, key=lambda x: x.final_score, reverse=True)
    
    variations = []
    used_captains = set()
    used_vice_captains = set()
    
    for i in range(min(num_variations, len(captain_candidates))):
        # Select captain (highest available scorer)
        captain = None
        for candidate in captain_candidates:
            if candidate.player_id not in used_captains:
                captain = candidate
                used_captains.add(candidate.player_id)
                break
        
        if not captain:
            break
        
        # Select vice-captain (prefer different role, exclude captain and used VCs)
        vc_candidates = [p for p in base_team_players if p != captain and p.player_id not in used_vice_captains]
        
        # If we've used all top candidates, reset and allow reuse but prefer unused ones
        if not vc_candidates:
            vc_candidates = [p for p in base_team_players if p != captain]
        
        # Sort candidates by score and prefer different role for diversity
        different_role_candidates = [p for p in vc_candidates if p.role != captain.role]
        same_role_candidates = [p for p in vc_candidates if p.role == captain.role]
        
        # Prioritize different role, then same role, both sorted by score
        if different_role_candidates:
            vice_captain = sorted(different_role_candidates, key=lambda x: x.final_score, reverse=True)[0]
        else:
            vice_captain = sorted(same_role_candidates, key=lambda x: x.final_score, reverse=True)[0]
        
        used_vice_captains.add(vice_captain.player_id)
        variations.append((captain, vice_captain))
    
    return variations

def generate_pack1_teams(base_team_players: List[PlayerForOptimization]) -> List[OptimalTeam]:
    """
    Generate Pack-1: Same 11 players with 3 different C/VC combinations
    
    Args:
        base_team_players: List of 11 optimally selected players
    
    Returns:
        List of 3 teams with same players but different C/VC
    """
    print("📦 Generating Pack-1: Same 11 players with different C/VC combinations...")
    
    # Generate 3 C/VC variations
    cv_variations = generate_captain_vice_captain_variations(base_team_players, 3)
    
    pack1_teams = []
    
    for i, (captain, vice_captain) in enumerate(cv_variations, 1):
        team = OptimalTeam(
            team_id=i,
            players=base_team_players.copy(),
            captain=captain,
            vice_captain=vice_captain,
            risk_level="Optimal",
            pack_type="Pack-1",
            strategy=f"C/VC Variation {i}"
        )
        pack1_teams.append(team)
        
        print(f"  ✅ Team {i}: C: {captain.name} | VC: {vice_captain.name} | Score: {team.total_score:.1f}")
    
    return pack1_teams

def generate_pack2_teams(players_for_opt: List[PlayerForOptimization]) -> List[OptimalTeam]:
    """
    Generate Pack-2: Alternative teams with different strategies
    
    Args:
        players_for_opt: All available players for selection
    
    Returns:
        List of 2-3 alternative teams with different strategies
    """
    print("📦 Generating Pack-2: Alternative teams with different strategies...")
    
    pack2_teams = []
    used_players = set()
    
    # Strategy 1: Risk-Adjusted (High consistency focus)
    risk_adjusted_players = apply_risk_profile_adjustments(players_for_opt, "Safe")
    team1 = generate_optimal_teams(risk_adjusted_players, 1, "Safe")[0] if generate_optimal_teams(risk_adjusted_players, 1, "Safe") else None
    
    if team1:
        team1.pack_type = "Pack-2"
        team1.strategy = "Risk-Adjusted"
        team1.team_id = 4
        pack2_teams.append(team1)
        used_players.update(p.player_id for p in team1.players)
        print(f"  ✅ Team 4 (Risk-Adjusted): Score: {team1.total_score:.1f}")
    
    # Strategy 2: Form-Based (Recent form focus)
    form_based_players = sorted(players_for_opt, key=lambda x: x.form_momentum + x.ema_score, reverse=True)
    available_form_players = [p for p in form_based_players if p.player_id not in used_players]
    
    if len(available_form_players) >= 11:
        team2_players = select_balanced_team_greedy(available_form_players[:20])  # Top 20 by form
        if len(team2_players) == 11:
            captain, vice_captain = select_captain_vice_captain(team2_players)
            team2 = OptimalTeam(
                team_id=5,
                players=team2_players,
                captain=captain,
                vice_captain=vice_captain,
                pack_type="Pack-2",
                strategy="Form-Based"
            )
            pack2_teams.append(team2)
            used_players.update(p.player_id for p in team2.players)
            print(f"  ✅ Team 5 (Form-Based): Score: {team2.total_score:.1f}")
    
    # Strategy 3: Value-Picks (Best value for credits)
    value_players = sorted(players_for_opt, key=lambda x: x.final_score / x.credits, reverse=True)
    available_value_players = [p for p in value_players if p.player_id not in used_players]
    
    if len(available_value_players) >= 11:
        team3_players = select_balanced_team_greedy(available_value_players[:20])  # Top 20 by value
        if len(team3_players) == 11:
            captain, vice_captain = select_captain_vice_captain(team3_players)
            team3 = OptimalTeam(
                team_id=6,
                players=team3_players,
                captain=captain,
                vice_captain=vice_captain,
                pack_type="Pack-2",
                strategy="Value-Picks"
            )
            pack2_teams.append(team3)
            print(f"  ✅ Team 6 (Value-Picks): Score: {team3.total_score:.1f}")
    
    return pack2_teams

def select_balanced_team_greedy(players: List[PlayerForOptimization], max_credits: float = 100.0) -> List[PlayerForOptimization]:
    """
    Select a balanced 11-player team using greedy approach with role constraints
    
    Args:
        players: List of available players (should be pre-sorted by desired criteria)
        max_credits: Maximum credit limit
    
    Returns:
        List of 11 selected players or empty list if unable to form team
    """
    selected_players = []
    current_credits = 0.0
    role_counts = {'bat': 0, 'bowl': 0, 'ar': 0, 'wk': 0}
    role_limits = {'bat': (2, 6), 'bowl': (2, 6), 'ar': (1, 4), 'wk': (1, 2)}
    
    # Helper function to get role key
    def get_role_key(role):
        role_lower = role.lower()
        if 'wk' in role_lower or 'wicket' in role_lower or 'keeper' in role_lower:
            return 'wk'
        elif 'allrounder' in role_lower or 'all-rounder' in role_lower:
            return 'ar'
        elif 'bowl' in role_lower:
            return 'bowl'
        else:
            return 'bat'
    
    # First pass: Ensure minimum requirements
    for role_key, (min_req, max_req) in role_limits.items():
        for player in players:
            if len(selected_players) >= 11:
                break
            
            player_role_key = get_role_key(player.role)
            if (player_role_key == role_key and 
                role_counts[role_key] < min_req and
                current_credits + player.credits <= max_credits + 5 and  # 5 credit buffer
                player not in selected_players):
                
                selected_players.append(player)
                current_credits += player.credits
                role_counts[role_key] += 1
    
    # Second pass: Fill remaining slots
    for player in players:
        if len(selected_players) >= 11:
            break
        
        if player in selected_players:
            continue
        
        player_role_key = get_role_key(player.role)
        min_req, max_req = role_limits[player_role_key]
        
        if (role_counts[player_role_key] < max_req and
            current_credits + player.credits <= max_credits + 5):
            
            selected_players.append(player)
            current_credits += player.credits
            role_counts[player_role_key] += 1
    
    return selected_players if len(selected_players) == 11 else []

def generate_hybrid_teams(player_features_list: List[PlayerFeatures],
                         match_format: str = "T20",
                         match_context: Dict[str, Any] = None) -> Dict[str, List[OptimalTeam]]:
    """
    Generate hybrid team strategy: Pack-1 (same players, different C/VC) + Pack-2 (alternative teams)
    
    Args:
        player_features_list: List of all player features
        match_format: Match format
        match_context: Match context
    
    Returns:
        Dict with 'Pack-1' and 'Pack-2' keys containing respective teams
    """
    if match_context is None:
        match_context = {}
    
    print("\n🎯 GENERATING HYBRID TEAM STRATEGY")
    print("="*60)
    
    # Prepare players for optimization
    players_for_opt = prepare_players_for_optimization(
        player_features_list, match_format, match_context
    )
    
    if len(players_for_opt) < 11:
        print(f"❌ Insufficient players ({len(players_for_opt)}) for team generation")
        return {'Pack-1': [], 'Pack-2': []}
    
    # Generate the base optimal team (11 players)
    print("\n🔍 Step 1: Generating base optimal team...")
    base_teams = generate_optimal_teams(players_for_opt, 1, "Balanced")
    
    if not base_teams:
        print("❌ Failed to generate base optimal team")
        return {'Pack-1': [], 'Pack-2': []}
    
    base_team_players = base_teams[0].players
    print(f"✅ Base team generated with {len(base_team_players)} players")
    
    # Generate Pack-1: Same players with different C/VC
    print("\n🔍 Step 2: Generating Pack-1...")
    pack1_teams = generate_pack1_teams(base_team_players)
    
    # Generate Pack-2: Alternative teams with different strategies
    print("\n🔍 Step 3: Generating Pack-2...")
    pack2_teams = generate_pack2_teams(players_for_opt)
    
    print(f"\n✅ HYBRID STRATEGY COMPLETE!")
    print(f"📦 Pack-1: {len(pack1_teams)} teams (same players, different C/VC)")
    print(f"📦 Pack-2: {len(pack2_teams)} teams (alternative strategies)")
    
    return {
        'Pack-1': pack1_teams,
        'Pack-2': pack2_teams
    }

def batch_generate_teams(player_features_list: List[PlayerFeatures],
                        match_format: str = "T20",
                        match_context: Dict[str, Any] = None,
                        num_teams: int = 3,
                        risk_profiles: List[str] = None) -> Dict[str, List[OptimalTeam]]:
    """
    Generate multiple teams with different risk profiles (legacy support)
    
    Args:
        player_features_list: List of all player features
        match_format: Match format
        match_context: Match context
        num_teams: Number of teams per risk profile
        risk_profiles: List of risk profiles to generate
    
    Returns:
        Dict mapping risk profiles to generated teams
    """
    if risk_profiles is None:
        risk_profiles = ['Safe', 'Balanced', 'High-Risk']
    
    if match_context is None:
        match_context = {}
    
    # Prepare players for optimization
    players_for_opt = prepare_players_for_optimization(
        player_features_list, match_format, match_context
    )
    
    all_teams = {}
    
    for risk_profile in risk_profiles:
        print(f"\n🎯 Generating {num_teams} teams with {risk_profile} risk profile...")
        teams = generate_optimal_teams(players_for_opt, num_teams, risk_profile)
        all_teams[risk_profile] = teams
        
        print(f"✅ Generated {len(teams)} {risk_profile} teams")
    
    return all_teams