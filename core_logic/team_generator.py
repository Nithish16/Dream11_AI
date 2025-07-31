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
    
    # Enhanced features
    ownership_prediction: float = 50.0  # Expected ownership percentage

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
    
    # Enhanced features
    confidence_score: float = 3.0  # 1-5 stars rating
    ownership_prediction: float = 50.0  # Expected ownership percentage
    contest_recommendation: str = "Both"  # Small, Grand, Both
    strategic_focus: str = "Balanced"  # Ceiling, Safety, Differential, etc.
    
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

def calculate_ownership_prediction(player: PlayerForOptimization, all_players: List[PlayerForOptimization]) -> float:
    """
    Calculate expected ownership percentage for a player
    Based on credits, performance, and player popularity
    """
    # Normalize final score (0-100 scale)
    max_score = max(p.final_score for p in all_players) if all_players else 100
    min_score = min(p.final_score for p in all_players) if all_players else 0
    score_range = max_score - min_score if max_score != min_score else 1
    normalized_score = ((player.final_score - min_score) / score_range) * 100
    
    # Credits factor (lower credits = higher ownership)
    credit_factor = max(0, (100 - player.credits * 10))  # Assuming credits are 0-10 scale
    
    # Role factor (popular roles get higher ownership)
    role_multiplier = 1.0
    if 'wk' in player.role.lower():
        role_multiplier = 1.3  # Wicket keepers are popular
    elif 'allrounder' in player.role.lower():
        role_multiplier = 1.2  # All-rounders are popular
    
    # Base ownership calculation
    base_ownership = (normalized_score * 0.6 + credit_factor * 0.4) * role_multiplier
    
    # Clamp between 5% and 95%
    return max(5.0, min(95.0, base_ownership))

def calculate_team_confidence_score(team: OptimalTeam) -> float:
    """
    Calculate confidence score (1-5 stars) for a team
    Based on consistency, risk level, and strategic focus
    """
    if not team.players:
        return 3.0
    
    # Factor 1: Average consistency of players
    avg_consistency = sum(p.consistency_score for p in team.players) / len(team.players)
    consistency_score = min(5.0, (avg_consistency / 20))  # Normalize to 5
    
    # Factor 2: Team balance (role distribution)
    role_balance = 1.0
    if len(team.batsmen) >= 3 and len(team.bowlers) >= 3 and len(team.wicket_keepers) >= 1:
        role_balance = 1.2
    
    # Factor 3: Strategy-specific adjustments
    strategy_bonus = 1.0
    if team.strategy == "Risk-Adjusted":
        strategy_bonus = 1.1  # Safer teams get slight bonus
    elif "Optimal" in team.strategy:
        strategy_bonus = 1.15  # Optimal strategies get higher confidence
    
    # Factor 4: Captain selection quality
    captain_bonus = 1.0
    if team.captain and team.captain.final_score > 50:
        captain_bonus = 1.1
    
    confidence = consistency_score * role_balance * strategy_bonus * captain_bonus
    return max(1.0, min(5.0, confidence))

def determine_contest_recommendation(team: OptimalTeam) -> str:
    """
    Determine contest recommendation based on team characteristics
    """
    if not team.players:
        return "Both"
    
    # Calculate average ownership
    avg_ownership = sum(p.ownership_prediction for p in team.players) / len(team.players)
    
    # Calculate risk level based on consistency
    avg_consistency = sum(p.consistency_score for p in team.players) / len(team.players)
    
    # Determine recommendation
    if avg_ownership < 40 and avg_consistency > 60:
        return "Grand"  # Low ownership, high consistency = good for large contests
    elif avg_ownership > 60 and avg_consistency > 70:
        return "Small"  # High ownership, high consistency = good for small contests
    else:
        return "Both"  # Balanced approach

def determine_strategic_focus(team: OptimalTeam, strategy: str) -> str:
    """
    Determine the strategic focus description for a team
    """
    if strategy == "Risk-Adjusted":
        return "Safety"
    elif strategy == "Form-Based":
        return "Ceiling"
    elif strategy == "Value-Picks":
        return "Differential"
    elif "C/VC Variation 1" in strategy:
        return "Ceiling"
    elif "C/VC Variation 2" in strategy:
        return "Safety"
    elif "C/VC Variation 3" in strategy:
        return "Differential"
    else:
        return "Balanced"

def generate_scenario_alternatives(team: OptimalTeam, all_players: List[PlayerForOptimization]) -> Dict[str, List[str]]:
    """
    Generate scenario planning alternatives for key players
    """
    scenarios = {
        "captain_alternatives": [],
        "vice_captain_alternatives": [], 
        "risky_player_substitutes": []
    }
    
    # Captain alternatives (top 3 captain candidates not already captain)
    captain_candidates = [p for p in all_players if p.is_captain_candidate and p != team.captain]
    captain_candidates.sort(key=lambda x: x.final_score, reverse=True)
    scenarios["captain_alternatives"] = [p.name for p in captain_candidates[:3]]
    
    # Vice-captain alternatives  
    vc_candidates = [p for p in all_players if p.is_vice_captain_candidate and p != team.vice_captain and p != team.captain]
    vc_candidates.sort(key=lambda x: x.final_score, reverse=True)
    scenarios["vice_captain_alternatives"] = [p.name for p in vc_candidates[:3]]
    
    # Risky player substitutes (players with low consistency in team)
    risky_players = [p for p in team.players if p.consistency_score < 40]
    for risky_player in risky_players[:2]:  # Max 2 risky players
        # Find similar role replacement
        role_replacements = [p for p in all_players 
                           if p.role == risky_player.role 
                           and p not in team.players 
                           and p.consistency_score > risky_player.consistency_score]
        if role_replacements:
            best_replacement = max(role_replacements, key=lambda x: x.final_score)
            scenarios["risky_player_substitutes"].append(f"{risky_player.name} → {best_replacement.name}")
    
    return scenarios

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
        
        # Determine team name (use from features if available, otherwise fallback)
        team_name = getattr(features, 'team_name', None) or team_mapping.get(features.player_id, None)
        
        # If still no team name, distribute players alternately between teams
        if not team_name or team_name == "Unknown":
            # Use player_id to consistently assign teams
            team_name = f"Team_{(features.player_id % 2) + 1}"
        
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
    
    # Calculate ownership predictions for all players
    for player in players_for_opt:
        player.ownership_prediction = calculate_ownership_prediction(player, players_for_opt)
    
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

def get_team_abbreviation(team_name: str) -> str:
    """
    Generate team abbreviation from team name
    """
    if not team_name or team_name == "Unknown" or team_name == "Unknown Team":
        return "UNK"
    
    # Common team abbreviations
    abbreviations = {
        'india': 'IND',
        'england': 'ENG', 
        'australia': 'AUS',
        'new zealand': 'NZ',
        'south africa': 'SA',
        'pakistan': 'PAK',
        'sri lanka': 'SL',
        'bangladesh': 'BAN',
        'west indies': 'WI',
        'afghanistan': 'AFG',
        'zimbabwe': 'ZIM',
        'ireland': 'IRE',
        'scotland': 'SCO',
        'netherlands': 'NED',
        'nepal': 'NEP',
        'oman': 'OMA',
        'united arab emirates': 'UAE',
        'qatar': 'QAT',
        'kuwait': 'KUW',
        'saudi arabia': 'SAU',
        'bahrain': 'BHR',
        'hong kong': 'HK',
        'singapore': 'SIN',
        'malaysia': 'MAL',
        'thailand': 'THA',
        'myanmar': 'MYA',
        'bhutan': 'BHU',
        'maldives': 'MDV',
        'usa': 'USA',
        'united states': 'USA',
        'canada': 'CAN',
        'bermuda': 'BER',
        'namibia': 'NAM',
        'kenya': 'KEN',
        'uganda': 'UGA',
        'tanzania': 'TAN',
        'botswana': 'BOT',
        'ghana': 'GHA',
        'nigeria': 'NIG',
        'gambia': 'GAM',
        'sierra leone': 'SLE',
        'malawi': 'MAW',
        'mozambique': 'MOZ',
        'rwanda': 'RWA',
        'lesotho': 'LES',
        'swaziland': 'SWA',
        'zambia': 'ZAM',
        'cameroon': 'CAM',
        'ivory coast': 'CIV',
        'mali': 'MLI',
        'burkina faso': 'BUR',
        'senegal': 'SEN',
        'madagascar': 'MAD',
        # IPL and domestic teams
        'mumbai indians': 'MI',
        'chennai super kings': 'CSK',
        'royal challengers bangalore': 'RCB',
        'kolkata knight riders': 'KKR',
        'delhi capitals': 'DC',
        'punjab kings': 'PK',
        'rajasthan royals': 'RR',
        'sunrisers hyderabad': 'SRH',
        'gujarat titans': 'GT',
        'lucknow super giants': 'LSG'
    }
    
    team_lower = team_name.lower().strip()
    
    # Check for exact match first
    if team_lower in abbreviations:
        return abbreviations[team_lower]
    
    # Check for partial matches
    for full_name, abbr in abbreviations.items():
        if full_name in team_lower or team_lower in full_name:
            return abbr
    
    # Fallback: Create abbreviation from first letters of words
    words = team_name.split()
    if len(words) == 1:
        # Single word: take first 3 characters
        return words[0][:3].upper()
    elif len(words) == 2:
        # Two words: first letter of each + first letter of second word
        return (words[0][0] + words[1][0] + words[1][1:2]).upper()[:3]
    else:
        # Multiple words: first letter of each word (max 3)
        return ''.join(word[0] for word in words[:3]).upper()

def print_team_summary(team: OptimalTeam) -> None:
    """Print comprehensive team summary with enhanced features"""
    stars = "⭐" * int(team.confidence_score)
    print(f"\n{'='*60}")
    print(f"🏆 {team.pack_type} TEAM {team.team_id} - {team.strategy.upper()}")
    print(f"{'='*60}")
    print(f"👑 Captain: {team.captain.name if team.captain else 'None'}")
    print(f"🥈 Vice Captain: {team.vice_captain.name if team.vice_captain else 'None'}")
    
    print(f"\n🎯 TEAM ANALYTICS:")
    print(f"   💎 Confidence Score: {team.confidence_score:.1f}/5.0 {stars}")
    print(f"   📊 Strategic Focus: {team.strategic_focus}")
    print(f"   🎪 Contest Recommendation: {team.contest_recommendation} Leagues")
    print(f"   📈 Expected Ownership: {team.ownership_prediction:.1f}%")
    print(f"   💰 Total Credits: {team.total_credits:.1f}/100")
    print(f"   🎯 Projected Score: {team.total_score:.1f}")
    
    print(f"\n📋 TEAM COMPOSITION:")
    print(f"🏏 Batsmen ({len(team.batsmen)}): {', '.join(f'{p.name} ({get_team_abbreviation(p.team)})' for p in team.batsmen)}")
    print(f"⚡ Bowlers ({len(team.bowlers)}): {', '.join(f'{p.name} ({get_team_abbreviation(p.team)})' for p in team.bowlers)}")
    print(f"🔄 All-rounders ({len(team.all_rounders)}): {', '.join(f'{p.name} ({get_team_abbreviation(p.team)})' for p in team.all_rounders)}")
    print(f"🧤 Wicket-keepers ({len(team.wicket_keepers)}): {', '.join(f'{p.name} ({get_team_abbreviation(p.team)})' for p in team.wicket_keepers)}")
    
    print(f"\n📈 DETAILED PLAYER LIST:")
    for i, player in enumerate(sorted(team.players, key=lambda x: x.final_score, reverse=True), 1):
        captain_indicator = " (C)" if player == team.captain else " (VC)" if player == team.vice_captain else ""
        ownership_indicator = f" [{player.ownership_prediction:.0f}% own]"
        team_abbr = get_team_abbreviation(player.team)
        print(f"  {i:2d}. {player.name:20s} ({team_abbr}) ({player.role:12s}){captain_indicator}{ownership_indicator}")
    
    print(f"{'='*60}")

def print_hybrid_teams_summary(hybrid_teams: Dict[str, List[OptimalTeam]]) -> None:
    """Print enhanced summary of all hybrid teams"""
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
            stars = "⭐" * int(team.confidence_score)
            strategy_info = f" ({team.strategy})" if team.strategy != "Optimal" else ""
            print(f"  🏆 Team {team.team_id}{strategy_info}:")
            print(f"     👑 C: {team.captain.name if team.captain else 'None'} | 🥈 VC: {team.vice_captain.name if team.vice_captain else 'None'}")
            print(f"     💎 {team.confidence_score:.1f}/5.0 {stars} | 🎪 {team.contest_recommendation} | 📊 {team.strategic_focus} | 🎯 {team.total_score:.1f}")
    
    print(f"\n{'='*60}")
    print("💡 ENHANCED STRATEGY EXPLANATION:")
    print("📦 Pack-1: Same optimal 11 players with strategic C/VC variations")
    print("   • Team 1: Highest Ceiling (Max Points Potential)")
    print("   • Team 2: Safest Choice (Consistent Performers)")  
    print("   • Team 3: Differential Pick (Low Ownership)")
    print("\n📦 Pack-2: Alternative teams with pitch-based strategies")
    print("   • Risk-Adjusted: Safety focus for consistent returns")
    print("   • Form-Based: Ceiling focus based on recent performance")
    print("   • Value-Picks: Differential focus with best credit value")
    print(f"\n🎯 CONTEST RECOMMENDATIONS:")
    print("   • Small: High ownership, high consistency teams")
    print("   • Grand: Low ownership, differential teams") 
    print("   • Both: Balanced approach for all contest types")
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
    Enhanced with strategic focus descriptions
    
    Args:
        base_team_players: List of 11 optimally selected players
    
    Returns:
        List of 3 teams with same players but different C/VC
    """
    print("📦 Generating Pack-1: Same 11 players with different C/VC combinations...")
    
    # Generate 3 C/VC variations
    cv_variations = generate_captain_vice_captain_variations(base_team_players, 3)
    
    pack1_teams = []
    strategy_descriptions = [
        "Highest Ceiling (Max Points Potential)", 
        "Safest Choice (Consistent Performers)",
        "Differential Pick (Low Ownership)"
    ]
    
    for i, (captain, vice_captain) in enumerate(cv_variations, 1):
        strategy = f"C/VC Variation {i}"
        team = OptimalTeam(
            team_id=i,
            players=base_team_players.copy(),
            captain=captain,
            vice_captain=vice_captain,
            risk_level="Optimal",
            pack_type="Pack-1",
            strategy=strategy
        )
        
        # Calculate enhanced features
        team.confidence_score = calculate_team_confidence_score(team)
        team.contest_recommendation = determine_contest_recommendation(team)
        team.strategic_focus = determine_strategic_focus(team, strategy)
        
        # Calculate team ownership prediction
        team.ownership_prediction = sum(p.ownership_prediction for p in team.players) / len(team.players)
        
        pack1_teams.append(team)
        
        print(f"  ✅ Team {i} ({strategy_descriptions[i-1]}): C: {captain.name} | VC: {vice_captain.name}")
        print(f"     Score: {team.total_score:.1f} | Confidence: {team.confidence_score:.1f}⭐ | Contest: {team.contest_recommendation}")
    
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
        
        # Calculate enhanced features
        team1.confidence_score = calculate_team_confidence_score(team1)
        team1.contest_recommendation = determine_contest_recommendation(team1)
        team1.strategic_focus = determine_strategic_focus(team1, "Risk-Adjusted")
        team1.ownership_prediction = sum(p.ownership_prediction for p in team1.players) / len(team1.players)
        
        pack2_teams.append(team1)
        used_players.update(p.player_id for p in team1.players)
        print(f"  ✅ Team 4 (Risk-Adjusted - Consistent Performers): Score: {team1.total_score:.1f}")
        print(f"     Confidence: {team1.confidence_score:.1f}⭐ | Contest: {team1.contest_recommendation} | Focus: {team1.strategic_focus}")
    
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
            
            # Calculate enhanced features
            team2.confidence_score = calculate_team_confidence_score(team2)
            team2.contest_recommendation = determine_contest_recommendation(team2)
            team2.strategic_focus = determine_strategic_focus(team2, "Form-Based")
            team2.ownership_prediction = sum(p.ownership_prediction for p in team2.players) / len(team2.players)
            
            pack2_teams.append(team2)
            used_players.update(p.player_id for p in team2.players)
            print(f"  ✅ Team 5 (Form-Based - Recent Performance): Score: {team2.total_score:.1f}")
            print(f"     Confidence: {team2.confidence_score:.1f}⭐ | Contest: {team2.contest_recommendation} | Focus: {team2.strategic_focus}")
    
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
            
            # Calculate enhanced features
            team3.confidence_score = calculate_team_confidence_score(team3)
            team3.contest_recommendation = determine_contest_recommendation(team3)
            team3.strategic_focus = determine_strategic_focus(team3, "Value-Picks")
            team3.ownership_prediction = sum(p.ownership_prediction for p in team3.players) / len(team3.players)
            
            pack2_teams.append(team3)
            print(f"  ✅ Team 6 (Value-Picks - Best Credit Value): Score: {team3.total_score:.1f}")
            print(f"     Confidence: {team3.confidence_score:.1f}⭐ | Contest: {team3.contest_recommendation} | Focus: {team3.strategic_focus}")
    
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
    
    # Create team mapping from player features to preserve actual team names
    team_mapping = {}
    for features in player_features_list:
        if hasattr(features, 'team_name') and features.team_name and features.team_name != "Unknown":
            team_mapping[features.player_id] = features.team_name
    
    # Prepare players for optimization
    players_for_opt = prepare_players_for_optimization(
        player_features_list, match_format, match_context, team_mapping
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
    
    # Create team mapping from player features to preserve actual team names
    team_mapping = {}
    for features in player_features_list:
        if hasattr(features, 'team_name') and features.team_name and features.team_name != "Unknown":
            team_mapping[features.player_id] = features.team_name
    
    # Prepare players for optimization
    players_for_opt = prepare_players_for_optimization(
        player_features_list, match_format, match_context, team_mapping
    )
    
    all_teams = {}
    
    for risk_profile in risk_profiles:
        print(f"\n🎯 Generating {num_teams} teams with {risk_profile} risk profile...")
        teams = generate_optimal_teams(players_for_opt, num_teams, risk_profile)
        all_teams[risk_profile] = teams
        
        print(f"✅ Generated {len(teams)} {risk_profile} teams")
    
    return all_teams