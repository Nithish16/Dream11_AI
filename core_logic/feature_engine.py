from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
import statistics

@dataclass
class PlayerFeatures:
    """Comprehensive player features for Dream11 prediction"""
    player_id: int
    player_name: str
    role: str
    team_name: str = "Unknown"
    
    # Core performance metrics
    ema_score: float = 0.0
    time_decayed_average: float = 0.0
    consistency_score: float = 0.0
    
    # Advanced metrics
    dynamic_opportunity_index: float = 0.0
    matchup_score: float = 1.0
    form_momentum: float = 0.0
    
    # Role-specific features
    batting_features: Dict[str, float] = None
    bowling_features: Dict[str, float] = None
    fielding_features: Dict[str, float] = None
    
    # Meta features
    injury_risk_factor: float = 0.0
    captain_vice_captain_probability: float = 0.0
    performance_rating: float = 0.0
    
    def __post_init__(self):
        if self.batting_features is None:
            self.batting_features = {}
        if self.bowling_features is None:
            self.bowling_features = {}
        if self.fielding_features is None:
            self.fielding_features = {}

def calculate_ema(points_series: List[float], alpha: float = 0.4) -> float:
    """
    Calculate Exponential Moving Average for recent performance
    
    Args:
        points_series: List of recent performance points (most recent first)
        alpha: Smoothing factor (0 < alpha <= 1)
    
    Returns:
        float: EMA score
    """
    if not points_series or len(points_series) == 0:
        return 0.0
    
    if len(points_series) == 1:
        return float(points_series[0])
    
    # Ensure alpha is in valid range
    alpha = max(0.1, min(1.0, alpha))
    
    # Start with the most recent value
    ema = float(points_series[0])
    
    # Calculate EMA iteratively
    for i in range(1, len(points_series)):
        ema = alpha * points_series[i] + (1 - alpha) * ema
    
    return round(ema, 2)

def calculate_time_decayed_average(performance_history: List[Dict[str, Any]]) -> float:
    """
    Calculate time-decayed average giving more weight to recent performances
    
    Args:
        performance_history: List of performance dictionaries with 'score' and 'date'
    
    Returns:
        float: Time-decayed average score
    """
    if not performance_history:
        return 0.0
    
    current_date = datetime.now()
    total_weighted_score = 0.0
    total_weights = 0.0
    
    for performance in performance_history:
        score = performance.get('score', 0.0)
        perf_date = performance.get('date')
        
        if perf_date is None:
            # If no date, assume recent with moderate weight
            weight = 0.5
        else:
            # Calculate days ago
            if isinstance(perf_date, str):
                try:
                    perf_date = datetime.fromisoformat(perf_date.replace('Z', '+00:00'))
                except:
                    perf_date = current_date - timedelta(days=30)  # Default fallback
            
            days_ago = (current_date - perf_date).days
            
            # Exponential decay: more recent = higher weight
            # Weight decays by half every 30 days
            weight = math.exp(-days_ago / 30.0)
        
        total_weighted_score += score * weight
        total_weights += weight
    
    if total_weights == 0:
        return 0.0
    
    return round(total_weighted_score / total_weights, 2)

def calculate_consistency_score(points_series: List[float]) -> float:
    """
    Calculate consistency score based on variance of recent performances
    
    Args:
        points_series: List of recent performance points
    
    Returns:
        float: Consistency score (0-100, higher is more consistent)
    """
    if not points_series or len(points_series) < 2:
        return 0.0
    
    # Calculate coefficient of variation (CV) using statistics module
    mean_score = statistics.mean(points_series)
    std_score = statistics.stdev(points_series) if len(points_series) > 1 else 0
    
    if mean_score == 0:
        return 0.0
    
    cv = std_score / mean_score
    
    # Convert CV to consistency score (lower CV = higher consistency)
    # Use inverse relationship with scaling
    consistency = max(0, 100 - (cv * 50))
    
    return round(consistency, 2)

def calculate_dynamic_opportunity_index(player_role: str, pitch_archetype: str, 
                                     match_format: str = "T20", 
                                     batting_position: int = 5) -> float:
    """
    Calculate dynamic opportunity index based on role, pitch, and conditions
    
    Args:
        player_role: Player's role (Batter, Bowler, etc.)
        pitch_archetype: Pitch type (Flat, Green, Turning, Variable)
        match_format: Match format (T20, ODI, Test)
        batting_position: Batting order position
    
    Returns:
        float: Opportunity index (0.5 to 2.0)
    """
    base_opportunity = 1.0
    role_lower = player_role.lower()
    pitch_lower = pitch_archetype.lower()
    format_lower = match_format.lower()
    
    # Role-based base opportunities
    role_multipliers = {
        'batter': 1.2,
        'batsman': 1.2,
        'bowler': 1.1,
        'bowling allrounder': 1.4,
        'batting allrounder': 1.3,
        'allrounder': 1.3,
        'wk-batter': 1.25,
        'wicket-keeper': 1.2
    }
    
    # Find matching role
    role_multiplier = 1.0
    for role_key, multiplier in role_multipliers.items():
        if role_key in role_lower:
            role_multiplier = multiplier
            break
    
    # Pitch archetype adjustments
    pitch_adjustments = {
        'flat': {
            'batter': 1.3, 'batsman': 1.3, 'wk-batter': 1.3,
            'bowler': 0.8, 'bowling allrounder': 0.9
        },
        'green': {
            'bowler': 1.4, 'bowling allrounder': 1.3,
            'batter': 0.8, 'batsman': 0.8, 'wk-batter': 0.8
        },
        'turning': {
            'bowling allrounder': 1.4, 'allrounder': 1.3,
            'bowler': 1.2, 'batter': 0.9, 'batsman': 0.9
        },
        'variable': {
            'allrounder': 1.2, 'bowling allrounder': 1.2,
            'batting allrounder': 1.1
        }
    }
    
    pitch_multiplier = 1.0
    if pitch_lower in pitch_adjustments:
        for role_key, adjustment in pitch_adjustments[pitch_lower].items():
            if role_key in role_lower:
                pitch_multiplier = adjustment
                break
    
    # Format-specific adjustments
    format_multipliers = {
        't20': {'bowler': 1.1, 'allrounder': 1.2},
        'odi': {'batter': 1.1, 'batsman': 1.1, 'allrounder': 1.15},
        'test': {'bowler': 1.2, 'batter': 1.05, 'batsman': 1.05}
    }
    
    format_multiplier = 1.0
    if format_lower in format_multipliers:
        for role_key, adjustment in format_multipliers[format_lower].items():
            if role_key in role_lower:
                format_multiplier = adjustment
                break
    
    # Batting position adjustment (for batters)
    position_multiplier = 1.0
    if 'bat' in role_lower:
        if batting_position <= 2:
            position_multiplier = 1.2  # Openers get more opportunity
        elif batting_position <= 4:
            position_multiplier = 1.1  # Top order
        elif batting_position >= 7:
            position_multiplier = 0.9  # Lower order
    
    # Calculate final opportunity index
    opportunity_index = (base_opportunity * role_multiplier * 
                        pitch_multiplier * format_multiplier * position_multiplier)
    
    # Clamp between 0.5 and 2.0
    opportunity_index = max(0.5, min(2.0, opportunity_index))
    
    return round(opportunity_index, 2)

def calculate_matchup_score(player_data: Dict[str, Any], 
                          opposition_squad: List[Dict[str, Any]]) -> float:
    """
    Calculate matchup score based on historical performance against opposition
    
    Args:
        player_data: Player's performance data
        opposition_squad: Opposition team squad data
    
    Returns:
        float: Matchup score (0.5 to 1.5, 1.0 is neutral)
    """
    # Placeholder implementation - in reality would analyze:
    # - Head-to-head records
    # - Performance against similar bowling/batting styles
    # - Venue-specific performance against this opposition
    # - Recent form against this team
    
    # For now, return neutral score with slight random variation
    # based on player characteristics
    base_score = 1.0
    
    # Add slight variation based on player role and opposition strength
    player_role = player_data.get('role', '').lower()
    
    # Simulate some matchup advantages/disadvantages
    if 'allrounder' in player_role:
        base_score += 0.1  # All-rounders generally have more matchup opportunities
    
    # Add small random factor for realistic variation
    import random
    random.seed(player_data.get('player_id', 0))  # Consistent randomness
    variation = random.uniform(-0.1, 0.1)
    
    matchup_score = base_score + variation
    
    # Clamp between 0.5 and 1.5
    matchup_score = max(0.5, min(1.5, matchup_score))
    
    return round(matchup_score, 2)

def calculate_form_momentum(recent_scores: List[float]) -> float:
    """
    Calculate form momentum based on recent performance trend
    
    Args:
        recent_scores: List of recent scores (most recent first)
    
    Returns:
        float: Form momentum (-1.0 to 1.0, positive indicates improving form)
    """
    if not recent_scores or len(recent_scores) < 3:
        return 0.0
    
    # Take last 5 games maximum
    scores = recent_scores[:5]
    
    # Calculate linear regression slope to determine trend
    n = len(scores)
    x = list(range(n))  # Time points (0 = most recent)
    y = scores
    
    # Calculate slope using least squares
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    
    # Normalize slope to -1 to 1 range
    # Reverse sign since x[0] is most recent (we want positive slope for improving)
    momentum = -slope / (max(y) - min(y) + 1)  # Normalize by score range
    momentum = max(-1.0, min(1.0, momentum))
    
    return round(momentum, 2)

def extract_batting_features(player_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract batting-specific features"""
    batting_stats = player_data.get('batting_stats', {})
    career_stats = player_data.get('career_stats', {})
    
    features = {
        'batting_average': 0.0,
        'strike_rate': 0.0,
        'boundary_percentage': 0.0,
        'dot_ball_percentage': 0.0,
        'acceleration_factor': 0.0,
        'pressure_performance': 0.0
    }
    
    # Extract basic stats
    if batting_stats:
        features['batting_average'] = batting_stats.get('average', 0.0)
        features['strike_rate'] = batting_stats.get('strikeRate', 0.0)
    
    # Calculate boundary percentage from recent matches
    recent_matches = career_stats.get('recentMatches', [])
    if recent_matches:
        total_runs = sum(match.get('runs', 0) for match in recent_matches)
        total_boundaries = sum(match.get('fours', 0) * 4 + match.get('sixes', 0) * 6 
                             for match in recent_matches)
        
        if total_runs > 0:
            features['boundary_percentage'] = (total_boundaries / total_runs) * 100
    
    return features

def extract_bowling_features(player_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract bowling-specific features"""
    bowling_stats = player_data.get('bowling_stats', {})
    
    features = {
        'bowling_average': 0.0,
        'economy_rate': 0.0,
        'wicket_taking_ability': 0.0,
        'death_over_performance': 0.0,
        'powerplay_performance': 0.0
    }
    
    if bowling_stats:
        features['bowling_average'] = bowling_stats.get('average', 0.0)
        features['economy_rate'] = bowling_stats.get('economyRate', 0.0)
        features['wicket_taking_ability'] = bowling_stats.get('strikeRate', 0.0)
    
    return features

def extract_fielding_features(player_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract fielding-specific features"""
    features = {
        'fielding_impact': 0.0,
        'catch_probability': 0.0,
        'run_out_involvement': 0.0
    }
    
    # Placeholder - would be based on fielding statistics
    # For now, assign baseline values based on role
    role = player_data.get('role', '').lower()
    
    if 'wk' in role or 'keeper' in role:
        features['fielding_impact'] = 1.2
        features['catch_probability'] = 0.8
    elif 'allrounder' in role:
        features['fielding_impact'] = 1.1
        features['catch_probability'] = 0.7
    else:
        features['fielding_impact'] = 1.0
        features['catch_probability'] = 0.6
    
    return features

def calculate_performance_score(features: PlayerFeatures, 
                              match_context: Dict[str, Any]) -> float:
    """
    Calculate performance score based purely on data analysis (no points system)
    
    Args:
        features: Calculated player features
        match_context: Match context (format, venue, etc.)
    
    Returns:
        float: Performance score based on statistical analysis
    """
    # Base performance score from historical data
    base_score = 50.0
    
    # Statistical components (no points-based calculation)
    historical_performance = (features.ema_score * 0.25 + 
                            features.time_decayed_average * 0.25 +
                            features.consistency_score * 0.15)
    
    # Match context impact
    opportunity_factor = features.dynamic_opportunity_index * 20
    matchup_factor = features.matchup_score * 15
    form_factor = features.form_momentum * 10
    
    # Role-based statistical adjustments
    role = features.role.lower()
    role_factor = 0
    if 'allrounder' in role:
        role_factor = 15  # More opportunities to contribute
    elif 'wk' in role:
        role_factor = 8   # Additional contributions beyond batting
    elif 'bowl' in role:
        role_factor = 5   # Bowling opportunities
    
    performance_score = (base_score + historical_performance + 
                       opportunity_factor + matchup_factor + form_factor + role_factor)
    
    # Normalize to reasonable range
    performance_score = max(10.0, min(100.0, performance_score))
    
    return round(performance_score, 1)

def generate_player_features(player_raw_data: Dict[str, Any], 
                           match_context: Dict[str, Any] = None,
                           opposition_squad: List[Dict[str, Any]] = None) -> PlayerFeatures:
    """
    Main function to generate comprehensive player features
    
    Args:
        player_raw_data: Raw player data from aggregation step
        match_context: Match context (pitch, format, venue, etc.)
        opposition_squad: Opposition team squad for matchup analysis
    
    Returns:
        PlayerFeatures: Comprehensive feature set for the player
    """
    if match_context is None:
        match_context = {}
    if opposition_squad is None:
        opposition_squad = []
    
    # Extract basic info
    player_id = player_raw_data.get('player_id', 0)
    player_name = player_raw_data.get('name', 'Unknown')
    role = player_raw_data.get('role', 'Unknown')
    team_name = player_raw_data.get('team_name', 'Unknown')
    
    # Extract performance history
    career_stats = player_raw_data.get('career_stats', {})
    recent_matches = career_stats.get('recentMatches', [])
    
    # Convert recent matches to points series (simplified)
    points_series = []
    performance_history = []
    
    for match in recent_matches:
        # Simple points calculation (runs + wickets*25 + catches*10)
        runs = match.get('runs', 0)
        wickets = match.get('wickets', 0)
        catches = match.get('catches', 0)
        
        points = runs + (wickets * 25) + (catches * 10)
        points_series.append(points)
        
        performance_history.append({
            'score': points,
            'date': match.get('date'),
            'match_id': match.get('matchId')
        })
    
    # Calculate core metrics
    ema_score = calculate_ema(points_series)
    time_decayed_avg = calculate_time_decayed_average(performance_history)
    consistency = calculate_consistency_score(points_series)
    
    # Calculate advanced metrics
    pitch_archetype = match_context.get('pitch_archetype', 'Variable')
    match_format = match_context.get('match_format', 'T20')
    
    opportunity_index = calculate_dynamic_opportunity_index(
        role, pitch_archetype, match_format
    )
    
    matchup_score = calculate_matchup_score(player_raw_data, opposition_squad)
    form_momentum = calculate_form_momentum(points_series)
    
    # Extract role-specific features
    batting_features = extract_batting_features(player_raw_data)
    bowling_features = extract_bowling_features(player_raw_data)
    fielding_features = extract_fielding_features(player_raw_data)
    
    # Create feature object
    features = PlayerFeatures(
        player_id=player_id,
        player_name=player_name,
        role=role,
        team_name=team_name,
        ema_score=ema_score,
        time_decayed_average=time_decayed_avg,
        consistency_score=consistency,
        dynamic_opportunity_index=opportunity_index,
        matchup_score=matchup_score,
        form_momentum=form_momentum,
        batting_features=batting_features,
        bowling_features=bowling_features,
        fielding_features=fielding_features
    )
    
    # Calculate performance score based on data analysis
    features.performance_rating = calculate_performance_score(
        features, match_context
    )
    
    # Calculate captain/vice-captain probability (simplified)
    features.captain_vice_captain_probability = min(
        100.0, (features.ema_score * 0.4 + features.consistency_score * 0.3 + 
                features.dynamic_opportunity_index * 20)
    )
    
    return features

def batch_generate_features(players_data: List[Dict[str, Any]], 
                          match_context: Dict[str, Any] = None) -> List[PlayerFeatures]:
    """
    Generate features for multiple players in batch
    
    Args:
        players_data: List of player raw data
        match_context: Match context information
    
    Returns:
        List[PlayerFeatures]: List of feature sets for all players
    """
    if match_context is None:
        match_context = {}
    
    features_list = []
    
    for player_data in players_data:
        try:
            features = generate_player_features(player_data, match_context)
            features_list.append(features)
        except Exception as e:
            print(f"Error generating features for player {player_data.get('name', 'Unknown')}: {e}")
            continue
    
    return features_list

def print_feature_summary(features: PlayerFeatures) -> None:
    """Print a comprehensive summary of player features"""
    print(f"\n{'='*60}")
    print(f"🏏 PLAYER FEATURES: {features.player_name}")
    print(f"{'='*60}")
    print(f"📊 Role: {features.role}")
    print(f"🎯 EMA Score: {features.ema_score}")
    print(f"⏰ Time Decayed Average: {features.time_decayed_average}")
    print(f"📈 Consistency: {features.consistency_score}%")
    print(f"🚀 Opportunity Index: {features.dynamic_opportunity_index}")
    print(f"⚔️  Matchup Score: {features.matchup_score}")
    print(f"📊 Form Momentum: {features.form_momentum}")
    print(f"💎 Performance Rating: {features.performance_rating}")
    print(f"👑 Captain Probability: {features.captain_vice_captain_probability:.1f}%")
    
    if features.batting_features:
        print(f"\n🏏 Batting Features:")
        for key, value in features.batting_features.items():
            print(f"   {key}: {value}")
    
    if features.bowling_features:
        print(f"\n⚡ Bowling Features:")
        for key, value in features.bowling_features.items():
            print(f"   {key}: {value}")
    
    print(f"{'='*60}")