#!/usr/bin/env python3
"""
Advanced Matchup Analysis Engine - Player vs Player/Team Analysis
Sophisticated head-to-head analysis and performance prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math
import statistics
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PlayerMatchup:
    """Individual player matchup analysis"""
    player_id: int
    player_name: str
    opposition_team: str
    
    # Historical performance
    matches_played: int = 0
    avg_score: float = 0.0
    strike_rate: float = 0.0
    economy_rate: float = 0.0
    wickets_per_match: float = 0.0
    
    # Recent form against opposition
    recent_form: List[float] = field(default_factory=list)
    last_5_avg: float = 0.0
    
    # Specific matchup factors
    vs_pace_performance: float = 0.0
    vs_spin_performance: float = 0.0
    vs_left_arm_performance: float = 0.0
    vs_right_arm_performance: float = 0.0
    
    # Venue specific
    venue_performance: float = 0.0
    home_away_factor: float = 1.0
    
    # Psychological factors
    pressure_handling: float = 0.5
    rivalry_factor: float = 1.0
    momentum_factor: float = 1.0

@dataclass
class TeamMatchup:
    """Team vs team matchup analysis"""
    team1_name: str
    team2_name: str
    
    # Historical head-to-head
    total_matches: int = 0
    team1_wins: int = 0
    team2_wins: int = 0
    draws: int = 0
    
    # Recent encounters
    recent_results: List[str] = field(default_factory=list)  # ['W', 'L', 'W', etc.]
    recent_margin: List[float] = field(default_factory=list)  # Victory margins
    
    # Strength analysis
    batting_strength_diff: float = 0.0
    bowling_strength_diff: float = 0.0
    fielding_strength_diff: float = 0.0
    
    # Tactical matchups
    pace_vs_batting: float = 0.0
    spin_vs_batting: float = 0.0
    death_bowling_advantage: float = 0.0
    powerplay_advantage: float = 0.0

@dataclass
class MatchupInsights:
    """Comprehensive matchup insights"""
    key_battles: List[Dict[str, Any]] = field(default_factory=list)
    tactical_advantages: Dict[str, float] = field(default_factory=dict)
    predicted_scores: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    opportunity_factors: List[str] = field(default_factory=list)

class AdvancedMatchupAnalyzer:
    """Advanced matchup analysis for cricket matches"""
    
    def __init__(self):
        self.bowling_types = {
            'pace': ['fast', 'medium_fast', 'medium'],
            'spin': ['leg_spin', 'off_spin', 'left_arm_spin']
        }
        
        self.matchup_weights = {
            'historical': 0.3,
            'recent_form': 0.25,
            'specific_matchup': 0.2,
            'venue_factor': 0.15,
            'psychological': 0.1
        }
        
        # Initialize historical database (in production, load from database)
        self.historical_database = self._initialize_historical_database()
    
    def _initialize_historical_database(self) -> Dict[str, Any]:
        """Initialize historical matchup database"""
        # This would be loaded from a comprehensive database in production
        return {
            'player_vs_team': defaultdict(list),
            'team_vs_team': defaultdict(dict),
            'venue_records': defaultdict(dict),
            'bowling_matchups': defaultdict(dict)
        }
    
    def analyze_player_matchups(self, players: List[Dict[str, Any]], 
                              opposition_team: str,
                              venue: str,
                              match_context: Dict[str, Any]) -> List[PlayerMatchup]:
        """Analyze individual player matchups against opposition"""
        
        matchups = []
        
        for player in players:
            player_id = player.get('player_id', 0)
            player_name = player.get('name', 'Unknown')
            
            # Create comprehensive matchup analysis
            matchup = self._analyze_individual_matchup(
                player, opposition_team, venue, match_context
            )
            
            matchups.append(matchup)
        
        return matchups
    
    def _analyze_individual_matchup(self, player: Dict[str, Any], 
                                  opposition_team: str,
                                  venue: str,
                                  match_context: Dict[str, Any]) -> PlayerMatchup:
        """Analyze individual player matchup"""
        
        player_id = player.get('player_id', 0)
        player_name = player.get('name', 'Unknown')
        
        # Initialize matchup
        matchup = PlayerMatchup(
            player_id=player_id,
            player_name=player_name,
            opposition_team=opposition_team
        )
        
        # Analyze historical performance against this team
        matchup = self._analyze_historical_vs_team(matchup, player, opposition_team)
        
        # Analyze recent form
        matchup = self._analyze_recent_form_vs_team(matchup, player, opposition_team)
        
        # Analyze specific bowling/batting matchups
        matchup = self._analyze_specific_matchups(matchup, player, opposition_team)
        
        # Analyze venue performance
        matchup = self._analyze_venue_performance(matchup, player, venue)
        
        # Calculate psychological factors
        matchup = self._calculate_psychological_factors(matchup, player, opposition_team, match_context)
        
        return matchup
    
    def _analyze_historical_vs_team(self, matchup: PlayerMatchup, 
                                  player: Dict[str, Any],
                                  opposition_team: str) -> PlayerMatchup:
        """Analyze historical performance against specific team"""
        
        # Get career stats
        career_stats = player.get('career_stats', {})
        recent_matches = career_stats.get('recentMatches', [])
        
        # Filter matches against this opposition
        vs_team_matches = []
        for match in recent_matches:
            if match.get('opposition', '').lower() == opposition_team.lower():
                vs_team_matches.append(match)
        
        if vs_team_matches:
            matchup.matches_played = len(vs_team_matches)
            
            # Calculate averages
            runs_list = [match.get('runs', 0) for match in vs_team_matches]
            balls_list = [match.get('balls_faced', 1) for match in vs_team_matches]
            wickets_list = [match.get('wickets', 0) for match in vs_team_matches]
            
            matchup.avg_score = np.mean(runs_list) if runs_list else 0
            
            # Calculate strike rate for batsmen
            if any(balls > 0 for balls in balls_list):
                total_runs = sum(runs_list)
                total_balls = sum(balls_list)
                matchup.strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
            
            # Calculate bowling stats
            if any(wickets > 0 for wickets in wickets_list):
                matchup.wickets_per_match = np.mean(wickets_list)
                
                overs_list = [match.get('overs', 0) for match in vs_team_matches]
                runs_conceded_list = [match.get('runs_conceded', 0) for match in vs_team_matches]
                
                if sum(overs_list) > 0:
                    matchup.economy_rate = sum(runs_conceded_list) / sum(overs_list)
        
        return matchup
    
    def _analyze_recent_form_vs_team(self, matchup: PlayerMatchup, 
                                   player: Dict[str, Any],
                                   opposition_team: str) -> PlayerMatchup:
        """Analyze recent form against specific team"""
        
        career_stats = player.get('career_stats', {})
        recent_matches = career_stats.get('recentMatches', [])
        
        # Get last 10 matches against this team
        vs_team_recent = []
        for match in recent_matches[-20:]:  # Look in last 20 matches
            if match.get('opposition', '').lower() == opposition_team.lower():
                fantasy_points = self._calculate_fantasy_points(match)
                vs_team_recent.append(fantasy_points)
                
                if len(vs_team_recent) >= 5:  # Last 5 matches against this team
                    break
        
        matchup.recent_form = vs_team_recent
        if vs_team_recent:
            matchup.last_5_avg = np.mean(vs_team_recent)
        
        return matchup
    
    def _calculate_fantasy_points(self, match: Dict[str, Any]) -> float:
        """Calculate fantasy points for a match"""
        runs = match.get('runs', 0)
        wickets = match.get('wickets', 0)
        catches = match.get('catches', 0)
        run_outs = match.get('run_outs', 0)
        
        points = runs + (wickets * 25) + (catches * 8) + (run_outs * 12)
        
        # Bonus points
        if runs >= 50:
            points += 8
        if runs >= 100:
            points += 16
        if wickets >= 3:
            points += 12
        if wickets >= 5:
            points += 24
        
        return points
    
    def _analyze_specific_matchups(self, matchup: PlayerMatchup, 
                                 player: Dict[str, Any],
                                 opposition_team: str) -> PlayerMatchup:
        """Analyze specific bowling type matchups"""
        
        # This would analyze performance against specific bowling types
        # For now, simulate based on player role and typical performance
        
        role = player.get('role', '').lower()
        batting_stats = player.get('batting_stats', {})
        
        if 'bat' in role:
            # Batsmen analysis
            base_avg = batting_stats.get('average', 35)
            
            # Simulate performance against different bowling types
            matchup.vs_pace_performance = base_avg * (0.9 + np.random.normal(0, 0.1))
            matchup.vs_spin_performance = base_avg * (0.95 + np.random.normal(0, 0.1))
            matchup.vs_left_arm_performance = base_avg * (0.85 + np.random.normal(0, 0.15))
            matchup.vs_right_arm_performance = base_avg * (1.0 + np.random.normal(0, 0.1))
            
        elif 'bowl' in role:
            # Bowlers analysis
            bowling_stats = player.get('bowling_stats', {})
            base_avg = bowling_stats.get('average', 30)
            
            # Performance against different batting strengths
            matchup.vs_pace_performance = 35 - (base_avg - 30)  # Lower is better for bowlers
            matchup.vs_spin_performance = 35 - (base_avg - 30)
        
        return matchup
    
    def _analyze_venue_performance(self, matchup: PlayerMatchup, 
                                 player: Dict[str, Any],
                                 venue: str) -> PlayerMatchup:
        """Analyze venue-specific performance"""
        
        # Get venue stats if available
        venue_stats = player.get('venue_stats', {})
        
        if venue.lower() in [v.lower() for v in venue_stats.keys()]:
            venue_data = venue_stats.get(venue, {})
            matchup.venue_performance = venue_data.get('average', 0)
        else:
            # Use overall average as baseline
            career_stats = player.get('career_stats', {})
            matchup.venue_performance = career_stats.get('average', 35)
        
        # Home/away factor
        player_team = player.get('team_name', '')
        if self._is_home_venue(player_team, venue):
            matchup.home_away_factor = 1.1  # 10% boost for home conditions
        else:
            matchup.home_away_factor = 0.95  # 5% reduction for away conditions
        
        return matchup
    
    def _is_home_venue(self, team_name: str, venue: str) -> bool:
        """Determine if venue is home for the team"""
        home_venues = {
            'india': ['wankhede', 'eden gardens', 'chinnaswamy', 'kotla', 'chepauk'],
            'australia': ['melbourne', 'sydney', 'adelaide', 'perth', 'gabba'],
            'england': ['lords', 'oval', 'old trafford', 'headingley', 'edgbaston'],
            'south africa': ['wanderers', 'newlands', 'centurion', 'durban'],
            'pakistan': ['karachi', 'lahore', 'rawalpindi'],
            'west indies': ['bridgetown', 'kingston', 'port of spain'],
            'new zealand': ['wellington', 'auckland', 'christchurch'],
            'sri lanka': ['colombo', 'galle', 'kandy'],
            'bangladesh': ['dhaka', 'chittagong']
        }
        
        team_lower = team_name.lower()
        venue_lower = venue.lower()
        
        for team, venues in home_venues.items():
            if team in team_lower:
                return any(v in venue_lower for v in venues)
        
        return False
    
    def _calculate_psychological_factors(self, matchup: PlayerMatchup, 
                                       player: Dict[str, Any],
                                       opposition_team: str,
                                       match_context: Dict[str, Any]) -> PlayerMatchup:
        """Calculate psychological factors"""
        
        # Pressure handling based on player experience
        career_stats = player.get('career_stats', {})
        total_matches = career_stats.get('matches', 0)
        
        if total_matches > 100:
            matchup.pressure_handling = 0.8
        elif total_matches > 50:
            matchup.pressure_handling = 0.6
        else:
            matchup.pressure_handling = 0.4
        
        # Rivalry factor for high-profile matchups
        high_profile_rivalries = [
            ('india', 'pakistan'), ('australia', 'england'),
            ('india', 'australia'), ('south africa', 'australia')
        ]
        
        player_team = player.get('team_name', '').lower()
        opposition_lower = opposition_team.lower()
        
        for team1, team2 in high_profile_rivalries:
            if ((team1 in player_team and team2 in opposition_lower) or 
                (team2 in player_team and team1 in opposition_lower)):
                matchup.rivalry_factor = 1.15  # 15% boost for rivalry matches
                break
        
        # Momentum factor based on recent overall form
        recent_form = player.get('recent_form', [])
        if len(recent_form) >= 3:
            recent_scores = [match.get('fantasy_points', 0) for match in recent_form[:3]]
            avg_recent = np.mean(recent_scores)
            
            if avg_recent > 60:
                matchup.momentum_factor = 1.1  # Good form
            elif avg_recent < 30:
                matchup.momentum_factor = 0.9  # Poor form
        
        return matchup
    
    def analyze_team_matchup(self, team1_players: List[Dict[str, Any]], 
                           team2_players: List[Dict[str, Any]],
                           team1_name: str, team2_name: str,
                           match_context: Dict[str, Any]) -> TeamMatchup:
        """Analyze team vs team matchup"""
        
        matchup = TeamMatchup(team1_name=team1_name, team2_name=team2_name)
        
        # Analyze historical head-to-head
        matchup = self._analyze_team_head_to_head(matchup, team1_name, team2_name)
        
        # Analyze strength differences
        matchup = self._analyze_strength_differences(matchup, team1_players, team2_players)
        
        # Analyze tactical matchups
        matchup = self._analyze_tactical_matchups(matchup, team1_players, team2_players)
        
        return matchup
    
    def _analyze_team_head_to_head(self, matchup: TeamMatchup, 
                                 team1_name: str, team2_name: str) -> TeamMatchup:
        """Analyze historical team head-to-head"""
        
        # This would query a comprehensive database
        # For now, simulate based on team strength
        matchup.total_matches = np.random.randint(10, 50)
        matchup.team1_wins = np.random.randint(0, matchup.total_matches)
        matchup.team2_wins = matchup.total_matches - matchup.team1_wins
        
        # Recent results (last 5 encounters)
        recent_count = min(5, matchup.total_matches)
        for _ in range(recent_count):
            result = np.random.choice(['W', 'L'], p=[0.5, 0.5])
            matchup.recent_results.append(result)
            matchup.recent_margin.append(np.random.uniform(5, 50))
        
        return matchup
    
    def _analyze_strength_differences(self, matchup: TeamMatchup, 
                                    team1_players: List[Dict[str, Any]], 
                                    team2_players: List[Dict[str, Any]]) -> TeamMatchup:
        """Analyze team strength differences"""
        
        # Calculate batting strength
        team1_batting = self._calculate_team_batting_strength(team1_players)
        team2_batting = self._calculate_team_batting_strength(team2_players)
        matchup.batting_strength_diff = team1_batting - team2_batting
        
        # Calculate bowling strength
        team1_bowling = self._calculate_team_bowling_strength(team1_players)
        team2_bowling = self._calculate_team_bowling_strength(team2_players)
        matchup.bowling_strength_diff = team1_bowling - team2_bowling
        
        # Calculate fielding strength (simplified)
        team1_fielding = self._calculate_team_fielding_strength(team1_players)
        team2_fielding = self._calculate_team_fielding_strength(team2_players)
        matchup.fielding_strength_diff = team1_fielding - team2_fielding
        
        return matchup
    
    def _calculate_team_batting_strength(self, players: List[Dict[str, Any]]) -> float:
        """Calculate team batting strength"""
        batting_strength = 0
        
        for player in players:
            role = player.get('role', '').lower()
            if 'bat' in role or 'allrounder' in role or 'wk' in role:
                batting_stats = player.get('batting_stats', {})
                avg = batting_stats.get('average', 25)
                sr = batting_stats.get('strike_rate', 120)
                
                # Combine average and strike rate
                player_strength = (avg * 0.7) + (sr / 5)
                batting_strength += player_strength
        
        return batting_strength
    
    def _calculate_team_bowling_strength(self, players: List[Dict[str, Any]]) -> float:
        """Calculate team bowling strength"""
        bowling_strength = 0
        
        for player in players:
            role = player.get('role', '').lower()
            if 'bowl' in role or 'allrounder' in role:
                bowling_stats = player.get('bowling_stats', {})
                avg = bowling_stats.get('average', 35)
                sr = bowling_stats.get('strike_rate', 25)
                economy = bowling_stats.get('economy', 7.5)
                
                # Lower values are better for bowling
                player_strength = 50 - avg + (30 - sr) + (10 - economy)
                bowling_strength += max(0, player_strength)
        
        return bowling_strength
    
    def _calculate_team_fielding_strength(self, players: List[Dict[str, Any]]) -> float:
        """Calculate team fielding strength"""
        fielding_strength = 0
        
        for player in players:
            # Simulate fielding ability based on role and stats
            role = player.get('role', '').lower()
            
            if 'wk' in role:
                fielding_strength += 8  # Wicket-keepers are usually good fielders
            elif 'allrounder' in role:
                fielding_strength += 7  # All-rounders are usually athletic
            else:
                fielding_strength += 6  # Base fielding ability
        
        return fielding_strength
    
    def _analyze_tactical_matchups(self, matchup: TeamMatchup, 
                                 team1_players: List[Dict[str, Any]], 
                                 team2_players: List[Dict[str, Any]]) -> TeamMatchup:
        """Analyze tactical matchups between teams"""
        
        # Pace bowlers vs batsmen
        team1_pace_quality = self._get_pace_bowling_quality(team1_players)
        team2_batting_vs_pace = self._get_batting_vs_pace_quality(team2_players)
        matchup.pace_vs_batting = team1_pace_quality - team2_batting_vs_pace
        
        # Spin bowlers vs batsmen
        team1_spin_quality = self._get_spin_bowling_quality(team1_players)
        team2_batting_vs_spin = self._get_batting_vs_spin_quality(team2_players)
        matchup.spin_vs_batting = team1_spin_quality - team2_batting_vs_spin
        
        # Death bowling analysis
        matchup.death_bowling_advantage = self._analyze_death_bowling_matchup(team1_players, team2_players)
        
        # Powerplay analysis
        matchup.powerplay_advantage = self._analyze_powerplay_matchup(team1_players, team2_players)
        
        return matchup
    
    def _get_pace_bowling_quality(self, players: List[Dict[str, Any]]) -> float:
        """Get pace bowling quality score"""
        pace_bowlers = []
        
        for player in players:
            role = player.get('role', '').lower()
            if 'fast' in role or 'pace' in role or 'medium' in role:
                bowling_stats = player.get('bowling_stats', {})
                avg = bowling_stats.get('average', 35)
                pace_bowlers.append(35 - avg)  # Lower average is better
        
        return np.mean(pace_bowlers) if pace_bowlers else 0
    
    def _get_spin_bowling_quality(self, players: List[Dict[str, Any]]) -> float:
        """Get spin bowling quality score"""
        spin_bowlers = []
        
        for player in players:
            role = player.get('role', '').lower()
            if 'spin' in role:
                bowling_stats = player.get('bowling_stats', {})
                avg = bowling_stats.get('average', 35)
                spin_bowlers.append(35 - avg)  # Lower average is better
        
        return np.mean(spin_bowlers) if spin_bowlers else 0
    
    def _get_batting_vs_pace_quality(self, players: List[Dict[str, Any]]) -> float:
        """Get batting quality against pace"""
        batsmen = []
        
        for player in players:
            role = player.get('role', '').lower()
            if 'bat' in role or 'allrounder' in role or 'wk' in role:
                batting_stats = player.get('batting_stats', {})
                avg = batting_stats.get('average', 25)
                batsmen.append(avg)
        
        return np.mean(batsmen) if batsmen else 25
    
    def _get_batting_vs_spin_quality(self, players: List[Dict[str, Any]]) -> float:
        """Get batting quality against spin"""
        # Similar to pace, but could be different based on specific spin stats
        return self._get_batting_vs_pace_quality(players)
    
    def _analyze_death_bowling_matchup(self, team1_players: List[Dict[str, Any]], 
                                     team2_players: List[Dict[str, Any]]) -> float:
        """Analyze death bowling advantage"""
        # Simplified death bowling analysis
        team1_death_bowlers = [p for p in team1_players if 'fast' in p.get('role', '').lower()]
        team2_death_batsmen = [p for p in team2_players if 'bat' in p.get('role', '').lower()]
        
        if team1_death_bowlers and team2_death_batsmen:
            death_bowling_strength = len(team1_death_bowlers) * 2
            death_batting_strength = len(team2_death_batsmen) * 1.5
            return death_bowling_strength - death_batting_strength
        
        return 0
    
    def _analyze_powerplay_matchup(self, team1_players: List[Dict[str, Any]], 
                                 team2_players: List[Dict[str, Any]]) -> float:
        """Analyze powerplay advantage"""
        # Simplified powerplay analysis
        team1_pp_bowlers = [p for p in team1_players if 'bowl' in p.get('role', '').lower()]
        team2_pp_batsmen = [p for p in team2_players if 'opener' in p.get('role', '').lower()]
        
        if team1_pp_bowlers and team2_pp_batsmen:
            pp_bowling_strength = len(team1_pp_bowlers) * 1.5
            pp_batting_strength = len(team2_pp_batsmen) * 2
            return pp_bowling_strength - pp_batting_strength
        
        return 0
    
    def generate_matchup_insights(self, player_matchups: List[PlayerMatchup], 
                                team_matchup: TeamMatchup,
                                match_context: Dict[str, Any]) -> MatchupInsights:
        """Generate comprehensive matchup insights"""
        
        insights = MatchupInsights()
        
        # Identify key battles
        insights.key_battles = self._identify_key_battles(player_matchups, team_matchup)
        
        # Calculate tactical advantages
        insights.tactical_advantages = self._calculate_tactical_advantages(team_matchup)
        
        # Predict team scores
        insights.predicted_scores = self._predict_team_scores(player_matchups, team_matchup, match_context)
        
        # Identify risk factors
        insights.risk_factors = self._identify_risk_factors(player_matchups, team_matchup)
        
        # Identify opportunity factors
        insights.opportunity_factors = self._identify_opportunity_factors(player_matchups, team_matchup)
        
        return insights
    
    def _identify_key_battles(self, player_matchups: List[PlayerMatchup], 
                            team_matchup: TeamMatchup) -> List[Dict[str, Any]]:
        """Identify key player battles"""
        
        key_battles = []
        
        # Find top batsmen vs top bowlers matchups
        top_batsmen = [m for m in player_matchups if 'bat' in m.player_name.lower() or m.avg_score > 40]
        top_bowlers = [m for m in player_matchups if 'bowl' in m.player_name.lower() or m.wickets_per_match > 1]
        
        # Create some key battle scenarios
        if top_batsmen and top_bowlers:
            for i, batsman in enumerate(top_batsmen[:3]):
                if i < len(top_bowlers):
                    key_battles.append({
                        'type': 'batsman_vs_bowler',
                        'players': [batsman.player_name, top_bowlers[i].player_name],
                        'significance': 'high',
                        'description': f'{batsman.player_name} vs {top_bowlers[i].player_name}'
                    })
        
        return key_battles
    
    def _calculate_tactical_advantages(self, team_matchup: TeamMatchup) -> Dict[str, float]:
        """Calculate tactical advantages"""
        
        return {
            'pace_bowling_advantage': team_matchup.pace_vs_batting,
            'spin_bowling_advantage': team_matchup.spin_vs_batting,
            'death_bowling_advantage': team_matchup.death_bowling_advantage,
            'powerplay_advantage': team_matchup.powerplay_advantage,
            'overall_batting_advantage': team_matchup.batting_strength_diff,
            'overall_bowling_advantage': team_matchup.bowling_strength_diff
        }
    
    def _predict_team_scores(self, player_matchups: List[PlayerMatchup], 
                           team_matchup: TeamMatchup,
                           match_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict team scores based on matchups"""
        
        # Base score prediction
        base_score = 300  # T20 base
        
        # Adjust based on batting advantage
        team1_predicted = base_score + (team_matchup.batting_strength_diff * 2)
        team2_predicted = base_score - (team_matchup.batting_strength_diff * 2)
        
        # Adjust for bowling strength
        team1_predicted -= (team_matchup.bowling_strength_diff * 1.5)
        team2_predicted += (team_matchup.bowling_strength_diff * 1.5)
        
        return {
            team_matchup.team1_name: max(120, min(400, team1_predicted)),
            team_matchup.team2_name: max(120, min(400, team2_predicted))
        }
    
    def _identify_risk_factors(self, player_matchups: List[PlayerMatchup], 
                             team_matchup: TeamMatchup) -> List[str]:
        """Identify risk factors"""
        
        risks = []
        
        # Check for players with poor recent form against opposition
        poor_form_players = [m for m in player_matchups if m.last_5_avg < 20 and m.matches_played >= 3]
        if poor_form_players:
            risks.append(f"Poor recent form against opposition: {', '.join([m.player_name for m in poor_form_players])}")
        
        # Check for tactical disadvantages
        if team_matchup.pace_vs_batting < -5:
            risks.append("Significant pace bowling disadvantage")
        
        if team_matchup.spin_vs_batting < -5:
            risks.append("Significant spin bowling disadvantage")
        
        return risks
    
    def _identify_opportunity_factors(self, player_matchups: List[PlayerMatchup], 
                                    team_matchup: TeamMatchup) -> List[str]:
        """Identify opportunity factors"""
        
        opportunities = []
        
        # Check for players with excellent form against opposition
        good_form_players = [m for m in player_matchups if m.last_5_avg > 60 and m.matches_played >= 3]
        if good_form_players:
            opportunities.append(f"Excellent form against opposition: {', '.join([m.player_name for m in good_form_players])}")
        
        # Check for tactical advantages
        if team_matchup.pace_vs_batting > 5:
            opportunities.append("Significant pace bowling advantage")
        
        if team_matchup.spin_vs_batting > 5:
            opportunities.append("Significant spin bowling advantage")
        
        return opportunities

# Export
__all__ = ['AdvancedMatchupAnalyzer', 'PlayerMatchup', 'TeamMatchup', 'MatchupInsights']