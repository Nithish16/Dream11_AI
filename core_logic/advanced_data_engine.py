#!/usr/bin/env python3
"""
Advanced Data Engine - Multi-Source Data Integration & Real-time Processing
Next-generation data pipeline with environmental intelligence and market data
"""

import asyncio
import aiohttp
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class EnvironmentalData:
    """Enhanced environmental conditions data"""
    temperature: float = 22.0
    humidity: float = 60.0
    wind_speed: float = 10.0
    wind_direction: str = "N"
    visibility: float = 10.0
    uv_index: float = 5.0
    air_quality_index: float = 50.0
    precipitation_probability: float = 0.0
    cloud_cover: float = 30.0
    dew_point: float = 15.0
    
    # Advanced metrics
    batting_favorability: float = 0.5  # 0-1 scale
    bowling_favorability: float = 0.5
    fielding_conditions: float = 0.5

@dataclass
class MarketIntelligence:
    """Market sentiment and betting intelligence"""
    player_ownership_percentage: float = 50.0
    betting_odds: float = 2.0
    market_sentiment: float = 0.0  # -1 to 1
    expert_consensus: float = 0.0
    social_media_buzz: float = 0.0
    injury_probability: float = 0.0
    form_momentum_market: float = 0.0

@dataclass
class PlayerIntelligence:
    """Comprehensive player intelligence profile"""
    player_id: int
    name: str
    
    # Core data
    performance_data: Dict[str, Any] = field(default_factory=dict)
    contextual_data: Dict[str, Any] = field(default_factory=dict)
    environmental_data: EnvironmentalData = field(default_factory=EnvironmentalData)
    market_data: MarketIntelligence = field(default_factory=MarketIntelligence)
    
    # Advanced metrics
    fatigue_index: float = 0.0
    travel_impact: float = 0.0
    psychological_state: float = 0.5
    injury_risk: float = 0.1
    captain_leadership_score: float = 0.5

class WeatherAPI:
    """Enhanced weather data integration"""
    
    def __init__(self):
        self.api_key = "demo_weather_key"  # Replace with actual API key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_weather_data(self, venue_lat: float, venue_lon: float, match_time: datetime) -> EnvironmentalData:
        """Fetch comprehensive weather data"""
        try:
            # Current weather
            current_url = f"{self.base_url}/weather?lat={venue_lat}&lon={venue_lon}&appid={self.api_key}&units=metric"
            
            # Forecast (if match is in future)
            forecast_url = f"{self.base_url}/forecast?lat={venue_lat}&lon={venue_lon}&appid={self.api_key}&units=metric"
            
            # For demo, return simulated data
            return self._generate_simulated_weather_data(venue_lat, venue_lon)
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return EnvironmentalData()
    
    def _generate_simulated_weather_data(self, lat: float, lon: float) -> EnvironmentalData:
        """Generate realistic weather simulation"""
        # Simulate based on location
        base_temp = 25.0 + (lat / 10)  # Rough temperature based on latitude
        
        return EnvironmentalData(
            temperature=base_temp + np.random.normal(0, 3),
            humidity=60 + np.random.normal(0, 15),
            wind_speed=10 + np.random.normal(0, 5),
            wind_direction=np.random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            visibility=9 + np.random.normal(0, 1),
            uv_index=5 + np.random.normal(0, 2),
            air_quality_index=50 + np.random.normal(0, 20),
            precipitation_probability=np.random.uniform(0, 0.3),
            cloud_cover=np.random.uniform(10, 70),
            dew_point=base_temp - 10 + np.random.normal(0, 2),
            batting_favorability=self._calculate_batting_favorability(base_temp, 60, 10),
            bowling_favorability=self._calculate_bowling_favorability(base_temp, 60, 10),
            fielding_conditions=self._calculate_fielding_conditions(10, 9)
        )
    
    def _calculate_batting_favorability(self, temp: float, humidity: float, wind: float) -> float:
        """Calculate how favorable conditions are for batting"""
        # Optimal batting: 22-28°C, 40-60% humidity, light wind
        temp_score = 1.0 - abs(temp - 25) / 15  # Peak at 25°C
        humidity_score = 1.0 - abs(humidity - 50) / 50  # Peak at 50%
        wind_score = max(0, 1.0 - wind / 20)  # Less wind is better
        
        return max(0, min(1, (temp_score + humidity_score + wind_score) / 3))
    
    def _calculate_bowling_favorability(self, temp: float, humidity: float, wind: float) -> float:
        """Calculate how favorable conditions are for bowling"""
        # Favorable for bowling: cooler temps, higher humidity, some wind
        temp_score = max(0, 1.0 - (temp - 15) / 20)  # Cooler is better
        humidity_score = humidity / 100  # Higher humidity helps swing
        wind_score = min(1, wind / 15)  # Some wind helps
        
        return max(0, min(1, (temp_score + humidity_score + wind_score) / 3))
    
    def _calculate_fielding_conditions(self, wind: float, visibility: float) -> float:
        """Calculate fielding condition quality"""
        wind_score = 1.0 - min(wind / 25, 1)  # Too much wind is bad
        visibility_score = visibility / 10  # Better visibility is good
        
        return max(0, min(1, (wind_score + visibility_score) / 2))

class MarketDataAPI:
    """Market intelligence and sentiment analysis"""
    
    def __init__(self):
        self.betting_apis = ["demo_betting_api"]
        self.social_apis = ["demo_social_api"]
    
    def get_market_intelligence(self, player_id: int, player_name: str) -> MarketIntelligence:
        """Fetch comprehensive market data"""
        try:
            # Simulate market data
            return self._generate_simulated_market_data(player_id, player_name)
        except Exception as e:
            print(f"Error fetching market data for {player_name}: {e}")
            return MarketIntelligence()
    
    def _generate_simulated_market_data(self, player_id: int, player_name: str) -> MarketIntelligence:
        """Generate realistic market simulation"""
        np.random.seed(player_id)  # Consistent randomness per player
        
        # Simulate ownership based on player popularity
        base_ownership = 30 + np.random.exponential(20)
        base_ownership = min(95, max(5, base_ownership))
        
        return MarketIntelligence(
            player_ownership_percentage=base_ownership,
            betting_odds=1.5 + np.random.exponential(1),
            market_sentiment=np.random.normal(0, 0.3),
            expert_consensus=np.random.normal(0, 0.2),
            social_media_buzz=np.random.uniform(-0.5, 0.5),
            injury_probability=np.random.beta(1, 10),  # Low probability
            form_momentum_market=np.random.normal(0, 0.25)
        )

class AdvancedDataEngine:
    """Next-generation multi-source data integration engine"""
    
    def __init__(self):
        self.weather_api = WeatherAPI()
        self.market_api = MarketDataAPI()
        self.data_fusion_weights = {
            'performance': 0.4,
            'contextual': 0.25,
            'environmental': 0.2,
            'market': 0.15
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'minimum_matches': 5,
            'data_freshness_hours': 24,
            'confidence_threshold': 0.7
        }
    
    async def create_comprehensive_player_profile(self, player_data: Dict[str, Any], 
                                                venue_coordinates: Tuple[float, float],
                                                match_time: datetime) -> PlayerIntelligence:
        """Create comprehensive player intelligence profile"""
        
        player_id = player_data.get('player_id', 0)
        player_name = player_data.get('name', 'Unknown')
        
        # Gather data from all sources concurrently
        tasks = [
            self._get_performance_data(player_data),
            self._get_contextual_data(player_data),
            self._get_environmental_data(venue_coordinates, match_time),
            self._get_market_data(player_id, player_name)
        ]
        
        try:
            performance_data, contextual_data, environmental_data, market_data = await asyncio.gather(*tasks)
            
            # Calculate advanced metrics
            fatigue_index = self._calculate_fatigue_index(performance_data)
            travel_impact = self._calculate_travel_impact(contextual_data)
            psychological_state = self._calculate_psychological_state(market_data, performance_data)
            injury_risk = self._calculate_injury_risk(performance_data, market_data)
            leadership_score = self._calculate_leadership_score(player_data, performance_data)
            
            return PlayerIntelligence(
                player_id=player_id,
                name=player_name,
                performance_data=performance_data,
                contextual_data=contextual_data,
                environmental_data=environmental_data,
                market_data=market_data,
                fatigue_index=fatigue_index,
                travel_impact=travel_impact,
                psychological_state=psychological_state,
                injury_risk=injury_risk,
                captain_leadership_score=leadership_score
            )
            
        except Exception as e:
            print(f"Error creating profile for {player_name}: {e}")
            return PlayerIntelligence(player_id=player_id, name=player_name)
    
    async def _get_performance_data(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced performance data extraction"""
        # Simulate async operation
        await asyncio.sleep(0.01)
        
        career_stats = player_data.get('career_stats', {})
        batting_stats = player_data.get('batting_stats', {})
        bowling_stats = player_data.get('bowling_stats', {})
        
        return {
            'career_stats': career_stats,
            'batting_stats': batting_stats,
            'bowling_stats': bowling_stats,
            'recent_form': self._extract_recent_form(career_stats),
            'venue_specific': self._extract_venue_stats(career_stats),
            'opposition_specific': self._extract_opposition_stats(career_stats)
        }
    
    async def _get_contextual_data(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced contextual data extraction"""
        await asyncio.sleep(0.01)
        
        return {
            'role': player_data.get('role', 'Unknown'),
            'team_name': player_data.get('team_name', 'Unknown'),
            'batting_position': self._estimate_batting_position(player_data),
            'bowling_type': self._identify_bowling_type(player_data),
            'captaincy_experience': self._assess_captaincy_experience(player_data),
            'tournament_experience': self._assess_tournament_experience(player_data)
        }
    
    async def _get_environmental_data(self, venue_coordinates: Tuple[float, float], 
                                    match_time: datetime) -> EnvironmentalData:
        """Get real-time environmental data"""
        await asyncio.sleep(0.01)
        
        if venue_coordinates:
            lat, lon = venue_coordinates
            return self.weather_api.get_weather_data(lat, lon, match_time)
        else:
            return EnvironmentalData()
    
    async def _get_market_data(self, player_id: int, player_name: str) -> MarketIntelligence:
        """Get real-time market intelligence"""
        await asyncio.sleep(0.01)
        
        return self.market_api.get_market_intelligence(player_id, player_name)
    
    def _extract_recent_form(self, career_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract recent form with enhanced analysis"""
        recent_matches = career_stats.get('recentMatches', [])
        
        form_data = []
        for match in recent_matches[-10:]:  # Last 10 matches
            # Calculate comprehensive form score
            runs = match.get('runs', 0)
            wickets = match.get('wickets', 0)
            catches = match.get('catches', 0)
            
            # Enhanced scoring
            fantasy_points = runs + (wickets * 25) + (catches * 8)
            
            # Add bonus points for milestones
            if runs >= 50:
                fantasy_points += 8
            if runs >= 100:
                fantasy_points += 16
            if wickets >= 3:
                fantasy_points += 12
            if wickets >= 5:
                fantasy_points += 24
            
            form_data.append({
                'match_id': match.get('matchId'),
                'runs': runs,
                'wickets': wickets,
                'catches': catches,
                'fantasy_points': fantasy_points,
                'date': match.get('date'),
                'opposition': match.get('opposition', 'Unknown')
            })
        
        return form_data
    
    def _extract_venue_stats(self, career_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract venue-specific performance"""
        # Placeholder - would analyze performance at specific venues
        return {
            'home_average': 45.0,
            'away_average': 38.0,
            'neutral_average': 42.0,
            'venue_familiarity': 0.6
        }
    
    def _extract_opposition_stats(self, career_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract opposition-specific performance"""
        # Placeholder - would analyze performance against specific teams
        return {
            'vs_pace_average': 42.0,
            'vs_spin_average': 48.0,
            'vs_top_teams': 40.0,
            'vs_similar_teams': 45.0
        }
    
    def _estimate_batting_position(self, player_data: Dict[str, Any]) -> int:
        """Estimate likely batting position"""
        role = player_data.get('role', '').lower()
        
        if 'opener' in role:
            return 1
        elif 'top' in role or 'batsman' in role:
            return 3
        elif 'middle' in role:
            return 5
        elif 'allrounder' in role:
            return 6
        elif 'wicket' in role or 'wk' in role:
            return 4
        else:
            return 7
    
    def _identify_bowling_type(self, player_data: Dict[str, Any]) -> str:
        """Identify bowling type from role"""
        role = player_data.get('role', '').lower()
        
        if 'fast' in role or 'pace' in role:
            return 'pace'
        elif 'spin' in role or 'leg' in role or 'off' in role:
            return 'spin'
        elif 'medium' in role:
            return 'medium_pace'
        else:
            return 'unknown'
    
    def _assess_captaincy_experience(self, player_data: Dict[str, Any]) -> float:
        """Assess captaincy experience level"""
        # Placeholder - would analyze captaincy records
        player_name = player_data.get('name', '').lower()
        
        # Common captains get higher scores
        if any(name in player_name for name in ['kohli', 'sharma', 'dhoni', 'smith', 'root']):
            return 0.9
        elif any(name in player_name for name in ['rahul', 'pant', 'pandya']):
            return 0.7
        else:
            return 0.3
    
    def _assess_tournament_experience(self, player_data: Dict[str, Any]) -> float:
        """Assess tournament experience level"""
        # Placeholder - would analyze tournament participation
        return 0.8  # Default high experience
    
    def _calculate_fatigue_index(self, performance_data: Dict[str, Any]) -> float:
        """Calculate player fatigue based on recent match density"""
        recent_form = performance_data.get('recent_form', [])
        
        # Calculate matches in last 30 days
        now = datetime.now()
        recent_matches = []
        
        for match in recent_form:
            match_date = match.get('date')
            if match_date:
                try:
                    if isinstance(match_date, str):
                        match_datetime = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                    else:
                        match_datetime = match_date
                    
                    days_ago = (now - match_datetime).days
                    if days_ago <= 30:
                        recent_matches.append((days_ago, match))
                except:
                    continue
        
        # Calculate fatigue based on match density
        if len(recent_matches) <= 2:
            return 0.0  # Well rested
        elif len(recent_matches) <= 4:
            return 0.3  # Moderate activity
        elif len(recent_matches) <= 6:
            return 0.6  # High activity
        else:
            return 0.9  # Very high fatigue risk
    
    def _calculate_travel_impact(self, contextual_data: Dict[str, Any]) -> float:
        """Calculate travel fatigue impact"""
        # Placeholder - would consider travel distance and time zones
        return np.random.uniform(0, 0.3)  # Random travel impact
    
    def _calculate_psychological_state(self, market_data: MarketIntelligence, 
                                     performance_data: Dict[str, Any]) -> float:
        """Calculate psychological state based on market sentiment and form"""
        
        # Market pressure factor
        ownership = market_data.player_ownership_percentage
        pressure_factor = min(ownership / 100, 1.0)  # High ownership = high pressure
        
        # Recent form factor
        recent_form = performance_data.get('recent_form', [])
        if len(recent_form) >= 3:
            recent_scores = [match.get('fantasy_points', 0) for match in recent_form[:3]]
            avg_recent = sum(recent_scores) / len(recent_scores)
            form_factor = min(avg_recent / 80, 1.0)  # Normalize to 0-1
        else:
            form_factor = 0.5
        
        # Social media buzz impact
        buzz_impact = abs(market_data.social_media_buzz) * 0.2
        
        # Combine factors (0 = poor state, 1 = excellent state)
        psychological_state = (form_factor * 0.6) - (pressure_factor * 0.3) - (buzz_impact * 0.1)
        
        return max(0, min(1, psychological_state))
    
    def _calculate_injury_risk(self, performance_data: Dict[str, Any], 
                             market_data: MarketIntelligence) -> float:
        """Calculate injury risk assessment"""
        
        # Base injury risk from market
        market_risk = market_data.injury_probability
        
        # Performance decline indicator
        recent_form = performance_data.get('recent_form', [])
        if len(recent_form) >= 5:
            recent_scores = [match.get('fantasy_points', 0) for match in recent_form[:5]]
            early_scores = [match.get('fantasy_points', 0) for match in recent_form[5:10]]
            
            if early_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                early_avg = sum(early_scores) / len(early_scores)
                
                # Significant decline might indicate injury
                if early_avg > 0 and recent_avg < early_avg * 0.7:
                    performance_risk = 0.3
                else:
                    performance_risk = 0.1
            else:
                performance_risk = 0.1
        else:
            performance_risk = 0.1
        
        # Combine risks
        total_risk = min(1.0, market_risk + performance_risk)
        
        return total_risk
    
    def _calculate_leadership_score(self, player_data: Dict[str, Any], 
                                  performance_data: Dict[str, Any]) -> float:
        """Calculate leadership/captaincy potential score"""
        
        # Base score from experience
        captaincy_exp = self._assess_captaincy_experience(player_data)
        tournament_exp = self._assess_tournament_experience(player_data)
        
        # Performance consistency
        recent_form = performance_data.get('recent_form', [])
        if len(recent_form) >= 5:
            scores = [match.get('fantasy_points', 0) for match in recent_form[:5]]
            if scores:
                consistency = 1 - (np.std(scores) / (np.mean(scores) + 1))
                consistency = max(0, min(1, consistency))
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        
        # Role factor - all-rounders often make good captains
        role = player_data.get('role', '').lower()
        if 'allrounder' in role:
            role_factor = 1.2
        elif 'batsman' in role:
            role_factor = 1.1
        elif 'wicket' in role:
            role_factor = 1.05
        else:
            role_factor = 1.0
        
        # Combine factors
        leadership_score = (captaincy_exp * 0.4 + tournament_exp * 0.3 + consistency * 0.3) * role_factor
        
        return max(0, min(1, leadership_score))

# Utility functions for coordinate lookup
def get_venue_coordinates(venue_name: str) -> Tuple[float, float]:
    """Get approximate coordinates for major cricket venues"""
    venue_coords = {
        'lords': (51.5294, -0.1726),
        'oval': (51.4816, -0.1148),
        'old trafford': (53.4568, -2.2913),
        'headingley': (53.8171, -1.5821),
        'edgbaston': (52.4559, -1.9021),
        'melbourne cricket ground': (-37.8200, 144.9834),
        'sydney cricket ground': (-33.8915, 151.2243),
        'adelaide oval': (-34.9159, 138.5959),
        'perth stadium': (-31.9505, 115.8605),
        'gabba': (-27.4848, 153.0388),
        'eden gardens': (22.5645, 88.3433),
        'wankhede': (19.0448, 72.8454),
        'chepauk': (13.0642, 80.2799),
        'chinnaswamy': (12.9784, 77.5996),
        'feroz shah kotla': (28.6385, 77.2408),
        'rajiv gandhi': (17.4030, 78.3489),
        'sawai mansingh': (26.8913, 75.8062),
        'green park': (26.4560, 80.3232),
        'brabourne': (19.0330, 72.8249),
        'centurion': (-25.7477, 28.2293),
        'wanderers': (-26.1965, 28.0436),
        'newlands': (-33.9716, 18.4675),
        'kingsmead': (-29.8292, 31.0219),
        'st georges park': (-33.8765, 25.6057)
    }
    
    venue_lower = venue_name.lower()
    for venue_key, coords in venue_coords.items():
        if venue_key in venue_lower:
            return coords
    
    # Default coordinates (London)
    return (51.5074, -0.1278)

# Export the main engine
__all__ = ['AdvancedDataEngine', 'PlayerIntelligence', 'EnvironmentalData', 'MarketIntelligence', 'get_venue_coordinates']