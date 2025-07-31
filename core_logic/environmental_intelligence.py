#!/usr/bin/env python3
"""
Environmental Intelligence Engine - Real-time Match Context Analysis
Advanced environmental factors integration for performance prediction
"""

import numpy as np
import pandas as pd
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class WeatherConditions:
    """Comprehensive weather conditions"""
    temperature: float = 25.0
    humidity: float = 60.0
    wind_speed: float = 10.0
    wind_direction: str = "N"
    visibility: float = 10.0
    precipitation: float = 0.0
    cloud_cover: float = 30.0
    dew_point: float = 15.0
    pressure: float = 1013.25
    uv_index: float = 5.0

@dataclass
class PitchConditions:
    """Pitch and ground conditions"""
    pitch_type: str = "balanced"
    grass_coverage: float = 0.7
    moisture_content: float = 0.4
    hardness_index: float = 0.6
    crack_density: float = 0.2
    wear_factor: float = 0.3
    bounce_consistency: float = 0.8
    turn_assistance: float = 0.5

@dataclass
class EnvironmentalContext:
    """Complete environmental context"""
    weather: WeatherConditions = field(default_factory=WeatherConditions)
    pitch: PitchConditions = field(default_factory=PitchConditions)
    venue_altitude: float = 100.0
    air_density: float = 1.225
    lighting_conditions: str = "daylight"
    match_timing: str = "afternoon"
    seasonal_factor: float = 0.5
    crowd_factor: float = 0.7

@dataclass
class PerformanceImpact:
    """Environmental performance impact scores"""
    batting_advantage: float = 0.5
    pace_bowling_advantage: float = 0.5
    spin_bowling_advantage: float = 0.5
    fielding_difficulty: float = 0.5
    
    # Specific impacts
    boundary_scoring_ease: float = 0.5
    swing_bowling_assistance: float = 0.5
    reverse_swing_likelihood: float = 0.3
    dew_factor_impact: float = 0.0

class EnvironmentalIntelligence:
    """Advanced environmental analysis for cricket performance prediction"""
    
    def __init__(self):
        self.venue_database = self._initialize_venue_database()
        self.weather_weights = {
            'temperature': 0.25,
            'humidity': 0.20,
            'wind': 0.15,
            'pressure': 0.10,
            'visibility': 0.10,
            'precipitation': 0.20
        }
        
        self.pitch_impact_matrix = self._initialize_pitch_impact_matrix()
        
    def _initialize_venue_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive venue database"""
        return {
            'melbourne_cricket_ground': {
                'coordinates': (-37.8200, 144.9834),
                'altitude': 35,
                'typical_conditions': {
                    'temperature_range': (15, 30),
                    'humidity_range': (40, 80),
                    'wind_factor': 'moderate',
                    'pitch_type': 'batting_friendly'
                },
                'historical_factors': {
                    'avg_first_innings': 320,
                    'batting_advantage': 0.7,
                    'pace_advantage': 0.6,
                    'spin_advantage': 0.4
                }
            },
            'lords': {
                'coordinates': (51.5294, -0.1726),
                'altitude': 69,
                'typical_conditions': {
                    'temperature_range': (12, 25),
                    'humidity_range': (60, 90),
                    'wind_factor': 'variable',
                    'pitch_type': 'seamer_friendly'
                },
                'historical_factors': {
                    'avg_first_innings': 280,
                    'batting_advantage': 0.5,
                    'pace_advantage': 0.8,
                    'spin_advantage': 0.3
                }
            },
            'wankhede': {
                'coordinates': (19.0448, 72.8454),
                'altitude': 14,
                'typical_conditions': {
                    'temperature_range': (20, 35),
                    'humidity_range': (70, 95),
                    'wind_factor': 'sea_breeze',
                    'pitch_type': 'batting_friendly'
                },
                'historical_factors': {
                    'avg_first_innings': 340,
                    'batting_advantage': 0.8,
                    'pace_advantage': 0.4,
                    'spin_advantage': 0.7
                }
            },
            'eden_gardens': {
                'coordinates': (22.5645, 88.3433),
                'altitude': 9,
                'typical_conditions': {
                    'temperature_range': (18, 38),
                    'humidity_range': (65, 90),
                    'wind_factor': 'low',
                    'pitch_type': 'spin_friendly'
                },
                'historical_factors': {
                    'avg_first_innings': 300,
                    'batting_advantage': 0.6,
                    'pace_advantage': 0.3,
                    'spin_advantage': 0.9
                }
            }
        }
    
    def _initialize_pitch_impact_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize pitch type impact matrix"""
        return {
            'green_top': {
                'batting_difficulty': 0.8,
                'pace_assistance': 0.9,
                'spin_assistance': 0.2,
                'bounce_predictability': 0.7
            },
            'dusty': {
                'batting_difficulty': 0.6,
                'pace_assistance': 0.3,
                'spin_assistance': 0.9,
                'bounce_predictability': 0.5
            },
            'flat_batting': {
                'batting_difficulty': 0.2,
                'pace_assistance': 0.3,
                'spin_assistance': 0.4,
                'bounce_predictability': 0.9
            },
            'balanced': {
                'batting_difficulty': 0.5,
                'pace_assistance': 0.6,
                'spin_assistance': 0.6,
                'bounce_predictability': 0.8
            }
        }
    
    async def analyze_environmental_context(self, venue_name: str, 
                                          match_datetime: datetime,
                                          weather_data: Optional[Dict[str, Any]] = None) -> EnvironmentalContext:
        """Analyze complete environmental context for the match"""
        
        # Get venue information
        venue_info = self._get_venue_info(venue_name)
        
        # Get weather conditions
        if weather_data:
            weather = self._parse_weather_data(weather_data)
        else:
            weather = await self._fetch_weather_data(venue_info['coordinates'], match_datetime)
        
        # Analyze pitch conditions
        pitch = self._analyze_pitch_conditions(venue_name, weather, match_datetime)
        
        # Calculate environmental factors
        altitude = venue_info.get('altitude', 100.0)
        air_density = self._calculate_air_density(weather.temperature, weather.pressure, altitude)
        
        # Determine match timing factors
        lighting_conditions = self._determine_lighting_conditions(match_datetime)
        match_timing = self._determine_match_timing(match_datetime)
        seasonal_factor = self._calculate_seasonal_factor(match_datetime, venue_info['coordinates'])
        crowd_factor = self._estimate_crowd_factor(venue_name, match_datetime)
        
        return EnvironmentalContext(
            weather=weather,
            pitch=pitch,
            venue_altitude=altitude,
            air_density=air_density,
            lighting_conditions=lighting_conditions,
            match_timing=match_timing,
            seasonal_factor=seasonal_factor,
            crowd_factor=crowd_factor
        )
    
    def _get_venue_info(self, venue_name: str) -> Dict[str, Any]:
        """Get venue information from database"""
        venue_key = venue_name.lower().replace(' ', '_')
        
        # Try exact match first
        if venue_key in self.venue_database:
            return self.venue_database[venue_key]
        
        # Try partial match
        for key, info in self.venue_database.items():
            if any(word in venue_key for word in key.split('_')):
                return info
        
        # Default venue info
        return {
            'coordinates': (0.0, 0.0),
            'altitude': 100,
            'typical_conditions': {
                'temperature_range': (15, 30),
                'humidity_range': (50, 80),
                'wind_factor': 'moderate',
                'pitch_type': 'balanced'
            },
            'historical_factors': {
                'avg_first_innings': 300,
                'batting_advantage': 0.5,
                'pace_advantage': 0.5,
                'spin_advantage': 0.5
            }
        }
    
    async def _fetch_weather_data(self, coordinates: Tuple[float, float], 
                                match_datetime: datetime) -> WeatherConditions:
        """Fetch real-time weather data"""
        try:
            # In production, integrate with weather API
            # For now, simulate realistic weather data
            lat, lon = coordinates
            
            # Simulate based on location and season
            return self._simulate_weather_conditions(lat, lon, match_datetime)
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return WeatherConditions()
    
    def _parse_weather_data(self, weather_data: Dict[str, Any]) -> WeatherConditions:
        """Parse weather data from API response"""
        return WeatherConditions(
            temperature=weather_data.get('temperature', 25.0),
            humidity=weather_data.get('humidity', 60.0),
            wind_speed=weather_data.get('wind_speed', 10.0),
            wind_direction=weather_data.get('wind_direction', 'N'),
            visibility=weather_data.get('visibility', 10.0),
            precipitation=weather_data.get('precipitation', 0.0),
            cloud_cover=weather_data.get('cloud_cover', 30.0),
            dew_point=weather_data.get('dew_point', 15.0),
            pressure=weather_data.get('pressure', 1013.25),
            uv_index=weather_data.get('uv_index', 5.0)
        )
    
    def _simulate_weather_conditions(self, lat: float, lon: float, 
                                   match_datetime: datetime) -> WeatherConditions:
        """Simulate realistic weather conditions"""
        
        # Base temperature based on latitude and season
        month = match_datetime.month
        season_factor = np.sin((month - 3) * np.pi / 6)  # Peak in summer
        
        if abs(lat) > 23.5:  # Outside tropics
            base_temp = 20 + 15 * season_factor - abs(lat - 23.5) * 0.3
        else:  # Tropics
            base_temp = 28 + 5 * season_factor
        
        # Add daily variation
        hour = match_datetime.hour
        daily_variation = 5 * np.sin((hour - 6) * np.pi / 12)
        
        temperature = base_temp + daily_variation + np.random.normal(0, 2)
        
        # Humidity inversely related to temperature in many climates
        humidity = max(30, min(95, 80 - (temperature - 20) * 1.5 + np.random.normal(0, 10)))
        
        # Wind speed - coastal areas tend to be windier
        if abs(lon) > 50:  # Coastal approximation
            wind_base = 15
        else:
            wind_base = 8
        
        wind_speed = max(0, wind_base + np.random.normal(0, 5))
        
        # Pressure variation
        pressure = 1013.25 + np.random.normal(0, 10)
        
        # Cloud cover affects visibility and UV
        cloud_cover = max(0, min(100, np.random.exponential(30)))
        visibility = max(2, 10 - cloud_cover * 0.05)
        uv_index = max(0, 8 - cloud_cover * 0.08)
        
        # Precipitation probability
        if cloud_cover > 70:
            precipitation = np.random.exponential(5)
        else:
            precipitation = 0
        
        return WeatherConditions(
            temperature=round(temperature, 1),
            humidity=round(humidity, 1),
            wind_speed=round(wind_speed, 1),
            wind_direction=np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            visibility=round(visibility, 1),
            precipitation=round(precipitation, 1),
            cloud_cover=round(cloud_cover, 1),
            dew_point=round(temperature - (100 - humidity) / 5, 1),
            pressure=round(pressure, 2),
            uv_index=round(uv_index, 1)
        )
    
    def _analyze_pitch_conditions(self, venue_name: str, weather: WeatherConditions, 
                                match_datetime: datetime) -> PitchConditions:
        """Analyze pitch conditions based on venue and weather"""
        
        venue_info = self._get_venue_info(venue_name)
        typical_pitch = venue_info['typical_conditions']['pitch_type']
        
        # Base pitch characteristics
        if typical_pitch == 'batting_friendly':
            base_pitch = PitchConditions(
                pitch_type='flat_batting',
                grass_coverage=0.3,
                moisture_content=0.2,
                hardness_index=0.8,
                crack_density=0.1,
                wear_factor=0.2,
                bounce_consistency=0.9,
                turn_assistance=0.3
            )
        elif typical_pitch == 'seamer_friendly':
            base_pitch = PitchConditions(
                pitch_type='green_top',
                grass_coverage=0.8,
                moisture_content=0.6,
                hardness_index=0.4,
                crack_density=0.2,
                wear_factor=0.3,
                bounce_consistency=0.7,
                turn_assistance=0.2
            )
        elif typical_pitch == 'spin_friendly':
            base_pitch = PitchConditions(
                pitch_type='dusty',
                grass_coverage=0.2,
                moisture_content=0.3,
                hardness_index=0.6,
                crack_density=0.7,
                wear_factor=0.5,
                bounce_consistency=0.5,
                turn_assistance=0.9
            )
        else:
            base_pitch = PitchConditions()
        
        # Adjust for weather conditions
        if weather.precipitation > 5:
            base_pitch.moisture_content = min(1.0, base_pitch.moisture_content + 0.3)
            base_pitch.hardness_index *= 0.7
            base_pitch.bounce_consistency *= 0.8
        
        if weather.humidity > 80:
            base_pitch.moisture_content = min(1.0, base_pitch.moisture_content + 0.2)
        
        if weather.temperature > 35:
            base_pitch.crack_density = min(1.0, base_pitch.crack_density + 0.2)
            base_pitch.hardness_index = min(1.0, base_pitch.hardness_index + 0.1)
        
        return base_pitch
    
    def _calculate_air_density(self, temperature: float, pressure: float, altitude: float) -> float:
        """Calculate air density for ball trajectory analysis"""
        # Standard atmospheric calculation
        temp_kelvin = temperature + 273.15
        pressure_pa = pressure * 100  # Convert hPa to Pa
        
        # Ideal gas law: ρ = P / (R * T)
        gas_constant = 287.05  # J/(kg·K) for dry air
        density = pressure_pa / (gas_constant * temp_kelvin)
        
        # Adjust for altitude
        altitude_factor = np.exp(-altitude / 8400)  # Scale height ~8.4km
        
        return density * altitude_factor
    
    def _determine_lighting_conditions(self, match_datetime: datetime) -> str:
        """Determine lighting conditions"""
        hour = match_datetime.hour
        
        if 6 <= hour < 18:
            return 'daylight'
        elif 18 <= hour < 20:
            return 'twilight'
        else:
            return 'floodlights'
    
    def _determine_match_timing(self, match_datetime: datetime) -> str:
        """Determine match timing category"""
        hour = match_datetime.hour
        
        if 9 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 20:
            return 'evening'
        else:
            return 'night'
    
    def _calculate_seasonal_factor(self, match_datetime: datetime, coordinates: Tuple[float, float]) -> float:
        """Calculate seasonal impact factor"""
        lat, _ = coordinates
        month = match_datetime.month
        
        # Northern hemisphere
        if lat > 0:
            if 4 <= month <= 9:  # Summer season
                return 0.8  # Generally better playing conditions
            else:  # Winter season
                return 0.4  # Challenging conditions
        else:  # Southern hemisphere
            if 10 <= month or month <= 3:  # Summer season
                return 0.8
            else:  # Winter season
                return 0.4
    
    def _estimate_crowd_factor(self, venue_name: str, match_datetime: datetime) -> float:
        """Estimate crowd impact factor"""
        # Weekend matches typically have larger crowds
        if match_datetime.weekday() >= 5:  # Saturday or Sunday
            base_crowd = 0.8
        else:
            base_crowd = 0.5
        
        # Major venues have higher crowd factors
        major_venues = ['lords', 'melbourne', 'wankhede', 'eden_gardens']
        if any(venue in venue_name.lower() for venue in major_venues):
            base_crowd += 0.2
        
        return min(1.0, base_crowd)
    
    def calculate_performance_impact(self, env_context: EnvironmentalContext, 
                                   player_role: str) -> PerformanceImpact:
        """Calculate environmental impact on player performance"""
        
        weather = env_context.weather
        pitch = env_context.pitch
        
        # Initialize impact scores
        impact = PerformanceImpact()
        
        # Batting advantage calculation
        impact.batting_advantage = self._calculate_batting_advantage(weather, pitch)
        
        # Bowling advantages
        if 'pace' in player_role.lower() or 'fast' in player_role.lower():
            impact.pace_bowling_advantage = self._calculate_pace_bowling_advantage(weather, pitch)
        
        if 'spin' in player_role.lower():
            impact.spin_bowling_advantage = self._calculate_spin_bowling_advantage(weather, pitch)
        
        # Fielding difficulty
        impact.fielding_difficulty = self._calculate_fielding_difficulty(weather, env_context)
        
        # Specific impacts
        impact.boundary_scoring_ease = self._calculate_boundary_ease(weather, env_context)
        impact.swing_bowling_assistance = self._calculate_swing_assistance(weather, pitch)
        impact.reverse_swing_likelihood = self._calculate_reverse_swing_likelihood(weather, pitch)
        impact.dew_factor_impact = self._calculate_dew_factor(weather, env_context)
        
        return impact
    
    def _calculate_batting_advantage(self, weather: WeatherConditions, pitch: PitchConditions) -> float:
        """Calculate batting advantage score"""
        
        # Temperature factor (optimal 20-30°C)
        temp_factor = 1.0 - abs(weather.temperature - 25) / 25
        temp_factor = max(0, temp_factor)
        
        # Wind factor (less wind better for batting)
        wind_factor = max(0, 1.0 - weather.wind_speed / 30)
        
        # Visibility factor
        visibility_factor = min(1.0, weather.visibility / 10)
        
        # Pitch factor
        pitch_factor = 1.0 - self.pitch_impact_matrix.get(pitch.pitch_type, {}).get('batting_difficulty', 0.5)
        
        # Combine factors
        batting_advantage = (temp_factor * 0.3 + wind_factor * 0.2 + 
                           visibility_factor * 0.2 + pitch_factor * 0.3)
        
        return max(0, min(1, batting_advantage))
    
    def _calculate_pace_bowling_advantage(self, weather: WeatherConditions, pitch: PitchConditions) -> float:
        """Calculate pace bowling advantage"""
        
        # Cloud cover helps swing
        cloud_factor = weather.cloud_cover / 100
        
        # Humidity helps swing
        humidity_factor = min(1.0, weather.humidity / 80)
        
        # Cooler temperatures help pace bowling
        temp_factor = max(0, 1.0 - (weather.temperature - 15) / 25)
        
        # Pitch assistance
        pitch_factor = self.pitch_impact_matrix.get(pitch.pitch_type, {}).get('pace_assistance', 0.5)
        
        pace_advantage = (cloud_factor * 0.25 + humidity_factor * 0.25 + 
                         temp_factor * 0.25 + pitch_factor * 0.25)
        
        return max(0, min(1, pace_advantage))
    
    def _calculate_spin_bowling_advantage(self, weather: WeatherConditions, pitch: PitchConditions) -> float:
        """Calculate spin bowling advantage"""
        
        # Dry conditions help spin
        humidity_factor = max(0, 1.0 - weather.humidity / 100)
        
        # Warmer temperatures help spin
        temp_factor = min(1.0, (weather.temperature - 15) / 25)
        
        # Pitch assistance
        pitch_factor = self.pitch_impact_matrix.get(pitch.pitch_type, {}).get('spin_assistance', 0.5)
        
        # Worn pitch helps spin
        wear_factor = pitch.wear_factor
        
        spin_advantage = (humidity_factor * 0.2 + temp_factor * 0.3 + 
                         pitch_factor * 0.3 + wear_factor * 0.2)
        
        return max(0, min(1, spin_advantage))
    
    def _calculate_fielding_difficulty(self, weather: WeatherConditions, 
                                     env_context: EnvironmentalContext) -> float:
        """Calculate fielding difficulty score"""
        
        # Wind makes fielding harder
        wind_factor = min(1.0, weather.wind_speed / 25)
        
        # Poor visibility makes fielding harder
        visibility_factor = max(0, 1.0 - weather.visibility / 10)
        
        # Precipitation makes fielding harder
        rain_factor = min(1.0, weather.precipitation / 10)
        
        # Lighting conditions
        lighting_factor = 0.0
        if env_context.lighting_conditions == 'twilight':
            lighting_factor = 0.3
        elif env_context.lighting_conditions == 'floodlights':
            lighting_factor = 0.2
        
        fielding_difficulty = (wind_factor * 0.3 + visibility_factor * 0.3 + 
                             rain_factor * 0.3 + lighting_factor * 0.1)
        
        return max(0, min(1, fielding_difficulty))
    
    def _calculate_boundary_ease(self, weather: WeatherConditions, 
                               env_context: EnvironmentalContext) -> float:
        """Calculate boundary scoring ease"""
        
        # Thin air (high altitude, hot weather) helps ball carry
        altitude_factor = min(1.0, env_context.venue_altitude / 1500)
        temp_factor = min(1.0, (weather.temperature - 15) / 25)
        
        # Tailwind helps boundaries
        wind_factor = min(1.0, weather.wind_speed / 20)
        
        # Air density factor
        standard_density = 1.225
        density_factor = max(0, 1.0 - env_context.air_density / standard_density)
        
        boundary_ease = (altitude_factor * 0.3 + temp_factor * 0.2 + 
                        wind_factor * 0.2 + density_factor * 0.3)
        
        return max(0, min(1, boundary_ease))
    
    def _calculate_swing_assistance(self, weather: WeatherConditions, pitch: PitchConditions) -> float:
        """Calculate swing bowling assistance"""
        
        # High humidity helps swing
        humidity_factor = min(1.0, weather.humidity / 80)
        
        # Cloud cover helps swing
        cloud_factor = weather.cloud_cover / 100
        
        # Cooler temperatures help swing
        temp_factor = max(0, 1.0 - (weather.temperature - 10) / 30)
        
        # New ball condition (simplified)
        ball_condition_factor = max(0, 1.0 - pitch.wear_factor)
        
        swing_assistance = (humidity_factor * 0.35 + cloud_factor * 0.25 + 
                          temp_factor * 0.25 + ball_condition_factor * 0.15)
        
        return max(0, min(1, swing_assistance))
    
    def _calculate_reverse_swing_likelihood(self, weather: WeatherConditions, pitch: PitchConditions) -> float:
        """Calculate reverse swing likelihood"""
        
        # Hot, dry conditions favor reverse swing
        temp_factor = min(1.0, (weather.temperature - 25) / 15)
        humidity_factor = max(0, 1.0 - weather.humidity / 100)
        
        # Abrasive pitch helps reverse swing
        abrasion_factor = pitch.crack_density
        
        # Later in innings (simplified by wear factor)
        innings_factor = pitch.wear_factor
        
        reverse_swing = (temp_factor * 0.3 + humidity_factor * 0.3 + 
                        abrasion_factor * 0.2 + innings_factor * 0.2)
        
        return max(0, min(1, reverse_swing))
    
    def _calculate_dew_factor(self, weather: WeatherConditions, env_context: EnvironmentalContext) -> float:
        """Calculate dew factor impact"""
        
        # Dew more likely in evening/night matches
        timing_factor = 0.0
        if env_context.match_timing in ['evening', 'night']:
            timing_factor = 0.8
        
        # High humidity increases dew likelihood
        humidity_factor = max(0, (weather.humidity - 70) / 30)
        
        # Temperature differential (cool nights after warm days)
        temp_factor = max(0, (25 - weather.temperature) / 15)
        
        dew_factor = timing_factor * humidity_factor * temp_factor
        
        return max(0, min(1, dew_factor))

# Utility functions for integration
def get_environmental_performance_multiplier(env_context: EnvironmentalContext, 
                                           player_role: str, 
                                           player_stats: Dict[str, Any]) -> float:
    """Get performance multiplier based on environmental conditions"""
    
    engine = EnvironmentalIntelligence()
    impact = engine.calculate_performance_impact(env_context, player_role)
    
    # Calculate role-specific multiplier
    if 'bat' in player_role.lower():
        multiplier = 0.8 + (impact.batting_advantage * 0.4)
    elif 'bowl' in player_role.lower():
        if 'pace' in player_role.lower() or 'fast' in player_role.lower():
            multiplier = 0.8 + (impact.pace_bowling_advantage * 0.4)
        else:  # spin bowler
            multiplier = 0.8 + (impact.spin_bowling_advantage * 0.4)
    elif 'allrounder' in player_role.lower():
        # Combine batting and bowling factors
        batting_mult = 0.8 + (impact.batting_advantage * 0.2)
        bowling_mult = 0.8 + (impact.pace_bowling_advantage * 0.2)
        multiplier = (batting_mult + bowling_mult) / 2
    else:
        multiplier = 1.0
    
    # Apply fielding impact
    multiplier *= (1.0 - impact.fielding_difficulty * 0.1)
    
    return max(0.5, min(1.5, multiplier))

# Export
__all__ = ['EnvironmentalIntelligence', 'EnvironmentalContext', 'WeatherConditions', 
           'PitchConditions', 'PerformanceImpact', 'get_environmental_performance_multiplier']