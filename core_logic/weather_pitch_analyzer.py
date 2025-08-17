#!/usr/bin/env python3
"""
Real-Time Weather and Pitch Condition Analysis System
Integrates weather data and pitch reports to influence team selection
"""

import requests
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os

# Load environment variables from .env file
def _load_env_variables():
    """Load environment variables from .env file"""
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        env_path = os.path.join(project_root, '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
    except Exception:
        pass

_load_env_variables()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherCondition(Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"

class PitchType(Enum):
    BATTING_FRIENDLY = "batting_friendly"
    BOWLING_FRIENDLY = "bowling_friendly"
    BALANCED = "balanced"
    SPINNER_FRIENDLY = "spinner_friendly"
    PACE_FRIENDLY = "pace_friendly"

@dataclass
class WeatherData:
    """Weather information for match location"""
    location: str
    temperature: float  # Celsius
    humidity: float     # Percentage
    wind_speed: float   # km/h
    wind_direction: str
    condition: WeatherCondition
    precipitation_chance: float  # Percentage
    visibility: float    # km
    pressure: float     # hPa
    
    # Match impact factors
    dew_factor: float = 0.0      # 0-1 scale
    swing_factor: float = 0.0    # 0-1 scale (high = more swing)
    spin_factor: float = 0.0     # 0-1 scale (high = more spin)
    
    timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = "openweather"

@dataclass
class PitchReport:
    """Pitch condition analysis"""
    venue: str
    pitch_type: PitchType
    
    # Pitch characteristics
    bounce: str = "medium"        # low, medium, high
    pace: str = "medium"          # slow, medium, fast  
    turn: str = "minimal"         # minimal, moderate, significant
    
    # Historical analysis
    avg_first_innings_score: float = 160.0
    avg_second_innings_score: float = 145.0
    
    # Specific insights
    favors_batsmen: bool = True
    favors_bowlers: bool = False
    favors_spinners: bool = False
    favors_pacers: bool = False
    
    # Toss impact
    toss_advantage: str = "batting"  # batting, bowling, neutral
    
    # Time-based factors
    day_performance: Dict[str, float] = field(default_factory=dict)  # session-wise scores
    
    confidence_score: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    
@dataclass
class MatchConditions:
    """Combined weather and pitch analysis for match impact"""
    match_id: str
    venue: str
    
    weather: WeatherData
    pitch: PitchReport
    
    # Combined impact analysis
    captain_preference: str = "batting"  # batting, bowling
    team_composition_bias: Dict[str, float] = field(default_factory=dict)
    
    # Player selection impact
    pace_bowler_advantage: float = 0.0    # -1 to 1 scale
    spin_bowler_advantage: float = 0.0    # -1 to 1 scale
    batsmen_advantage: float = 0.0        # -1 to 1 scale
    wicket_keeper_advantage: float = 0.0  # -1 to 1 scale
    
    # Strategy recommendations
    power_play_strategy: str = "aggressive"  # aggressive, conservative
    death_over_strategy: str = "pace"        # pace, spin, mixed
    
    analysis_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class WeatherPitchAnalyzer:
    """Comprehensive weather and pitch condition analyzer"""
    
    def __init__(self, db_path: str = "weather_pitch_analysis.db"):
        self.db_path = db_path
        self.cache = {}
        self.cache_duration = timedelta(hours=2)  # Cache for 2 hours
        
        # Weather API configuration - WeatherAPI.com (primary), OpenWeatherMap (fallback)
        self.weatherapi_key = os.getenv('WEATHERAPI_KEY')
        self.weatherapi_base_url = "https://api.weatherapi.com/v1"
        self.openweather_key = os.getenv('OPENWEATHER_API_KEY')
        self.openweather_base_url = "https://api.openweathermap.org/data/2.5"
        
        self._init_database()
        
    def _init_database(self):
        """Initialize the weather and pitch database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Weather data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    location TEXT NOT NULL,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction TEXT,
                    condition TEXT,
                    precipitation_chance REAL,
                    dew_factor REAL,
                    swing_factor REAL,
                    spin_factor REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT
                )
            ''')
            
            # Pitch reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pitch_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    venue TEXT NOT NULL,
                    pitch_type TEXT,
                    bounce TEXT,
                    pace TEXT,
                    turn TEXT,
                    avg_first_innings_score REAL,
                    avg_second_innings_score REAL,
                    favors_batsmen BOOLEAN,
                    favors_bowlers BOOLEAN,
                    favors_spinners BOOLEAN,
                    favors_pacers BOOLEAN,
                    toss_advantage TEXT,
                    confidence_score REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Match conditions analysis table  
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_conditions (
                    match_id TEXT PRIMARY KEY,
                    venue TEXT,
                    captain_preference TEXT,
                    pace_bowler_advantage REAL,
                    spin_bowler_advantage REAL,
                    batsmen_advantage REAL,
                    wicket_keeper_advantage REAL,
                    power_play_strategy TEXT,
                    death_over_strategy TEXT,
                    analysis_confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Weather and pitch analysis database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing weather/pitch database: {e}")
    
    def get_match_conditions(self, match_id: str, venue: str, location: str = None) -> MatchConditions:
        """Get comprehensive match conditions analysis"""
        cache_key = f"{match_id}_{venue}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                return cached_data
        
        # Get weather data
        weather = self._get_weather_data(location or venue)
        
        # Get pitch report
        pitch = self._get_pitch_report(venue, match_id)
        
        # Combine analysis
        conditions = self._analyze_combined_conditions(match_id, venue, weather, pitch)
        
        # Cache result
        self.cache[cache_key] = (conditions, datetime.now())
        
        # Save to database
        self._save_conditions_to_database(conditions)
        
        return conditions
    
    def _get_weather_data(self, location: str) -> WeatherData:
        """Fetch real-time weather data using WeatherAPI.com (primary) or OpenWeatherMap (fallback)"""
        # Try WeatherAPI.com first
        if self.weatherapi_key:
            try:
                weather_url = f"{self.weatherapi_base_url}/current.json"
                params = {
                    'key': self.weatherapi_key,
                    'q': location,
                    'aqi': 'no'
                }
                
                response = requests.get(weather_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_weatherapi_data(data, location)
                else:
                    logger.warning(f"⚠️ WeatherAPI.com returned {response.status_code}, trying OpenWeatherMap")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error with WeatherAPI.com: {e}, trying OpenWeatherMap")
        
        # Fallback to OpenWeatherMap
        if self.openweather_key:
            try:
                weather_url = f"{self.openweather_base_url}/weather"
                params = {
                    'q': location,
                    'appid': self.openweather_key,
                    'units': 'metric'
                }
                
                response = requests.get(weather_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_openweather_data(data, location)
                else:
                    logger.warning(f"⚠️ OpenWeatherMap returned {response.status_code}, using fallback")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error with OpenWeatherMap: {e}, using fallback")
        
        logger.warning("⚠️ No weather API keys configured, using venue-specific fallback")
        return self._get_fallback_weather(location)
    
    def _parse_weatherapi_data(self, data: Dict, location: str) -> WeatherData:
        """Parse WeatherAPI.com response into WeatherData object"""
        try:
            current = data['current']
            
            # Map weather condition
            condition_text = current.get('condition', {}).get('text', 'cloudy').lower()
            if 'rain' in condition_text:
                condition = WeatherCondition.LIGHT_RAIN if 'light' in condition_text else WeatherCondition.HEAVY_RAIN
            elif 'thunder' in condition_text:
                condition = WeatherCondition.THUNDERSTORM
            elif 'clear' in condition_text or 'sunny' in condition_text:
                condition = WeatherCondition.SUNNY
            elif 'overcast' in condition_text:
                condition = WeatherCondition.OVERCAST
            else:
                condition = WeatherCondition.CLOUDY
            
            # Calculate cricket-specific factors
            temp = current.get('temp_c', 25.0)
            humidity = current.get('humidity', 65.0)
            wind_speed = current.get('wind_kph', 10.0)
            
            dew_factor = min(1.0, humidity / 100.0 * (1 - abs(temp - 20) / 40))
            swing_factor = min(1.0, (humidity / 100.0) * (wind_speed / 20.0) * 0.8)
            spin_factor = max(0.1, 1.0 - swing_factor)
            
            return WeatherData(
                location=location,
                temperature=temp,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=current.get('wind_dir', 'SW'),
                condition=condition,
                precipitation_chance=current.get('precip_mm', 0) * 10,  # Convert mm to percentage
                visibility=current.get('vis_km', 10.0),
                pressure=current.get('pressure_mb', 1013.0),
                dew_factor=dew_factor,
                swing_factor=swing_factor,
                spin_factor=spin_factor,
                data_source="weatherapi.com"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing WeatherAPI.com data: {e}")
            return self._get_fallback_weather(location)
    
    def _parse_openweather_data(self, data: Dict, location: str) -> WeatherData:
        """Parse weather API response into WeatherData object"""
        try:
            main = data.get('main', {})
            wind = data.get('wind', {})
            weather = data.get('weather', [{}])[0]
            
            # Map weather condition
            condition_map = {
                'clear': WeatherCondition.SUNNY,
                'clouds': WeatherCondition.CLOUDY,
                'rain': WeatherCondition.LIGHT_RAIN,
                'thunderstorm': WeatherCondition.THUNDERSTORM,
                'drizzle': WeatherCondition.LIGHT_RAIN,
                'mist': WeatherCondition.OVERCAST,
                'fog': WeatherCondition.OVERCAST
            }
            
            condition = condition_map.get(weather.get('main', '').lower(), WeatherCondition.CLOUDY)
            
            # Calculate cricket-specific factors
            humidity = main.get('humidity', 50)
            temp = main.get('temp', 25)
            wind_speed = wind.get('speed', 0) * 3.6  # Convert m/s to km/h
            
            # Cricket impact calculations
            dew_factor = self._calculate_dew_factor(temp, humidity)
            swing_factor = self._calculate_swing_factor(condition, humidity, temp)
            spin_factor = self._calculate_spin_factor(humidity, wind_speed)
            
            return WeatherData(
                location=location,
                temperature=temp,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=self._get_wind_direction(wind.get('deg', 0)),
                condition=condition,
                precipitation_chance=data.get('clouds', {}).get('all', 0),
                visibility=data.get('visibility', 10000) / 1000,  # Convert to km
                pressure=main.get('pressure', 1013),
                dew_factor=dew_factor,
                swing_factor=swing_factor,
                spin_factor=spin_factor
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing weather data: {e}")
            return self._get_fallback_weather(location)
    
    def _get_fallback_weather(self, location: str) -> WeatherData:
        """Generate venue-specific realistic fallback weather data"""
        # Venue-specific weather patterns
        venue_weather = {
            "lord's": {"temp": 22.0, "humidity": 70.0, "wind": 12.0, "swing": 0.6},
            "the oval": {"temp": 23.0, "humidity": 65.0, "wind": 10.0, "swing": 0.5},
            "old trafford": {"temp": 20.0, "humidity": 75.0, "wind": 15.0, "swing": 0.7},
            "edgbaston": {"temp": 21.0, "humidity": 68.0, "wind": 11.0, "swing": 0.5},
            "headingley": {"temp": 19.0, "humidity": 72.0, "wind": 14.0, "swing": 0.6},
            "mcg": {"temp": 28.0, "humidity": 55.0, "wind": 8.0, "swing": 0.3},
            "scg": {"temp": 30.0, "humidity": 60.0, "wind": 12.0, "swing": 0.4},
            "wankhede": {"temp": 32.0, "humidity": 80.0, "wind": 8.0, "swing": 0.3},
            "eden gardens": {"temp": 31.0, "humidity": 85.0, "wind": 6.0, "swing": 0.2},
        }
        
        venue_key = location.lower()
        venue_data = venue_weather.get(venue_key, venue_weather.get("lord's"))
        
        return WeatherData(
            location=location,
            temperature=venue_data["temp"],
            humidity=venue_data["humidity"],
            wind_speed=venue_data["wind"],
            wind_direction="SW",
            condition=WeatherCondition.CLOUDY,
            precipitation_chance=20.0,
            visibility=10.0,
            pressure=1013.0,
            dew_factor=venue_data["humidity"] / 100.0 * 0.5,
            swing_factor=venue_data["swing"],
            spin_factor=0.8 - venue_data["swing"],  # inverse relationship
            data_source="venue_specific_fallback"
        )
    
    def _get_pitch_report(self, venue: str, match_id: str) -> PitchReport:
        """Get pitch analysis for venue"""
        # Try to get from database first
        cached_report = self._get_pitch_from_database(venue)
        if cached_report and not self._is_pitch_data_stale(cached_report):
            return cached_report
        
        # Generate pitch analysis based on venue knowledge
        return self._analyze_venue_pitch(venue, match_id)
    
    def _get_pitch_from_database(self, venue: str) -> Optional[PitchReport]:
        """Get pitch report from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM pitch_reports 
                WHERE venue = ? 
                ORDER BY last_updated DESC 
                LIMIT 1
            ''', (venue,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return PitchReport(
                    venue=row[2],
                    pitch_type=PitchType(row[3]) if row[3] else PitchType.BALANCED,
                    bounce=row[4] or "medium",
                    pace=row[5] or "medium", 
                    turn=row[6] or "minimal",
                    avg_first_innings_score=row[7] or 160.0,
                    avg_second_innings_score=row[8] or 145.0,
                    favors_batsmen=bool(row[9]),
                    favors_bowlers=bool(row[10]),
                    favors_spinners=bool(row[11]),
                    favors_pacers=bool(row[12]),
                    toss_advantage=row[13] or "batting",
                    confidence_score=row[14] or 0.5,
                    last_updated=datetime.fromisoformat(row[15])
                )
        except Exception as e:
            logger.warning(f"⚠️ Error getting pitch data from database: {e}")
        
        return None
    
    def _analyze_venue_pitch(self, venue: str, match_id: str) -> PitchReport:
        """Analyze pitch characteristics based on venue knowledge"""
        venue_lower = venue.lower()
        
        # Venue-specific pitch characteristics (based on cricket knowledge)
        venue_characteristics = {
            # High-scoring venues
            'wankhede': {
                'type': PitchType.BATTING_FRIENDLY,
                'avg_first': 180, 'avg_second': 160,
                'favors_batsmen': True, 'pace': 'fast'
            },
            'chinnaswamy': {
                'type': PitchType.BATTING_FRIENDLY, 
                'avg_first': 175, 'avg_second': 155,
                'favors_batsmen': True, 'pace': 'medium'
            },
            'delhi': {
                'type': PitchType.SPINNER_FRIENDLY,
                'avg_first': 160, 'avg_second': 140,
                'favors_spinners': True, 'turn': 'moderate'
            },
            'kolkata': {
                'type': PitchType.SPINNER_FRIENDLY,
                'avg_first': 155, 'avg_second': 135,
                'favors_spinners': True, 'turn': 'significant'
            },
            # Bowling-friendly venues
            'mohali': {
                'type': PitchType.PACE_FRIENDLY,
                'avg_first': 150, 'avg_second': 130,
                'favors_pacers': True, 'bounce': 'high'
            },
            'dubai': {
                'type': PitchType.BOWLING_FRIENDLY,
                'avg_first': 145, 'avg_second': 125,
                'favors_bowlers': True, 'pace': 'slow'
            }
        }
        
        # Find matching venue characteristics
        characteristics = None
        for venue_key, char in venue_characteristics.items():
            if venue_key in venue_lower:
                characteristics = char
                break
        
        # Default characteristics if venue not found
        if not characteristics:
            characteristics = {
                'type': PitchType.BALANCED,
                'avg_first': 160, 'avg_second': 145,
                'favors_batsmen': True, 'pace': 'medium'
            }
        
        pitch_report = PitchReport(
            venue=venue,
            pitch_type=characteristics['type'],
            bounce=characteristics.get('bounce', 'medium'),
            pace=characteristics.get('pace', 'medium'),
            turn=characteristics.get('turn', 'minimal'),
            avg_first_innings_score=characteristics['avg_first'],
            avg_second_innings_score=characteristics['avg_second'],
            favors_batsmen=characteristics.get('favors_batsmen', False),
            favors_bowlers=characteristics.get('favors_bowlers', False),
            favors_spinners=characteristics.get('favors_spinners', False),
            favors_pacers=characteristics.get('favors_pacers', False),
            confidence_score=0.7 if characteristics != venue_characteristics.get('default') else 0.3
        )
        
        # Save to database
        self._save_pitch_to_database(pitch_report, match_id)
        
        return pitch_report
    
    def _analyze_combined_conditions(self, match_id: str, venue: str, 
                                   weather: WeatherData, pitch: PitchReport) -> MatchConditions:
        """Analyze combined weather and pitch impact on match"""
        
        # Calculate player selection advantages
        pace_advantage = self._calculate_pace_advantage(weather, pitch)
        spin_advantage = self._calculate_spin_advantage(weather, pitch)
        batting_advantage = self._calculate_batting_advantage(weather, pitch)
        keeper_advantage = self._calculate_keeper_advantage(weather, pitch)
        
        # Determine captain preference
        captain_pref = "batting" if batting_advantage > 0 else "bowling"
        
        # Strategy recommendations
        power_play_strategy = "aggressive" if batting_advantage > 0.2 else "conservative"
        death_strategy = "pace" if pace_advantage > spin_advantage else "spin"
        
        return MatchConditions(
            match_id=match_id,
            venue=venue,
            weather=weather,
            pitch=pitch,
            captain_preference=captain_pref,
            pace_bowler_advantage=pace_advantage,
            spin_bowler_advantage=spin_advantage,
            batsmen_advantage=batting_advantage,
            wicket_keeper_advantage=keeper_advantage,
            power_play_strategy=power_play_strategy,
            death_over_strategy=death_strategy,
            analysis_confidence=min(weather.swing_factor + pitch.confidence_score, 1.0) / 2
        )
    
    def _calculate_pace_advantage(self, weather: WeatherData, pitch: PitchReport) -> float:
        """Calculate pace bowler advantage (-1 to 1)"""
        advantage = 0.0
        
        # Weather factors
        if weather.swing_factor > 0.6:
            advantage += 0.3
        if weather.humidity > 70:
            advantage += 0.2
        if weather.condition in [WeatherCondition.CLOUDY, WeatherCondition.OVERCAST]:
            advantage += 0.2
        
        # Pitch factors
        if pitch.favors_pacers:
            advantage += 0.4
        if pitch.bounce == "high":
            advantage += 0.2
        if pitch.pace == "fast":
            advantage += 0.2
        
        return min(max(advantage - 0.5, -1.0), 1.0)
    
    def _calculate_spin_advantage(self, weather: WeatherData, pitch: PitchReport) -> float:
        """Calculate spin bowler advantage (-1 to 1)"""
        advantage = 0.0
        
        # Weather factors
        if weather.spin_factor > 0.5:
            advantage += 0.3
        if weather.humidity < 50:
            advantage += 0.2
        
        # Pitch factors
        if pitch.favors_spinners:
            advantage += 0.4
        if pitch.turn in ["moderate", "significant"]:
            advantage += 0.3
        if pitch.pace == "slow":
            advantage += 0.2
        
        return min(max(advantage - 0.4, -1.0), 1.0)
    
    def _calculate_batting_advantage(self, weather: WeatherData, pitch: PitchReport) -> float:
        """Calculate batting advantage (-1 to 1)"""
        advantage = 0.0
        
        # Weather factors
        if weather.condition == WeatherCondition.SUNNY:
            advantage += 0.2
        if weather.wind_speed < 15:
            advantage += 0.1
        
        # Pitch factors
        if pitch.favors_batsmen:
            advantage += 0.4
        if pitch.avg_first_innings_score > 170:
            advantage += 0.3
        
        return min(max(advantage - 0.3, -1.0), 1.0)
    
    def _calculate_keeper_advantage(self, weather: WeatherData, pitch: PitchReport) -> float:
        """Calculate wicket-keeper advantage"""
        advantage = 0.0
        
        # Higher dew factor means more chances
        if weather.dew_factor > 0.5:
            advantage += 0.2
        
        # More swing/spin means more edges
        if weather.swing_factor > 0.6 or weather.spin_factor > 0.6:
            advantage += 0.2
        
        return min(advantage, 0.5)
    
    def _calculate_dew_factor(self, temp: float, humidity: float) -> float:
        """Calculate dew formation likelihood (0-1)"""
        if temp < 15 and humidity > 80:
            return 0.9
        elif temp < 20 and humidity > 70:
            return 0.6
        elif humidity > 85:
            return 0.4
        else:
            return max(0, (humidity - 50) / 50)
    
    def _calculate_swing_factor(self, condition: WeatherCondition, humidity: float, temp: float) -> float:
        """Calculate swing bowling advantage (0-1)"""
        base_swing = 0.3
        
        if condition in [WeatherCondition.CLOUDY, WeatherCondition.OVERCAST]:
            base_swing += 0.3
        if humidity > 70:
            base_swing += 0.2
        if 15 <= temp <= 25:
            base_swing += 0.2
        
        return min(base_swing, 1.0)
    
    def _calculate_spin_factor(self, humidity: float, wind_speed: float) -> float:
        """Calculate spin bowling advantage (0-1)"""
        base_spin = 0.3
        
        if humidity < 60:
            base_spin += 0.2
        if wind_speed < 10:
            base_spin += 0.2
        
        return min(base_spin, 1.0)
    
    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind direction degrees to compass direction"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return directions[int((degrees + 22.5) / 45) % 8]
    
    def _is_pitch_data_stale(self, pitch: PitchReport) -> bool:
        """Check if pitch data needs refresh"""
        age = datetime.now() - pitch.last_updated
        return age > timedelta(days=7)  # Pitch data valid for 7 days
    
    def _save_conditions_to_database(self, conditions: MatchConditions):
        """Save combined conditions analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO match_conditions 
                (match_id, venue, captain_preference, pace_bowler_advantage, 
                 spin_bowler_advantage, batsmen_advantage, wicket_keeper_advantage,
                 power_play_strategy, death_over_strategy, analysis_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                conditions.match_id, conditions.venue, conditions.captain_preference,
                conditions.pace_bowler_advantage, conditions.spin_bowler_advantage,
                conditions.batsmen_advantage, conditions.wicket_keeper_advantage,
                conditions.power_play_strategy, conditions.death_over_strategy,
                conditions.analysis_confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Error saving conditions to database: {e}")
    
    def _save_pitch_to_database(self, pitch: PitchReport, match_id: str):
        """Save pitch report to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pitch_reports 
                (match_id, venue, pitch_type, bounce, pace, turn,
                 avg_first_innings_score, avg_second_innings_score,
                 favors_batsmen, favors_bowlers, favors_spinners, favors_pacers,
                 toss_advantage, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id, pitch.venue, pitch.pitch_type.value, pitch.bounce, pitch.pace, pitch.turn,
                pitch.avg_first_innings_score, pitch.avg_second_innings_score,
                pitch.favors_batsmen, pitch.favors_bowlers, pitch.favors_spinners, pitch.favors_pacers,
                pitch.toss_advantage, pitch.confidence_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Error saving pitch report: {e}")

# Global instance
_weather_pitch_analyzer = None

def get_weather_pitch_analyzer() -> WeatherPitchAnalyzer:
    """Get global weather and pitch analyzer instance"""
    global _weather_pitch_analyzer
    if _weather_pitch_analyzer is None:
        _weather_pitch_analyzer = WeatherPitchAnalyzer()
    return _weather_pitch_analyzer

def get_match_conditions(match_id: str, venue: str, location: str = None) -> MatchConditions:
    """Main interface for getting match conditions"""
    analyzer = get_weather_pitch_analyzer()
    return analyzer.get_match_conditions(match_id, venue, location)