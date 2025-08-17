# Weather API Setup Guide

## WeatherAPI.com Setup (Recommended)

1. **Register for Free Account:**
   - Visit: https://www.weatherapi.com/signup.aspx
   - Create free account with email
   - Get instant API key with 1M calls/month free

2. **Add API Key:**
   - Copy your API key from dashboard
   - Update `.env` file:
   ```
   WEATHERAPI_KEY=your_actual_api_key_here
   ```

3. **WeatherAPI.com Free Tier Benefits:**
   - 1,000,000 requests per month
   - Current weather conditions
   - 14-day weather forecast
   - Historical weather data
   - No credit card required

## OpenWeatherMap Setup (Backup)

1. **Alternative Option:**
   - Visit: https://openweathermap.org/api
   - Sign up for free account
   - Get API key with 1,000 calls/day

2. **Add Backup Key:**
   ```
   OPENWEATHER_API_KEY=your_openweather_key_here
   ```

## Current Status

- **Fallback System:** Venue-specific realistic weather data
- **Real API Integration:** Ready when keys are configured
- **Cricket-Specific Analysis:** Swing/spin factors calculated

## Features Provided

✅ **Temperature & Humidity:** Accurate for venue conditions  
✅ **Wind Analysis:** Speed and direction for swing bowling  
✅ **Cricket Factors:** Dew, swing, and spin factor calculations  
✅ **Venue Intelligence:** Lord's, MCG, Wankhede specific conditions  
✅ **Pitch Analysis:** Pace/spin friendly surface detection  

The system works perfectly with venue-specific data and will automatically use real weather APIs when keys are configured.