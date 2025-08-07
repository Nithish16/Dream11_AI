#!/usr/bin/env python3
"""
SIMPLE OPTIMIZED DAILY SYSTEM
Based on user's excellent suggestion: 6 AM discovery + precise scheduling

Key improvement: One discovery call + exact timing per match
"""

import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from smart_api_manager import SmartAPIManager

class SimpleOptimizedSystem:
    """
    User's optimized approach - simple and effective
    """
    
    def __init__(self):
        self.api_manager = SmartAPIManager()
        self.todays_matches = []
        
    def midnight_discovery_12am(self):
        """12 AM (Midnight): Discover all matches for today and schedule predictions"""
        
        print(f"\n🌙 MIDNIGHT DAILY DISCOVERY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Check API quota
        usage = self.api_manager.get_current_usage()
        if usage['quota_remaining'] < 10:
            print("⚠️ Low API quota - skipping discovery")
            return
        
        try:
            # Single API call to get all upcoming matches
            from utils.api_client import fetch_upcoming_matches
            
            print("🔍 Making single discovery API call...")
            upcoming_data = self.api_manager.smart_api_call(
                api_function=fetch_upcoming_matches,
                endpoint_name='upcoming_matches',
                priority=2
            )
            
            if not upcoming_data:
                print("📭 No upcoming matches found")
                return
            
            # Process today's matches only
            todays_matches = self.extract_todays_matches(upcoming_data)
            
            if not todays_matches:
                print("📅 No matches scheduled for today")
                return
            
            # Schedule predictions for each match
            scheduled_count = self.schedule_predictions(todays_matches)
            
            print(f"\n✅ MIDNIGHT DISCOVERY COMPLETE:")
            print(f"   🎯 Matches found for today: {len(todays_matches)}")
            print(f"   ⏰ Predictions scheduled: {scheduled_count}")
            print(f"   🌐 API calls used: 1 (discovery only)")
            print(f"   🔋 Quota remaining: {self.api_manager.get_current_usage()['quota_remaining']}")
            print(f"   🌍 Coverage: Full 24-hour window (00:00-23:59)")
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
    
    def extract_todays_matches(self, upcoming_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract only today's matches from API response"""
        
        today = datetime.now().date()
        todays_matches = []
        
        print(f"📅 Looking for matches on {today}")
        
        try:
            for match_type in upcoming_data.get('typeMatches', []):
                for series in match_type.get('seriesMatches', []):
                    for match_wrapper in series.get('seriesAdWrapper', {}).get('matches', []):
                        match_info = match_wrapper.get('matchInfo', {})
                        
                        # Extract match details
                        match_id = str(match_info.get('matchId', ''))
                        start_date = match_info.get('startDate', '')
                        team1 = match_info.get('team1', {}).get('teamName', 'Team1')
                        team2 = match_info.get('team2', {}).get('teamName', 'Team2')
                        format_type = match_info.get('matchFormat', 'Unknown')
                        
                        if not match_id or not start_date:
                            continue
                        
                        # Parse date - handle different formats
                        try:
                            if 'T' in start_date:
                                match_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                            else:
                                # Handle other date formats
                                continue
                        except:
                            continue
                        
                        # Check if match is today
                        if match_datetime.date() == today:
                            
                            # Check if match is in future (at least 30 minutes from now)
                            hours_until = (match_datetime - datetime.now()).total_seconds() / 3600
                            if 0.5 <= hours_until <= 24:  # Between 30 minutes and 24 hours
                                
                                match_record = {
                                    'match_id': match_id,
                                    'teams': f"{team1} vs {team2}",
                                    'format': format_type,
                                    'start_time': match_datetime,
                                    'hours_until': hours_until,
                                    'priority': self.get_format_priority(format_type)
                                }
                                
                                todays_matches.append(match_record)
                                print(f"   ✅ {team1} vs {team2} ({format_type}) at {match_datetime.strftime('%H:%M')}")
        
        except Exception as e:
            print(f"⚠️ Error extracting matches: {e}")
        
        # Sort by priority and start time
        todays_matches.sort(key=lambda x: (x['priority'], x['start_time']), reverse=True)
        
        return todays_matches
    
    def get_format_priority(self, format_type: str) -> int:
        """Get priority score for match format"""
        priorities = {
            'T20': 10, 'T20I': 10,
            'The Hundred': 9,
            'ODI': 8, 'ODIM': 8,
            'Test': 6
        }
        return priorities.get(format_type, 5)
    
    def schedule_predictions(self, matches: List[Dict[str, Any]]) -> int:
        """Schedule predictions exactly 20 minutes before each match"""
        
        # Limit to top priority matches based on API quota
        usage = self.api_manager.get_current_usage()
        max_matches = min(len(matches), usage['quota_remaining'] // 3, 10)  # 3 calls per match, max 10
        
        priority_matches = matches[:max_matches]
        scheduled_count = 0
        
        print(f"\n⏰ SCHEDULING PREDICTIONS:")
        print(f"   📊 Total matches today: {len(matches)}")
        print(f"   🎯 Scheduling top {len(priority_matches)} matches")
        
        for match in priority_matches:
            try:
                # Calculate prediction time (20 minutes before match start)
                prediction_time = match['start_time'] - timedelta(minutes=20)
                
                # Only schedule if prediction time is in the future
                if prediction_time > datetime.now():
                    
                    # Schedule the prediction
                    job_time = prediction_time.strftime('%H:%M')
                    
                    schedule.every().day.at(job_time).do(
                        self.execute_prediction,
                        match['match_id'],
                        match['teams']
                    ).tag(f"predict_{match['match_id']}")
                    
                    scheduled_count += 1
                    
                    print(f"   🎯 {match['teams']}")
                    print(f"      📅 Starts: {match['start_time'].strftime('%H:%M')}")
                    print(f"      ⏰ Predict: {prediction_time.strftime('%H:%M')}")
                    print(f"      🏏 Format: {match['format']}")
                
            except Exception as e:
                print(f"   ❌ Failed to schedule {match['match_id']}: {e}")
        
        return scheduled_count
    
    def execute_prediction(self, match_id: str, teams: str):
        """Execute prediction exactly 20 minutes before match"""
        
        print(f"\n🎯 EXECUTING PREDICTION")
        print(f"⚽ Match: {teams}")
        print(f"🆔 ID: {match_id}")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Check quota before prediction
        usage = self.api_manager.get_current_usage()
        if usage['quota_remaining'] < 3:
            print("⚠️ Insufficient quota - skipping prediction")
            return
        
        try:
            # Execute main prediction
            import subprocess
            result = subprocess.run(
                ['python3', 'dream11_ai.py', match_id, '5'],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Prediction completed successfully!")
                # Store in learning system
                from ai_learning_system import log_prediction
                # log_prediction(match_id, [])  # Will be populated by dream11_ai.py
            else:
                print(f"❌ Prediction failed: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Prediction timeout - match may have already started")
        except Exception as e:
            print(f"❌ Error executing prediction: {e}")
    
    def show_todays_plan(self):
        """Show what would happen if we ran discovery now"""
        
        print("🧪 TESTING: What matches would be found for today?")
        print("-" * 50)
        
        try:
            from utils.api_client import fetch_upcoming_matches
            
            # Use cached data if available
            upcoming_data = self.api_manager.smart_api_call(
                api_function=fetch_upcoming_matches,
                endpoint_name='upcoming_matches',
                priority=5
            )
            
            if upcoming_data:
                matches = self.extract_todays_matches(upcoming_data)
                
                if matches:
                    print(f"\n📊 SUMMARY:")
                    print(f"   🎯 Matches found: {len(matches)}")
                    print(f"   ⏰ Would schedule: {min(len(matches), 10)} predictions")
                    print(f"   🌐 API calls needed: ~{len(matches) * 3} for all predictions")
                    print(f"   🔋 Current quota: {self.api_manager.get_current_usage()['quota_remaining']}")
                else:
                    print("📅 No matches found for today")
            else:
                print("❌ Could not fetch match data")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def start_system(self):
        """Start the optimized system"""
        
        print("🌙 OPTIMIZED DAILY PREDICTION SYSTEM")
        print("="*45)
        print("✨ Your Superior Approach:")
        print("   • 00:00 (Midnight): Single discovery call")
        print("   • Schedule all day's predictions")
        print("   • Execute exactly 20 min before each match")
        print("   • Minimal API usage, maximum precision")
        print()
        
        # Schedule daily discovery at midnight
        schedule.every().day.at("00:00").do(self.midnight_discovery_12am)
        
        print("⏰ System scheduled:")
        print("   • 00:00 (Midnight): Daily match discovery")
        print("   • Auto-schedule: 20 min before each match")
        print()
        print("🔄 Running... (Press Ctrl+C to stop)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("\n⏹️ System stopped")

def main():
    """Main function"""
    import sys
    
    system = SimpleOptimizedSystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'test':
            # Test discovery without scheduling
            system.show_todays_plan()
            
        elif command == 'discovery':
            # Run discovery now (as if it's midnight)
            system.midnight_discovery_12am()
            
        elif command == 'start':
            # Start full system
            system.start_system()
            
    else:
        print("🌙 OPTIMIZED SYSTEM OPTIONS:")
        print("="*35)
        print("python3 simple_optimized_system.py test      # Test what matches would be found")
        print("python3 simple_optimized_system.py discovery # Run midnight discovery now")
        print("python3 simple_optimized_system.py start     # Start full system (00:00 discovery)")

if __name__ == "__main__":
    main()
