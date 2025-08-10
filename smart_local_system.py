#!/usr/bin/env python3
"""
SMART LOCAL SYSTEM - Realistic for personal laptop usage
Designed to work with YOUR schedule, not against it

Key Features:
- Runs when YOU open the project (not 24/7)
- Queues predictions for execution when convenient
- Batch learning when available
- Graceful offline operation
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import requests

class SmartLocalSystem:
    """
    Intelligent local system that works with real-world laptop usage
    """
    
    def __init__(self):
        self.db_path = "smart_local_predictions.db"
        self.queue_file = "prediction_queue.json"
        self.setup_database()
        
    def setup_database(self):
        """Setup local prediction database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS local_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL UNIQUE,
                teams TEXT,
                format TEXT,
                start_time TIMESTAMP,
                optimal_prediction_time TIMESTAMP,
                queued_time TIMESTAMP,
                executed_time TIMESTAMP,
                status TEXT DEFAULT 'queued',
                api_calls_used INTEGER DEFAULT 0,
                learning_processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS local_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date DATE,
                discovery_time TIMESTAMP,
                matches_found INTEGER DEFAULT 0,
                predictions_queued INTEGER DEFAULT 0,
                predictions_executed INTEGER DEFAULT 0,
                learning_sessions INTEGER DEFAULT 0,
                total_api_calls INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Smart local system database initialized")
    
    def has_internet(self) -> bool:
        """Check if internet connection is available"""
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except requests.exceptions.RequestException:
            return False
    
    def daily_discovery_when_available(self):
        """
        Run daily discovery when YOU open the project
        This replaces the midnight automated discovery
        """
        
        print(f"\n🏠 SMART LOCAL DISCOVERY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print("🎯 Running when convenient for your schedule")
        
        if not self.has_internet():
            print("❌ No internet connection - skipping discovery")
            print("💡 Try again when internet is available")
            return
        
        try:
            # Check if we already did discovery today
            today = datetime.now().date()
            if self.already_discovered_today(today):
                print(f"✅ Already discovered matches for {today}")
                self.show_todays_queue()
                return
            
            # Discover matches for today
            matches = self.discover_todays_matches()
            
            if not matches:
                print("📅 No matches found for today")
                self.log_session(today, 0, 0)
                return
            
            # Queue predictions for later execution
            queued_count = self.queue_predictions(matches)
            
            # Log session
            self.log_session(today, len(matches), queued_count)
            
            print(f"\n✅ DISCOVERY COMPLETE:")
            print(f"   🎯 Matches found: {len(matches)}")
            print(f"   ⏰ Predictions queued: {queued_count}")
            print(f"   🏠 Ready for execution when convenient")
            
            self.show_execution_guidance()
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
    
    def already_discovered_today(self, date) -> bool:
        """Check if we already ran discovery today"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id FROM local_sessions 
            WHERE session_date = ?
        ''', (date,))
        
        exists = cursor.fetchone()
        conn.close()
        return exists is not None
    
    def discover_todays_matches(self) -> List[Dict[str, Any]]:
        """Discover matches using the smart API manager"""
        
        from smart_api_manager import SmartAPIManager
        
        api_manager = SmartAPIManager()
        
        try:
            from utils.api_client import fetch_upcoming_matches
            
            print("🔍 Making discovery API call...")
            upcoming_data = api_manager.smart_api_call(
                api_function=fetch_upcoming_matches,
                endpoint_name='upcoming_matches',
                priority=2
            )
            
            if not upcoming_data:
                return []
            
            # Extract today's matches
            today = datetime.now().date()
            todays_matches = []
            
            for match_type in upcoming_data.get('typeMatches', []):
                for series in match_type.get('seriesMatches', []):
                    for match_wrapper in series.get('seriesAdWrapper', {}).get('matches', []):
                        match_info = match_wrapper.get('matchInfo', {})
                        
                        match_id = str(match_info.get('matchId', ''))
                        start_date = match_info.get('startDate', '')
                        team1 = match_info.get('team1', {}).get('teamName', 'Team1')
                        team2 = match_info.get('team2', {}).get('teamName', 'Team2')
                        format_type = match_info.get('matchFormat', 'Unknown')
                        
                        if not match_id or not start_date:
                            continue
                        
                        try:
                            if 'T' in start_date:
                                # ISO format
                                match_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                            elif start_date.isdigit():
                                # Unix timestamp in milliseconds
                                timestamp_seconds = int(start_date) / 1000
                                match_datetime = datetime.fromtimestamp(timestamp_seconds)
                            else:
                                # Unknown format
                                print(f"   ⚠️ Unknown date format: {start_date}")
                                continue
                        except Exception as e:
                            print(f"   ❌ Date parse error for {start_date}: {e}")
                            continue
                        
                        # Include matches for today and tomorrow (in case of timezone differences)
                        match_date = match_datetime.date()
                        tomorrow = today + timedelta(days=1)
                        
                        if match_date in [today, tomorrow]:
                            hours_until = (match_datetime - datetime.now()).total_seconds() / 3600
                            if 0.5 <= hours_until <= 36:  # Extended window for flexibility
                                
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
            
            # Sort by priority and start time
            todays_matches.sort(key=lambda x: (x['priority'], x['start_time']), reverse=True)
            
            return todays_matches
            
        except Exception as e:
            print(f"⚠️ Error discovering matches: {e}")
            return []
    
    def get_format_priority(self, format_type: str) -> int:
        """Get priority score for match format"""
        priorities = {
            'T20': 10, 'T20I': 10,
            'The Hundred': 9,
            'ODI': 8, 'ODIM': 8,
            'Test': 6
        }
        return priorities.get(format_type, 5)
    
    def queue_predictions(self, matches: List[Dict[str, Any]]) -> int:
        """Queue predictions for later execution"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        queued_count = 0
        
        # Limit to reasonable number for local execution
        max_matches = min(len(matches), 8)  # 8 matches max per day for local
        priority_matches = matches[:max_matches]
        
        for match in priority_matches:
            try:
                # Calculate optimal prediction time (20 minutes before)
                optimal_time = match['start_time'] - timedelta(minutes=20)
                
                # Only queue if prediction time is in the future
                if optimal_time > datetime.now():
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO local_predictions
                        (match_id, teams, format, start_time, optimal_prediction_time, queued_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        match['match_id'],
                        match['teams'],
                        match['format'],
                        match['start_time'],
                        optimal_time,
                        datetime.now()
                    ))
                    
                    queued_count += 1
                    
            except Exception as e:
                print(f"   ❌ Failed to queue {match['match_id']}: {e}")
        
        conn.commit()
        conn.close()
        
        return queued_count
    
    def show_execution_guidance(self):
        """Show user how to execute queued predictions"""
        
        print(f"\n💡 EXECUTION GUIDANCE:")
        print("="*30)
        print("🔄 Throughout the day, when convenient:")
        print("   python3 smart_local_system.py execute")
        print()
        print("📊 Check queue status anytime:")
        print("   python3 smart_local_system.py status")
        print()
        print("🧠 Weekend batch learning:")
        print("   python3 smart_local_system.py learn")
    
    def execute_available_predictions(self):
        """Execute predictions that are ready and convenient"""
        
        print(f"\n🎯 EXECUTING AVAILABLE PREDICTIONS - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        if not self.has_internet():
            print("❌ No internet connection - skipping execution")
            return
        
        # Get ready predictions
        ready_predictions = self.get_ready_predictions()
        
        if not ready_predictions:
            print("📭 No predictions ready for execution")
            self.show_next_prediction_time()
            return
        
        print(f"🎯 Found {len(ready_predictions)} predictions ready to execute")
        
        # Execute up to 3 predictions at a time (to be considerate of API)
        executed_count = 0
        max_executions = 3
        
        for prediction in ready_predictions[:max_executions]:
            
            print(f"\n🏏 Executing: {prediction['teams']}")
            print(f"   ⏰ Optimal time was: {prediction['optimal_prediction_time']}")
            
            try:
                # Execute prediction
                result = subprocess.run(
                    ['python3', 'dream11_ai.py', prediction['match_id'], '5'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    print(f"   ✅ Prediction successful!")
                    self.mark_prediction_executed(prediction['match_id'], True, 3)  # Estimate 3 API calls
                    executed_count += 1
                else:
                    print(f"   ❌ Prediction failed: {result.stderr[:100]}")
                    self.mark_prediction_executed(prediction['match_id'], False, 1)
                
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Prediction timeout")
                self.mark_prediction_executed(prediction['match_id'], False, 0)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.mark_prediction_executed(prediction['match_id'], False, 0)
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   ✅ Executed: {executed_count} predictions")
        print(f"   ⏳ Remaining: {len(ready_predictions) - executed_count}")
        
        if len(ready_predictions) > max_executions:
            print(f"   💡 Run again later to execute remaining predictions")
    
    def get_ready_predictions(self) -> List[Dict[str, Any]]:
        """Get predictions that are ready for execution"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        
        # Get predictions that are:
        # 1. Still queued (not executed)
        # 2. Past their optimal time (or within 5 minutes)
        # 3. Match hasn't started yet
        
        cursor.execute('''
            SELECT * FROM local_predictions
            WHERE status = 'queued'
            AND optimal_prediction_time <= ?
            AND start_time > ?
            ORDER BY optimal_prediction_time
        ''', (now + timedelta(minutes=5), now))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        columns = ['id', 'match_id', 'teams', 'format', 'start_time', 
                  'optimal_prediction_time', 'queued_time', 'executed_time', 
                  'status', 'api_calls_used', 'learning_processed', 'created_at']
        
        predictions = []
        for row in rows:
            pred_dict = dict(zip(columns, row))
            # Convert timestamp strings back to datetime objects
            pred_dict['start_time'] = datetime.fromisoformat(pred_dict['start_time'])
            pred_dict['optimal_prediction_time'] = datetime.fromisoformat(pred_dict['optimal_prediction_time'])
            predictions.append(pred_dict)
        
        return predictions
    
    def show_next_prediction_time(self):
        """Show when the next prediction will be ready"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT teams, optimal_prediction_time FROM local_predictions
            WHERE status = 'queued'
            ORDER BY optimal_prediction_time
            LIMIT 1
        ''')
        
        next_pred = cursor.fetchone()
        conn.close()
        
        if next_pred:
            next_time = datetime.fromisoformat(next_pred[1])
            time_until = next_time - datetime.now()
            
            if time_until.total_seconds() > 0:
                hours = int(time_until.total_seconds() // 3600)
                minutes = int((time_until.total_seconds() % 3600) // 60)
                print(f"⏰ Next prediction ready in {hours}h {minutes}m: {next_pred[0]}")
            else:
                print(f"🎯 Next prediction ready now: {next_pred[0]}")
    
    def mark_prediction_executed(self, match_id: str, success: bool, api_calls: int):
        """Mark prediction as executed"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        status = 'completed' if success else 'failed'
        
        cursor.execute('''
            UPDATE local_predictions
            SET executed_time = ?, status = ?, api_calls_used = ?
            WHERE match_id = ?
        ''', (datetime.now(), status, api_calls, match_id))
        
        conn.commit()
        conn.close()
    
    def show_status(self):
        """Show current system status"""
        
        print(f"\n📊 SMART LOCAL SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Today's summary
        today = datetime.now().date()
        
        cursor.execute('''
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN status = 'queued' THEN 1 END) as queued,
                   COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                   COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
            FROM local_predictions
            WHERE DATE(queued_time) = ?
        ''', (today,))
        
        stats = cursor.fetchone()
        
        print(f"📅 TODAY'S PREDICTIONS:")
        print(f"   📝 Total queued: {stats[0]}")
        print(f"   ⏳ Waiting: {stats[1]}")
        print(f"   ✅ Completed: {stats[2]}")
        print(f"   ❌ Failed: {stats[3]}")
        
        # Ready predictions
        ready = self.get_ready_predictions()
        print(f"\n🎯 READY NOW: {len(ready)} predictions")
        
        if ready:
            for pred in ready[:3]:
                time_diff = datetime.now() - pred['optimal_prediction_time']
                status = "NOW" if time_diff.total_seconds() > 0 else f"in {-int(time_diff.total_seconds()/60)}min"
                print(f"   🏏 {pred['teams']} - {status}")
        
        # Upcoming predictions
        cursor.execute('''
            SELECT teams, optimal_prediction_time FROM local_predictions
            WHERE status = 'queued' AND optimal_prediction_time > ?
            ORDER BY optimal_prediction_time
            LIMIT 3
        ''', (datetime.now(),))
        
        upcoming = cursor.fetchall()
        
        if upcoming:
            print(f"\n⏰ UPCOMING:")
            for teams, opt_time in upcoming:
                opt_datetime = datetime.fromisoformat(opt_time)
                time_until = opt_datetime - datetime.now()
                hours = int(time_until.total_seconds() // 3600)
                minutes = int((time_until.total_seconds() % 3600) // 60)
                print(f"   🏏 {teams} - in {hours}h {minutes}m")
        
        # Internet status
        internet_status = "✅ Connected" if self.has_internet() else "❌ Offline"
        print(f"\n🌐 Internet: {internet_status}")
        
        conn.close()
    
    def log_session(self, date, matches_found: int, predictions_queued: int):
        """Log session activity"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO local_sessions
            (session_date, discovery_time, matches_found, predictions_queued)
            VALUES (?, ?, ?, ?)
        ''', (date, datetime.now(), matches_found, predictions_queued))
        
        conn.commit()
        conn.close()
    
    def show_todays_queue(self):
        """Show today's queued predictions"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        cursor.execute('''
            SELECT teams, optimal_prediction_time, status
            FROM local_predictions
            WHERE DATE(queued_time) = ?
            ORDER BY optimal_prediction_time
        ''', (today,))
        
        predictions = cursor.fetchall()
        
        if predictions:
            print(f"\n📋 TODAY'S QUEUE ({len(predictions)} predictions):")
            for teams, opt_time, status in predictions:
                opt_datetime = datetime.fromisoformat(opt_time)
                status_emoji = {'queued': '⏳', 'completed': '✅', 'failed': '❌'}.get(status, '❓')
                print(f"   {status_emoji} {teams} - {opt_datetime.strftime('%H:%M')}")
        
        conn.close()

def main():
    """Main function with smart local options"""
    import sys
    
    system = SmartLocalSystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'discovery':
            # Run discovery when convenient
            system.daily_discovery_when_available()
            
        elif command == 'execute':
            # Execute ready predictions
            system.execute_available_predictions()
            
        elif command == 'status':
            # Show current status
            system.show_status()
            
        elif command == 'queue':
            # Show today's queue
            system.show_todays_queue()
            
    else:
        print("🏠 SMART LOCAL SYSTEM - Works with YOUR schedule")
        print("="*50)
        print("Commands:")
        print("  discovery  # Discover today's matches (run once daily)")
        print("  execute    # Execute ready predictions (run when convenient)")
        print("  status     # Show system status")
        print("  queue      # Show today's queue")
        print()
        print("💡 Typical daily workflow:")
        print("  1. Morning: python3 smart_local_system.py discovery")
        print("  2. During day: python3 smart_local_system.py execute")
        print("  3. Anytime: python3 smart_local_system.py status")

if __name__ == "__main__":
    main()
