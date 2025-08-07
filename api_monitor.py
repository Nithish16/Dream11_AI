#!/usr/bin/env python3
"""
API Optimization Monitor - Real-time Dashboard
Monitor API usage, cache performance, and cost savings in real-time
"""

import time
import json
from datetime import datetime
from utils.api_client import (
    get_api_optimization_status, print_optimization_report, 
    clear_cache, invalidate_live_data, RATE_LIMITING_ENABLED
)
from utils.predictive_cache import run_manual_cache_warming

def display_live_dashboard():
    """Display real-time API optimization dashboard"""
    
    while True:
        # Clear screen
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        
        print("🚀 DREAM11 AI - API OPTIMIZATION DASHBOARD")
        print("=" * 70)
        print(f"📅 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if not RATE_LIMITING_ENABLED:
            print("❌ API Optimization: DISABLED")
            print("💡 Install aiohttp and restart to enable optimization")
            print()
            print("Press Ctrl+C to exit")
            time.sleep(5)
            continue
        
        status = get_api_optimization_status()
        rate_limit = status["rate_limiting"]
        cache = status["caching"]
        perf = status["performance_summary"]
        
        # Rate Limiting Status
        print("🛡️ RATE LIMITING STATUS:")
        print(f"   Status: {'🟢 Available' if rate_limit['can_make_request'] else '🔴 Limited'}")
        print(f"   Tokens Available: {rate_limit['tokens_available']:.1f}/10.0")
        print(f"   Requests This Minute: {rate_limit['requests_this_minute']}/100")
        print(f"   Requests This Hour: {rate_limit['requests_this_hour']}/1000")
        print(f"   Requests Today: {rate_limit['requests_today']}/10000")
        print(f"   Total Requests Made: {rate_limit['total_requests_made']}")
        
        if rate_limit['wait_time_seconds'] > 0:
            print(f"   ⏰ Wait Time: {rate_limit['wait_time_seconds']:.1f}s")
        
        print()
        
        # Caching Performance
        print("⚡ CACHE PERFORMANCE:")
        print(f"   Hit Rate: {cache['hit_rate_percent']:.1f}% {'🟢' if cache['hit_rate_percent'] > 70 else '🟡' if cache['hit_rate_percent'] > 50 else '🔴'}")
        print(f"   Memory Cache: {cache['memory_entries']} entries ({cache['memory_usage_mb']:.2f}MB)")
        print(f"   Disk Cache: {cache['disk_entries']} entries")
        print(f"   Cache Hits: {cache['memory_hits']} memory + {cache['disk_hits']} disk")
        print(f"   Cache Misses: {cache['misses']}")
        
        print()
        
        # Performance Metrics
        print("💰 COST & PERFORMANCE:")
        print(f"   API Calls Saved: {perf['api_calls_saved']}")
        print(f"   Cost Reduction: {perf['cost_savings_percent']:.1f}%")
        print(f"   Monthly Savings: ${perf['estimated_monthly_savings_usd']:.2f}")
        print(f"   Performance: {perf['performance_improvement']}")
        
        print()
        
        # Status Indicators
        print("📊 SYSTEM HEALTH:")
        health_score = calculate_health_score(rate_limit, cache, perf)
        print(f"   Overall Health: {health_score['status']} ({health_score['score']:.1f}/10.0)")
        
        for issue in health_score['issues']:
            print(f"   ⚠️ {issue}")
        
        for success in health_score['successes']:
            print(f"   ✅ {success}")
        
        print()
        print("=" * 70)
        print("🔧 Commands: [C]lear cache, [I]nvalidate live data, [W]arm cache, [R]eport, [Q]uit")
        print("Press Ctrl+C to exit dashboard mode")
        
        # Update every 5 seconds
        time.sleep(5)

def calculate_health_score(rate_limit, cache, perf):
    """Calculate overall system health score"""
    score = 10.0
    issues = []
    successes = []
    
    # Rate limiting health
    if not rate_limit['can_make_request']:
        score -= 3.0
        issues.append("Rate limited - requests blocked")
    elif rate_limit['tokens_available'] < 2:
        score -= 1.0
        issues.append("Low token availability")
    else:
        successes.append("Rate limiting healthy")
    
    # Cache performance
    hit_rate = cache['hit_rate_percent']
    if hit_rate < 30:
        score -= 2.0
        issues.append(f"Low cache hit rate ({hit_rate:.1f}%)")
    elif hit_rate < 50:
        score -= 1.0
        issues.append(f"Below-average cache hit rate ({hit_rate:.1f}%)")
    elif hit_rate > 70:
        successes.append(f"Excellent cache hit rate ({hit_rate:.1f}%)")
    else:
        successes.append(f"Good cache hit rate ({hit_rate:.1f}%)")
    
    # Memory usage
    if cache['memory_usage_mb'] > 80:
        score -= 1.0
        issues.append("High memory usage")
    
    # Request volume
    if rate_limit['requests_today'] > 8000:
        score -= 1.0
        issues.append("High daily API usage")
    elif rate_limit['requests_today'] < 100:
        successes.append("Efficient API usage")
    
    # Cost savings
    if perf['cost_savings_percent'] > 60:
        successes.append(f"Excellent cost savings ({perf['cost_savings_percent']:.1f}%)")
    elif perf['cost_savings_percent'] < 20:
        issues.append(f"Low cost savings ({perf['cost_savings_percent']:.1f}%)")
    
    # Determine status
    if score >= 8.5:
        status = "🟢 EXCELLENT"
    elif score >= 7.0:
        status = "🟡 GOOD"
    elif score >= 5.0:
        status = "🟠 ACCEPTABLE"
    else:
        status = "🔴 NEEDS ATTENTION"
    
    return {
        "score": max(0, score),
        "status": status,
        "issues": issues,
        "successes": successes
    }

def interactive_monitor():
    """Interactive monitoring with commands"""
    print("🚀 DREAM11 AI - Interactive API Monitor")
    print("=" * 50)
    
    while True:
        print("\n📋 Available Commands:")
        print("  1. [S] Status Report")
        print("  2. [D] Live Dashboard")
        print("  3. [C] Clear Expired Cache")
        print("  4. [I] Invalidate Live Data")
        print("  5. [W] Warm Cache (Predictive)")
        print("  6. [R] Full Optimization Report")
        print("  7. [E] Export Metrics")
        print("  8. [Q] Quit")
        
        choice = input("\n💡 Enter command: ").upper().strip()
        
        if choice == 'S':
            show_status_report()
        elif choice == 'D':
            try:
                display_live_dashboard()
            except KeyboardInterrupt:
                print("\n📊 Exited dashboard mode")
        elif choice == 'C':
            clear_cache()
        elif choice == 'I':
            invalidate_live_data()
        elif choice == 'W':
            print("🔮 Starting predictive cache warming...")
            run_manual_cache_warming()
        elif choice == 'R':
            print_optimization_report()
        elif choice == 'E':
            export_metrics()
        elif choice == 'Q':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid command. Please try again.")

def show_status_report():
    """Show current status report"""
    if not RATE_LIMITING_ENABLED:
        print("❌ API optimization not available")
        return
    
    status = get_api_optimization_status()
    rate_limit = status["rate_limiting"]
    cache = status["caching"]
    perf = status["performance_summary"]
    
    print("\n📊 CURRENT STATUS SNAPSHOT")
    print("-" * 40)
    print(f"🛡️ Rate Limiter: {'✅ Healthy' if rate_limit['can_make_request'] else '🚫 Limited'}")
    print(f"⚡ Cache: {cache['hit_rate_percent']:.1f}% hit rate")
    print(f"💰 Savings: {perf['cost_savings_percent']:.1f}% cost reduction")
    print(f"🚀 API Calls: {rate_limit['total_requests_made']} total")
    
    health = calculate_health_score(rate_limit, cache, perf)
    print(f"📈 Health Score: {health['status']} ({health['score']:.1f}/10.0)")

def export_metrics():
    """Export current metrics to JSON file"""
    try:
        if not RATE_LIMITING_ENABLED:
            print("❌ API optimization not available for export")
            return
        
        status = get_api_optimization_status()
        
        # Add timestamp and additional metadata
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "system_status": status,
            "health_analysis": calculate_health_score(
                status["rate_limiting"], 
                status["caching"], 
                status["performance_summary"]
            )
        }
        
        filename = f"api_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Metrics exported to: {filename}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def main():
    """Main function with command line interface"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'dashboard':
            try:
                display_live_dashboard()
            except KeyboardInterrupt:
                print("\n👋 Dashboard closed")
        elif command == 'status':
            show_status_report()
        elif command == 'report':
            print_optimization_report()
        elif command == 'clear':
            clear_cache()
        elif command == 'warm':
            run_manual_cache_warming()
        elif command == 'export':
            export_metrics()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: dashboard, status, report, clear, warm, export")
    else:
        interactive_monitor()

if __name__ == "__main__":
    main()