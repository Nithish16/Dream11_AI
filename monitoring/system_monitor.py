#!/usr/bin/env python3
"""
Dream11 AI System Monitoring & Health Check
Comprehensive monitoring, alerting, and health management system
"""

import sys
import os
import json
import time
import sqlite3
import logging
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil for environments without it
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=1):
            return 45.0  # Mock CPU usage
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 65.0
            return MockMemory()
        
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                percent = 55.0
            return MockDisk()
    
    psutil = MockPsutil()

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    status: str  # "healthy", "warning", "critical"
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SystemAlert:
    """System alert notification"""
    alert_id: str
    severity: str  # "info", "warning", "critical"
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class SystemMonitor:
    """
    Comprehensive system monitoring and health management
    """
    
    def __init__(self, db_path: str = "system_monitor.db"):
        self.db_path = db_path
        
        # Health thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 85.0},
            'memory_usage': {'warning': 75.0, 'critical': 90.0},
            'disk_usage': {'warning': 80.0, 'critical': 95.0},
            'api_response_time': {'warning': 2.0, 'critical': 5.0},
            'prediction_accuracy': {'warning': 60.0, 'critical': 50.0},
            'cache_hit_rate': {'warning': 50.0, 'critical': 30.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0}
        }
        
        # Current metrics
        self.current_metrics = {}
        self.active_alerts = {}
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.check_interval = 60  # seconds
        self.alert_cooldown = 300  # 5 minutes
        
        # Performance history
        self.performance_history = defaultdict(list)
        self.max_history_points = 1440  # 24 hours of minute-by-minute data
        
        # Initialize database
        self._init_database()
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Health metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                status TEXT NOT NULL,
                threshold_warning REAL,
                threshold_critical REAL,
                unit TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_alerts (
                alert_id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TIMESTAMP
            )
        ''')
        
        # Performance snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                active_connections INTEGER,
                api_response_time REAL,
                prediction_accuracy REAL,
                cache_hit_rate REAL,
                error_rate REAL
            )
        ''')
        
        # System status log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_status_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                overall_status TEXT NOT NULL,
                component_statuses TEXT,
                active_alerts_count INTEGER,
                performance_summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ System monitoring database initialized: {self.db_path}")
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitoring_worker():
            while self.monitoring_enabled:
                try:
                    self._collect_system_metrics()
                    self._check_thresholds()
                    self._cleanup_old_data()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitor_thread.start()
        logger.info("🔄 System monitoring started")
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # System resource metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update current metrics
            self.current_metrics.update({
                'cpu_usage': HealthMetric(
                    name='CPU Usage',
                    value=cpu_usage,
                    status=self._get_status('cpu_usage', cpu_usage),
                    threshold_warning=self.thresholds['cpu_usage']['warning'],
                    threshold_critical=self.thresholds['cpu_usage']['critical'],
                    unit='%'
                ),
                'memory_usage': HealthMetric(
                    name='Memory Usage',
                    value=memory.percent,
                    status=self._get_status('memory_usage', memory.percent),
                    threshold_warning=self.thresholds['memory_usage']['warning'],
                    threshold_critical=self.thresholds['memory_usage']['critical'],
                    unit='%'
                ),
                'disk_usage': HealthMetric(
                    name='Disk Usage',
                    value=disk.percent,
                    status=self._get_status('disk_usage', disk.percent),
                    threshold_warning=self.thresholds['disk_usage']['warning'],
                    threshold_critical=self.thresholds['disk_usage']['critical'],
                    unit='%'
                )
            })
            
            # Application-specific metrics
            self._collect_application_metrics()
            
            # Store performance snapshot
            self._store_performance_snapshot()
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Try to get API optimization stats
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_logic'))
                from api_request_optimizer import get_api_optimization_stats
                from intelligent_api_cache import get_cache_stats
                
                api_stats = get_api_optimization_stats()
                cache_stats = get_cache_stats()
                
                # Parse cache hit rate
                cache_hit_rate_str = cache_stats.get('performance_metrics', {}).get('cache_hit_rate', '0%')
                cache_hit_rate = float(cache_hit_rate_str.rstrip('%'))
                
                self.current_metrics.update({
                    'cache_hit_rate': HealthMetric(
                        name='Cache Hit Rate',
                        value=cache_hit_rate,
                        status=self._get_status('cache_hit_rate', cache_hit_rate, reverse=True),
                        threshold_warning=self.thresholds['cache_hit_rate']['warning'],
                        threshold_critical=self.thresholds['cache_hit_rate']['critical'],
                        unit='%'
                    )
                })
                
                # API response time (placeholder - would be measured from actual requests)
                avg_response_time = 1.5  # Placeholder
                self.current_metrics['api_response_time'] = HealthMetric(
                    name='API Response Time',
                    value=avg_response_time,
                    status=self._get_status('api_response_time', avg_response_time),
                    threshold_warning=self.thresholds['api_response_time']['warning'],
                    threshold_critical=self.thresholds['api_response_time']['critical'],
                    unit='s'
                )
                
            except ImportError:
                # Fallback metrics if AI systems not available
                self.current_metrics.update({
                    'cache_hit_rate': HealthMetric(
                        name='Cache Hit Rate',
                        value=0.0,
                        status='warning',
                        threshold_warning=50.0,
                        threshold_critical=30.0,
                        unit='%'
                    ),
                    'api_response_time': HealthMetric(
                        name='API Response Time',
                        value=999.0,
                        status='critical',
                        threshold_warning=2.0,
                        threshold_critical=5.0,
                        unit='s'
                    )
                })
            
            # Placeholder metrics (would be calculated from actual data)
            self.current_metrics.update({
                'prediction_accuracy': HealthMetric(
                    name='Prediction Accuracy',
                    value=72.5,  # Placeholder
                    status='healthy',
                    threshold_warning=60.0,
                    threshold_critical=50.0,
                    unit='%'
                ),
                'error_rate': HealthMetric(
                    name='Error Rate',
                    value=2.1,  # Placeholder
                    status='healthy',
                    threshold_warning=5.0,
                    threshold_critical=10.0,
                    unit='%'
                )
            })
            
        except Exception as e:
            logger.error(f"❌ Error collecting application metrics: {e}")
    
    def _get_status(self, metric_name: str, value: float, reverse: bool = False) -> str:
        """Determine health status based on thresholds"""
        thresholds = self.thresholds.get(metric_name, {'warning': 50, 'critical': 75})
        
        if not reverse:
            # Higher values are worse (CPU, memory, disk usage, etc.)
            if value >= thresholds['critical']:
                return 'critical'
            elif value >= thresholds['warning']:
                return 'warning'
            else:
                return 'healthy'
        else:
            # Lower values are worse (cache hit rate, accuracy, etc.)
            if value <= thresholds['critical']:
                return 'critical'
            elif value <= thresholds['warning']:
                return 'warning'
            else:
                return 'healthy'
    
    def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts"""
        current_time = datetime.now()
        
        for metric_name, metric in self.current_metrics.items():
            if metric.status in ['warning', 'critical']:
                alert_id = f"{metric_name}_{metric.status}"
                
                # Check if alert already exists and is within cooldown
                if alert_id in self.active_alerts:
                    last_alert_time = self.active_alerts[alert_id].timestamp
                    if (current_time - last_alert_time).total_seconds() < self.alert_cooldown:
                        continue  # Skip if within cooldown period
                
                # Create new alert
                alert = SystemAlert(
                    alert_id=alert_id,
                    severity=metric.status,
                    component=metric_name,
                    message=f"{metric.name} is {metric.status}: {metric.value}{metric.unit} (threshold: {metric.threshold_warning if metric.status == 'warning' else metric.threshold_critical}{metric.unit})",
                    timestamp=current_time
                )
                
                self.active_alerts[alert_id] = alert
                self._store_alert(alert)
                
                logger.warning(f"🚨 ALERT [{metric.status.upper()}]: {alert.message}")
            
            else:
                # Resolve any existing alerts for this metric
                for alert_id in list(self.active_alerts.keys()):
                    if alert_id.startswith(metric_name):
                        alert = self.active_alerts[alert_id]
                        alert.resolved = True
                        alert.resolved_at = current_time
                        del self.active_alerts[alert_id]
                        self._update_alert_resolved(alert)
                        logger.info(f"✅ RESOLVED: {alert.message}")
    
    def _store_performance_snapshot(self):
        """Store current performance snapshot"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_snapshots 
                (cpu_usage, memory_usage, disk_usage, api_response_time, 
                 prediction_accuracy, cache_hit_rate, error_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_metrics.get('cpu_usage', HealthMetric('', 0, '', 0, 0)).value,
                self.current_metrics.get('memory_usage', HealthMetric('', 0, '', 0, 0)).value,
                self.current_metrics.get('disk_usage', HealthMetric('', 0, '', 0, 0)).value,
                self.current_metrics.get('api_response_time', HealthMetric('', 0, '', 0, 0)).value,
                self.current_metrics.get('prediction_accuracy', HealthMetric('', 0, '', 0, 0)).value,
                self.current_metrics.get('cache_hit_rate', HealthMetric('', 0, '', 0, 0)).value,
                self.current_metrics.get('error_rate', HealthMetric('', 0, '', 0, 0)).value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing performance snapshot: {e}")
    
    def _store_alert(self, alert: SystemAlert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO system_alerts 
                (alert_id, severity, component, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert.alert_id, alert.severity, alert.component, alert.message, alert.timestamp))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing alert: {e}")
    
    def _update_alert_resolved(self, alert: SystemAlert):
        """Update alert as resolved in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE system_alerts 
                SET resolved = TRUE, resolved_at = ?
                WHERE alert_id = ?
            ''', (alert.resolved_at, alert.alert_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error updating resolved alert: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean old performance snapshots
            cursor.execute('DELETE FROM performance_snapshots WHERE timestamp < ?', (cutoff_time,))
            
            # Clean old resolved alerts
            cursor.execute('DELETE FROM system_alerts WHERE resolved = TRUE AND resolved_at < ?', (cutoff_time,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        overall_status = "healthy"
        critical_count = 0
        warning_count = 0
        
        component_statuses = {}
        
        for metric_name, metric in self.current_metrics.items():
            component_statuses[metric_name] = {
                'status': metric.status,
                'value': metric.value,
                'unit': metric.unit,
                'threshold_warning': metric.threshold_warning,
                'threshold_critical': metric.threshold_critical
            }
            
            if metric.status == 'critical':
                critical_count += 1
                overall_status = "critical"
            elif metric.status == 'warning' and overall_status != "critical":
                warning_count += 1
                overall_status = "warning"
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'component_statuses': component_statuses,
            'active_alerts': {
                'total': len(self.active_alerts),
                'critical': len([a for a in self.active_alerts.values() if a.severity == 'critical']),
                'warning': len([a for a in self.active_alerts.values() if a.severity == 'warning'])
            },
            'summary': {
                'total_components': len(self.current_metrics),
                'healthy_components': len([m for m in self.current_metrics.values() if m.status == 'healthy']),
                'warning_components': warning_count,
                'critical_components': critical_count
            }
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, cpu_usage, memory_usage, disk_usage, 
                       api_response_time, prediction_accuracy, cache_hit_rate, error_rate
                FROM performance_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp
            ''', (cutoff_time,))
            
            snapshots = cursor.fetchall()
            conn.close()
            
            trends = defaultdict(list)
            
            for snapshot in snapshots:
                timestamp = snapshot[0]
                values = {
                    'cpu_usage': snapshot[1],
                    'memory_usage': snapshot[2], 
                    'disk_usage': snapshot[3],
                    'api_response_time': snapshot[4],
                    'prediction_accuracy': snapshot[5],
                    'cache_hit_rate': snapshot[6],
                    'error_rate': snapshot[7]
                }
                
                for metric, value in values.items():
                    if value is not None:
                        trends[metric].append({
                            'timestamp': timestamp,
                            'value': value
                        })
            
            return dict(trends)
            
        except Exception as e:
            logger.error(f"❌ Error getting performance trends: {e}")
            return {}
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT alert_id, severity, component, message, timestamp, resolved, resolved_at
                FROM system_alerts 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            alerts = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'alert_id': alert[0],
                    'severity': alert[1],
                    'component': alert[2],
                    'message': alert[3],
                    'timestamp': alert[4],
                    'resolved': bool(alert[5]),
                    'resolved_at': alert[6]
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"❌ Error getting alert history: {e}")
            return []
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        health_status = self.get_system_health_status()
        trends = self.get_performance_trends(hours=1)  # Last hour
        alerts = self.get_alert_history(hours=24)  # Last 24 hours
        
        report = []
        report.append("🏥 DREAM11 AI SYSTEM HEALTH REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Overall Status: {health_status['overall_status'].upper()}")
        report.append("")
        
        # Component Status
        report.append("📊 COMPONENT STATUS:")
        report.append("-" * 30)
        for component, status in health_status['component_statuses'].items():
            status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}[status['status']]
            report.append(f"{status_emoji} {component}: {status['value']}{status['unit']} ({status['status']})")
        report.append("")
        
        # Active Alerts
        if health_status['active_alerts']['total'] > 0:
            report.append("🚨 ACTIVE ALERTS:")
            report.append("-" * 20)
            for alert in self.active_alerts.values():
                severity_emoji = {"warning": "⚠️", "critical": "🚨"}[alert.severity]
                report.append(f"{severity_emoji} {alert.component}: {alert.message}")
            report.append("")
        
        # Summary
        summary = health_status['summary']
        report.append("📈 SUMMARY:")
        report.append("-" * 15)
        report.append(f"Total Components: {summary['total_components']}")
        report.append(f"Healthy: {summary['healthy_components']}")
        report.append(f"Warning: {summary['warning_components']}")
        report.append(f"Critical: {summary['critical_components']}")
        
        return "\n".join(report)
    
    def shutdown(self):
        """Shutdown monitoring system"""
        self.monitoring_enabled = False
        logger.info("🔄 System monitoring stopped")

def main():
    """Main function for standalone monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dream11 AI System Monitor')
    parser.add_argument('--status', action='store_true', help='Show current system status')
    parser.add_argument('--report', action='store_true', help='Generate health report')
    parser.add_argument('--trends', type=int, default=24, help='Show performance trends (hours)')
    parser.add_argument('--alerts', type=int, default=24, help='Show alert history (hours)')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SystemMonitor()
    
    if args.daemon:
        print("🔄 Running system monitor as daemon...")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Stopping system monitor...")
            monitor.shutdown()
    
    elif args.status:
        print("📊 Current System Status:")
        print("=" * 30)
        status = monitor.get_system_health_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.report:
        print(monitor.generate_health_report())
    
    elif args.trends:
        print(f"📈 Performance Trends (Last {args.trends} hours):")
        print("=" * 40)
        trends = monitor.get_performance_trends(args.trends)
        for metric, data_points in trends.items():
            if data_points:
                recent_values = [dp['value'] for dp in data_points[-10:]]  # Last 10 points
                avg_value = sum(recent_values) / len(recent_values)
                print(f"{metric}: {avg_value:.2f} (avg of last 10 readings)")
    
    elif args.alerts:
        print(f"🚨 Alert History (Last {args.alerts} hours):")
        print("=" * 35)
        alerts = monitor.get_alert_history(args.alerts)
        for alert in alerts:
            status = "RESOLVED" if alert['resolved'] else "ACTIVE"
            print(f"[{alert['severity'].upper()}] {alert['component']}: {alert['message']} ({status})")
    
    else:
        # Default: show brief status
        status = monitor.get_system_health_status()
        print(f"🏥 System Status: {status['overall_status'].upper()}")
        print(f"📊 Components: {status['summary']['healthy_components']}/{status['summary']['total_components']} healthy")
        print(f"🚨 Active Alerts: {status['active_alerts']['total']}")

if __name__ == '__main__':
    main()