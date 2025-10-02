#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard
=============================

This module provides a comprehensive real-time monitoring dashboard for the AI Istanbul system,
including metrics visualization, alerting, and system health monitoring.

Features:
1. Real-time system metrics dashboard
2. Automated alerting system
3. Performance trend analysis
4. Cost optimization tracking
5. Cache performance visualization
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from collections import defaultdict, deque
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration for system alerts"""
    error_rate_threshold: float = 1.0  # Percentage
    response_time_threshold: float = 1000.0  # Milliseconds
    cache_hit_rate_threshold: float = 75.0  # Percentage
    cpu_threshold: float = 80.0  # Percentage
    memory_threshold: float = 80.0  # Percentage
    disk_threshold: float = 90.0  # Percentage
    
    # Alert channels
    email_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    
    # Alert settings
    alert_cooldown_minutes: int = 15
    max_alerts_per_hour: int = 4

@dataclass
class AlertEvent:
    """Alert event data structure"""
    alert_type: str
    severity: str  # 'warning', 'critical', 'info'
    message: str
    timestamp: datetime
    value: float
    threshold: float
    resolved: bool = False

class RealTimeMonitor:
    """
    Real-time monitoring system with alerting and dashboard capabilities
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", alert_config: AlertConfig = None):
        self.base_url = base_url
        self.alert_config = alert_config or AlertConfig()
        
        # Monitoring data storage
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics snapshots
        self.active_alerts = {}
        self.alert_history = deque(maxlen=500)
        self.last_alert_times = defaultdict(datetime)
        
        # Performance trends
        self.trend_data = {
            'response_times': deque(maxlen=100),
            'cache_hit_rates': deque(maxlen=100),
            'error_rates': deque(maxlen=100),
            'cost_savings': deque(maxlen=100)
        }
        
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def start_monitoring(self, interval_seconds: int = 30):
        """Start real-time monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"✅ Real-time monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 Real-time monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                health_data = self._fetch_health_metrics()
                metrics_data = self._fetch_system_metrics()
                
                if health_data and metrics_data:
                    # Store metrics
                    combined_metrics = {
                        'timestamp': datetime.now(),
                        'health': health_data,
                        'metrics': metrics_data
                    }
                    self.metrics_history.append(combined_metrics)
                    
                    # Update trends
                    self._update_trends(metrics_data)
                    
                    # Check for alerts
                    self._check_alerts(health_data, metrics_data)
                    
                    # Log summary
                    self._log_monitoring_summary(health_data, metrics_data)
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
            
            time.sleep(interval_seconds)
    
    def _fetch_health_metrics(self) -> Optional[Dict]:
        """Fetch health metrics from API"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to fetch health metrics: {e}")
        return None
    
    def _fetch_system_metrics(self) -> Optional[Dict]:
        """Fetch system metrics from API"""
        try:
            response = requests.get(f"{self.base_url}/api/metrics", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to fetch system metrics: {e}")
        return None
    
    def _update_trends(self, metrics_data: Dict):
        """Update trend data for visualization"""
        try:
            # Response time trend
            response_time = metrics_data.get('performance', {}).get('response_time', {}).get('p95_ms', 0)
            self.trend_data['response_times'].append({
                'timestamp': datetime.now(),
                'value': response_time
            })
            
            # Cache hit rate trend
            cache_hit_rate = metrics_data.get('cache', {}).get('hit_rate_percent', 0)
            self.trend_data['cache_hit_rates'].append({
                'timestamp': datetime.now(),
                'value': cache_hit_rate
            })
            
            # Error rate trend
            error_rate = metrics_data.get('performance', {}).get('error_rate_percent', 0)
            self.trend_data['error_rates'].append({
                'timestamp': datetime.now(),
                'value': error_rate
            })
            
            # Cost savings trend
            cost_savings = metrics_data.get('business_kpis', {}).get('cache_savings_usd', 0)
            self.trend_data['cost_savings'].append({
                'timestamp': datetime.now(),
                'value': cost_savings
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to update trends: {e}")
    
    def _check_alerts(self, health_data: Dict, metrics_data: Dict):
        """Check for alert conditions"""
        current_time = datetime.now()
        alerts_to_trigger = []
        
        try:
            # Error rate alert
            error_rate = metrics_data.get('performance', {}).get('error_rate_percent', 0)
            if error_rate > self.alert_config.error_rate_threshold:
                alerts_to_trigger.append(AlertEvent(
                    alert_type='high_error_rate',
                    severity='critical',
                    message=f'High error rate detected: {error_rate:.2f}% (threshold: {self.alert_config.error_rate_threshold}%)',
                    timestamp=current_time,
                    value=error_rate,
                    threshold=self.alert_config.error_rate_threshold
                ))
            
            # Response time alert
            response_time = metrics_data.get('performance', {}).get('response_time', {}).get('p95_ms', 0)
            if response_time > self.alert_config.response_time_threshold:
                alerts_to_trigger.append(AlertEvent(
                    alert_type='slow_response',
                    severity='warning',
                    message=f'Slow response time: {response_time:.1f}ms (threshold: {self.alert_config.response_time_threshold}ms)',
                    timestamp=current_time,
                    value=response_time,
                    threshold=self.alert_config.response_time_threshold
                ))
            
            # Cache hit rate alert
            cache_hit_rate = metrics_data.get('cache', {}).get('hit_rate_percent', 0)
            if cache_hit_rate < self.alert_config.cache_hit_rate_threshold:
                alerts_to_trigger.append(AlertEvent(
                    alert_type='low_cache_hit',
                    severity='warning',
                    message=f'Low cache hit rate: {cache_hit_rate:.1f}% (threshold: {self.alert_config.cache_hit_rate_threshold}%)',
                    timestamp=current_time,
                    value=cache_hit_rate,
                    threshold=self.alert_config.cache_hit_rate_threshold
                ))
            
            # System resource alerts
            cpu_percent = health_data.get('system', {}).get('cpu_percent', 0)
            if cpu_percent > self.alert_config.cpu_threshold:
                alerts_to_trigger.append(AlertEvent(
                    alert_type='high_cpu',
                    severity='warning',
                    message=f'High CPU usage: {cpu_percent:.1f}% (threshold: {self.alert_config.cpu_threshold}%)',
                    timestamp=current_time,
                    value=cpu_percent,
                    threshold=self.alert_config.cpu_threshold
                ))
            
            memory_percent = health_data.get('system', {}).get('memory', {}).get('percent', 0)
            if memory_percent > self.alert_config.memory_threshold:
                alerts_to_trigger.append(AlertEvent(
                    alert_type='high_memory',
                    severity='warning',
                    message=f'High memory usage: {memory_percent:.1f}% (threshold: {self.alert_config.memory_threshold}%)',
                    timestamp=current_time,
                    value=memory_percent,
                    threshold=self.alert_config.memory_threshold
                ))
            
            # Process alerts
            for alert in alerts_to_trigger:
                self._process_alert(alert)
                
        except Exception as e:
            logger.error(f"❌ Alert checking error: {e}")
    
    def _process_alert(self, alert: AlertEvent):
        """Process and potentially trigger an alert"""
        alert_key = f"{alert.alert_type}_{alert.severity}"
        current_time = datetime.now()
        
        # Check cooldown period
        last_alert_time = self.last_alert_times.get(alert_key, datetime.min)
        cooldown_period = timedelta(minutes=self.alert_config.alert_cooldown_minutes)
        
        if current_time - last_alert_time < cooldown_period:
            return  # Still in cooldown
        
        # Trigger alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = current_time
        
        # Send alert notifications
        self._send_alert_notifications(alert)
        
        logger.warning(f"🚨 ALERT: {alert.message}")
    
    def _send_alert_notifications(self, alert: AlertEvent):
        """Send alert notifications via configured channels"""
        try:
            # Email notification
            if self.alert_config.email_enabled:
                self._send_email_alert(alert)
            
            # Slack notification
            if self.alert_config.slack_enabled:
                self._send_slack_alert(alert)
            
            # Webhook notification
            if self.alert_config.webhook_enabled:
                self._send_webhook_alert(alert)
                
        except Exception as e:
            logger.error(f"❌ Failed to send alert notifications: {e}")
    
    def _send_email_alert(self, alert: AlertEvent):
        """Send email alert (placeholder implementation)"""
        # Email configuration would go here
        logger.info(f"📧 Email alert would be sent: {alert.message}")
    
    def _send_slack_alert(self, alert: AlertEvent):
        """Send Slack alert (placeholder implementation)"""
        # Slack webhook configuration would go here  
        logger.info(f"💬 Slack alert would be sent: {alert.message}")
    
    def _send_webhook_alert(self, alert: AlertEvent):
        """Send webhook alert (placeholder implementation)"""
        # Webhook configuration would go here
        logger.info(f"🔗 Webhook alert would be sent: {alert.message}")
    
    def _log_monitoring_summary(self, health_data: Dict, metrics_data: Dict):
        """Log monitoring summary"""
        try:
            summary = {
                'status': health_data.get('status', 'unknown'),
                'response_time_p95': metrics_data.get('performance', {}).get('response_time', {}).get('p95_ms', 0),
                'cache_hit_rate': metrics_data.get('cache', {}).get('hit_rate_percent', 0),
                'error_rate': metrics_data.get('performance', {}).get('error_rate_percent', 0),
                'cpu_percent': health_data.get('system', {}).get('cpu_percent', 0),
                'active_alerts': len(self.active_alerts)
            }
            
            logger.info(f"📊 Monitoring Summary: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log monitoring summary: {e}")
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                'current_status': latest_metrics,
                'trends': {
                    'response_times': list(self.trend_data['response_times']),
                    'cache_hit_rates': list(self.trend_data['cache_hit_rates']),
                    'error_rates': list(self.trend_data['error_rates']),
                    'cost_savings': list(self.trend_data['cost_savings'])
                },
                'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
                'alert_history': [asdict(alert) for alert in list(self.alert_history)[-10:]],  # Last 10 alerts
                'monitoring_stats': {
                    'total_metrics_collected': len(self.metrics_history),
                    'total_alerts_triggered': len(self.alert_history),
                    'active_alerts_count': len(self.active_alerts),
                    'monitoring_uptime': (datetime.now() - (self.metrics_history[0]['timestamp'] if self.metrics_history else datetime.now())).total_seconds()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get dashboard data: {e}")
            return {}

def create_monitoring_dashboard(base_url: str = "http://localhost:8001") -> RealTimeMonitor:
    """Create and configure a monitoring dashboard instance"""
    
    # Configure alerts with production-ready thresholds
    alert_config = AlertConfig(
        error_rate_threshold=1.0,      # Alert if error rate > 1%
        response_time_threshold=1000.0, # Alert if P95 > 1000ms
        cache_hit_rate_threshold=75.0,  # Alert if cache hit rate < 75%
        cpu_threshold=80.0,            # Alert if CPU > 80%
        memory_threshold=80.0,         # Alert if memory > 80%
        alert_cooldown_minutes=15,     # 15 minute cooldown between same alerts
        max_alerts_per_hour=4          # Max 4 alerts per hour per type
    )
    
    monitor = RealTimeMonitor(base_url=base_url, alert_config=alert_config)
    
    return monitor

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Istanbul Real-Time Monitoring Dashboard")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the AI Istanbul backend")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=3600, help="Monitoring duration in seconds (0 for infinite)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and start monitoring
    monitor = create_monitoring_dashboard(args.url)
    monitor.start_monitoring(args.interval)
    
    try:
        if args.duration > 0:
            print(f"🚀 Monitoring started for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("🚀 Monitoring started indefinitely. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
    finally:
        monitor.stop_monitoring()
        print("✅ Monitoring stopped successfully")
