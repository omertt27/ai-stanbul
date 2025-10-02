#!/usr/bin/env python3
"""
Production Alert Configuration System
====================================

This module configures production-ready alerts for the AI Istanbul system,
integrating with the existing admin dashboard and monitoring infrastructure.

Features:
1. Real-time alert thresholds (error rate >1%, response time >1000ms)
2. Integration with existing admin dashboard
3. Email, Slack, and webhook notifications
4. Alert aggregation and rate limiting
5. Business KPI monitoring and alerts
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 4
    enabled: bool = True

@dataclass
class AlertEvent:
    """Alert event data"""
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class AlertConfiguration:
    """Complete alert configuration"""
    # Performance alerts
    error_rate_threshold: float = 1.0          # Alert if error rate > 1%
    response_time_threshold: float = 1000.0    # Alert if P95 response > 1000ms
    cache_hit_rate_threshold: float = 75.0     # Alert if cache hit rate < 75%
    
    # Resource alerts
    cpu_threshold: float = 80.0                # Alert if CPU > 80%
    memory_threshold: float = 80.0             # Alert if memory > 80%
    disk_threshold: float = 90.0               # Alert if disk > 90%
    
    # Business KPI alerts
    cost_savings_threshold: float = 150.0      # Alert if cost savings < $150/month
    api_cost_threshold: float = 500.0          # Alert if API costs > $500/month
    
    # Notification settings
    email_enabled: bool = True
    slack_enabled: bool = True
    webhook_enabled: bool = True
    
    # Email configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_from: str = ""
    email_to: List[str] = None
    email_password: str = ""
    
    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    # Webhook configuration
    webhook_url: str = ""

class ProductionAlertManager:
    """
    Production-ready alert management system
    """
    
    def __init__(self, config: AlertConfiguration):
        self.config = config
        
        # Alert state management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_counts = defaultdict(int)  # For rate limiting
        self.last_alert_times = defaultdict(datetime)
        
        # Alert rules
        self.alert_rules = self._initialize_alert_rules()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Dashboard integration
        self.dashboard_callbacks = []
        
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """Initialize production alert rules"""
        return [
            # Critical performance alerts
            AlertRule(
                name="high_error_rate",
                description="High error rate detected",
                metric_name="error_rate_percent",
                threshold_value=self.config.error_rate_threshold,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                cooldown_minutes=10,
                max_alerts_per_hour=6
            ),
            AlertRule(
                name="slow_response_time",
                description="Response time exceeds threshold",
                metric_name="p95_response_ms",
                threshold_value=self.config.response_time_threshold,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_minutes=15,
                max_alerts_per_hour=4
            ),
            AlertRule(
                name="low_cache_hit_rate",
                description="Cache hit rate below target",
                metric_name="cache_hit_rate_percent",
                threshold_value=self.config.cache_hit_rate_threshold,
                comparison="lt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK, AlertChannel.DASHBOARD],
                cooldown_minutes=20,
                max_alerts_per_hour=3
            ),
            
            # System resource alerts
            AlertRule(
                name="high_cpu_usage",
                description="High CPU usage detected",
                metric_name="cpu_percent",
                threshold_value=self.config.cpu_threshold,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_minutes=15,
                max_alerts_per_hour=4
            ),
            AlertRule(
                name="high_memory_usage",
                description="High memory usage detected",
                metric_name="memory_percent",
                threshold_value=self.config.memory_threshold,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_minutes=15,
                max_alerts_per_hour=4
            ),
            AlertRule(
                name="high_disk_usage",
                description="High disk usage detected",
                metric_name="disk_percent",
                threshold_value=self.config.disk_threshold,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                cooldown_minutes=30,
                max_alerts_per_hour=2
            ),
            
            # Business KPI alerts
            AlertRule(
                name="low_cost_savings",
                description="Cost savings below target",
                metric_name="cost_savings_usd",
                threshold_value=self.config.cost_savings_threshold,
                comparison="lt",
                severity=AlertSeverity.INFO,
                channels=[AlertChannel.DASHBOARD],
                cooldown_minutes=60,
                max_alerts_per_hour=1
            ),
            AlertRule(
                name="high_api_costs",
                description="API costs exceeding budget",
                metric_name="api_costs_usd",
                threshold_value=self.config.api_cost_threshold,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
                cooldown_minutes=30,
                max_alerts_per_hour=2
            )
        ]
    
    def start_monitoring(self, metrics_callback: Callable[[], Dict]):
        """Start alert monitoring with metrics callback"""
        if self.monitoring_active:
            logger.warning("Alert monitoring already active")
            return
        
        self.monitoring_active = True
        self.metrics_callback = metrics_callback
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("✅ Production alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Main alert monitoring loop"""
        while self.monitoring_active:
            try:
                # Get current metrics
                metrics = self.metrics_callback() if hasattr(self, 'metrics_callback') else {}
                
                if metrics:
                    # Check all alert rules
                    triggered_alerts = self._check_alert_rules(metrics)
                    
                    # Process triggered alerts
                    for alert in triggered_alerts:
                        self._process_alert(alert)
                    
                    # Check for resolved alerts
                    self._check_resolved_alerts(metrics)
                
                # Clean up old alert counts (for rate limiting)
                self._cleanup_alert_counts()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Alert monitoring error: {e}")
                time.sleep(60)  # Wait longer before retrying
    
    def _check_alert_rules(self, metrics: Dict) -> List[AlertEvent]:
        """Check all alert rules against current metrics"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Get metric value
            metric_value = self._get_metric_value(metrics, rule.metric_name)
            if metric_value is None:
                continue
            
            # Check if alert should trigger
            should_trigger = self._evaluate_rule(rule, metric_value)
            
            if should_trigger:
                # Check rate limiting and cooldown
                if self._should_send_alert(rule):
                    alert = AlertEvent(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=self._format_alert_message(rule, metric_value),
                        metric_value=metric_value,
                        threshold_value=rule.threshold_value,
                        timestamp=datetime.now()
                    )
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def _get_metric_value(self, metrics: Dict, metric_name: str) -> Optional[float]:
        """Extract metric value from metrics dict"""
        try:
            # Handle nested metric paths
            if '.' in metric_name:
                parts = metric_name.split('.')
                value = metrics
                for part in parts:
                    value = value.get(part, {})
                return float(value) if isinstance(value, (int, float)) else None
            else:
                # Map common metric names to their paths in the metrics structure
                metric_mappings = {
                    'error_rate_percent': 'performance.error_rate_percent',
                    'p95_response_ms': 'performance.response_time.p95_ms',
                    'cache_hit_rate_percent': 'cache.hit_rate_percent',
                    'cpu_percent': 'system.cpu_percent',
                    'memory_percent': 'system.memory_percent',
                    'disk_percent': 'system.disk_percent',
                    'cost_savings_usd': 'business_kpis.cache_savings_usd',
                    'api_costs_usd': 'business_kpis.api_costs_usd'
                }
                
                mapped_path = metric_mappings.get(metric_name, metric_name)
                return self._get_metric_value(metrics, mapped_path)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to get metric value for {metric_name}: {e}")
            return None
    
    def _evaluate_rule(self, rule: AlertRule, metric_value: float) -> bool:
        """Evaluate if alert rule should trigger"""
        if rule.comparison == "gt":
            return metric_value > rule.threshold_value
        elif rule.comparison == "lt":
            return metric_value < rule.threshold_value
        elif rule.comparison == "gte":
            return metric_value >= rule.threshold_value
        elif rule.comparison == "lte":
            return metric_value <= rule.threshold_value
        elif rule.comparison == "eq":
            return abs(metric_value - rule.threshold_value) < 0.001
        else:
            return False
    
    def _should_send_alert(self, rule: AlertRule) -> bool:
        """Check if alert should be sent based on rate limiting and cooldown"""
        current_time = datetime.now()
        
        # Check cooldown period
        last_alert_time = self.last_alert_times.get(rule.name, datetime.min)
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        
        if current_time - last_alert_time < cooldown_period:
            return False
        
        # Check hourly rate limit
        hour_key = f"{rule.name}_{current_time.hour}"
        if self.alert_counts[hour_key] >= rule.max_alerts_per_hour:
            return False
        
        return True
    
    def _format_alert_message(self, rule: AlertRule, metric_value: float) -> str:
        """Format alert message"""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        emoji = severity_emoji.get(rule.severity, "📊")
        
        return (f"{emoji} {rule.description}: "
               f"{rule.metric_name} = {metric_value:.2f} "
               f"(threshold: {rule.threshold_value:.2f})")
    
    def _process_alert(self, alert: AlertEvent):
        """Process a triggered alert"""
        try:
            # Add to active alerts
            self.active_alerts[alert.rule_name] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Update rate limiting counters
            current_time = datetime.now()
            hour_key = f"{alert.rule_name}_{current_time.hour}"
            self.alert_counts[hour_key] += 1
            self.last_alert_times[alert.rule_name] = current_time
            
            # Get alert rule for notification channels
            rule = next((r for r in self.alert_rules if r.name == alert.rule_name), None)
            if not rule:
                return
            
            # Send notifications
            self._send_alert_notifications(alert, rule)
            
            # Update dashboard
            self._update_dashboard(alert)
            
            logger.warning(f"🚨 ALERT TRIGGERED: {alert.message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process alert: {e}")
    
    def _send_alert_notifications(self, alert: AlertEvent, rule: AlertRule):
        """Send alert notifications via configured channels"""
        try:
            for channel in rule.channels:
                if channel == AlertChannel.EMAIL and self.config.email_enabled:
                    self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK and self.config.slack_enabled:
                    self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK and self.config.webhook_enabled:
                    self._send_webhook_alert(alert)
                # Dashboard notifications are handled separately
                
        except Exception as e:
            logger.error(f"❌ Failed to send alert notifications: {e}")
    
    def _send_email_alert(self, alert: AlertEvent):
        """Send email alert notification"""
        try:
            if not self.config.email_from or not self.config.email_to:
                logger.warning("⚠️ Email configuration incomplete")
                return
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[AI Istanbul {alert.severity.value.upper()}] {alert.rule_name}"
            
            # Email body
            body = f"""
AI Istanbul System Alert

Alert: {alert.message}
Severity: {alert.severity.value.upper()}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Metric Value: {alert.metric_value:.2f}
Threshold: {alert.threshold_value:.2f}

Please investigate this alert immediately.

---
AI Istanbul Monitoring System
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_from, self.config.email_password)
                server.send_message(msg)
            
            logger.info(f"📧 Email alert sent for {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert: AlertEvent):
        """Send Slack alert notification"""
        try:
            if not self.config.slack_webhook_url:
                logger.warning("⚠️ Slack webhook URL not configured")
                return
            
            # Color coding based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",      # Green
                AlertSeverity.WARNING: "#ff9500",   # Orange
                AlertSeverity.CRITICAL: "#ff0000"   # Red
            }
            
            # Slack message payload
            payload = {
                "channel": self.config.slack_channel,
                "username": "AI Istanbul Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#cccccc"),
                        "title": f"{alert.severity.value.upper()}: {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Metric Value", "value": f"{alert.metric_value:.2f}", "short": True},
                            {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True},
                            {"title": "Timestamp", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": False}
                        ],
                        "footer": "AI Istanbul Monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(self.config.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"💬 Slack alert sent for {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send Slack alert: {e}")
    
    def _send_webhook_alert(self, alert: AlertEvent):
        """Send webhook alert notification"""
        try:
            if not self.config.webhook_url:
                logger.warning("⚠️ Webhook URL not configured")
                return
            
            # Webhook payload
            payload = {
                "alert": {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "metric_value": alert.metric_value,
                    "threshold_value": alert.threshold_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                },
                "system": "ai_istanbul",
                "version": "1.0"
            }
            
            # Send webhook
            response = requests.post(self.config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"🔗 Webhook alert sent for {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send webhook alert: {e}")
    
    def _update_dashboard(self, alert: AlertEvent):
        """Update dashboard with new alert"""
        try:
            # Call registered dashboard callbacks
            for callback in self.dashboard_callbacks:
                callback(alert)
                
        except Exception as e:
            logger.error(f"❌ Failed to update dashboard: {e}")
    
    def _check_resolved_alerts(self, metrics: Dict):
        """Check if any active alerts have been resolved"""
        try:
            resolved_alerts = []
            
            for rule_name, alert in self.active_alerts.items():
                rule = next((r for r in self.alert_rules if r.name == rule_name), None)
                if not rule:
                    continue
                
                # Get current metric value
                metric_value = self._get_metric_value(metrics, rule.metric_name)
                if metric_value is None:
                    continue
                
                # Check if alert condition is no longer met
                is_resolved = not self._evaluate_rule(rule, metric_value)
                
                if is_resolved:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    resolved_alerts.append(rule_name)
                    
                    logger.info(f"✅ Alert resolved: {rule_name}")
            
            # Remove resolved alerts from active alerts
            for rule_name in resolved_alerts:
                del self.active_alerts[rule_name]
                
        except Exception as e:
            logger.error(f"❌ Failed to check resolved alerts: {e}")
    
    def _cleanup_alert_counts(self):
        """Clean up old alert counts for rate limiting"""
        try:
            current_hour = datetime.now().hour
            keys_to_remove = []
            
            for key in self.alert_counts.keys():
                if key.endswith(f"_{current_hour}"):
                    continue  # Keep current hour
                
                # Remove counts older than 1 hour
                try:
                    hour = int(key.split('_')[-1])
                    if abs(hour - current_hour) > 1:
                        keys_to_remove.append(key)
                except (ValueError, IndexError):
                    keys_to_remove.append(key)  # Invalid key format
            
            for key in keys_to_remove:
                del self.alert_counts[key]
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup alert counts: {e}")
    
    def register_dashboard_callback(self, callback: Callable[[AlertEvent], None]):
        """Register callback for dashboard updates"""
        self.dashboard_callbacks.append(callback)
    
    def get_alert_status(self) -> Dict:
        """Get current alert status"""
        try:
            return {
                'active_alerts': {name: asdict(alert) for name, alert in self.active_alerts.items()},
                'alert_history': [asdict(alert) for alert in list(self.alert_history)[-10:]],  # Last 10
                'alert_rules': [asdict(rule) for rule in self.alert_rules],
                'monitoring_active': self.monitoring_active,
                'total_alerts_triggered': len(self.alert_history),
                'active_alerts_count': len(self.active_alerts)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get alert status: {e}")
            return {}
    
    def update_alert_rule(self, rule_name: str, **kwargs):
        """Update an existing alert rule"""
        try:
            rule = next((r for r in self.alert_rules if r.name == rule_name), None)
            if not rule:
                logger.error(f"❌ Alert rule not found: {rule_name}")
                return False
            
            # Update rule properties
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            logger.info(f"✅ Alert rule updated: {rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update alert rule: {e}")
            return False

def create_production_alert_manager(config: AlertConfiguration = None) -> ProductionAlertManager:
    """Create production alert manager with default configuration"""
    if config is None:
        config = AlertConfiguration()
    
    manager = ProductionAlertManager(config)
    logger.info("✅ Production alert manager created")
    return manager

# Integration with existing admin dashboard
def integrate_with_admin_dashboard(alert_manager: ProductionAlertManager):
    """Integrate alert manager with existing admin dashboard"""
    try:
        # This would integrate with the existing admin dashboard
        # For now, we'll create a callback that logs dashboard updates
        def dashboard_callback(alert: AlertEvent):
            logger.info(f"📊 Dashboard updated with alert: {alert.rule_name}")
        
        alert_manager.register_dashboard_callback(dashboard_callback)
        logger.info("✅ Alert manager integrated with admin dashboard")
        
    except Exception as e:
        logger.error(f"❌ Failed to integrate with admin dashboard: {e}")

# Example usage and configuration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Istanbul Production Alert Manager")
    parser.add_argument("--config", help="Path to alert configuration JSON file")
    parser.add_argument("--test", action="store_true", help="Run test alerts")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = AlertConfiguration(**config_data)
    else:
        config = AlertConfiguration()
    
    # Create alert manager
    alert_manager = create_production_alert_manager(config)
    
    # Integrate with dashboard
    integrate_with_admin_dashboard(alert_manager)
    
    if args.test:
        # Run test alerts
        print("🧪 Running test alerts...")
        test_metrics = {
            'performance': {'error_rate_percent': 2.5, 'response_time': {'p95_ms': 1200}},
            'cache': {'hit_rate_percent': 70},
            'system': {'cpu_percent': 85, 'memory_percent': 75, 'disk_percent': 95},
            'business_kpis': {'cache_savings_usd': 100, 'api_costs_usd': 600}
        }
        
        # Simulate metrics callback
        def test_metrics_callback():
            return test_metrics
        
        alert_manager.start_monitoring(test_metrics_callback)
        time.sleep(60)  # Run for 1 minute
        alert_manager.stop_monitoring()
        
        print("✅ Test alerts completed")
    else:
        print("🚀 Production alert manager ready. Integrate with your metrics source.")
        print("Example integration:")
        print("  manager.start_monitoring(your_metrics_callback)")
