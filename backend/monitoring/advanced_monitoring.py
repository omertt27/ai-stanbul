"""
Advanced Monitoring System for AI Istanbul
Comprehensive monitoring with metrics, alerting, and performance tracking
"""

import os
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import asyncio
from contextlib import asynccontextmanager
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Performance metrics storage
performance_metrics = defaultdict(deque)
system_metrics = defaultdict(deque)
error_metrics = defaultdict(int)
alert_history = deque(maxlen=1000)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    response_time: float
    error_rate: float
    
@dataclass
class AlertConfig:
    """Alert configuration"""
    name: str
    threshold: float
    comparison: str  # gt, lt, eq
    metric_path: str
    severity: str  # critical, warning, info
    cooldown_minutes: int = 15
    
@dataclass
class Alert:
    """Alert instance"""
    config: AlertConfig
    timestamp: datetime
    value: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AdvancedMonitoring:
    """Advanced monitoring system with real-time metrics and alerting"""
    
    def __init__(self):
        self.is_running = False
        self.metrics_interval = 30  # seconds
        self.alert_configs = []
        self.active_alerts = {}
        self.webhook_urls = []
        self.email_config = {}
        self.setup_default_alerts()
        
    def setup_default_alerts(self):
        """Setup default monitoring alerts"""
        default_alerts = [
            AlertConfig("high_cpu_usage", 80.0, "gt", "cpu_usage", "warning", 5),
            AlertConfig("critical_cpu_usage", 90.0, "gt", "cpu_usage", "critical", 15),
            AlertConfig("high_memory_usage", 85.0, "gt", "memory_usage", "warning", 10),
            AlertConfig("critical_memory_usage", 95.0, "gt", "memory_usage", "critical", 15),
            AlertConfig("high_error_rate", 5.0, "gt", "error_rate", "warning", 5),
            AlertConfig("critical_error_rate", 10.0, "gt", "error_rate", "critical", 10),
            AlertConfig("slow_response_time", 2.0, "gt", "response_time", "warning", 5),
            AlertConfig("critical_response_time", 5.0, "gt", "response_time", "critical", 10),
            AlertConfig("disk_space_low", 85.0, "gt", "disk_usage", "warning", 30),
            AlertConfig("disk_space_critical", 95.0, "gt", "disk_usage", "critical", 60),
        ]
        self.alert_configs.extend(default_alerts)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Active connections (approximate)
            active_connections = len(psutil.net_connections(kind='inet'))
            
            # Calculate error rate from recent metrics
            recent_errors = sum(list(error_metrics.values())[-10:])  # Last 10 error counts
            error_rate = recent_errors / max(1, len(error_metrics)) * 100
            
            # Calculate average response time from recent metrics
            recent_response_times = list(performance_metrics.get('response_time', []))[-10:]
            avg_response_time = sum(recent_response_times) / max(1, len(recent_response_times))
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_io,
                active_connections=active_connections,
                response_time=avg_response_time,
                error_rate=error_rate
            )
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return None
    
    def store_metrics(self, metrics: SystemMetrics):
        """Store metrics in memory (with rotation)"""
        if not metrics:
            return
            
        # Store with rotation (keep last 1000 entries)
        system_metrics['cpu_usage'].append(metrics.cpu_usage)
        system_metrics['memory_usage'].append(metrics.memory_usage)
        system_metrics['disk_usage'].append(metrics.disk_usage)
        system_metrics['response_time'].append(metrics.response_time)
        system_metrics['error_rate'].append(metrics.error_rate)
        system_metrics['active_connections'].append(metrics.active_connections)
        
        # Rotate old data
        for key in system_metrics:
            if len(system_metrics[key]) > 1000:
                system_metrics[key].popleft()
    
    def check_alerts(self, metrics: SystemMetrics):
        """Check all alert conditions"""
        if not metrics:
            return
            
        current_time = datetime.utcnow()
        
        for config in self.alert_configs:
            try:
                # Get metric value
                metric_value = getattr(metrics, config.metric_path, 0)
                
                # Check condition
                triggered = self._evaluate_condition(metric_value, config.threshold, config.comparison)
                
                alert_key = f"{config.name}_{config.metric_path}"
                
                if triggered:
                    # Check if already active and within cooldown
                    if alert_key in self.active_alerts:
                        last_alert_time = self.active_alerts[alert_key].timestamp
                        if (current_time - last_alert_time).total_seconds() < config.cooldown_minutes * 60:
                            continue  # Still in cooldown
                    
                    # Create new alert
                    alert = Alert(
                        config=config,
                        timestamp=current_time,
                        value=metric_value,
                        message=f"{config.name}: {config.metric_path} = {metric_value:.2f} (threshold: {config.threshold})"
                    )
                    
                    self.active_alerts[alert_key] = alert
                    alert_history.append(alert)
                    
                    # Send alert notifications
                    asyncio.create_task(self._send_alert_notifications(alert))
                    
                else:
                    # Check if we need to resolve an active alert
                    if alert_key in self.active_alerts and not self.active_alerts[alert_key].resolved:
                        self.active_alerts[alert_key].resolved = True
                        self.active_alerts[alert_key].resolved_at = current_time
                        
            except Exception as e:
                logging.error(f"Error checking alert {config.name}: {e}")
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition"""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "eq":
            return abs(value - threshold) < 0.01
        return False
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via configured channels"""
        try:
            alert_data = {
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.config.severity,
                "name": alert.config.name,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.config.threshold
            }
            
            # Send to webhooks
            for webhook_url in self.webhook_urls:
                await self._send_webhook_alert(webhook_url, alert_data)
            
            # Send email if configured
            if self.email_config:
                await self._send_email_alert(alert_data)
                
        except Exception as e:
            logging.error(f"Error sending alert notifications: {e}")
    
    async def _send_webhook_alert(self, webhook_url: str, alert_data: Dict):
        """Send alert to webhook endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=alert_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logging.info(f"Alert sent to webhook: {webhook_url}")
                    else:
                        logging.warning(f"Webhook alert failed: {response.status}")
        except Exception as e:
            logging.error(f"Webhook notification error: {e}")
    
    async def _send_email_alert(self, alert_data: Dict):
        """Send alert via email"""
        try:
            if not all(key in self.email_config for key in ['smtp_server', 'smtp_port', 'username', 'password', 'to_email']):
                return
                
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"[AI Istanbul] {alert_data['severity'].upper()} Alert: {alert_data['name']}"
            
            body = f"""
            Alert Details:
            - Name: {alert_data['name']}
            - Severity: {alert_data['severity']}
            - Message: {alert_data['message']}
            - Timestamp: {alert_data['timestamp']}
            - Current Value: {alert_data['value']}
            - Threshold: {alert_data['threshold']}
            
            Please investigate immediately if this is a critical alert.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['username'], self.email_config['to_email'], text)
            server.quit()
            
            logging.info("Alert email sent successfully")
            
        except Exception as e:
            logging.error(f"Email notification error: {e}")
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        self.is_running = True
        logging.info("Advanced monitoring started")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = self.collect_system_metrics()
                if metrics:
                    self.store_metrics(metrics)
                    self.check_alerts(metrics)
                    
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.is_running = False
        logging.info("Advanced monitoring stopped")
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Calculate recent averages
            recent_cpu = list(system_metrics.get('cpu_usage', []))[-min(hours*120, len(system_metrics.get('cpu_usage', []))):]
            recent_memory = list(system_metrics.get('memory_usage', []))[-min(hours*120, len(system_metrics.get('memory_usage', []))):]
            recent_response_time = list(system_metrics.get('response_time', []))[-min(hours*120, len(system_metrics.get('response_time', []))):]
            
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "period_hours": hours,
                "system": {
                    "cpu_usage": {
                        "current": recent_cpu[-1] if recent_cpu else 0,
                        "average": sum(recent_cpu) / max(1, len(recent_cpu)),
                        "max": max(recent_cpu) if recent_cpu else 0,
                        "min": min(recent_cpu) if recent_cpu else 0
                    },
                    "memory_usage": {
                        "current": recent_memory[-1] if recent_memory else 0,
                        "average": sum(recent_memory) / max(1, len(recent_memory)),
                        "max": max(recent_memory) if recent_memory else 0,
                        "min": min(recent_memory) if recent_memory else 0
                    },
                    "response_time": {
                        "current": recent_response_time[-1] if recent_response_time else 0,
                        "average": sum(recent_response_time) / max(1, len(recent_response_time)),
                        "max": max(recent_response_time) if recent_response_time else 0,
                        "min": min(recent_response_time) if recent_response_time else 0
                    }
                },
                "alerts": {
                    "active": len([a for a in self.active_alerts.values() if not a.resolved]),
                    "total_today": len([a for a in alert_history if a.timestamp > cutoff_time]),
                    "critical_today": len([a for a in alert_history if a.timestamp > cutoff_time and a.config.severity == "critical"])
                },
                "error_metrics": dict(error_metrics)
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating metrics summary: {e}")
            return {"error": str(e)}
    
    def configure_webhooks(self, webhook_urls: List[str]):
        """Configure webhook URLs for alerts"""
        self.webhook_urls = webhook_urls
        logging.info(f"Configured {len(webhook_urls)} webhook endpoints")
    
    def configure_email_alerts(self, smtp_config: Dict[str, str]):
        """Configure email alerts"""
        required_keys = ['smtp_server', 'smtp_port', 'username', 'password', 'to_email']
        if all(key in smtp_config for key in required_keys):
            self.email_config = smtp_config
            logging.info("Email alerts configured successfully")
        else:
            logging.error(f"Email configuration missing required keys: {required_keys}")

# Global monitoring instance
advanced_monitor = AdvancedMonitoring()

# Performance monitoring decorators
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_metrics[f"{operation_name}_duration"].append(duration)
                performance_metrics[f"{operation_name}_success"].append(1)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_metrics[f"{operation_name}_duration"].append(duration)
                performance_metrics[f"{operation_name}_error"].append(1)
                error_metrics[f"{operation_name}_errors"] += 1
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_metrics[f"{operation_name}_duration"].append(duration)
                performance_metrics[f"{operation_name}_success"].append(1)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_metrics[f"{operation_name}_duration"].append(duration)
                performance_metrics[f"{operation_name}_error"].append(1)
                error_metrics[f"{operation_name}_errors"] += 1
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def log_error_metric(error_type: str, details: str = ""):
    """Log an error metric"""
    error_metrics[error_type] += 1
    logging.error(f"Error metric logged: {error_type} - {details}")

def log_performance_metric(metric_name: str, value: float):
    """Log a custom performance metric"""
    performance_metrics[metric_name].append(value)
    if len(performance_metrics[metric_name]) > 1000:
        performance_metrics[metric_name].popleft()

# Export functions
__all__ = [
    'advanced_monitor',
    'monitor_performance', 
    'log_error_metric',
    'log_performance_metric',
    'SystemMetrics',
    'Alert',
    'AlertConfig'
]
