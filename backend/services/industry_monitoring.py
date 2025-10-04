"""
Industry-Level Monitoring and Observability System
=================================================

Production-grade monitoring, logging, metrics, alerting, and observability
for the AI Istanbul system. Implements enterprise-level standards.
"""

import asyncio
import json
import time
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import redis
import psutil
import traceback
from contextlib import contextmanager
import uuid
import hashlib
import sqlite3

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Industry-standard metric structure"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Industry-standard alert structure"""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceTrace:
    """Distributed tracing for performance monitoring"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    status: str
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

class IndustryMonitoringSystem:
    """
    Enterprise-grade monitoring system with:
    - Real-time metrics collection
    - Distributed tracing
    - Alerting and notifications
    - Performance analytics
    - Security monitoring
    - Business intelligence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Core components
        self.metrics_storage = defaultdict(list)
        self.alerts_storage = []
        self.traces_storage = defaultdict(list)
        self.active_spans = {}
        
        # Performance monitoring
        self.response_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.throughput_counter = defaultdict(int)
        
        # System monitoring
        self.system_metrics = {}
        self.health_checks = {}
        
        # Security monitoring
        self.security_events = deque(maxlen=1000)
        self.rate_limiting = defaultdict(deque)
        
        # Business metrics
        self.business_metrics = defaultdict(float)
        self.user_analytics = defaultdict(dict)
        
        # Alerting
        self.alert_rules = []
        self.notification_channels = []
        
        # Initialize storage
        self._initialize_storage()
        
        # Start background monitoring
        self.monitoring_active = True
        self._start_background_monitoring()
        
        logger.info("🚀 Industry Monitoring System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            "metrics_retention_hours": 24,
            "traces_retention_hours": 6,
            "alerts_retention_days": 30,
            "monitoring_interval_seconds": 60,
            "health_check_interval_seconds": 30,
            "security_scan_interval_seconds": 300,
            "enable_distributed_tracing": True,
            "enable_real_time_alerts": True,
            "max_metrics_per_hour": 10000,
            "database_path": "monitoring.db"
        }
    
    def _initialize_storage(self):
        """Initialize persistent storage for monitoring data"""
        try:
            db_path = self.config.get("database_path", "monitoring.db")
            self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            self.db_lock = threading.Lock()
            
            # Create tables
            cursor = self.db_connection.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    labels TEXT
                );
                
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT NOT NULL,
                    span_id TEXT PRIMARY KEY,
                    parent_span_id TEXT,
                    operation_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    tags TEXT,
                    logs TEXT
                );
                
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source_ip TEXT,
                    user_agent TEXT,
                    request_data TEXT,
                    timestamp TEXT NOT NULL,
                    blocked INTEGER DEFAULT 0
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
                CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces(trace_id);
                CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_events(timestamp);
            """)
            self.db_connection.commit()
            logger.info("📊 Monitoring database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring storage: {e}")
            self.db_connection = None
    
    # =================== METRICS COLLECTION ===================
    
    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     tags: Optional[Dict[str, str]] = None,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric with industry-standard format"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            labels=labels or {}
        )
        
        # Store in memory for real-time access
        self.metrics_storage[name].append(metric)
        
        # Persist to database
        self._persist_metric(metric)
        
        # Check alert rules
        self._check_metric_alerts(metric)
    
    def increment_counter(self, name: str, value: float = 1.0,
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float,
                  tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float,
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    @contextmanager
    def time_operation(self, operation_name: str,
                      tags: Optional[Dict[str, str]] = None):
        """Time an operation and record as metric"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_metric(
                f"{operation_name}_duration_ms",
                duration_ms,
                MetricType.TIMER,
                tags
            )
    
    # =================== DISTRIBUTED TRACING ===================
    
    def start_trace(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new distributed trace"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        trace = PerformanceTrace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration_ms=None,
            status="started",
            tags=tags or {}
        )
        
        self.active_spans[span_id] = trace
        return span_id
    
    def start_span(self, operation_name: str, parent_span_id: str,
                   tags: Optional[Dict[str, str]] = None) -> str:
        """Start a child span in an existing trace"""
        if parent_span_id not in self.active_spans:
            return self.start_trace(operation_name, tags)
        
        parent_trace = self.active_spans[parent_span_id]
        span_id = str(uuid.uuid4())
        
        trace = PerformanceTrace(
            trace_id=parent_trace.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration_ms=None,
            status="started",
            tags=tags or {}
        )
        
        self.active_spans[span_id] = trace
        return span_id
    
    def finish_span(self, span_id: str, status: str = "success"):
        """Finish a span and calculate duration"""
        if span_id not in self.active_spans:
            return
        
        trace = self.active_spans[span_id]
        trace.end_time = datetime.now()
        trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
        trace.status = status
        
        # Store completed trace
        self.traces_storage[trace.trace_id].append(trace)
        self._persist_trace(trace)
        
        # Remove from active spans
        del self.active_spans[span_id]
        
        # Record performance metrics
        self.record_metric(
            f"trace_{trace.operation_name}_duration_ms",
            trace.duration_ms,
            MetricType.TIMER,
            {"status": status}
        )
    
    def add_span_log(self, span_id: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Add a log entry to a span"""
        if span_id not in self.active_spans:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "data": data or {}
        }
        
        self.active_spans[span_id].logs.append(log_entry)
    
    # =================== ALERTING SYSTEM ===================
    
    def create_alert(self, name: str, severity: AlertSeverity, message: str,
                    source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert"""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts_storage.append(alert)
        self._persist_alert(alert)
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        logger.warning(f"🚨 ALERT [{severity.value.upper()}]: {name} - {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        for alert in self.alerts_storage:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self._update_alert_in_db(alert)
                logger.info(f"✅ Alert resolved: {alert.name}")
                break
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, List[Metric]]], bool],
                      severity: AlertSeverity, message: str):
        """Add a custom alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message,
            "last_triggered": None
        }
        self.alert_rules.append(rule)
    
    # =================== SECURITY MONITORING ===================
    
    def log_security_event(self, event_type: str, severity: AlertSeverity,
                          source_ip: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          request_data: Optional[Dict[str, Any]] = None,
                          blocked: bool = False):
        """Log a security event"""
        event = {
            "event_type": event_type,
            "severity": severity.value,
            "source_ip": source_ip,
            "user_agent": user_agent,
            "request_data": json.dumps(request_data) if request_data else None,
            "timestamp": datetime.now().isoformat(),
            "blocked": 1 if blocked else 0
        }
        
        self.security_events.append(event)
        self._persist_security_event(event)
        
        # Create alert for high-severity security events
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self.create_alert(
                f"Security Event: {event_type}",
                severity,
                f"Security event detected from {source_ip}: {event_type}",
                "security_monitor",
                event
            )
    
    def check_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old entries
        self.rate_limiting[identifier] = deque([
            timestamp for timestamp in self.rate_limiting[identifier]
            if timestamp > window_start
        ])
        
        # Check current rate
        current_count = len(self.rate_limiting[identifier])
        if current_count >= limit:
            self.log_security_event(
                "rate_limit_exceeded",
                AlertSeverity.MEDIUM,
                source_ip=identifier,
                request_data={"current_count": current_count, "limit": limit},
                blocked=True
            )
            return False
        
        # Record this request
        self.rate_limiting[identifier].append(now)
        return True
    
    # =================== SYSTEM MONITORING ===================
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system_cpu_percent", cpu_percent, {"host": "localhost"})
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_percent", memory.percent, {"host": "localhost"})
            self.set_gauge("system_memory_used_bytes", memory.used, {"host": "localhost"})
            self.set_gauge("system_memory_available_bytes", memory.available, {"host": "localhost"})
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.set_gauge("system_disk_percent", (disk.used / disk.total) * 100, {"host": "localhost"})
            self.set_gauge("system_disk_used_bytes", disk.used, {"host": "localhost"})
            self.set_gauge("system_disk_free_bytes", disk.free, {"host": "localhost"})
            
            # Network metrics
            network = psutil.net_io_counters()
            self.increment_counter("system_network_bytes_sent", network.bytes_sent, {"host": "localhost"})
            self.increment_counter("system_network_bytes_recv", network.bytes_recv, {"host": "localhost"})
            
            # Process metrics
            process = psutil.Process()
            self.set_gauge("process_memory_rss_bytes", process.memory_info().rss, {"process": "ai_istanbul"})
            self.set_gauge("process_cpu_percent", process.cpu_percent(), {"process": "ai_istanbul"})
            
            # Store system metrics for health checks
            self.system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "process_cpu_percent": process.cpu_percent()
            }
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # System resource checks
        if self.system_metrics.get("cpu_percent", 0) > 80:
            health_status["checks"]["cpu"] = {"status": "warning", "message": "High CPU usage"}
            health_status["overall_status"] = "degraded"
        else:
            health_status["checks"]["cpu"] = {"status": "healthy", "message": "CPU usage normal"}
        
        if self.system_metrics.get("memory_percent", 0) > 90:
            health_status["checks"]["memory"] = {"status": "critical", "message": "High memory usage"}
            health_status["overall_status"] = "unhealthy"
        else:
            health_status["checks"]["memory"] = {"status": "healthy", "message": "Memory usage normal"}
        
        if self.system_metrics.get("disk_percent", 0) > 85:
            health_status["checks"]["disk"] = {"status": "warning", "message": "Low disk space"}
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"
        else:
            health_status["checks"]["disk"] = {"status": "healthy", "message": "Disk space adequate"}
        
        # Database connectivity check
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                health_status["checks"]["database"] = {"status": "healthy", "message": "Database accessible"}
            else:
                health_status["checks"]["database"] = {"status": "critical", "message": "Database not connected"}
                health_status["overall_status"] = "unhealthy"
        except Exception as e:
            health_status["checks"]["database"] = {"status": "critical", "message": f"Database error: {e}"}
            health_status["overall_status"] = "unhealthy"
        
        # Redis connectivity check (if available)
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            health_status["checks"]["redis"] = {"status": "healthy", "message": "Redis accessible"}
        except Exception:
            health_status["checks"]["redis"] = {"status": "warning", "message": "Redis not available"}
        
        # Alert on unhealthy status
        if health_status["overall_status"] == "unhealthy":
            self.create_alert(
                "System Health Critical",
                AlertSeverity.CRITICAL,
                "System health check failed - immediate attention required",
                "health_monitor",
                health_status
            )
        elif health_status["overall_status"] == "degraded":
            self.create_alert(
                "System Health Degraded",
                AlertSeverity.MEDIUM,
                "System health check shows degraded performance",
                "health_monitor",
                health_status
            )
        
        self.health_checks = health_status
        return health_status
    
    # =================== BUSINESS ANALYTICS ===================
    
    def track_user_interaction(self, user_id: str, interaction_type: str,
                              metadata: Optional[Dict[str, Any]] = None):
        """Track user interactions for business analytics"""
        self.increment_counter(
            "user_interactions_total",
            tags={"interaction_type": interaction_type}
        )
        
        # Update user analytics
        if user_id not in self.user_analytics:
            self.user_analytics[user_id] = {
                "first_seen": datetime.now().isoformat(),
                "interactions": defaultdict(int),
                "total_interactions": 0
            }
        
        self.user_analytics[user_id]["interactions"][interaction_type] += 1
        self.user_analytics[user_id]["total_interactions"] += 1
        self.user_analytics[user_id]["last_seen"] = datetime.now().isoformat()
    
    def track_business_metric(self, metric_name: str, value: float,
                            tags: Optional[Dict[str, str]] = None):
        """Track business-specific metrics"""
        self.business_metrics[metric_name] = value
        self.record_metric(f"business_{metric_name}", value, MetricType.GAUGE, tags)
    
    # =================== ANALYTICS & REPORTING ===================
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        summary = {
            "time_range": f"Last {hours} hour(s)",
            "metrics": {},
            "alerts_count": 0,
            "traces_count": 0
        }
        
        # Summarize metrics
        for metric_name, metrics_list in self.metrics_storage.items():
            recent_metrics = [m for m in metrics_list if m.timestamp > cutoff_time]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        
        # Count recent alerts
        recent_alerts = [a for a in self.alerts_storage if a.timestamp > cutoff_time]
        summary["alerts_count"] = len(recent_alerts)
        
        # Count recent traces
        recent_traces = 0
        for trace_list in self.traces_storage.values():
            recent_traces += len([t for t in trace_list if t.start_time > cutoff_time])
        summary["traces_count"] = recent_traces
        
        return summary
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        return {
            "system_health": self.health_checks,
            "system_metrics": self.system_metrics,
            "recent_alerts": [asdict(a) for a in self.alerts_storage[-10:]],
            "active_traces": len(self.active_spans),
            "metrics_summary": self.get_metrics_summary(1),
            "business_metrics": dict(self.business_metrics),
            "security_events_count": len(self.security_events),
            "rate_limiting_active": len(self.rate_limiting)
        }
    
    # =================== PERSISTENCE METHODS ===================
    
    def _persist_metric(self, metric: Metric):
        """Persist metric to database"""
        if not self.db_connection:
            return
        
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO metrics (name, value, metric_type, timestamp, tags, labels)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags),
                    json.dumps(metric.labels)
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"❌ Error persisting metric: {e}")
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to database"""
        if not self.db_connection:
            return
        
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO alerts (id, name, severity, message, timestamp, source, resolved, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.name,
                    alert.severity.value,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.source,
                    0,
                    json.dumps(alert.metadata)
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"❌ Error persisting alert: {e}")
    
    def _persist_trace(self, trace: PerformanceTrace):
        """Persist trace to database"""
        if not self.db_connection:
            return
        
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO traces (trace_id, span_id, parent_span_id, operation_name, 
                                      start_time, end_time, duration_ms, status, tags, logs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.trace_id,
                    trace.span_id,
                    trace.parent_span_id,
                    trace.operation_name,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.duration_ms,
                    trace.status,
                    json.dumps(trace.tags),
                    json.dumps(trace.logs)
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"❌ Error persisting trace: {e}")
    
    def _persist_security_event(self, event: Dict[str, Any]):
        """Persist security event to database"""
        if not self.db_connection:
            return
        
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO security_events (event_type, severity, source_ip, user_agent, 
                                               request_data, timestamp, blocked)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event["event_type"],
                    event["severity"],
                    event["source_ip"],
                    event["user_agent"],
                    event["request_data"],
                    event["timestamp"],
                    event["blocked"]
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"❌ Error persisting security event: {e}")
    
    # =================== BACKGROUND MONITORING ===================
    
    def _start_background_monitoring(self):
        """Start background monitoring threads"""
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self.collect_system_metrics()
                    self.run_health_checks()
                    self._check_alert_rules()
                    self._cleanup_old_data()
                    
                    time.sleep(self.config.get("monitoring_interval_seconds", 60))
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait before retrying
        
        def security_scan_loop():
            while self.monitoring_active:
                try:
                    self._run_security_scans()
                    time.sleep(self.config.get("security_scan_interval_seconds", 300))
                except Exception as e:
                    logger.error(f"❌ Error in security scan loop: {e}")
                    time.sleep(300)
        
        # Start background threads
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True, name="monitoring")
        security_thread = threading.Thread(target=security_scan_loop, daemon=True, name="security")
        
        monitoring_thread.start()
        security_thread.start()
        
        logger.info("🔄 Background monitoring threads started")
    
    def _check_metric_alerts(self, metric: Metric):
        """Check if metric triggers any alerts"""
        # Example: CPU usage alert
        if metric.name == "system_cpu_percent" and metric.value > 90:
            self.create_alert(
                "High CPU Usage",
                AlertSeverity.HIGH,
                f"CPU usage is {metric.value:.1f}%",
                "system_monitor"
            )
        
        # Example: Memory usage alert
        if metric.name == "system_memory_percent" and metric.value > 95:
            self.create_alert(
                "Critical Memory Usage",
                AlertSeverity.CRITICAL,
                f"Memory usage is {metric.value:.1f}%",
                "system_monitor"
            )
    
    def _check_alert_rules(self):
        """Check custom alert rules"""
        for rule in self.alert_rules:
            try:
                if rule["condition"](self.metrics_storage):
                    # Avoid duplicate alerts (5-minute cooldown)
                    now = datetime.now()
                    if (rule["last_triggered"] is None or 
                        (now - rule["last_triggered"]).total_seconds() > 300):
                        
                        self.create_alert(
                            rule["name"],
                            rule["severity"],
                            rule["message"],
                            "custom_rule"
                        )
                        rule["last_triggered"] = now
            except Exception as e:
                logger.error(f"❌ Error checking alert rule {rule['name']}: {e}")
    
    def _run_security_scans(self):
        """Run periodic security scans"""
        try:
            # Check for suspicious patterns in recent security events
            recent_events = list(self.security_events)[-100:]  # Last 100 events
            
            # Group events by source IP
            ip_events = defaultdict(list)
            for event in recent_events:
                if isinstance(event, dict) and event.get("source_ip"):
                    ip_events[event["source_ip"]].append(event)
            
            # Check for suspicious IPs
            for ip, events in ip_events.items():
                if len(events) > 10:  # More than 10 security events from same IP
                    self.create_alert(
                        "Suspicious IP Activity",
                        AlertSeverity.HIGH,
                        f"IP {ip} has generated {len(events)} security events",
                        "security_scanner",
                        {"ip": ip, "event_count": len(events)}
                    )
            
        except Exception as e:
            logger.error(f"❌ Error in security scan: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Clean up old metrics in memory
            cutoff_time = datetime.now() - timedelta(hours=self.config.get("metrics_retention_hours", 24))
            for metric_name in list(self.metrics_storage.keys()):
                self.metrics_storage[metric_name] = [
                    m for m in self.metrics_storage[metric_name] 
                    if m.timestamp > cutoff_time
                ]
            
            # Clean up old traces
            trace_cutoff = datetime.now() - timedelta(hours=self.config.get("traces_retention_hours", 6))
            for trace_id in list(self.traces_storage.keys()):
                self.traces_storage[trace_id] = [
                    t for t in self.traces_storage[trace_id]
                    if t.start_time > trace_cutoff
                ]
            
            # Clean up database (run weekly)
            if datetime.now().hour == 2 and datetime.now().minute < 5:  # 2 AM window
                self._cleanup_database()
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")
    
    def _cleanup_database(self):
        """Clean up old database records"""
        if not self.db_connection:
            return
        
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                
                # Clean up old metrics
                metrics_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (metrics_cutoff,))
                
                # Clean up old resolved alerts
                alerts_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute("DELETE FROM alerts WHERE resolved = 1 AND timestamp < ?", (alerts_cutoff,))
                
                # Clean up old traces
                traces_cutoff = (datetime.now() - timedelta(days=3)).isoformat()
                cursor.execute("DELETE FROM traces WHERE start_time < ?", (traces_cutoff,))
                
                # Clean up old security events
                security_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute("DELETE FROM security_events WHERE timestamp < ?", (security_cutoff,))
                
                self.db_connection.commit()
                logger.info("🧹 Database cleanup completed")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up database: {e}")
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        # This would integrate with notification services like:
        # - Email (SMTP)
        # - Slack webhooks
        # - PagerDuty
        # - SMS services
        # - Discord webhooks
        
        # For now, just log the alert
        logger.critical(f"🚨 ALERT NOTIFICATION: [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
    
    def _update_alert_in_db(self, alert: Alert):
        """Update alert status in database"""
        if not self.db_connection:
            return
        
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    UPDATE alerts SET resolved = ?, resolved_at = ? WHERE id = ?
                """, (1, alert.resolved_at.isoformat() if alert.resolved_at else None, alert.id))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"❌ Error updating alert in database: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the monitoring system"""
        logger.info("🛑 Shutting down Industry Monitoring System")
        self.monitoring_active = False
        
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("👋 Industry Monitoring System shutdown complete")

# Global monitoring instance
_monitoring_instance = None

def get_monitoring_system() -> IndustryMonitoringSystem:
    """Get the global monitoring system instance"""
    global _monitoring_instance
    if _monitoring_instance is None:
        _monitoring_instance = IndustryMonitoringSystem()
    return _monitoring_instance

# Convenience functions for common operations
def record_api_request(endpoint: str, method: str, status_code: int, 
                      duration_ms: float, user_id: Optional[str] = None):
    """Record an API request with full monitoring"""
    monitoring = get_monitoring_system()
    
    # Record metrics
    monitoring.increment_counter("api_requests_total", tags={
        "endpoint": endpoint,
        "method": method,
        "status_code": str(status_code)
    })
    
    monitoring.record_histogram("api_request_duration_ms", duration_ms, tags={
        "endpoint": endpoint,
        "method": method
    })
    
    # Track user interaction
    if user_id:
        monitoring.track_user_interaction(user_id, "api_request", {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code
        })
    
    # Log errors
    if status_code >= 400:
        monitoring.increment_counter("api_errors_total", tags={
            "endpoint": endpoint,
            "status_code": str(status_code)
        })

def monitor_operation(operation_name: str):
    """Decorator to monitor an operation with distributed tracing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            span_id = monitoring.start_trace(operation_name)
            
            try:
                with monitoring.time_operation(operation_name):
                    result = func(*args, **kwargs)
                
                monitoring.finish_span(span_id, "success")
                return result
                
            except Exception as e:
                monitoring.finish_span(span_id, "error")
                monitoring.log_security_event(
                    "operation_error",
                    AlertSeverity.MEDIUM,
                    request_data={"operation": operation_name, "error": str(e)}
                )
                raise
        
        return wrapper
    return decorator

# Initialize monitoring system when module is imported
if __name__ != "__main__":
    # Auto-initialize in production
    try:
        _monitoring_instance = IndustryMonitoringSystem()
        logger.info("🚀 Industry Monitoring System auto-initialized")
    except Exception as e:
        logger.error(f"❌ Failed to auto-initialize monitoring: {e}")
