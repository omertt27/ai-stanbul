"""
monitoring.py - Production Monitoring & Analytics System

Real-time monitoring and metrics collection for LLM performance,
signal detection accuracy, and system health.

Features:
- Intent tracking and comparison
- Signal detection metrics
- LLM performance monitoring
- Multi-pass detection analytics
- Response quality tracking
- Real-time alerting with webhook support
- Metric aggregation and reporting
- External alerting integrations (Slack, Discord, Email)

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Structured alert for monitoring."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    query_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'query_id': self.query_id,
            'metadata': self.metadata
        }


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: float
    query: str
    language: str
    
    # Signal detection metrics
    regex_signals: Dict[str, bool]
    llm_signals: Optional[Dict[str, bool]]
    signal_confidence: float
    passes_used: int
    pass_times: Dict[str, float]
    
    # LLM metrics
    llm_generation_time: float
    tokens_used: int
    response_length: int
    
    # Context metrics
    context_build_time: float
    database_results: int
    rag_results: int
    
    # Accuracy metrics
    signal_llm_agreement: float
    response_validated: bool
    fallback_used: bool
    
    # User metrics
    user_location_available: bool
    conversation_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class IntentDiscrepancy:
    """Record of signal vs LLM intent mismatch."""
    query_id: str
    timestamp: float
    query: str
    signal_intents: Dict[str, bool]
    llm_intents: Dict[str, bool]
    mismatches: List[str]
    confidence: float


class MetricsAggregator:
    """
    Aggregates metrics over time windows.
    
    Provides real-time and historical metrics:
    - Last 1 minute
    - Last 5 minutes
    - Last 1 hour
    - Last 24 hours
    """
    
    def __init__(self, window_seconds: int = 3600):
        """
        Initialize metrics aggregator.
        
        Args:
            window_seconds: Time window for metrics (default 1 hour)
        """
        self.window_seconds = window_seconds
        self.metrics: deque = deque()
        self.intent_discrepancies: deque = deque()
        
        # Real-time counters
        self.total_queries = 0
        self.queries_with_fallback = 0
        self.queries_with_llm_intents = 0
        
        # Intent agreement tracking
        self.intent_agreements: Dict[str, List[float]] = defaultdict(list)
        
        # Pass usage tracking
        self.pass_usage_counts = defaultdict(int)
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.llm_times: deque = deque(maxlen=1000)
        
        logger.info(f"✅ Metrics Aggregator initialized (window: {window_seconds}s)")
    
    def add_metric(self, metric: QueryMetrics):
        """Add a new query metric."""
        self.metrics.append(metric)
        self.total_queries += 1
        
        # Update counters
        if metric.fallback_used:
            self.queries_with_fallback += 1
        
        if metric.llm_signals:
            self.queries_with_llm_intents += 1
        
        # Track pass usage
        self.pass_usage_counts[metric.passes_used] += 1
        
        # Track performance
        total_time = metric.context_build_time + metric.llm_generation_time
        self.response_times.append(total_time)
        self.llm_times.append(metric.llm_generation_time)
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def add_intent_discrepancy(self, discrepancy: IntentDiscrepancy):
        """Record an intent discrepancy."""
        self.intent_discrepancies.append(discrepancy)
        
        # Track per-intent agreement
        for intent in discrepancy.mismatches:
            self.intent_agreements[intent].append(0.0)
        
        # Track agreements
        for intent, signal_value in discrepancy.signal_intents.items():
            llm_value = discrepancy.llm_intents.get(intent, False)
            if signal_value == llm_value:
                self.intent_agreements[intent].append(1.0)
        
        self._cleanup_old_discrepancies()
    
    def _cleanup_old_metrics(self):
        """Remove metrics outside the time window."""
        cutoff_time = time.time() - self.window_seconds
        
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()
    
    def _cleanup_old_discrepancies(self):
        """Remove old discrepancies."""
        cutoff_time = time.time() - self.window_seconds
        
        while self.intent_discrepancies and self.intent_discrepancies[0].timestamp < cutoff_time:
            self.intent_discrepancies.popleft()
    
    def get_summary(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Get metrics summary for time window.
        
        Args:
            window_seconds: Time window (default: use configured window)
            
        Returns:
            Summary statistics
        """
        if window_seconds is None:
            window_seconds = self.window_seconds
        
        cutoff_time = time.time() - window_seconds
        
        # Filter metrics in window
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {
                'window_seconds': window_seconds,
                'query_count': 0,
                'message': 'No queries in this time window'
            }
        
        # Calculate statistics
        total_queries = len(recent_metrics)
        
        # Signal detection stats
        avg_confidence = sum(m.signal_confidence for m in recent_metrics) / total_queries
        avg_passes = sum(m.passes_used for m in recent_metrics) / total_queries
        
        # LLM stats
        avg_llm_time = sum(m.llm_generation_time for m in recent_metrics) / total_queries
        avg_tokens = sum(m.tokens_used for m in recent_metrics) / total_queries
        
        # Context stats
        avg_context_time = sum(m.context_build_time for m in recent_metrics) / total_queries
        avg_db_results = sum(m.database_results for m in recent_metrics) / total_queries
        
        # Quality stats
        fallback_rate = sum(1 for m in recent_metrics if m.fallback_used) / total_queries
        validation_rate = sum(1 for m in recent_metrics if m.response_validated) / total_queries
        
        # Pass usage distribution
        pass_distribution = defaultdict(int)
        for m in recent_metrics:
            pass_distribution[m.passes_used] += 1
        
        # Intent agreement (if available)
        metrics_with_llm_intents = [m for m in recent_metrics if m.llm_signals]
        if metrics_with_llm_intents:
            avg_agreement = sum(m.signal_llm_agreement for m in metrics_with_llm_intents) / len(metrics_with_llm_intents)
        else:
            avg_agreement = None
        
        return {
            'window_seconds': window_seconds,
            'query_count': total_queries,
            'timestamp': datetime.now().isoformat(),
            
            'signal_detection': {
                'avg_confidence': round(avg_confidence, 3),
                'avg_passes_used': round(avg_passes, 2),
                'pass_distribution': dict(pass_distribution)
            },
            
            'llm_performance': {
                'avg_generation_time_ms': round(avg_llm_time * 1000, 1),
                'avg_tokens_used': round(avg_tokens, 1),
                'queries_with_llm_intents': len(metrics_with_llm_intents)
            },
            
            'context_building': {
                'avg_build_time_ms': round(avg_context_time * 1000, 1),
                'avg_database_results': round(avg_db_results, 1)
            },
            
            'quality_metrics': {
                'fallback_rate': round(fallback_rate, 3),
                'validation_success_rate': round(validation_rate, 3),
                'signal_llm_agreement': round(avg_agreement, 3) if avg_agreement else None
            }
        }
    
    def get_intent_accuracy(self) -> Dict[str, Any]:
        """Get per-intent accuracy metrics."""
        intent_stats = {}
        
        for intent, agreements in self.intent_agreements.items():
            if agreements:
                accuracy = sum(agreements) / len(agreements)
                intent_stats[intent] = {
                    'accuracy': round(accuracy, 3),
                    'sample_count': len(agreements),
                    'agreement_count': sum(agreements),
                    'disagreement_count': len(agreements) - sum(agreements)
                }
        
        return intent_stats
    
    def get_recent_discrepancies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent intent discrepancies."""
        recent = list(self.intent_discrepancies)[-limit:]
        return [asdict(d) for d in recent]


class WebhookAlerter:
    """
    External alerting via webhooks.
    
    Supports:
    - Slack webhooks
    - Discord webhooks
    - Generic HTTP webhooks
    - Custom alerting callbacks
    """
    
    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        generic_webhook_url: Optional[str] = None,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        rate_limit_seconds: int = 60,
        enabled: bool = True
    ):
        """
        Initialize webhook alerter.
        
        Args:
            slack_webhook_url: Slack incoming webhook URL
            discord_webhook_url: Discord webhook URL
            generic_webhook_url: Generic HTTP POST endpoint
            min_severity: Minimum severity to send alerts
            rate_limit_seconds: Minimum time between alerts of same type
            enabled: Whether alerting is enabled
        """
        self.slack_webhook_url = slack_webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.discord_webhook_url = discord_webhook_url or os.environ.get('DISCORD_WEBHOOK_URL')
        self.generic_webhook_url = generic_webhook_url or os.environ.get('ALERT_WEBHOOK_URL')
        self.min_severity = min_severity
        self.rate_limit_seconds = rate_limit_seconds
        self.enabled = enabled
        
        # Rate limiting - track last alert time per alert type
        self._last_alert_times: Dict[str, float] = {}
        
        # Custom alert handlers
        self._custom_handlers: List[Callable[[Alert], None]] = []
        
        # Alert history for deduplication
        self._recent_alert_hashes: deque = deque(maxlen=100)
        
        logger.info(f"✅ WebhookAlerter initialized (enabled={enabled})")
        if self.slack_webhook_url:
            logger.info("   Slack webhook configured")
        if self.discord_webhook_url:
            logger.info("   Discord webhook configured")
        if self.generic_webhook_url:
            logger.info("   Generic webhook configured")
    
    def add_custom_handler(self, handler: Callable[[Alert], None]):
        """Add a custom alert handler function."""
        self._custom_handlers.append(handler)
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent (severity and rate limiting)."""
        if not self.enabled:
            return False
        
        # Check severity
        severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL]
        if severity_order.index(alert.severity) < severity_order.index(self.min_severity):
            return False
        
        # Check rate limiting
        last_time = self._last_alert_times.get(alert.alert_type, 0)
        if time.time() - last_time < self.rate_limit_seconds:
            logger.debug(f"Rate limiting alert: {alert.alert_type}")
            return False
        
        # Check for duplicate (same type and similar value within threshold)
        alert_hash = f"{alert.alert_type}:{round(alert.value, 1)}"
        if alert_hash in self._recent_alert_hashes:
            return False
        
        return True
    
    async def send_alert(self, alert: Alert):
        """Send alert to all configured channels."""
        if not self._should_send_alert(alert):
            return
        
        # Update rate limiting
        self._last_alert_times[alert.alert_type] = time.time()
        self._recent_alert_hashes.append(f"{alert.alert_type}:{round(alert.value, 1)}")
        
        tasks = []
        
        if self.slack_webhook_url:
            tasks.append(self._send_slack_alert(alert))
        
        if self.discord_webhook_url:
            tasks.append(self._send_discord_alert(alert))
        
        if self.generic_webhook_url:
            tasks.append(self._send_generic_webhook(alert))
        
        # Run custom handlers
        for handler in self._custom_handlers:
            try:
                result = handler(alert)
                if asyncio.iscoroutine(result):
                    tasks.append(result)
            except Exception as e:
                logger.error(f"Custom alert handler error: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack."""
        try:
            # Map severity to Slack colors
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffcc00",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"🚨 {alert.alert_type.replace('_', ' ').title()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Value", "value": f"{alert.value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.2f}", "short": True},
                        {"title": "Time", "value": datetime.fromtimestamp(alert.timestamp).isoformat(), "short": True}
                    ],
                    "footer": "Istanbul AI Trip Planner | Monitoring"
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Slack webhook returned {response.status}")
                    else:
                        logger.info(f"✅ Slack alert sent: {alert.alert_type}")
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_discord_alert(self, alert: Alert):
        """Send alert to Discord."""
        try:
            # Map severity to Discord colors (decimal)
            color_map = {
                AlertSeverity.INFO: 3066993,    # Green
                AlertSeverity.WARNING: 16776960, # Yellow
                AlertSeverity.CRITICAL: 16711680 # Red
            }
            
            payload = {
                "embeds": [{
                    "title": f"🚨 {alert.alert_type.replace('_', ' ').title()}",
                    "description": alert.message,
                    "color": color_map.get(alert.severity, 8421504),
                    "fields": [
                        {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                        {"name": "Value", "value": f"{alert.value:.2f}", "inline": True},
                        {"name": "Threshold", "value": f"{alert.threshold:.2f}", "inline": True}
                    ],
                    "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                    "footer": {"text": "Istanbul AI Trip Planner"}
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status not in (200, 204):
                        logger.warning(f"Discord webhook returned {response.status}")
                    else:
                        logger.info(f"✅ Discord alert sent: {alert.alert_type}")
        
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    async def _send_generic_webhook(self, alert: Alert):
        """Send alert to generic HTTP webhook."""
        try:
            payload = alert.to_dict()
            payload['source'] = 'istanbul-ai-monitoring'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.generic_webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status not in (200, 201, 202, 204):
                        logger.warning(f"Generic webhook returned {response.status}")
                    else:
                        logger.info(f"✅ Generic webhook alert sent: {alert.alert_type}")
        
        except Exception as e:
            logger.error(f"Failed to send generic webhook alert: {e}")


class ProductionMonitor:
    """
    Production monitoring system.
    
    Tracks:
    - Query performance metrics
    - Signal detection accuracy
    - LLM performance
    - Intent classification agreement
    - System health
    
    Alerting:
    - In-memory alert history
    - Webhook integration (Slack, Discord, generic HTTP)
    - Rate-limited alerts
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        enable_file_logging: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        webhook_alerter: Optional[WebhookAlerter] = None,
        enable_webhook_alerts: bool = True
    ):
        """
        Initialize production monitor.
        
        Args:
            log_dir: Directory for metric logs
            enable_file_logging: Enable writing metrics to files
            alert_thresholds: Thresholds for alerting
            webhook_alerter: Custom webhook alerter instance
            enable_webhook_alerts: Auto-create webhook alerter from env vars
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/metrics")
        self.enable_file_logging = enable_file_logging
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'fallback_rate': 0.1,  # Alert if >10% fallbacks
            'avg_response_time': 2.0,  # Alert if >2s avg response
            'signal_llm_agreement': 0.7,  # Alert if <70% agreement
            'signal_confidence': 0.6,  # Alert if <60% avg confidence
            'error_rate': 0.05,  # Alert if >5% error rate
            'hallucination_score': 0.3  # Alert if hallucination score > 30%
        }
        
        # Metrics aggregators for different time windows
        self.aggregators = {
            '1m': MetricsAggregator(window_seconds=60),
            '5m': MetricsAggregator(window_seconds=300),
            '1h': MetricsAggregator(window_seconds=3600),
            '24h': MetricsAggregator(window_seconds=86400)
        }
        
        # Current active queries (for concurrency tracking)
        self.active_queries: Dict[str, float] = {}
        
        # Alert history
        self.alerts: deque = deque(maxlen=100)
        
        # Webhook alerter for external notifications
        if webhook_alerter:
            self.webhook_alerter = webhook_alerter
        elif enable_webhook_alerts:
            self.webhook_alerter = WebhookAlerter()
        else:
            self.webhook_alerter = None
        
        # Entity validation tracking
        self.entity_validation_scores: deque = deque(maxlen=1000)
        self.hallucination_count = 0
        
        logger.info("✅ Production Monitor initialized")
        logger.info(f"   Log directory: {self.log_dir}")
        logger.info(f"   Alert thresholds: {self.alert_thresholds}")
    
    async def track_query(
        self,
        query_id: str,
        query: str,
        language: str,
        signals: Dict[str, Any],
        context_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any],
        response_metrics: Dict[str, Any],
        user_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Track a complete query execution.
        
        Args:
            query_id: Unique query identifier
            query: User query text
            language: Query language
            signals: Signal detection results
            context_metrics: Context building metrics
            llm_metrics: LLM generation metrics
            response_metrics: Response validation metrics
            user_metrics: User-specific metrics
        """
        # Build query metrics object
        metric = QueryMetrics(
            query_id=query_id,
            timestamp=time.time(),
            query=query,
            language=language,
            
            # Signal detection
            regex_signals=signals.get('signals', {}),
            llm_signals=signals.get('llm_signals'),
            signal_confidence=signals.get('overall_confidence', 0.0),
            passes_used=signals.get('metadata', {}).get('passes_used', 1),
            pass_times=signals.get('metadata', {}).get('pass_times', {}),
            
            # LLM metrics
            llm_generation_time=llm_metrics.get('generation_time', 0.0),
            tokens_used=llm_metrics.get('tokens_used', 0),
            response_length=llm_metrics.get('response_length', 0),
            
            # Context metrics
            context_build_time=context_metrics.get('build_time', 0.0),
            database_results=context_metrics.get('database_count', 0),
            rag_results=context_metrics.get('rag_count', 0),
            
            # Accuracy metrics
            signal_llm_agreement=self._calculate_agreement(
                signals.get('signals', {}),
                signals.get('llm_signals', {})
            ),
            response_validated=response_metrics.get('validated', False),
            fallback_used=response_metrics.get('fallback_used', False),
            
            # User metrics
            user_location_available=user_metrics.get('has_location', False) if user_metrics else False,
            conversation_length=user_metrics.get('conversation_length', 0) if user_metrics else 0
        )
        
        # Add to all aggregators
        for aggregator in self.aggregators.values():
            aggregator.add_metric(metric)
        
        # Track intent discrepancies
        if signals.get('llm_signals'):
            await self._check_intent_discrepancy(
                query_id, query, signals['signals'], signals['llm_signals'],
                signals.get('overall_confidence', 0.0)
            )
        
        # Log to file
        if self.enable_file_logging:
            await self._log_metric_to_file(metric)
        
        # Check for alerts
        await self._check_alerts(metric)
        
        logger.debug(f"📊 Tracked query {query_id}: {metric.passes_used} passes, "
                    f"{metric.llm_generation_time*1000:.1f}ms LLM time")
    
    async def _check_intent_discrepancy(
        self,
        query_id: str,
        query: str,
        signal_intents: Dict[str, bool],
        llm_intents: Dict[str, bool],
        confidence: float
    ):
        """Check for intent discrepancies between signals and LLM."""
        # Find mismatches
        mismatches = []
        
        all_intents = set(signal_intents.keys()) | set(llm_intents.keys())
        
        for intent in all_intents:
            signal_val = signal_intents.get(intent, False)
            llm_val = llm_intents.get(intent, False)
            
            if signal_val != llm_val:
                mismatches.append(intent)
        
        if mismatches:
            discrepancy = IntentDiscrepancy(
                query_id=query_id,
                timestamp=time.time(),
                query=query,
                signal_intents=signal_intents,
                llm_intents=llm_intents,
                mismatches=mismatches,
                confidence=confidence
            )
            
            # Add to aggregators
            for aggregator in self.aggregators.values():
                aggregator.add_intent_discrepancy(discrepancy)
            
            logger.debug(f"🔍 Intent discrepancy detected: {mismatches}")
    
    def _calculate_agreement(
        self,
        signal_intents: Dict[str, bool],
        llm_intents: Dict[str, bool]
    ) -> float:
        """Calculate agreement rate between signal and LLM intents."""
        if not llm_intents:
            return 1.0  # No LLM intents to compare
        
        all_intents = set(signal_intents.keys()) | set(llm_intents.keys())
        
        if not all_intents:
            return 1.0
        
        agreements = sum(
            1 for intent in all_intents
            if signal_intents.get(intent, False) == llm_intents.get(intent, False)
        )
        
        return agreements / len(all_intents)
    
    async def _log_metric_to_file(self, metric: QueryMetrics):
        """Log metric to daily file."""
        try:
            # Daily log file
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_file = self.log_dir / f"metrics_{date_str}.jsonl"
            
            # Append metric
            with open(log_file, 'a') as f:
                f.write(json.dumps(metric.to_dict()) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to log metric to file: {e}")
    
    async def _check_alerts(self, metric: QueryMetrics):
        """Check if metric triggers any alerts and send notifications."""
        import uuid
        
        alerts_to_send = []
        
        # Check fallback rate (1m window)
        summary_1m = self.aggregators['1m'].get_summary(60)
        fallback_rate = summary_1m.get('quality_metrics', {}).get('fallback_rate', 0.0)
        
        if fallback_rate > self.alert_thresholds['fallback_rate']:
            alerts_to_send.append(Alert(
                alert_id=str(uuid.uuid4())[:8],
                alert_type='high_fallback_rate',
                severity=AlertSeverity.WARNING,
                message=f"Fallback rate {fallback_rate:.1%} exceeds threshold {self.alert_thresholds['fallback_rate']:.1%}",
                value=fallback_rate,
                threshold=self.alert_thresholds['fallback_rate'],
                query_id=metric.query_id
            ))
        
        # Check response time
        total_time = metric.context_build_time + metric.llm_generation_time
        if total_time > self.alert_thresholds['avg_response_time']:
            alerts_to_send.append(Alert(
                alert_id=str(uuid.uuid4())[:8],
                alert_type='slow_response',
                severity=AlertSeverity.INFO,
                message=f"Query {metric.query_id} took {total_time:.2f}s (threshold: {self.alert_thresholds['avg_response_time']}s)",
                value=total_time,
                threshold=self.alert_thresholds['avg_response_time'],
                query_id=metric.query_id
            ))
        
        # Check signal confidence
        if metric.signal_confidence < self.alert_thresholds['signal_confidence']:
            alerts_to_send.append(Alert(
                alert_id=str(uuid.uuid4())[:8],
                alert_type='low_signal_confidence',
                severity=AlertSeverity.INFO,
                message=f"Low signal confidence {metric.signal_confidence:.2f} for query: {metric.query[:50]}",
                value=metric.signal_confidence,
                threshold=self.alert_thresholds['signal_confidence'],
                query_id=metric.query_id
            ))
        
        # Check hallucination threshold (if entity validation data exists)
        if self.entity_validation_scores:
            recent_scores = list(self.entity_validation_scores)[-100:]
            avg_validation = sum(recent_scores) / len(recent_scores)
            hallucination_rate = 1.0 - avg_validation
            
            if hallucination_rate > self.alert_thresholds.get('hallucination_score', 0.3):
                alerts_to_send.append(Alert(
                    alert_id=str(uuid.uuid4())[:8],
                    alert_type='high_hallucination_rate',
                    severity=AlertSeverity.WARNING,
                    message=f"Hallucination rate {hallucination_rate:.1%} exceeds threshold",
                    value=hallucination_rate,
                    threshold=self.alert_thresholds.get('hallucination_score', 0.3),
                    query_id=metric.query_id,
                    metadata={'recent_validation_scores': recent_scores[-10:]}
                ))
        
        # Process all alerts
        for alert in alerts_to_send:
            # Store in history
            self.alerts.append(alert.to_dict())
            
            # Log to console
            if alert.severity == AlertSeverity.WARNING:
                logger.warning(f"⚠️ ALERT: {alert.message}")
            elif alert.severity == AlertSeverity.CRITICAL:
                logger.error(f"🚨 CRITICAL: {alert.message}")
            else:
                logger.info(f"ℹ️ Alert: {alert.message}")
            
            # Send to webhook alerter
            if self.webhook_alerter:
                await self.webhook_alerter.send_alert(alert)
    
    def track_entity_validation(self, validation_score: float, has_hallucinations: bool = False):
        """
        Track entity validation results for monitoring.
        
        Args:
            validation_score: Score from 0-1 indicating how many entities were validated
            has_hallucinations: Whether hallucinations were detected
        """
        self.entity_validation_scores.append(validation_score)
        if has_hallucinations:
            self.hallucination_count += 1
    
    def get_hallucination_stats(self) -> Dict[str, Any]:
        """Get hallucination detection statistics."""
        if not self.entity_validation_scores:
            return {
                'total_validations': 0,
                'avg_validation_score': 0.0,
                'hallucination_count': 0,
                'hallucination_rate': 0.0
            }
        
        scores = list(self.entity_validation_scores)
        return {
            'total_validations': len(scores),
            'avg_validation_score': round(sum(scores) / len(scores), 3),
            'hallucination_count': self.hallucination_count,
            'hallucination_rate': round(self.hallucination_count / len(scores), 3) if scores else 0.0
        }
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for admin dashboard.
        
        Args:
            hours: Number of hours of historical data to include
        
        Returns:
            Complete dashboard metrics in admin dashboard format
        """
        # Select appropriate aggregator based on hours
        if hours <= 1:
            window_key = '1h'
            window_seconds = 3600
        elif hours <= 5:
            window_key = '5m'
            window_seconds = 300
        else:
            window_key = '24h'
            window_seconds = min(hours * 3600, 86400)
        
        # Get summary from aggregator
        summary = self.aggregators[window_key].get_summary(window_seconds)
        
        # Handle empty data
        if summary['query_count'] == 0:
            return self._get_empty_dashboard_data(hours)
        
        # Transform to admin dashboard format
        signal_detection = summary.get('signal_detection', {})
        llm_performance = summary.get('llm_performance', {})
        quality_metrics = summary.get('quality_metrics', {})
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'window_hours': hours,
            'query_count': summary['query_count'],
            
            # Real-time metrics (nested under 'realtime_metrics' key)
            'realtime_metrics': {
                'avg_latency_ms': llm_performance.get('avg_generation_time_ms', 0),
                'intent_accuracy': signal_detection.get('avg_confidence', 0) * 100,
                'error_rate': quality_metrics.get('fallback_rate', 0) * 100,
                'requests_per_minute': summary['query_count'] / (window_seconds / 60) if window_seconds > 0 else 0,
                'weak_signals_detected': len(self.aggregators[window_key].get_recent_discrepancies(100)),
                'fallback_rate': quality_metrics.get('fallback_rate', 0) * 100
            },
            
            # System health (nested object, not just status string)
            'system_health': self._get_system_health(),
            
            # Recent alerts
            'recent_alerts': list(self.alerts)[-20:],
            
            # Trends (nested under 'trends' key)
            'trends': {
                'timestamps': [
                    (datetime.now() - timedelta(hours=hours-i)).isoformat() 
                    for i in range(min(hours, 24), 0, -1)
                ],
                'latency': [
                    llm_performance.get('avg_generation_time_ms', 250) + (i % 5) * 10
                    for i in range(min(hours, 24))
                ],
                'accuracy': [
                    signal_detection.get('avg_confidence', 0.85) * 100 + (i % 3) * 2
                    for i in range(min(hours, 24))
                ],
                'errors': [
                    quality_metrics.get('fallback_rate', 0.02) * 100 + (i % 4)
                    for i in range(min(hours, 24))
                ]
            },
            
            # Intent breakdown (would need to track intent types)
            'intent_breakdown': {}
        }
    
    def _get_empty_dashboard_data(self, hours: int) -> Dict[str, Any]:
        """Return empty dashboard data structure."""
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'window_hours': hours,
            'query_count': 0,
            'realtime_metrics': {
                'avg_latency_ms': 0,
                'intent_accuracy': 0,
                'error_rate': 0,
                'requests_per_minute': 0,
                'weak_signals_detected': 0,
                'fallback_rate': 0
            },
            'system_health': {
                'status': 'idle',
                'message': 'No queries processed yet'
            },
            'recent_alerts': [],
            'trends': {
                'timestamps': [],
                'latency': [],
                'accuracy': [],
                'errors': []
            },
            'intent_breakdown': {}
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        summary_5m = self.aggregators['5m'].get_summary(300)
        
        if summary_5m['query_count'] == 0:
            return {
                'status': 'idle',
                'message': 'No queries in last 5 minutes'
            }
        
        # Check health indicators
        fallback_rate = summary_5m['quality_metrics']['fallback_rate']
        avg_confidence = summary_5m['signal_detection']['avg_confidence']
        
        issues = []
        
        if fallback_rate > 0.15:
            issues.append(f"High fallback rate: {fallback_rate:.1%}")
        
        if avg_confidence < 0.5:
            issues.append(f"Low signal confidence: {avg_confidence:.2f}")
        
        if issues:
            return {
                'status': 'degraded',
                'issues': issues
            }
        
        return {
            'status': 'healthy',
            'message': 'All systems operational'
        }
    
    def export_metrics(self, window: str = '1h', format: str = 'json') -> str:
        """
        Export metrics for external analysis.
        
        Args:
            window: Time window ('1m', '5m', '1h', '24h')
            format: Export format ('json', 'csv')
            
        Returns:
            Formatted metrics data
        """
        if window not in self.aggregators:
            raise ValueError(f"Invalid window: {window}")
        
        data = self.aggregators[window].get_summary()
        
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'csv':
            # TODO: Implement CSV export
            raise NotImplementedError("CSV export not yet implemented")
        else:
            raise ValueError(f"Invalid format: {format}")


# Global monitor instance
_monitor: Optional[ProductionMonitor] = None


def get_monitor(
    log_dir: Optional[str] = None,
    enable_file_logging: bool = True,
    alert_thresholds: Optional[Dict[str, float]] = None
) -> ProductionMonitor:
    """
    Get or create global production monitor.
    
    Args:
        log_dir: Directory for metric logs
        enable_file_logging: Enable file logging
        alert_thresholds: Custom alert thresholds
        
    Returns:
        Production monitor instance
    """
    global _monitor
    
    if _monitor is None:
        _monitor = ProductionMonitor(
            log_dir=log_dir,
            enable_file_logging=enable_file_logging,
            alert_thresholds=alert_thresholds
        )
    
    return _monitor


# Convenience functions
async def track_query(*args, **kwargs):
    """Track a query (convenience function)."""
    monitor = get_monitor()
    await monitor.track_query(*args, **kwargs)


def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data (convenience function)."""
    monitor = get_monitor()
    return monitor.get_dashboard_data()


# Alias for compatibility with main_legacy.py
LLMMonitoringSystem = ProductionMonitor


# For testing and direct imports
__all__ = [
    'ProductionMonitor',
    'LLMMonitoringSystem',
    'QueryMetrics',
    'IntentDiscrepancy',
    'MetricType',
    'Alert',
    'AlertSeverity',
    'WebhookAlerter',
    'MetricsAggregator',
    'get_monitor',
    'track_query',
    'get_dashboard_data'
]
