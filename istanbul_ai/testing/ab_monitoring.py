"""
A/B Testing Real-Time Monitoring Module

Provides real-time monitoring, dashboards, and alerts for A/B experiments:
- Live metrics streaming
- Performance tracking
- Anomaly detection
- Early stopping rules
- Real-time dashboards

Part of Phase 4: A/B Testing Framework
Author: Istanbul AI Team
Date: October 31, 2025
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metric at a point in time"""
    timestamp: datetime
    variant_id: str
    metric_name: str
    value: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'variant_id': self.variant_id,
            'metric_name': self.metric_name,
            'value': self.value,
            'sample_count': self.sample_count
        }


@dataclass
class Alert:
    """Monitoring alert"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    experiment_id: str
    variant_id: Optional[str]
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'experiment_id': self.experiment_id,
            'variant_id': self.variant_id,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value
        }


@dataclass
class MonitoringConfig:
    """Configuration for experiment monitoring"""
    update_interval_seconds: int = 60
    window_size_minutes: int = 5
    min_samples_per_window: int = 10
    anomaly_std_threshold: float = 3.0
    min_effect_size_threshold: float = 0.2
    early_stop_confidence_threshold: float = 99.0
    enable_alerts: bool = True
    enable_early_stopping: bool = False


class RealTimeMetricsTracker:
    """Track metrics in real-time with sliding windows"""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize real-time tracker
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        
        # Sliding windows: variant -> metric -> deque of (timestamp, value)
        self.windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Cumulative stats: variant -> metric -> list of all values
        self.cumulative: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Latest snapshots
        self.latest_snapshots: Dict[str, MetricSnapshot] = {}
        
        # Baseline stats (first N samples)
        self.baselines: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.baseline_samples = 100
        
        logger.info(f"✅ Real-time tracker initialized (window: {config.window_size_minutes}m)")
    
    def record_metric(self, variant_id: str, metric_name: str, value: float):
        """
        Record new metric value
        
        Args:
            variant_id: Variant identifier
            metric_name: Metric name
            value: Metric value
        """
        timestamp = datetime.now()
        
        # Add to sliding window
        window = self.windows[variant_id][metric_name]
        window.append((timestamp, value))
        
        # Add to cumulative
        self.cumulative[variant_id][metric_name].append(value)
        
        # Remove old entries from window
        cutoff = timestamp - timedelta(minutes=self.config.window_size_minutes)
        while window and window[0][0] < cutoff:
            window.popleft()
        
        # Update baseline if needed
        cumulative_values = self.cumulative[variant_id][metric_name]
        if len(cumulative_values) <= self.baseline_samples:
            self._update_baseline(variant_id, metric_name, cumulative_values)
        
        # Create snapshot
        values = [v for _, v in window]
        snapshot = MetricSnapshot(
            timestamp=timestamp,
            variant_id=variant_id,
            metric_name=metric_name,
            value=statistics.mean(values) if values else value,
            sample_count=len(values)
        )
        
        key = f"{variant_id}:{metric_name}"
        self.latest_snapshots[key] = snapshot
    
    def _update_baseline(self, variant_id: str, metric_name: str, values: List[float]):
        """Update baseline statistics"""
        if len(values) < 2:
            return
        
        if variant_id not in self.baselines:
            self.baselines[variant_id] = {}
        
        self.baselines[variant_id][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values)
        }
    
    def get_current_stats(self, variant_id: str, metric_name: str) -> Dict[str, float]:
        """
        Get current statistics for variant and metric
        
        Args:
            variant_id: Variant identifier
            metric_name: Metric name
            
        Returns:
            Dictionary with current stats
        """
        window = self.windows[variant_id][metric_name]
        if not window:
            return {'count': 0}
        
        values = [v for _, v in window]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'recent': values[-1] if values else 0.0
        }
    
    def detect_anomaly(self, variant_id: str, metric_name: str) -> Optional[str]:
        """
        Detect anomalies using baseline comparison
        
        Args:
            variant_id: Variant identifier
            metric_name: Metric name
            
        Returns:
            Anomaly description or None
        """
        # Need baseline
        if variant_id not in self.baselines:
            return None
        
        if metric_name not in self.baselines[variant_id]:
            return None
        
        baseline = self.baselines[variant_id][metric_name]
        current = self.get_current_stats(variant_id, metric_name)
        
        if current['count'] < self.config.min_samples_per_window:
            return None
        
        # Check if current mean is outside baseline threshold
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        current_mean = current['mean']
        
        if baseline_std > 0:
            z_score = abs(current_mean - baseline_mean) / baseline_std
            
            if z_score > self.config.anomaly_std_threshold:
                direction = "higher" if current_mean > baseline_mean else "lower"
                return f"Anomaly detected: {metric_name} is {z_score:.1f} std devs {direction} than baseline"
        
        return None
    
    def get_time_series(
        self, 
        variant_id: str, 
        metric_name: str,
        duration_minutes: int = 60
    ) -> List[Tuple[datetime, float]]:
        """
        Get time series data for metric
        
        Args:
            variant_id: Variant identifier
            metric_name: Metric name
            duration_minutes: Duration to retrieve
            
        Returns:
            List of (timestamp, value) tuples
        """
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        cumulative = self.cumulative[variant_id][metric_name]
        
        # For now, return aggregated by minute
        # In production, would store timestamps with values
        result = []
        if cumulative:
            # Create synthetic time series for visualization
            interval = duration_minutes / min(len(cumulative), 100)
            for i, value in enumerate(cumulative[-100:]):
                timestamp = cutoff + timedelta(minutes=i * interval)
                result.append((timestamp, value))
        
        return result


class ExperimentMonitor:
    """Monitor A/B experiment in real-time"""
    
    def __init__(self, experiment_id: str, config: Optional[MonitoringConfig] = None):
        """
        Initialize experiment monitor
        
        Args:
            experiment_id: Experiment to monitor
            config: Monitoring configuration
        """
        self.experiment_id = experiment_id
        self.config = config or MonitoringConfig()
        self.tracker = RealTimeMetricsTracker(self.config)
        self.alerts: List[Alert] = []
        self.start_time = datetime.now()
        self.is_running = True
        
        logger.info(f"✅ Monitor initialized for experiment: {experiment_id}")
    
    def record_event(self, variant_id: str, metrics: Dict[str, float]):
        """
        Record event metrics
        
        Args:
            variant_id: Variant that generated the event
            metrics: Dictionary of metric_name -> value
        """
        for metric_name, value in metrics.items():
            self.tracker.record_metric(variant_id, metric_name, value)
        
        # Check for anomalies
        if self.config.enable_alerts:
            self._check_for_anomalies(variant_id, metrics.keys())
    
    def _check_for_anomalies(self, variant_id: str, metric_names: List[str]):
        """Check for anomalies and create alerts"""
        for metric_name in metric_names:
            anomaly = self.tracker.detect_anomaly(variant_id, metric_name)
            if anomaly:
                alert = Alert(
                    timestamp=datetime.now(),
                    severity='warning',
                    experiment_id=self.experiment_id,
                    variant_id=variant_id,
                    message=anomaly,
                    metric_name=metric_name
                )
                self.alerts.append(alert)
                logger.warning(f"⚠️ {anomaly}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard
        
        Returns:
            Dictionary with dashboard data
        """
        # Get current stats for all variants and metrics
        variants_data = {}
        
        for variant_id in self.tracker.cumulative.keys():
            metrics_data = {}
            for metric_name in self.tracker.cumulative[variant_id].keys():
                stats = self.tracker.get_current_stats(variant_id, metric_name)
                time_series = self.tracker.get_time_series(variant_id, metric_name, 60)
                
                metrics_data[metric_name] = {
                    'current': stats,
                    'time_series': [
                        {'timestamp': ts.isoformat(), 'value': val}
                        for ts, val in time_series
                    ]
                }
            
            variants_data[variant_id] = metrics_data
        
        # Recent alerts
        recent_alerts = [
            alert.to_dict() 
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat(),
            'uptime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'is_running': self.is_running,
            'variants': variants_data,
            'recent_alerts': recent_alerts,
            'alert_count': len(self.alerts)
        }
    
    def should_stop_early(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if experiment should stop early
        
        Returns:
            Tuple of (should_stop, reason, winner)
        """
        if not self.config.enable_early_stopping:
            return False, None, None
        
        # Need at least 2 variants with sufficient data
        variant_ids = list(self.tracker.cumulative.keys())
        if len(variant_ids) < 2:
            return False, None, None
        
        # Check each metric across variants
        # For simplicity, use first metric
        all_metrics = set()
        for variant_id in variant_ids:
            all_metrics.update(self.tracker.cumulative[variant_id].keys())
        
        if not all_metrics:
            return False, None, None
        
        metric_name = list(all_metrics)[0]
        
        # Get values for each variant
        variant_values = {}
        for variant_id in variant_ids:
            values = self.tracker.cumulative[variant_id].get(metric_name, [])
            if len(values) >= 100:  # Minimum samples
                variant_values[variant_id] = values
        
        if len(variant_values) < 2:
            return False, None, None
        
        # Compare variants (simplified - just compare means)
        variant_means = {
            vid: statistics.mean(values)
            for vid, values in variant_values.items()
        }
        
        winner = max(variant_means.items(), key=lambda x: x[1])[0]
        winner_mean = variant_means[winner]
        
        # Check if winner is significantly better
        for vid, mean in variant_means.items():
            if vid != winner:
                # Calculate relative difference
                diff_pct = abs(winner_mean - mean) / mean * 100 if mean > 0 else 0
                
                # If large difference, consider early stopping
                if diff_pct > 20:  # 20% better
                    reason = f"Variant {winner} is {diff_pct:.1f}% better than {vid}"
                    return True, reason, winner
        
        return False, None, None
    
    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"\n{'='*80}",
            f"EXPERIMENT MONITOR: {self.experiment_id}",
            f"{'='*80}",
            f"Uptime: {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes",
            f"Status: {'🟢 Running' if self.is_running else '🔴 Stopped'}",
            f"\nVariants:"
        ]
        
        for variant_id in self.tracker.cumulative.keys():
            lines.append(f"\n  {variant_id}:")
            for metric_name in self.tracker.cumulative[variant_id].keys():
                stats = self.tracker.get_current_stats(variant_id, metric_name)
                lines.append(
                    f"    {metric_name}: "
                    f"mean={stats.get('mean', 0):.4f}, "
                    f"samples={stats.get('count', 0)}"
                )
        
        if self.alerts:
            lines.append(f"\n⚠️ Alerts ({len(self.alerts)} total, showing last 5):")
            for alert in self.alerts[-5:]:
                lines.append(
                    f"  [{alert.severity.upper()}] {alert.timestamp.strftime('%H:%M:%S')} - "
                    f"{alert.message}"
                )
        
        lines.append(f"\n{'='*80}\n")
        return '\n'.join(lines)
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to JSON"""
        data = self.get_dashboard_data()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported monitoring data to {filepath}")


class MonitoringDashboard:
    """Simple console-based monitoring dashboard"""
    
    def __init__(self, monitor: ExperimentMonitor):
        """
        Initialize dashboard
        
        Args:
            monitor: Experiment monitor
        """
        self.monitor = monitor
    
    def display(self):
        """Display dashboard in console"""
        import os
        
        # Clear screen (Unix/Linux/Mac)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Display summary
        print(self.monitor.generate_summary())
        
        # Check for early stopping
        should_stop, reason, winner = self.monitor.should_stop_early()
        if should_stop:
            print(f"\n🛑 EARLY STOPPING RECOMMENDED:")
            print(f"   Reason: {reason}")
            print(f"   Winner: {winner}\n")
    
    def run_live(self, update_interval: int = 5):
        """
        Run live dashboard with auto-refresh
        
        Args:
            update_interval: Seconds between refreshes
        """
        print("🚀 Starting live monitoring dashboard...")
        print(f"   Refresh interval: {update_interval}s")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while self.monitor.is_running:
                self.display()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped by user")


# Factory functions
def create_experiment_monitor(
    experiment_id: str,
    config: Optional[MonitoringConfig] = None
) -> ExperimentMonitor:
    """
    Create experiment monitor
    
    Args:
        experiment_id: Experiment to monitor
        config: Optional monitoring configuration
        
    Returns:
        ExperimentMonitor instance
    """
    return ExperimentMonitor(experiment_id, config)


def create_monitoring_dashboard(monitor: ExperimentMonitor) -> MonitoringDashboard:
    """
    Create monitoring dashboard
    
    Args:
        monitor: Experiment monitor
        
    Returns:
        MonitoringDashboard instance
    """
    return MonitoringDashboard(monitor)
