#!/usr/bin/env python3
"""
Real-Time Backend Performance Monitor
Monitors logs, metrics, and patterns in the AI Istanbul backend
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import subprocess

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.ml_production_monitor import get_production_monitor
    ML_MONITOR_AVAILABLE = True
except ImportError:
    ML_MONITOR_AVAILABLE = False
    print("⚠️ ML Production Monitor not available")

try:
    from scripts.user_feedback_collector import get_feedback_collector
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    print("⚠️ Feedback Collector not available")


class BackendPerformanceMonitor:
    """Real-time monitoring of backend performance"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.log_dir = Path("logs/ml_production")
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
    def check_backend_status(self):
        """Check if backend is running"""
        try:
            import requests
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Backend is running at {self.backend_url}")
                return True
            else:
                print(f"⚠️ Backend returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend not accessible: {e}")
            return False
    
    def get_ml_metrics(self):
        """Get ML monitoring metrics"""
        if not ML_MONITOR_AVAILABLE:
            return None
        
        try:
            monitor = get_production_monitor()
            metrics = monitor.get_current_metrics()
            return metrics
        except Exception as e:
            print(f"⚠️ Error getting ML metrics: {e}")
            return None
    
    def get_feedback_summary(self):
        """Get feedback collector summary"""
        if not FEEDBACK_AVAILABLE:
            return None
        
        try:
            collector = get_feedback_collector()
            summary = collector.get_summary()
            return summary
        except Exception as e:
            print(f"⚠️ Error getting feedback: {e}")
            return None
    
    def read_recent_logs(self, n_lines: int = 50):
        """Read recent log entries"""
        if not self.metrics_file.exists():
            return []
        
        logs = []
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-n_lines:]:
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Error reading logs: {e}")
        
        return logs
    
    def analyze_recent_patterns(self, logs: list):
        """Analyze patterns in recent logs"""
        if not logs:
            return {
                'total_queries': 0,
                'intents': {},
                'avg_confidence': 0,
                'low_confidence_count': 0,
                'avg_latency': 0
            }
        
        intent_counts = Counter()
        confidences = []
        latencies = []
        low_confidence_queries = []
        
        for log in logs:
            if 'predicted_intent' in log:
                intent_counts[log['predicted_intent']] += 1
            
            if 'confidence' in log:
                conf = log['confidence']
                confidences.append(conf)
                if conf < 0.7:
                    low_confidence_queries.append({
                        'query': log.get('query', 'N/A'),
                        'intent': log.get('predicted_intent', 'N/A'),
                        'confidence': conf
                    })
            
            if 'latency_ms' in log:
                latencies.append(log['latency_ms'])
        
        return {
            'total_queries': len(logs),
            'intents': dict(intent_counts.most_common(10)),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'low_confidence_count': len(low_confidence_queries),
            'low_confidence_queries': low_confidence_queries[:10],
            'avg_latency': sum(latencies) / len(latencies) if latencies else 0
        }
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "AI ISTANBUL BACKEND PERFORMANCE MONITOR" + " " * 19 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Check backend status
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🔧 BACKEND STATUS")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        backend_running = self.check_backend_status()
        print()
        
        # ML Metrics
        if ML_MONITOR_AVAILABLE:
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("📊 ML PRODUCTION METRICS")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            metrics = self.get_ml_metrics()
            if metrics and metrics.get('status') != 'No predictions yet':
                print(f"Total Predictions: {metrics['total_predictions']}")
                print(f"Period Start:      {metrics.get('period_start', 'N/A')}")
                print()
                
                perf = metrics.get('performance', {})
                print(f"⚡ Performance:")
                print(f"   • Avg Latency: {perf.get('avg_latency_ms', 'N/A')} ms")
                print(f"   • P95 Latency: {perf.get('p95_latency_ms', 'N/A')} ms")
                print(f"   • P99 Latency: {perf.get('p99_latency_ms', 'N/A')} ms")
                print(f"   • Accuracy:    {perf.get('accuracy', 'N/A')}")
                print()
                
                print(f"🎯 Top Intents:")
                for intent_data in metrics.get('top_intents', [])[:5]:
                    print(f"   • {intent_data['intent']:<25} {intent_data['count']:>5} ({intent_data['percentage']}%)")
                print()
                
                quality = metrics.get('quality_metrics', {})
                print(f"⚠️ Quality Metrics:")
                print(f"   • Low Confidence:     {quality.get('low_confidence_count', 0)} ({quality.get('low_confidence_rate', 0)}%)")
                print(f"   • Misclassifications: {quality.get('misclassification_count', 0)}")
                print(f"   • Errors:             {quality.get('error_count', 0)}")
                print()
                
                feedback = metrics.get('user_feedback', {})
                if feedback.get('total_feedback', 0) > 0:
                    print(f"👤 User Feedback:")
                    print(f"   • Total:   {feedback.get('total_feedback', 0)}")
                    print(f"   • Correct: {feedback.get('correct', 0)}")
                    print(f"   • Wrong:   {feedback.get('wrong', 0)}")
                    print(f"   • Partial: {feedback.get('partial', 0)}")
                    print()
            else:
                print("No predictions recorded yet. Start using the backend to see metrics.\n")
        
        # Recent logs analysis
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📝 RECENT ACTIVITY (Last 50 queries)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        recent_logs = self.read_recent_logs(50)
        if recent_logs:
            patterns = self.analyze_recent_patterns(recent_logs)
            print(f"Total Queries:      {patterns['total_queries']}")
            print(f"Avg Confidence:     {patterns['avg_confidence']:.2%}")
            print(f"Avg Latency:        {patterns['avg_latency']:.2f} ms")
            print(f"Low Confidence (<70%): {patterns['low_confidence_count']}")
            print()
            
            if patterns['intents']:
                print("Intent Distribution:")
                for intent, count in list(patterns['intents'].items())[:8]:
                    percentage = (count / patterns['total_queries']) * 100
                    bar = '█' * int(percentage / 3)
                    print(f"   {intent:<25} {count:>3} ({percentage:>5.1f}%) {bar}")
                print()
            
            if patterns['low_confidence_queries']:
                print("⚠️ Low Confidence Queries (sample):")
                for lc in patterns['low_confidence_queries'][:5]:
                    print(f"   • \"{lc['query'][:50]}...\"")
                    print(f"     Intent: {lc['intent']} (conf: {lc['confidence']:.2%})")
                print()
        else:
            print("No recent activity found.\n")
        
        # Feedback summary
        if FEEDBACK_AVAILABLE:
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("💬 USER FEEDBACK SUMMARY")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            feedback_summary = self.get_feedback_summary()
            if feedback_summary:
                print(f"Total Feedback:   {feedback_summary.get('total_feedback', 0)}")
                print(f"Positive:         {feedback_summary.get('positive', 0)}")
                print(f"Negative:         {feedback_summary.get('negative', 0)}")
                print(f"Neutral:          {feedback_summary.get('neutral', 0)}")
                print()
        
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("\n💡 Press Ctrl+C to exit | Auto-refresh every 10 seconds\n")
    
    def watch(self, interval: int = 10):
        """Watch mode - continuously refresh dashboard"""
        print("🔄 Starting watch mode...")
        print(f"Refreshing every {interval} seconds\n")
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive report"""
        if not ML_MONITOR_AVAILABLE:
            print("❌ ML Monitor not available. Cannot generate report.")
            return
        
        monitor = get_production_monitor()
        report = monitor.generate_report(output_file)
        
        print(report)
        
        if output_file:
            print(f"\n✅ Report saved to: {output_file}")
    
    def tail_logs(self, n_lines: int = 20, follow: bool = False):
        """Tail recent logs"""
        if not self.metrics_file.exists():
            print("❌ No log file found. Start using the backend to generate logs.")
            return
        
        print(f"📜 Tailing last {n_lines} log entries from {self.metrics_file}\n")
        print("━" * 80)
        
        if follow:
            # Use tail -f
            try:
                subprocess.run(['tail', '-f', str(self.metrics_file)])
            except KeyboardInterrupt:
                print("\n\n👋 Stopped tailing logs.")
        else:
            # Read last n lines
            logs = self.read_recent_logs(n_lines)
            for log in logs:
                timestamp = log.get('timestamp', 'N/A')
                query = log.get('query', 'N/A')
                intent = log.get('predicted_intent', 'N/A')
                confidence = log.get('confidence', 0)
                latency = log.get('latency_ms', 0)
                
                print(f"[{timestamp}]")
                print(f"  Query: {query[:60]}...")
                print(f"  Intent: {intent} | Confidence: {confidence:.2%} | Latency: {latency:.2f}ms")
                print("━" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor AI Istanbul Backend Performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current dashboard
  python scripts/monitor_backend_performance.py
  
  # Watch mode (auto-refresh every 10s)
  python scripts/monitor_backend_performance.py --watch
  
  # Custom refresh interval
  python scripts/monitor_backend_performance.py --watch --interval 5
  
  # Generate detailed report
  python scripts/monitor_backend_performance.py --report
  
  # Save report to file
  python scripts/monitor_backend_performance.py --report --output logs/report.txt
  
  # Tail recent logs
  python scripts/monitor_backend_performance.py --tail 30
  
  # Follow logs (like tail -f)
  python scripts/monitor_backend_performance.py --tail --follow
        """
    )
    
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Watch mode - continuously refresh dashboard'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=10,
        help='Refresh interval in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate detailed monitoring report'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for report'
    )
    
    parser.add_argument(
        '--tail', '-t',
        type=int,
        nargs='?',
        const=20,
        help='Tail recent log entries (default: 20)'
    )
    
    parser.add_argument(
        '--follow', '-f',
        action='store_true',
        help='Follow log file (use with --tail)'
    )
    
    parser.add_argument(
        '--backend-url',
        type=str,
        default='http://localhost:8000',
        help='Backend URL (default: http://localhost:8000)'
    )
    
    args = parser.parse_args()
    
    monitor = BackendPerformanceMonitor(backend_url=args.backend_url)
    
    if args.report:
        monitor.generate_report(output_file=args.output)
    elif args.tail is not None:
        monitor.tail_logs(n_lines=args.tail, follow=args.follow)
    elif args.watch:
        monitor.watch(interval=args.interval)
    else:
        monitor.display_dashboard()


if __name__ == '__main__':
    main()
