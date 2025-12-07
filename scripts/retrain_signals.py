#!/usr/bin/env python3
"""
retrain_signals.py - Signal Detection Retraining Script

CLI tool to trigger signal detection retraining from feedback loop data.

Usage:
    python scripts/retrain_signals.py                    # Check status
    python scripts/retrain_signals.py --retrain          # Trigger retraining
    python scripts/retrain_signals.py --report           # Generate report only
    python scripts/retrain_signals.py --analyze restaurant  # Analyze specific intent

Author: AI Istanbul Team
Date: December 2025
"""

import sys
import argparse
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.llm.feedback_trainer import FeedbackTrainer
from backend.services.llm.signals import SignalDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║     Signal Detection Retraining Tool (Phase 2)            ║
║     Feedback Loop Continuous Improvement System           ║
╚═══════════════════════════════════════════════════════════╝
    """)


def check_status(trainer: FeedbackTrainer):
    """Check and display current status."""
    print("\n📊 Current Status\n" + "="*60)
    
    stats = trainer.get_statistics()
    readiness = trainer.get_retraining_readiness()
    
    print(f"Total training samples: {stats['total_samples']}")
    print(f"Total comparisons: {stats['total_comparisons'] if hasattr(stats, 'total_comparisons') else 'N/A'}")
    print(f"Recent samples (last 7 days): {stats.get('recent_samples', 'N/A')}")
    print(f"\nLast retrain: {stats.get('last_retrain', 'Never')}")
    print(f"Patterns suggested: {stats.get('patterns_suggested', 0)}")
    
    print(f"\n🎯 Retraining Readiness\n" + "="*60)
    print(f"Status: {'✅ READY' if readiness['is_ready'] else '⏳ NOT READY'}")
    print(f"Total discrepancies: {readiness['total_discrepancies']}")
    print(f"Minimum required: {readiness['min_required']}")
    print(f"Recommendation: {readiness['recommendation']}")
    
    if readiness['discrepancies_by_intent']:
        print(f"\n📈 Discrepancies by Intent\n" + "="*60)
        for intent, count in sorted(
            readiness['discrepancies_by_intent'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {intent:30s}: {count:4d} misses")
    
    print("\n" + "="*60)


def analyze_intent(trainer: FeedbackTrainer, intent: str, language: str = 'en'):
    """Analyze a specific intent."""
    print(f"\n🔍 Analyzing Intent: {intent} ({language})\n" + "="*60)
    
    # Convert to signal format
    signal_name = f"needs_{intent}" if not intent.startswith('needs_') else intent
    
    analysis = trainer.analyze_missed_patterns(signal_name, language)
    
    if analysis['status'] == 'insufficient_data':
        print(f"❌ Insufficient data for analysis")
        print(f"   Samples: {analysis['sample_count']} (need at least 3)")
        return
    
    print(f"✅ Analysis complete")
    print(f"   Samples analyzed: {analysis['sample_count']}")
    print(f"   Keyword candidates: {len(analysis['keyword_candidates'])}")
    print(f"   Suggested patterns: {len(analysis['suggested_patterns'])}")
    
    if analysis['keyword_candidates']:
        print(f"\n📝 Top Keyword Candidates\n" + "-"*60)
        for i, candidate in enumerate(analysis['keyword_candidates'][:10], 1):
            print(
                f"  {i}. {candidate['text']:20s} "
                f"({candidate['type']:8s}) - "
                f"{candidate['occurrences']:3d} occurrences "
                f"({candidate['frequency']*100:.1f}%)"
            )
    
    if analysis['suggested_patterns']:
        print(f"\n🎯 Top Suggested Patterns\n" + "-"*60)
        for i, pattern in enumerate(analysis['suggested_patterns'][:10], 1):
            print(f"  {i}. Confidence: {pattern['confidence']*100:.1f}%")
            print(f"     Pattern: {pattern['pattern']}")
            print(f"     Description: {pattern['description']}")
            print(f"     Matches: {pattern['matches']}/{pattern['total_samples']}")
            print()
    
    if analysis['example_queries']:
        print(f"\n📋 Example Queries\n" + "-"*60)
        for i, query in enumerate(analysis['example_queries'], 1):
            print(f"  {i}. {query}")
    
    print("\n" + "="*60)


def generate_report(trainer: FeedbackTrainer):
    """Generate and display retraining report."""
    print(f"\n📊 Generating Retraining Report...\n" + "="*60)
    
    report = trainer.generate_retraining_report()
    
    print(f"✅ Report generated at: {report['generated_at']}")
    print(f"\n📈 Statistics\n" + "-"*60)
    print(f"  Total samples: {report['statistics']['total_samples']}")
    print(f"  Total discrepancies: {report['readiness']['total_discrepancies']}")
    
    if report['intent_analyses']:
        print(f"\n🎯 Intent Analyses\n" + "-"*60)
        for intent_lang, analysis in report['intent_analyses'].items():
            print(f"\n  {intent_lang}:")
            print(f"    Samples: {analysis['sample_count']}")
            print(f"    Patterns: {len(analysis['suggested_patterns'])}")
            if analysis['suggested_patterns']:
                top_pattern = analysis['suggested_patterns'][0]
                print(f"    Top pattern confidence: {top_pattern['confidence']*100:.1f}%")
    
    print(f"\n💾 Full report saved to: backend/services/llm/retraining_report.json")
    print("\n" + "="*60)


def perform_retrain(trainer: FeedbackTrainer, signal_detector: SignalDetector):
    """Perform retraining."""
    print(f"\n🔄 Starting Retraining Process...\n" + "="*60)
    
    # Check readiness
    readiness = trainer.get_retraining_readiness()
    if not readiness['is_ready']:
        print(f"❌ Cannot retrain: {readiness['recommendation']}")
        return False
    
    print("✅ Readiness check passed")
    
    # Generate report
    print("\n📊 Generating analysis report...")
    report = trainer.generate_retraining_report()
    print(f"✅ Analyzed {len(report['intent_analyses'])} intent/language combinations")
    
    # Export learned patterns
    print("\n💾 Exporting learned patterns...")
    success = trainer.export_learned_patterns()
    
    if not success:
        print("❌ Failed to export patterns")
        return False
    
    print(f"✅ Exported {trainer.stats['patterns_suggested']} patterns")
    
    # Reload patterns in signal detector
    print("\n🔄 Reloading signal detector patterns...")
    reload_success = signal_detector.reload_patterns()
    
    if not reload_success:
        print("⚠️  Warning: Failed to reload patterns in signal detector")
        print("   Patterns are saved but not yet active")
        print("   Restart the application to use new patterns")
    else:
        print("✅ Patterns reloaded successfully")
    
    print(f"\n✅ Retraining Complete!\n" + "="*60)
    print(f"Summary:")
    print(f"  - Training samples used: {report['statistics']['total_samples']}")
    print(f"  - Patterns generated: {trainer.stats['patterns_suggested']}")
    print(f"  - Patterns file: backend/services/llm/learned_patterns.json")
    print(f"  - Report file: backend/services/llm/retraining_report.json")
    print("\n" + "="*60)
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Signal Detection Retraining Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        Check current status
  %(prog)s --retrain              Trigger retraining
  %(prog)s --report               Generate full report
  %(prog)s --analyze restaurant   Analyze restaurant intent
  %(prog)s --analyze restaurant --language tr  Analyze in Turkish
        """
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Trigger retraining process'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate retraining report only'
    )
    
    parser.add_argument(
        '--analyze',
        metavar='INTENT',
        help='Analyze specific intent (e.g., restaurant, attraction)'
    )
    
    parser.add_argument(
        '--language',
        default='en',
        choices=['en', 'tr', 'ar', 'de', 'fr', 'ru'],
        help='Language for analysis (default: en)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Initialize trainer
    print("🔧 Initializing feedback trainer...")
    trainer = FeedbackTrainer()
    
    # Initialize signal detector (for pattern reloading)
    signal_detector = SignalDetector()
    
    # Execute command
    if args.retrain:
        perform_retrain(trainer, signal_detector)
    elif args.report:
        generate_report(trainer)
    elif args.analyze:
        analyze_intent(trainer, args.analyze, args.language)
    else:
        # Default: show status
        check_status(trainer)
        print("\nℹ️  Use --help for more options")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)
