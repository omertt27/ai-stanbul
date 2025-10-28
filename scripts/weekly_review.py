#!/usr/bin/env python3
"""
Generate weekly feedback statistics report
"""

from user_feedback_collection_system import get_feedback_collector
import json
from datetime import datetime
from pathlib import Path

# Create reports directory
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

collector = get_feedback_collector()

# Get statistics
stats = collector.get_feedback_summary(days=7)
report = collector.get_misclassification_report()

# Create report
report_data = {
    'date': datetime.now().isoformat(),
    'period': 'Last 7 days',
    'total_feedback': stats['total'],
    'misclassifications': len(stats['misclassifications']),
    'corrections': len(stats.get('corrections', [])),
    'low_confidence': len(stats.get('low_confidence_predictions', [])),
    'by_function': dict(stats['by_function']),
    'by_language': dict(stats['by_language']),
    'by_intent': dict(stats.get('by_intent', {})),
    'confused_pairs': report['most_confused_pairs'][:10],
    'recommendations': []
}

# Add recommendations
if report_data['misclassifications'] > 20:
    report_data['recommendations'].append(
        "⚠️  High misclassification rate - Consider reviewing intent definitions"
    )

if report_data['low_confidence'] > 30:
    report_data['recommendations'].append(
        "⚠️  Many low-confidence predictions - May need more training data"
    )

# Check for confused intent pairs
if report['most_confused_pairs']:
    top_confused = report['most_confused_pairs'][0]
    if top_confused[2] > 5:  # More than 5 confusions
        report_data['recommendations'].append(
            f"⚠️  Intents '{top_confused[0]}' and '{top_confused[1]}' are frequently confused"
        )

if not report_data['recommendations']:
    report_data['recommendations'].append("✅ No major issues detected")

# Save report
report_file = reports_dir / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
with open(report_file, 'w') as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

# Print summary
print("=" * 80)
print("📊 WEEKLY FEEDBACK REPORT")
print("=" * 80)
print(f"\n📅 Period: {report_data['period']}")
print(f"📝 Generated: {report_data['date']}\n")

print("📈 Summary:")
print(f"   Total feedback: {report_data['total_feedback']}")
print(f"   Misclassifications: {report_data['misclassifications']}")
print(f"   Corrections: {report_data['corrections']}")
print(f"   Low confidence: {report_data['low_confidence']}")
print()

print("🔧 By Function (Top 5):")
top_functions = sorted(report_data['by_function'].items(), key=lambda x: x[1], reverse=True)[:5]
for func, count in top_functions:
    print(f"   {func:30s}: {count:3d}")
print()

print("🌍 By Language:")
for lang, count in report_data['by_language'].items():
    print(f"   {lang:10s}: {count:3d}")
print()

if report_data['confused_pairs']:
    print("🔀 Most Confused Intent Pairs:")
    for correct, predicted, count in report_data['confused_pairs'][:5]:
        print(f"   {correct:15s} ↔️  {predicted:15s}: {count:2d} times")
    print()

print("💡 Recommendations:")
for rec in report_data['recommendations']:
    print(f"   {rec}")
print()

print(f"📁 Report saved: {report_file}")
print("=" * 80)
