#!/usr/bin/env python3
"""
Check if model retraining is needed based on feedback data
Returns 0 if retraining recommended, 1 otherwise
"""

from user_feedback_collection_system import get_feedback_collector
import sys

# Thresholds
MIN_CORRECTIONS = 50  # Minimum corrections to trigger retraining
MIN_MISCLASSIFICATION_RATE = 0.05  # 5%
MIN_LOW_CONFIDENCE_RATE = 0.10  # 10%

collector = get_feedback_collector()
stats = collector.get_feedback_summary(days=30)

total = stats['total']
corrections = len(stats.get('corrections', []))
misclassifications = len(stats.get('misclassifications', []))
low_confidence = len(stats.get('low_confidence_predictions', []))

print(f"📊 Feedback Summary (Last 30 days):")
print(f"   Total feedback: {total}")
print(f"   Corrections: {corrections}")
print(f"   Misclassifications: {misclassifications}")
print(f"   Low confidence: {low_confidence}")
print()

# Calculate rates
misclass_rate = misclassifications / total if total > 0 else 0
low_conf_rate = low_confidence / total if total > 0 else 0

print(f"📈 Metrics:")
print(f"   Misclassification rate: {misclass_rate:.2%}")
print(f"   Low confidence rate: {low_conf_rate:.2%}")
print()

# Check if retraining is needed
should_retrain = False
reasons = []

if corrections >= MIN_CORRECTIONS:
    should_retrain = True
    reasons.append(f"✅ Enough corrections ({corrections} >= {MIN_CORRECTIONS})")
else:
    reasons.append(f"❌ Not enough corrections ({corrections} < {MIN_CORRECTIONS})")

if misclass_rate >= MIN_MISCLASSIFICATION_RATE:
    should_retrain = True
    reasons.append(f"✅ High misclassification rate ({misclass_rate:.2%})")

if low_conf_rate >= MIN_LOW_CONFIDENCE_RATE:
    should_retrain = True
    reasons.append(f"✅ High low-confidence rate ({low_conf_rate:.2%})")

print("🎯 Decision:")
for reason in reasons:
    print(f"   {reason}")
print()

if should_retrain:
    print("✅ RETRAINING RECOMMENDED")
    sys.exit(0)  # Success - proceed with retraining
else:
    print("⏸️  RETRAINING NOT NEEDED YET")
    sys.exit(1)  # Don't retrain
