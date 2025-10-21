#!/usr/bin/env python3
"""
Check if extended training is complete and show results
"""

import os
import json
import time
from datetime import datetime

print("🔍 Phase 2 Extended Training Status Checker")
print("=" * 80)
print()

# Check if results file exists
results_file = 'phase2_extended_v2_results.json'
model_file = 'phase2_extended_v2_model.pth'
log_file = 'phase2_extended_training_v2.log'

if os.path.exists(results_file):
    print("✅ TRAINING COMPLETE!")
    print()
    
    # Load and display results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print()
    print(f"🎯 Accuracy: {results['accuracy_percent']}%")
    print(f"   Correct: {results['correct_predictions']}/{results['total_predictions']}")
    print()
    print(f"⚡ Performance:")
    print(f"   Average Latency: {results['avg_latency_ms']}ms")
    print(f"   P95 Latency: {results['p95_latency_ms']}ms")
    print(f"   P99 Latency: {results['p99_latency_ms']}ms")
    print()
    print(f"⏱️  Training Time:")
    print(f"   Duration: {results['training_duration_minutes']:.1f} minutes")
    print(f"   Total Epochs: {results['total_epochs']}")
    print()
    
    accuracy = results['accuracy_percent']
    if accuracy >= 90:
        print("🎉 TARGET ACHIEVED! ≥90% accuracy - Excellent!")
    elif accuracy >= 85:
        print("🎉 EXCELLENT! ≥85% accuracy - Production ready!")
    elif accuracy >= 80:
        print("✅ VERY GOOD! ≥80% accuracy - Close to target")
    else:
        print("📈 GOOD PROGRESS - Continue training or add more data")
    print()
    
    # Show accuracy by intent
    print("📊 Accuracy by Intent:")
    print("-" * 80)
    intent_acc = results.get('intent_accuracy', {})
    for intent, metrics in sorted(intent_acc.items()):
        acc = metrics['accuracy_percent']
        status = "✅" if acc >= 80 else "⚠️" if acc >= 50 else "❌"
        print(f"{status} {intent:20s} {acc:5.1f}% ({metrics['correct']}/{metrics['total']})")
    print()
    
    # Show errors if any
    predictions = results.get('predictions', [])
    errors = [p for p in predictions if not p['correct']]
    if errors:
        print("❌ Errors Analysis:")
        print("-" * 80)
        for err in errors[:10]:  # Show first 10 errors
            print(f"Query: '{err['query'][:50]}...'")
            print(f"  Expected: {err['expected']}")
            print(f"  Got: {err['predicted']} ({err['confidence']}% confidence)")
            print()
    else:
        print("✅ No errors! Perfect accuracy!")
        print()
    
    print("=" * 80)
    print("📂 Generated Files:")
    print(f"   ✅ {model_file}")
    print(f"   ✅ {results_file}")
    print()
    
elif os.path.exists(log_file):
    print("🔄 TRAINING IN PROGRESS...")
    print()
    
    # Try to estimate progress
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Count logging steps (every 10 steps)
    if 'loss' in log_content.lower():
        # Try to find the last progress line
        lines = log_content.split('\n')
        progress_lines = [l for l in lines if '%|' in l and 'it/s' in l]
        if progress_lines:
            last_line = progress_lines[-1]
            print(f"Last Progress: {last_line.strip()}")
            print()
    
    # Check how long it's been running
    if '🎯 Starting extended training...' in log_content:
        print("✅ Training started successfully")
        print("⏱️  Estimated time: ~27 minutes")
        print()
        print("💡 Monitor with: tail -f phase2_extended_training_v2.log")
        print()
    else:
        print("⚠️  Training may not have started yet...")
        print()
    
else:
    print("⚠️  No training log found yet")
    print()
    print("To start training:")
    print("  python3 phase2_extended_training_v2.py")
    print()

print("=" * 80)
