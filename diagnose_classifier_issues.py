"""
Diagnose issues with the trained intent classifier.
Analyze training data distribution, model behavior, and potential problems.
"""

import json
from collections import Counter, defaultdict
import numpy as np

def analyze_training_data():
    """Analyze the training data for potential issues"""
    print("="*80)
    print("TRAINING DATA ANALYSIS")
    print("="*80)
    
    with open("comprehensive_training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\nTotal samples: {len(data)}")
    
    # Intent distribution (data format is [text, intent])
    intent_counts = Counter(sample[1] for sample in data)
    print(f"\n--- Intent Distribution ---")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"{intent:20s}: {count:4d} ({percentage:5.2f}%)")
    
    # Check for imbalance
    max_count = max(intent_counts.values())
    min_count = min(intent_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}x (max: {max_count}, min: {min_count})")
    
    if imbalance_ratio > 10:
        print("⚠️  SEVERE CLASS IMBALANCE DETECTED")
    elif imbalance_ratio > 5:
        print("⚠️  MODERATE CLASS IMBALANCE DETECTED")
    
    # Language distribution by intent
    print(f"\n--- Language Distribution by Intent ---")
    lang_by_intent = defaultdict(lambda: {"turkish": 0, "english": 0, "unknown": 0})
    
    for sample in data:
        intent = sample[1]  # Format is [text, intent]
        text = sample[0]
        
        # Simple language detection
        has_turkish_chars = any(char in text for char in "ığüşöçİĞÜŞÖÇ")
        
        if has_turkish_chars:
            lang_by_intent[intent]["turkish"] += 1
        else:
            # Check for common Turkish words without special chars
            turkish_words = ["bana", "nerede", "nasil", "var", "yok", "iyi", "guzel"]
            if any(word in text.lower() for word in turkish_words):
                lang_by_intent[intent]["turkish"] += 1
            else:
                lang_by_intent[intent]["english"] += 1
    
    for intent in sorted(lang_by_intent.keys()):
        stats = lang_by_intent[intent]
        total = stats["turkish"] + stats["english"]
        if total > 0:
            tr_pct = (stats["turkish"] / total) * 100
            en_pct = (stats["english"] / total) * 100
            print(f"{intent:20s}: TR: {stats['turkish']:3d} ({tr_pct:5.1f}%)  EN: {stats['english']:3d} ({en_pct:5.1f}%)")
    
    # Check for duplicate or near-duplicate samples
    print(f"\n--- Checking for Duplicates ---")
    text_counts = Counter(sample[0].lower().strip() for sample in data)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} duplicate texts")
        for text, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - '{text[:50]}...' appears {count} times")
    else:
        print("✓ No duplicate texts found")
    
    # Analyze sample quality for problematic intents
    print(f"\n--- Analyzing Problematic Intents ---")
    problematic_intents = ["transportation", "restaurant", "attraction", "weather", "route_planning"]
    
    for intent in problematic_intents:
        samples = [s for s in data if s[1] == intent]
        print(f"\n{intent.upper()} ({len(samples)} samples):")
        
        # Show a few examples
        turkish_samples = [s for s in samples if any(c in s[0] for c in "ığüşöçİĞÜŞÖÇ")]
        english_samples = [s for s in samples if s not in turkish_samples]
        
        print(f"  Turkish: {len(turkish_samples)}, English: {len(english_samples)}")
        
        if turkish_samples:
            print(f"  Example TR: {turkish_samples[0][0][:60]}...")
        if english_samples:
            print(f"  Example EN: {english_samples[0][0][:60]}...")
    
    return intent_counts, lang_by_intent

def analyze_model_predictions():
    """Analyze the model's predictions from evaluation"""
    print("\n" + "="*80)
    print("MODEL PREDICTION ANALYSIS")
    print("="*80)
    
    with open("models/istanbul_intent_classifier_finetuned/evaluation_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    predictions = results["predictions"]
    
    # Confusion patterns
    print(f"\n--- Confusion Patterns ---")
    confusion = defaultdict(lambda: defaultdict(int))
    
    for pred in predictions:
        if not pred["correct"]:
            confusion[pred["expected"]][pred["predicted"]] += 1
    
    for expected in sorted(confusion.keys()):
        print(f"\n{expected}:")
        for predicted, count in sorted(confusion[expected].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  → {predicted}: {count} times")
    
    # Confidence analysis
    print(f"\n--- Confidence Analysis ---")
    correct_preds = [p for p in predictions if p["correct"]]
    incorrect_preds = [p for p in predictions if not p["correct"]]
    
    if correct_preds:
        avg_conf_correct = np.mean([p["confidence"] for p in correct_preds])
        print(f"Average confidence (correct): {avg_conf_correct:.3f}")
    
    if incorrect_preds:
        avg_conf_incorrect = np.mean([p["confidence"] for p in incorrect_preds])
        print(f"Average confidence (incorrect): {avg_conf_incorrect:.3f}")
    
    # Check if model is just confused or over-confident
    high_conf_wrong = [p for p in incorrect_preds if p["confidence"] > 0.5]
    print(f"\nHigh-confidence mistakes (>50%): {len(high_conf_wrong)}/{len(incorrect_preds)}")
    
    if high_conf_wrong:
        print("Examples:")
        for pred in high_conf_wrong[:3]:
            print(f"  '{pred['query'][:50]}...'")
            print(f"  Expected: {pred['expected']}, Predicted: {pred['predicted']} ({pred['confidence']:.3f})")

def identify_issues():
    """Identify specific issues and recommend solutions"""
    print("\n" + "="*80)
    print("IDENTIFIED ISSUES & RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    
    # Analyze the data
    with open("comprehensive_training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    intent_counts = Counter(sample[1] for sample in data)
    
    # Issue 1: Class imbalance
    max_count = max(intent_counts.values())
    min_count = min(intent_counts.values())
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 5:
        issues.append({
            "issue": "Severe Class Imbalance",
            "severity": "HIGH",
            "description": f"Imbalance ratio of {imbalance_ratio:.1f}x between most and least common intents",
            "recommendation": "Use class weights during training or oversample minority classes"
        })
    
    # Issue 2: Low sample count for some intents
    small_intents = [intent for intent, count in intent_counts.items() if count < 30]
    if small_intents:
        issues.append({
            "issue": "Insufficient Training Data",
            "severity": "HIGH",
            "description": f"{len(small_intents)} intents have <30 samples: {', '.join(small_intents)}",
            "recommendation": "Add more diverse training examples for these intents"
        })
    
    # Issue 3: English weakness
    with open("models/istanbul_intent_classifier_finetuned/evaluation_results.json", "r") as f:
        results = json.load(f)
    
    english_acc = results["by_language"]["English"]["correct"] / results["by_language"]["English"]["total"]
    
    if english_acc < 0.3:
        issues.append({
            "issue": "Poor English Performance",
            "severity": "CRITICAL",
            "description": f"Only {english_acc:.1%} accuracy on English queries",
            "recommendation": "Add more high-quality English training examples with diverse phrasings"
        })
    
    # Issue 4: Model architecture
    issues.append({
        "issue": "Model Selection",
        "severity": "MEDIUM",
        "description": "DistilBERT may not be optimal for Turkish-English bilingual tasks",
        "recommendation": "Consider XLM-RoBERTa, mBERT, or BERTurk for better multilingual support"
    })
    
    # Print issues
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. [{issue['severity']}] {issue['issue']}")
        print(f"   Problem: {issue['description']}")
        print(f"   Solution: {issue['recommendation']}")
    
    return issues

def main():
    print("\n" + "="*80)
    print("INTENT CLASSIFIER DIAGNOSTIC REPORT")
    print("="*80)
    
    # Analyze training data
    intent_counts, lang_by_intent = analyze_training_data()
    
    # Analyze predictions
    analyze_model_predictions()
    
    # Identify issues
    issues = identify_issues()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
The neural intent classifier is experiencing significant performance issues:

1. Overall accuracy: 33.3% (far below the 85%+ target)
2. Critical English weakness: Only 9.1% accuracy on English queries
3. Turkish performance: 51.7% (better but still inadequate)
4. Training showed overfitting (99%+ train accuracy, 51% validation)

ROOT CAUSES:
- Class imbalance in training data
- Insufficient English examples for key intents
- Possible model architecture mismatch for bilingual task
- Training data may have quality/diversity issues

IMMEDIATE ACTIONS NEEDED:
1. Add 500+ high-quality English examples across all intents
2. Balance training data distribution (aim for 50+ samples per intent)
3. Consider switching to XLM-RoBERTa or mBERT
4. Implement class weighting during training
5. Add more diverse query patterns for underperforming intents

STATUS: Model is NOT ready for production use.
""")
    
    # Save diagnostic report
    diagnostic = {
        "timestamp": "2025-11-02",
        "overall_accuracy": 0.333,
        "english_accuracy": 0.091,
        "turkish_accuracy": 0.517,
        "intent_distribution": dict(intent_counts),
        "issues": issues,
        "status": "FAILED - Requires significant improvement"
    }
    
    with open("NEURAL_CLASSIFIER_DIAGNOSTIC_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(diagnostic, f, indent=2, ensure_ascii=False)
    
    print("✓ Full diagnostic report saved to: NEURAL_CLASSIFIER_DIAGNOSTIC_REPORT.json")

if __name__ == "__main__":
    main()
