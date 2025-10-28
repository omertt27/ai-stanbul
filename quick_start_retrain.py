#!/usr/bin/env python3
"""
Quick Start: Retrain Model with English Expansion & Feedback
Combines English expansion data with user feedback for comprehensive retraining
"""

import json
import sys
from pathlib import Path


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def merge_datasets():
    """Merge English expansion with user feedback data"""
    print("=" * 80)
    print("MERGING DATASETS FOR RETRAINING")
    print("=" * 80)
    print()
    
    datasets = []
    sources = []
    
    # 1. Load English expansion data
    english_file = 'english_expanded_training_data.json'
    if Path(english_file).exists():
        print(f"✅ Loading {english_file}...")
        english_data = load_json(english_file)
        english_examples = english_data.get('training_data', english_data)
        datasets.append(english_examples)
        sources.append(('english_expansion', len(english_examples)))
        print(f"   Loaded {len(english_examples)} examples\n")
    else:
        print(f"⚠️ {english_file} not found, skipping...\n")
    
    # 2. Load feedback-based retraining data
    feedback_file = 'data/retraining_data.json'
    if Path(feedback_file).exists():
        print(f"✅ Loading {feedback_file}...")
        feedback_data = load_json(feedback_file)
        feedback_examples = feedback_data.get('training_data', feedback_data)
        datasets.append(feedback_examples)
        sources.append(('user_feedback', len(feedback_examples)))
        print(f"   Loaded {len(feedback_examples)} examples\n")
    else:
        print(f"⚠️ {feedback_file} not found, skipping...\n")
    
    # 3. Merge all datasets
    if not datasets:
        print("❌ No datasets found to merge!")
        sys.exit(1)
    
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    
    print(f"📊 Merged Dataset Statistics:")
    print(f"   Total examples: {len(merged)}")
    for source_name, count in sources:
        print(f"   - {source_name}: {count} examples")
    print()
    
    # Count by intent
    from collections import Counter
    intent_counts = Counter(item['intent'] for item in merged)
    
    print(f"📈 Distribution by Intent (Top 15):")
    for intent, count in intent_counts.most_common(15):
        bar = "█" * min(50, count // 10)
        print(f"   {intent:20s} {count:4d} {bar}")
    print()
    
    # Save merged dataset
    output_file = 'final_training_data_merged.json'
    output_data = {
        'training_data': merged,
        'metadata': {
            'total_examples': len(merged),
            'sources': dict(sources),
            'num_intents': len(intent_counts),
            'intents': sorted(list(set(item['intent'] for item in merged)))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Merged dataset saved to: {output_file}")
    print()
    
    return output_file


def print_next_steps(merged_file):
    """Print next steps"""
    print("=" * 80)
    print("✅ DATASET MERGE COMPLETE!")
    print("=" * 80)
    print()
    print("🎯 Next Steps:")
    print()
    print("1. Review the merged dataset:")
    print(f"   cat {merged_file} | jq '.metadata'")
    print()
    print("2. Train the model:")
    print(f"   python train_intent_classifier.py \\")
    print(f"     --data-file {merged_file} \\")
    print(f"     --epochs 20 \\")
    print(f"     --batch-size 16 \\")
    print(f"     --output-dir models/distilbert_intent_classifier_v2")
    print()
    print("3. Test the new model:")
    print(f"   python distilbert_intent_inference.py")
    print()
    print("4. Compare performance:")
    print(f"   # Test English queries")
    print(f"   # Compare with current model accuracy")
    print()
    print("5. Deploy if improved:")
    print(f"   mv models/distilbert_intent_classifier models/distilbert_intent_classifier_backup")
    print(f"   mv models/distilbert_intent_classifier_v2 models/distilbert_intent_classifier")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 QUICK START: DATASET MERGE & RETRAIN")
    print("=" * 80)
    print("\nCombining English expansion + user feedback for retraining\n")
    
    merged_file = merge_datasets()
    print_next_steps(merged_file)
    
    print("💡 Tip: Run this script regularly (weekly/monthly) to incorporate")
    print("   new feedback and keep improving the model!\n")
