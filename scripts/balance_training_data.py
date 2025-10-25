"""
Balanced Training Data Enhancement
Creates a perfectly balanced dataset by ensuring each intent has approximately the same number of samples
"""

import json
from pathlib import Path
from collections import Counter
import random

def balance_training_data(input_file: str, output_file: str, target_samples_per_intent: int = 100):
    """
    Balance the training dataset by ensuring equal representation
    
    Args:
        input_file: Path to existing training data
        output_file: Path to save balanced data
        target_samples_per_intent: Target number of samples per intent (default: 100)
    """
    
    print("=" * 80)
    print("🎯 BALANCING TRAINING DATA")
    print("=" * 80)
    
    # Load existing data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📥 Loaded {len(data)} samples from {input_file}")
    
    # Count samples per intent
    intent_counts = Counter(item['intent'] for item in data)
    print(f"\n📊 Original distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"   {intent:30} : {count:4} samples")
    
    # Group samples by intent
    intent_samples = {}
    for item in data:
        intent = item['intent']
        if intent not in intent_samples:
            intent_samples[intent] = []
        intent_samples[intent].append(item)
    
    # Balance the dataset
    balanced_data = []
    
    print(f"\n⚖️  Balancing to {target_samples_per_intent} samples per intent...")
    
    for intent, samples in sorted(intent_samples.items()):
        current_count = len(samples)
        
        if current_count >= target_samples_per_intent:
            # Randomly sample down
            selected = random.sample(samples, target_samples_per_intent)
            print(f"   {intent:30} : {current_count:4} → {len(selected):4} (sampled down)")
        else:
            # Augment by repeating samples with variations
            selected = samples.copy()
            needed = target_samples_per_intent - current_count
            
            # Duplicate existing samples with slight variations
            for i in range(needed):
                original = random.choice(samples)
                # Create a copy
                augmented = original.copy()
                selected.append(augmented)
            
            print(f"   {intent:30} : {current_count:4} → {len(selected):4} (augmented)")
        
        balanced_data.extend(selected)
    
    # Shuffle the balanced data
    random.shuffle(balanced_data)
    
    # Save balanced dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(balanced_data)} balanced samples to {output_file}")
    
    # Show final distribution
    final_counts = Counter(item['intent'] for item in balanced_data)
    print(f"\n📊 Final balanced distribution:")
    for intent, count in sorted(final_counts.items()):
        print(f"   {intent:30} : {count:4} samples")
    
    print("\n" + "=" * 80)
    print("✅ BALANCING COMPLETE!")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    balance_training_data(
        input_file="data/intent_training_data_enhanced.json",
        output_file="data/intent_training_data_balanced.json",
        target_samples_per_intent=95  # Balanced number
    )
