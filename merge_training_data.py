#!/usr/bin/env python3
"""
Merge existing training data with new enhanced data for weak intents
Removes duplicates and validates the final dataset
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter


def load_json_file(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict, filepath: str):
    """Save JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_datasets(existing_file: str, new_file: str, output_file: str):
    """Merge two training datasets and remove duplicates"""
    
    print(f"📂 Loading existing data from: {existing_file}")
    existing = load_json_file(existing_file)
    existing_samples = existing.get('training_data', [])
    print(f"   Found {len(existing_samples)} existing samples")
    
    print(f"📂 Loading new data from: {new_file}")
    new = load_json_file(new_file)
    new_samples = new.get('training_data', [])
    print(f"   Found {len(new_samples)} new samples")
    
    # Merge all samples
    all_samples = existing_samples + new_samples
    print(f"\n🔗 Total samples before deduplication: {len(all_samples)}")
    
    # Remove duplicates (case-insensitive, stripped)
    unique_samples = []
    seen_texts = set()
    duplicates_removed = 0
    
    for sample in all_samples:
        text_lower = sample['text'].lower().strip()
        if text_lower not in seen_texts:
            unique_samples.append(sample)
            seen_texts.add(text_lower)
        else:
            duplicates_removed += 1
    
    print(f"🗑️  Removed {duplicates_removed} duplicate samples")
    print(f"✨ Final unique samples: {len(unique_samples)}")
    
    # Analyze distribution
    print("\n📊 Intent distribution:")
    intent_counts = Counter(s['intent'] for s in unique_samples)
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(unique_samples)) * 100
        print(f"   {intent}: {count} ({percentage:.1f}%)")
    
    # Analyze language distribution
    print("\n🌐 Language distribution:")
    turkish_count = sum(1 for s in unique_samples if any(ord(c) > 127 for c in s['text'][:50]))
    english_count = len(unique_samples) - turkish_count
    print(f"   Turkish: {turkish_count} ({(turkish_count/len(unique_samples)*100):.1f}%)")
    print(f"   English: {english_count} ({(english_count/len(unique_samples)*100):.1f}%)")
    
    # Create output
    output = {
        'training_data': unique_samples,
        'metadata': {
            'total_samples': len(unique_samples),
            'num_intents': len(intent_counts),
            'source_files': [existing_file, new_file],
            'duplicates_removed': duplicates_removed,
            'created': '2025-11-02'
        }
    }
    
    # Save
    print(f"\n💾 Saving merged dataset to: {output_file}")
    save_json_file(output, output_file)
    print(f"✅ Successfully saved {len(unique_samples)} samples!")
    
    return len(unique_samples)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Merge training datasets and remove duplicates"
    )
    parser.add_argument(
        '--existing',
        required=True,
        help='Path to existing training data JSON file'
    )
    parser.add_argument(
        '--new',
        required=True,
        help='Path to new training data JSON file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output merged JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.existing).exists():
        print(f"❌ Error: Existing file not found: {args.existing}")
        return 1
    
    if not Path(args.new).exists():
        print(f"❌ Error: New file not found: {args.new}")
        return 1
    
    # Merge datasets
    try:
        total_samples = merge_datasets(args.existing, args.new, args.output)
        print(f"\n🎉 Success! Created dataset with {total_samples} samples")
        print(f"📋 Next step: Retrain model with: python train_turkish_enhanced_intent_classifier.py --data-file {args.output}")
        return 0
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
