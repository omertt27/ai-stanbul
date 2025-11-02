#!/usr/bin/env python3
"""
Dataset Integration & Balancing Script
Integrates english_expanded_training_data.json to create a balanced bilingual dataset
"""

import json
import random
from collections import Counter
from typing import Dict, List

def is_turkish_text(text: str) -> bool:
    """Improved Turkish detection"""
    if not isinstance(text, str):
        return False
    
    # Turkish-specific characters
    turkish_chars = set('çğıöşüÇĞİÖŞÜ')
    if any(c in turkish_chars for c in text):
        return True
    
    # Common Turkish words (more comprehensive)
    turkish_words = {
        've', 'mi', 'ne', 'nerede', 'nasıl', 'için', 'ile', 'var', 'bir', 'bu', 'şu',
        'ben', 'sen', 'bana', 'sana', 'hangi', 'kaç', 'kim', 'niye', 'niçin',
        'lütfen', 'teşekkür', 'merhaba', 'günaydın', 'iyi', 'kötü', 'büyük', 'küçük',
        'yeni', 'eski', 'yakın', 'uzak', 'var', 'yok', 'gitmek', 'gelmek', 'görmek',
        'yapmak', 'olmak', 'bilmek', 'söylemek', 'müze', 'yer', 'zaman', 'gün'
    }
    
    # Tokenize and check
    lower_text = text.lower()
    words = lower_text.split()
    
    # If any word is Turkish, consider it Turkish
    return any(word in turkish_words for word in words)


def normalize_dataset_format(data: any) -> List[Dict]:
    """Convert various dataset formats to standard format"""
    examples = []
    
    if isinstance(data, dict) and 'training_data' in data:
        raw_examples = data['training_data']
    elif isinstance(data, list):
        raw_examples = data
    else:
        return []
    
    for item in raw_examples:
        if isinstance(item, dict):
            # Already in correct format
            if 'text' in item and 'intent' in item:
                examples.append({
                    'text': item['text'],
                    'intent': item['intent']
                })
            elif 'query' in item and 'label' in item:
                examples.append({
                    'text': item['query'],
                    'intent': item['label']
                })
        elif isinstance(item, list) and len(item) == 2:
            # [text, intent] format
            examples.append({
                'text': item[0],
                'intent': item[1]
            })
    
    return examples


def analyze_dataset(examples: List[Dict], dataset_name: str):
    """Analyze and print dataset statistics"""
    print(f"\n{'=' * 80}")
    print(f"📊 {dataset_name}")
    print('=' * 80)
    
    total = len(examples)
    print(f"Total Examples: {total}")
    
    # Language distribution
    turkish_count = sum(1 for ex in examples if is_turkish_text(ex['text']))
    english_count = total - turkish_count
    
    print(f"\n🌐 Language Distribution:")
    print(f"   🇹🇷 Turkish: {turkish_count:4d} ({turkish_count/total*100:5.1f}%)")
    print(f"   🇬🇧 English: {english_count:4d} ({english_count/total*100:5.1f}%)")
    
    # Intent distribution
    intent_counts = Counter(ex['intent'] for ex in examples)
    print(f"\n🎯 Intent Distribution:")
    print(f"   Unique Intents: {len(intent_counts)}")
    print(f"   Top 5 Intents:")
    for intent, count in intent_counts.most_common(5):
        print(f"      {intent:25s}: {count:4d} ({count/total*100:5.1f}%)")
    
    # Check balance
    min_count = min(intent_counts.values())
    max_count = max(intent_counts.values())
    balance_ratio = min_count / max_count if max_count > 0 else 0
    print(f"\n⚖️  Balance Score: {balance_ratio:.2f} (0=poor, 1=perfect)")
    print(f"   Min per intent: {min_count}")
    print(f"   Max per intent: {max_count}")
    
    return {
        'total': total,
        'turkish': turkish_count,
        'english': english_count,
        'intents': len(intent_counts),
        'balance': balance_ratio
    }


def integrate_datasets():
    """Main integration function"""
    print("\n" + "=" * 80)
    print("🔄 BILINGUAL DATASET INTEGRATION")
    print("=" * 80)
    
    # Load english_expanded_training_data.json (the source of truth)
    print("\n📥 Loading english_expanded_training_data.json...")
    with open('english_expanded_training_data.json', 'r', encoding='utf-8') as f:
        english_data = json.load(f)
    
    # Normalize format
    examples = normalize_dataset_format(english_data)
    
    # Analyze
    stats = analyze_dataset(examples, "ENGLISH_EXPANDED_TRAINING_DATA.JSON")
    
    # Check if we need better balance
    print(f"\n{'=' * 80}")
    print("📊 BALANCE ASSESSMENT")
    print('=' * 80)
    
    turkish_ratio = stats['turkish'] / stats['total']
    english_ratio = stats['english'] / stats['total']
    
    print(f"\nCurrent ratio: {turkish_ratio:.1%} Turkish / {english_ratio:.1%} English")
    
    if english_ratio >= 0.40 and english_ratio <= 0.60:
        print("✅ Dataset is well-balanced for bilingual training!")
        recommendation = "Use as-is"
    elif english_ratio < 0.40:
        print("⚠️  Dataset needs more English examples")
        recommendation = "Add more English data"
    else:
        print("⚠️  Dataset needs more Turkish examples")
        recommendation = "Add more Turkish data"
    
    print(f"Recommendation: {recommendation}")
    
    # Save as the new comprehensive dataset
    print(f"\n{'=' * 80}")
    print("💾 SAVING INTEGRATED DATASET")
    print('=' * 80)
    
    output_data = {
        'training_data': examples,
        'metadata': {
            'total_examples': len(examples),
            'turkish_examples': stats['turkish'],
            'english_examples': stats['english'],
            'num_intents': stats['intents'],
            'balance_score': stats['balance'],
            'integration_source': 'english_expanded_training_data.json',
            'integrated_at': '2025-01-08',
            'ready_for_training': True
        }
    }
    
    # Save to comprehensive_training_data_v2.json (keep original as backup)
    output_file = 'comprehensive_training_data_v2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   Total examples: {len(examples)}")
    print(f"   Turkish: {stats['turkish']} ({turkish_ratio:.1%})")
    print(f"   English: {stats['english']} ({english_ratio:.1%})")
    
    # Also update the original comprehensive_training_data.json
    print(f"\n📝 Updating comprehensive_training_data.json...")
    with open('comprehensive_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("✅ Updated comprehensive_training_data.json")
    
    print(f"\n{'=' * 80}")
    print("✅ INTEGRATION COMPLETE!")
    print('=' * 80)
    print(f"\n🎯 Next Steps:")
    print(f"   1. Verify the data: head -100 {output_file}")
    print(f"   2. Train the model: python train_turkish_enhanced_intent_classifier.py")
    print(f"   3. Test bilingual performance")
    print()
    
    return output_file


if __name__ == "__main__":
    integrate_datasets()
