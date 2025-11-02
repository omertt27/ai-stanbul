#!/usr/bin/env python3
"""
Intent Consolidation Script
Remaps 30 intents → 10 core intents for better accuracy and focus
"""

import json
from collections import Counter
from typing import Dict, List

# Intent mapping: old_intent → new_intent
INTENT_MAPPING = {
    # 1. RESTAURANT (consolidates food & dining)
    'restaurant': 'restaurant',
    'food': 'restaurant',
    'nightlife': 'restaurant',  # Food-related nightlife
    
    # 2. ATTRACTION (consolidates tourist spots)
    'attraction': 'attraction',
    'museum': 'attraction',
    'romantic': 'attraction',
    'family_activities': 'attraction',
    'shopping': 'attraction',  # Shopping destinations
    
    # 3. NEIGHBORHOOD (district guides)
    'neighborhoods': 'neighborhood',
    
    # 4. TRANSPORTATION (getting around)
    'transportation': 'transportation',
    'gps_navigation': 'transportation',
    
    # 5. DAILY_TALKS (conversational)
    'greeting': 'daily_talks',
    'farewell': 'daily_talks',
    'thanks': 'daily_talks',
    'help': 'daily_talks',
    
    # 6. HIDDEN_GEMS (local tips)
    'hidden_gems': 'hidden_gems',
    'local_tips': 'hidden_gems',
    
    # 7. WEATHER
    'weather': 'weather',
    
    # 8. EVENTS (activities & entertainment)
    'events': 'events',
    'cultural_info': 'events',  # Cultural events
    
    # 9. ROUTE_PLANNING
    'route_planning': 'route_planning',
    'booking': 'route_planning',  # Trip booking/planning
    
    # 10. GENERAL_INFO (catch-all)
    'general_info': 'general_info',
    'history': 'general_info',
    'price_info': 'general_info',
    'accommodation': 'general_info',
    'emergency': 'general_info',
    'recommendation': 'general_info',
    'budget': 'general_info',
    'luxury': 'general_info',
}

# Define the 10 core intents
CORE_INTENTS = [
    'restaurant',
    'attraction',
    'neighborhood',
    'transportation',
    'daily_talks',
    'hidden_gems',
    'weather',
    'events',
    'route_planning',
    'general_info'
]


def load_training_data(filepath: str) -> List[Dict]:
    """Load training data from JSON file"""
    print(f"\n📥 Loading training data from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and 'training_data' in data:
        examples = data['training_data']
        metadata = data.get('metadata', {})
        print(f"   Metadata: {metadata}")
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    # Normalize to dict format
    normalized = []
    for item in examples:
        if isinstance(item, dict):
            text = item.get('text') or item.get('query', '')
            intent = item.get('intent') or item.get('label', '')
            if text and intent:
                normalized.append({'text': text, 'intent': intent})
        elif isinstance(item, list) and len(item) == 2:
            normalized.append({'text': item[0], 'intent': item[1]})
    
    print(f"✅ Loaded {len(normalized)} examples")
    return normalized


def analyze_original_distribution(examples: List[Dict]):
    """Analyze the original intent distribution"""
    print(f"\n{'=' * 80}")
    print("📊 ORIGINAL INTENT DISTRIBUTION (30 Intents)")
    print('=' * 80)
    
    intent_counts = Counter(ex['intent'] for ex in examples)
    
    print(f"\n{'Intent':<25} {'Count':>8} {'Percentage':>12}")
    print('-' * 50)
    
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(examples) * 100
        print(f"{intent:<25} {count:>8} {pct:>11.1f}%")
    
    print(f"\nTotal Intents: {len(intent_counts)}")
    print(f"Total Examples: {len(examples)}")
    print(f"Average per intent: {len(examples) / len(intent_counts):.1f}")


def remap_intents(examples: List[Dict]) -> List[Dict]:
    """Remap 30 intents to 10 core intents"""
    print(f"\n{'=' * 80}")
    print("🔄 REMAPPING INTENTS (30 → 10)")
    print('=' * 80)
    
    remapped = []
    unmapped = []
    
    for ex in examples:
        old_intent = ex['intent']
        
        if old_intent in INTENT_MAPPING:
            new_intent = INTENT_MAPPING[old_intent]
            remapped.append({
                'text': ex['text'],
                'intent': new_intent,
                'original_intent': old_intent
            })
        else:
            # Unknown intent - map to general_info
            print(f"⚠️  Unknown intent: {old_intent}")
            unmapped.append(old_intent)
            remapped.append({
                'text': ex['text'],
                'intent': 'general_info',
                'original_intent': old_intent
            })
    
    print(f"\n✅ Remapped {len(remapped)} examples")
    if unmapped:
        print(f"⚠️  {len(unmapped)} examples had unmapped intents")
        print(f"   Unmapped intents: {set(unmapped)}")
    
    return remapped


def analyze_remapped_distribution(examples: List[Dict]):
    """Analyze the remapped intent distribution"""
    print(f"\n{'=' * 80}")
    print("📊 REMAPPED INTENT DISTRIBUTION (10 Core Intents)")
    print('=' * 80)
    
    intent_counts = Counter(ex['intent'] for ex in examples)
    
    print(f"\n{'Intent':<25} {'Count':>8} {'Percentage':>12} {'Target':>10}")
    print('-' * 60)
    
    targets = {
        'restaurant': 13,
        'attraction': 13,
        'neighborhood': 12,
        'transportation': 12,
        'daily_talks': 9,
        'hidden_gems': 10,
        'weather': 9,
        'events': 10,
        'route_planning': 9,
        'general_info': 9,
    }
    
    for intent in CORE_INTENTS:
        count = intent_counts[intent]
        pct = count / len(examples) * 100
        target_pct = targets.get(intent, 10)
        status = "✅" if abs(pct - target_pct) < 3 else "⚠️"
        print(f"{intent:<25} {count:>8} {pct:>11.1f}% {target_pct:>9}% {status}")
    
    print(f"\nTotal Intents: {len(intent_counts)}")
    print(f"Total Examples: {len(examples)}")
    print(f"Average per intent: {len(examples) / len(intent_counts):.1f}")
    
    # Check balance
    min_count = min(intent_counts.values())
    max_count = max(intent_counts.values())
    balance_ratio = min_count / max_count
    print(f"\n⚖️  Balance Score: {balance_ratio:.2f} (0=poor, 1=perfect)")
    print(f"   Min per intent: {min_count}")
    print(f"   Max per intent: {max_count}")
    print(f"   Ratio: {max_count/min_count:.1f}x")


def detect_language(text: str) -> str:
    """Simple Turkish/English detection"""
    turkish_chars = set('çğıöşüÇĞİÖŞÜ')
    turkish_words = {
        've', 'mi', 'ne', 'nerede', 'nasıl', 'için', 'ile', 'var', 'bir', 'bu', 'şu',
        'ben', 'sen', 'bana', 'sana', 'hangi', 'kaç', 'kim', 'niye', 'niçin',
        'lütfen', 'teşekkür', 'merhaba', 'günaydın', 'iyi', 'kötü'
    }
    
    if any(c in turkish_chars for c in text):
        return 'turkish'
    
    words = text.lower().split()
    if any(word in turkish_words for word in words):
        return 'turkish'
    
    return 'english'


def analyze_language_balance(examples: List[Dict]):
    """Analyze language balance per intent"""
    print(f"\n{'=' * 80}")
    print("🌐 LANGUAGE BALANCE PER INTENT")
    print('=' * 80)
    
    print(f"\n{'Intent':<25} {'Turkish':>10} {'English':>10} {'Balance'}")
    print('-' * 60)
    
    for intent in CORE_INTENTS:
        intent_examples = [ex for ex in examples if ex['intent'] == intent]
        
        turkish_count = sum(1 for ex in intent_examples if detect_language(ex['text']) == 'turkish')
        english_count = len(intent_examples) - turkish_count
        
        if len(intent_examples) > 0:
            tr_pct = turkish_count / len(intent_examples) * 100
            en_pct = english_count / len(intent_examples) * 100
            balance = min(tr_pct, en_pct) / max(tr_pct, en_pct) if max(tr_pct, en_pct) > 0 else 0
            
            status = "✅" if balance > 0.7 else "⚠️"
            print(f"{intent:<25} {turkish_count:>10} {english_count:>10} {balance:>8.2f} {status}")


def save_remapped_data(examples: List[Dict], output_file: str):
    """Save remapped training data"""
    print(f"\n{'=' * 80}")
    print("💾 SAVING REMAPPED DATA")
    print('=' * 80)
    
    # Prepare data in the expected format
    training_data = [
        {'text': ex['text'], 'intent': ex['intent']}
        for ex in examples
    ]
    
    # Count statistics
    intent_counts = Counter(ex['intent'] for ex in examples)
    turkish_count = sum(1 for ex in examples if detect_language(ex['text']) == 'turkish')
    english_count = len(examples) - turkish_count
    
    output_data = {
        'training_data': training_data,
        'metadata': {
            'total_examples': len(examples),
            'num_intents': len(CORE_INTENTS),
            'intents': CORE_INTENTS,
            'turkish_examples': turkish_count,
            'english_examples': english_count,
            'turkish_percentage': turkish_count / len(examples) * 100,
            'english_percentage': english_count / len(examples) * 100,
            'intent_distribution': dict(intent_counts),
            'remapping_date': '2025-11-02',
            'remapping_version': '1.0',
            'source_intents': 30,
            'target_intents': 10,
            'consolidation_strategy': 'functional_grouping'
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   Total examples: {len(examples)}")
    print(f"   Intents: {len(CORE_INTENTS)}")
    print(f"   Turkish: {turkish_count} ({turkish_count/len(examples)*100:.1f}%)")
    print(f"   English: {english_count} ({english_count/len(examples)*100:.1f}%)")


def create_intent_mapping_file():
    """Create intent_mapping.json for the model"""
    mapping_data = {
        'intents': CORE_INTENTS,
        'num_labels': len(CORE_INTENTS),
        'intent_to_id': {intent: idx for idx, intent in enumerate(CORE_INTENTS)},
        'id_to_intent': {idx: intent for idx, intent in enumerate(CORE_INTENTS)},
        'consolidation_mapping': INTENT_MAPPING,
        'version': '2.0',
        'date': '2025-11-02'
    }
    
    output_file = 'intent_mapping_10_core.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Intent mapping saved to: {output_file}")


def main():
    """Main consolidation process"""
    print("\n" + "=" * 80)
    print("🎯 INTENT CONSOLIDATION: 30 → 10 CORE INTENTS")
    print("=" * 80)
    
    # Load original data
    input_file = 'comprehensive_training_data.json'
    examples = load_training_data(input_file)
    
    # Analyze original distribution
    analyze_original_distribution(examples)
    
    # Remap intents
    remapped_examples = remap_intents(examples)
    
    # Analyze remapped distribution
    analyze_remapped_distribution(remapped_examples)
    
    # Analyze language balance
    analyze_language_balance(remapped_examples)
    
    # Save remapped data
    output_file = 'comprehensive_training_data_10_intents.json'
    save_remapped_data(remapped_examples, output_file)
    
    # Create intent mapping file
    create_intent_mapping_file()
    
    print(f"\n{'=' * 80}")
    print("✅ INTENT CONSOLIDATION COMPLETE!")
    print('=' * 80)
    
    print("\n🎯 Next Steps:")
    print("   1. Review the remapped data")
    print("   2. Update training script to use 10 intents")
    print("   3. Retrain the model:")
    print("      python3 train_turkish_enhanced_intent_classifier.py \\")
    print("        --data-file comprehensive_training_data_10_intents.json \\")
    print("        --num-intents 10")
    print()


if __name__ == "__main__":
    main()
