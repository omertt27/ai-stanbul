#!/usr/bin/env python3
"""
Parse and Expand Existing Training Data for Intent Classification
==================================================================

This script takes your existing comprehensive_training_data.json and expands it
with variations, paraphrases, and additional context to create a robust dataset
for training the DistilBERT intent classifier.
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# Turkish paraphrase patterns for expansion
TURKISH_VARIATIONS = {
    'greeting': [
        'Merhaba', 'Selam', 'Selamlar', 'İyi günler', 'Günaydın', 'İyi akşamlar',
        'Nasılsın', 'Nasıl gidiyor', 'Ne haber', 'İyi misin', 'Naber',
        'Selam dostum', 'Hey', 'Hoş geldin', 'Hoşça kal'
    ],
    'thanks': [
        'Teşekkürler', 'Sağol', 'Teşekkür ederim', 'Çok teşekkürler',
        'Sağ ol', 'Çok sağol', 'Eyvallah', 'Mersi', 'Teşekkür',
        'Minnettarım', 'Çok yardımcı oldun', 'Süpersin'
    ],
    'farewell': [
        'Görüşürüz', 'Hoşça kal', 'Bay bay', 'Güle güle', 'İyi günler',
        'Kendine iyi bak', 'Sonra görüşürüz', 'Hoşçakal', 'Bye',
        'Haydi hoşça kal', 'İyi akşamlar', 'İyi geceler'
    ],
    'help': [
        'Yardım et', 'Yardıma ihtiyacım var', 'Bana yardım edebilir misin',
        'Yardım', 'Help', 'Yardımcı olur musun', 'Ne yapmalıyım',
        'Nasıl yapabilirim', 'Bilgi ver', 'Destek lazım'
    ],
}

# English variations
ENGLISH_VARIATIONS = {
    'greeting': [
        'Hello', 'Hi', 'Hey', 'Good morning', 'Good afternoon', 'Good evening',
        'How are you', 'How\'s it going', 'What\'s up', 'Greetings',
        'Nice to meet you', 'Hey there', 'Hiya'
    ],
    'thanks': [
        'Thanks', 'Thank you', 'Thanks a lot', 'Thank you so much',
        'Much appreciated', 'Cheers', 'Thanks!', 'Appreciate it',
        'Thank you very much', 'That\'s helpful'
    ],
    'farewell': [
        'Goodbye', 'Bye', 'See you', 'See you later', 'Take care',
        'Catch you later', 'Until next time', 'Bye bye', 'Have a good day',
        'Good night', 'Farewell'
    ],
    'help': [
        'Help', 'I need help', 'Can you help me', 'Assist me',
        'I need assistance', 'Help me please', 'What should I do',
        'How can I', 'Support needed', 'Give me information'
    ],
}

# Question patterns for different intents
QUESTION_PATTERNS_TR = {
    'attraction': [
        'nerede', 'neresi', 'hangi', 'göster', 'öner', 'tavsiye et',
        'neler var', 'ne var', 'gezilecek', 'görülecek', 'ziyaret',
        'gitmek istiyorum', 'görmek istiyorum', 'bul'
    ],
    'restaurant': [
        'restoran', 'yemek', 'lokanta', 'meze', 'kahvaltı', 'yemek yiyebileceğim',
        'nerede yemek yenir', 'açım', 'yemek önerisi', 'mekan'
    ],
    'transportation': [
        'nasıl giderim', 'nasıl gidilir', 'ulaşım', 'metro', 'otobüs', 'tramvay',
        'taksi', 'nasıl ulaşırım', 'yol tarifi', 'gitmek için', 'araç'
    ],
    'weather': [
        'hava durumu', 'hava nasıl', 'yağmur', 'güneşli', 'sıcak', 'soğuk',
        'hava', 'bugün hava', 'yarın hava', 'tahmin'
    ],
    'neighborhood': [
        'semt', 'mahalle', 'bölge', 'civarda', 'yakınında', 'etrafında',
        'hangi semtte', 'nerede kalmalıyım', 'alan'
    ],
    'events': [
        'etkinlik', 'konser', 'festival', 'gösteri', 'organizasyon', 'program',
        'ne var', 'neler oluyor', 'bugün ne var', 'yarın ne var'
    ],
    'route_planning': [
        'plan yap', 'rota', 'güzergah', 'tur planla', 'program yap',
        'rotam', 'nasıl gezerim', 'sırayla', 'plan'
    ],
}

QUESTION_PATTERNS_EN = {
    'attraction': [
        'where', 'which', 'show me', 'recommend', 'suggest',
        'what is there', 'places to visit', 'to see', 'want to visit',
        'want to see', 'find', 'tourist spots'
    ],
    'restaurant': [
        'restaurant', 'food', 'eat', 'dining', 'breakfast', 'lunch', 'dinner',
        'where to eat', 'hungry', 'food recommendation', 'place to eat'
    ],
    'transportation': [
        'how to get', 'how to go', 'transport', 'metro', 'bus', 'tram',
        'taxi', 'how to reach', 'directions', 'route', 'travel'
    ],
    'weather': [
        'weather', 'how is the weather', 'rain', 'sunny', 'hot', 'cold',
        'temperature', 'today weather', 'tomorrow weather', 'forecast'
    ],
    'neighborhood': [
        'neighborhood', 'district', 'area', 'nearby', 'around',
        'which area', 'where to stay', 'region'
    ],
    'events': [
        'event', 'concert', 'festival', 'show', 'performance', 'program',
        'what\'s on', 'happening', 'today events', 'tomorrow events'
    ],
    'route_planning': [
        'plan', 'route', 'itinerary', 'tour', 'schedule',
        'my route', 'how to explore', 'in order', 'planning'
    ],
}


def load_existing_data(file_path):
    """Load existing training data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert [text, intent] format to dict format
    formatted_data = []
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            formatted_data.append({
                'text': item[0],
                'intent': item[1]
            })
        elif isinstance(item, dict):
            formatted_data.append(item)
    
    return formatted_data


def detect_language(text):
    """Simple language detection based on Turkish characters"""
    turkish_chars = set('çğıöşüÇĞİÖŞÜ')
    return 'tr' if any(char in turkish_chars for char in text) else 'en'


def expand_with_variations(data):
    """Expand dataset with variations and paraphrases"""
    expanded_data = []
    intent_groups = defaultdict(list)
    
    # Group by intent
    for item in data:
        intent_groups[item['intent']].append(item)
    
    # Add original data
    for item in data:
        if 'language' not in item:
            item['language'] = detect_language(item['text'])
        expanded_data.append(item)
    
    print(f"📊 Original data: {len(data)} examples")
    print(f"📊 Found {len(intent_groups)} unique intents\n")
    
    # Add special variations for common intents
    for intent, variations_dict in [('greeting', TURKISH_VARIATIONS['greeting']), 
                                     ('greeting', ENGLISH_VARIATIONS['greeting'])]:
        lang = 'tr' if variations_dict == TURKISH_VARIATIONS.get('greeting', []) else 'en'
        for variation in variations_dict:
            if not any(d['text'].lower() == variation.lower() for d in expanded_data):
                expanded_data.append({
                    'text': variation,
                    'intent': intent,
                    'language': lang,
                    'category': 'daily_talk'
                })
    
    # Add thanks variations
    for intent, variations_dict in [('thanks', TURKISH_VARIATIONS['thanks']),
                                     ('thanks', ENGLISH_VARIATIONS['thanks'])]:
        lang = 'tr' if variations_dict == TURKISH_VARIATIONS.get('thanks', []) else 'en'
        for variation in variations_dict:
            if not any(d['text'].lower() == variation.lower() for d in expanded_data):
                expanded_data.append({
                    'text': variation,
                    'intent': intent,
                    'language': lang,
                    'category': 'daily_talk'
                })
    
    # Add farewell variations
    for intent, variations_dict in [('farewell', TURKISH_VARIATIONS['farewell']),
                                     ('farewell', ENGLISH_VARIATIONS['farewell'])]:
        lang = 'tr' if variations_dict == TURKISH_VARIATIONS.get('farewell', []) else 'en'
        for variation in variations_dict:
            if not any(d['text'].lower() == variation.lower() for d in expanded_data):
                expanded_data.append({
                    'text': variation,
                    'intent': intent,
                    'language': lang,
                    'category': 'daily_talk'
                })
    
    # Add help variations
    for intent, variations_dict in [('help', TURKISH_VARIATIONS['help']),
                                     ('help', ENGLISH_VARIATIONS['help'])]:
        lang = 'tr' if variations_dict == TURKISH_VARIATIONS.get('help', []) else 'en'
        for variation in variations_dict:
            if not any(d['text'].lower() == variation.lower() for d in expanded_data):
                expanded_data.append({
                    'text': variation,
                    'intent': intent,
                    'language': lang,
                    'category': 'daily_talk'
                })
    
    print(f"✅ Expanded to {len(expanded_data)} examples")
    
    return expanded_data


def analyze_dataset(data):
    """Analyze and print dataset statistics"""
    intent_counts = Counter(item['intent'] for item in data)
    language_counts = Counter(item.get('language', 'unknown') for item in data)
    
    print("\n" + "="*80)
    print("📊 DATASET STATISTICS")
    print("="*80)
    print(f"\n📝 Total examples: {len(data)}")
    print(f"🏷️  Unique intents: {len(intent_counts)}")
    print(f"🌍 Languages: {dict(language_counts)}\n")
    
    print("Intent Distribution:")
    print("-" * 60)
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(50, count)
        print(f"  {intent:25s} {count:4d} {bar}")
    
    print("\n" + "="*80)
    
    # Check for intents with too few examples
    low_count_intents = [(intent, count) for intent, count in intent_counts.items() if count < 20]
    if low_count_intents:
        print("⚠️  WARNING: Intents with less than 20 examples:")
        for intent, count in sorted(low_count_intents, key=lambda x: x[1]):
            print(f"   {intent}: {count} examples (recommend at least 50)")
        print()
    
    return intent_counts


def save_expanded_data(data, output_file):
    """Save expanded training data"""
    output_data = {
        'training_data': data,
        'metadata': {
            'total_examples': len(data),
            'num_intents': len(set(item['intent'] for item in data)),
            'intents': sorted(list(set(item['intent'] for item in data))),
            'languages': list(set(item.get('language', 'unknown') for item in data))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved expanded data to: {output_file}")
    print(f"   Total examples: {len(data)}")
    print(f"   Intents: {len(output_data['metadata']['intents'])}")


def main():
    input_file = 'comprehensive_training_data.json'
    output_file = 'expanded_intent_training_data.json'
    
    print("\n" + "="*80)
    print("🚀 PARSING AND EXPANDING TRAINING DATA")
    print("="*80 + "\n")
    
    # Load existing data
    print(f"📥 Loading data from: {input_file}")
    data = load_existing_data(input_file)
    
    # Expand with variations
    print("\n🔄 Expanding dataset with variations...")
    expanded_data = expand_with_variations(data)
    
    # Analyze dataset
    intent_counts = analyze_dataset(expanded_data)
    
    # Save expanded data
    print("\n💾 Saving expanded dataset...")
    save_expanded_data(expanded_data, output_file)
    
    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*80)
    print(f"\n📂 Original file: {input_file} ({len(data)} examples)")
    print(f"📂 Expanded file: {output_file} ({len(expanded_data)} examples)")
    print(f"📈 Expansion ratio: {len(expanded_data)/len(data):.2f}x")
    print(f"\n💡 Next step: Train model with:")
    print(f"   python3 train_distilbert_intent_classifier.py --data-file {output_file} --epochs 15")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
