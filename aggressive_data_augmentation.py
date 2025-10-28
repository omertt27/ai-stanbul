#!/usr/bin/env python3
"""
Aggressive Data Augmentation for Intent Classification
=======================================================

This script aggressively augments the training data to reach at least
50-100 examples per intent for robust transformer training.
"""

import json
import random
from collections import Counter, defaultdict

# Augmentation templates per intent type
AUGMENTATION_TEMPLATES = {
    'attraction': {
        'tr': [
            "{place} nerede?",
            "{place} hakkında bilgi ver",
            "{place} nasıl bir yer?",
            "{place} gezilir mi?",
            "{place}'i ziyaret etmek istiyorum",
            "{place} görmeye değer mi?",
            "{place}'e gitmeli miyim?",
            "{place} nerede bulunuyor?",
            "{place} ile ilgili bilgi",
            "{place} hakkında ne biliyorsun?"
        ],
        'en': [
            "Where is {place}?",
            "Tell me about {place}",
            "What is {place} like?",
            "Is {place} worth visiting?",
            "I want to visit {place}",
            "Should I go to {place}?",
            "Information about {place}",
            "Where can I find {place}?",
            "What do you know about {place}?",
            "Is {place} interesting?"
        ]
    },
    'restaurant': {
        'tr': [
            "{food} nerede yenir?",
            "{food} için restoran öner",
            "İyi bir {food} restoranı",
            "{food} yemek istiyorum",
            "{food} nerede bulabilirim?",
            "En iyi {food} nerede?",
            "{food} için mekan",
            "{food} önerisi",
            "Güzel {food} restoranı",
            "{food} yiyebileceğim yer"
        ],
        'en': [
            "Where to eat {food}?",
            "Recommend a {food} restaurant",
            "Good {food} restaurant",
            "I want to eat {food}",
            "Where can I find {food}?",
            "Best {food} place",
            "Place for {food}",
            "{food} recommendation",
            "Nice {food} restaurant",
            "Where to have {food}"
        ]
    },
    'transportation': {
        'tr': [
            "{place}'e nasıl giderim?",
            "{place} için ulaşım",
            "{place}'e metro ile gidilir mi?",
            "{place}'e nasıl ulaşırım?",
            "{place} yolu",
            "{place} için yol tarifi",
            "{place}'e otobüs var mı?",
            "{place}'e taksi ücretli",
            "{place} uzakta mı?",
            "{place}'e yürüyerek gidilir mi?"
        ],
        'en': [
            "How to get to {place}?",
            "Transport to {place}",
            "Can I take metro to {place}?",
            "How to reach {place}?",
            "Route to {place}",
            "Directions to {place}",
            "Is there a bus to {place}?",
            "Taxi to {place}",
            "Is {place} far?",
            "Can I walk to {place}?"
        ]
    },
    'weather': {
        'tr': [
            "{time} hava nasıl?",
            "{time} yağmur yağar mı?",
            "{time} hava durumu",
            "{time} sıcak olur mu?",
            "{time} için hava tahmini",
            "{time} güneşli mi olacak?",
            "{time} soğuk mu?",
            "{time} nasıl bir hava?",
            "{time} hava güzel mi?",
            "{time} için ne giysem?"
        ],
        'en': [
            "How's the weather {time}?",
            "Will it rain {time}?",
            "Weather forecast {time}",
            "Will it be hot {time}?",
            "Weather prediction {time}",
            "Will it be sunny {time}?",
            "Is it cold {time}?",
            "What's the weather like {time}?",
            "Is the weather nice {time}?",
            "What should I wear {time}?"
        ]
    },
    'neighborhood': {
        'tr': [
            "{area} nasıl bir semt?",
            "{area} hakkında bilgi",
            "{area} güvenli mi?",
            "{area}'da ne var?",
            "{area} gezilir mi?",
            "{area}'da kalmalı mıyım?",
            "{area} hangi semtte?",
            "{area} nerede?",
            "{area} canlı mı?",
            "{area} turist için uygun mu?"
        ],
        'en': [
            "What is {area} like?",
            "Information about {area}",
            "Is {area} safe?",
            "What's in {area}?",
            "Should I visit {area}?",
            "Should I stay in {area}?",
            "Where is {area}?",
            "Tell me about {area}",
            "Is {area} lively?",
            "Is {area} good for tourists?"
        ]
    },
    'events': {
        'tr': [
            "{time} etkinlik var mı?",
            "{time} ne oluyor?",
            "{time} konser var mı?",
            "{time} için program",
            "{time} festival var mı?",
            "{time} ne yapılır?",
            "{time} gidilebilecek etkinlik",
            "{time} için öneriler",
            "{time} ne izlenebilir?",
            "{time} açık etkinlik"
        ],
        'en': [
            "Any events {time}?",
            "What's happening {time}?",
            "Concerts {time}?",
            "Program for {time}",
            "Any festivals {time}?",
            "What to do {time}?",
            "Events to attend {time}",
            "Recommendations for {time}",
            "What to watch {time}?",
            "Outdoor events {time}"
        ]
    }
}

# Sample entities for template filling
SAMPLE_ENTITIES = {
    'place': {
        'tr': ['Ayasofya', 'Topkapı Sarayı', 'Kapalıçarşı', 'Sultanahmet', 'Galata Kulesi',
               'Dolmabahçe Sarayı', 'Yerebatan Sarnıcı', 'Taksim', 'Beyoğlu', 'Kadıköy',
               'Ortaköy', 'Bebek', 'Beşiktaş', 'Eminönü', 'Üsküdar'],
        'en': ['Hagia Sophia', 'Topkapi Palace', 'Grand Bazaar', 'Sultanahmet', 'Galata Tower',
               'Dolmabahce Palace', 'Basilica Cistern', 'Taksim', 'Beyoglu', 'Kadikoy',
               'Ortakoy', 'Bebek', 'Besiktas', 'Eminonu', 'Uskudar']
    },
    'food': {
        'tr': ['kebap', 'baklava', 'meze', 'balık', 'İskender', 'döner', 'mantı',
               'köfte', 'lahmacun', 'pide', 'börek', 'çorba', 'kumpir', 'simit'],
        'en': ['kebab', 'baklava', 'meze', 'fish', 'Iskender', 'doner', 'manti',
               'kofte', 'lahmacun', 'pide', 'borek', 'soup', 'kumpir', 'simit']
    },
    'area': {
        'tr': ['Beyoğlu', 'Kadıköy', 'Beşiktaş', 'Şişli', 'Nişantaşı', 'Ortaköy',
               'Sultanahmet', 'Taksim', 'Üsküdar', 'Fatih', 'Bebek', 'Arnavutköy'],
        'en': ['Beyoglu', 'Kadikoy', 'Besiktas', 'Sisli', 'Nisantasi', 'Ortakoy',
               'Sultanahmet', 'Taksim', 'Uskudar', 'Fatih', 'Bebek', 'Arnavutkoy']
    },
    'time': {
        'tr': ['bugün', 'yarın', 'bu hafta', 'hafta sonu', 'bu akşam', 'yarın akşam',
               'pazartesi', 'cumartesi', 'pazar', 'bu ay', 'gelecek hafta'],
        'en': ['today', 'tomorrow', 'this week', 'weekend', 'tonight', 'tomorrow night',
               'monday', 'saturday', 'sunday', 'this month', 'next week']
    }
}


def load_data(file_path):
    """Load training data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'training_data' in data:
        return data['training_data']
    return data


def detect_language(text):
    """Detect language"""
    turkish_chars = set('çğıöşüÇĞİÖŞÜ')
    return 'tr' if any(char in turkish_chars for char in text) else 'en'


def augment_with_templates(data, target_per_intent=80):
    """Augment data using templates"""
    augmented = list(data)  # Start with original data
    intent_groups = defaultdict(list)
    
    # Group by intent
    for item in data:
        intent_groups[item['intent']].append(item)
    
    print(f"\n🔄 Augmenting intents with less than {target_per_intent} examples...")
    
    # For each intent category that has templates
    for intent_category, templates_dict in AUGMENTATION_TEMPLATES.items():
        # Find all intents that match this category
        matching_intents = [intent for intent in intent_groups.keys() 
                          if intent_category in intent or intent in intent_category]
        
        for intent in matching_intents:
            current_count = len(intent_groups[intent])
            
            if current_count >= target_per_intent:
                continue
            
            needed = target_per_intent - current_count
            print(f"   {intent}: {current_count} → {target_per_intent} (+{needed})")
            
            # Generate new examples using templates
            added = 0
            attempts = 0
            max_attempts = needed * 5
            
            while added < needed and attempts < max_attempts:
                attempts += 1
                
                # Pick a language
                lang = random.choice(['tr', 'en'])
                
                # Get templates for this category and language
                if lang in templates_dict:
                    template = random.choice(templates_dict[lang])
                    
                    # Find required entities in template
                    if '{place}' in template:
                        entity = random.choice(SAMPLE_ENTITIES['place'][lang])
                        text = template.format(place=entity)
                    elif '{food}' in template:
                        entity = random.choice(SAMPLE_ENTITIES['food'][lang])
                        text = template.format(food=entity)
                    elif '{area}' in template:
                        entity = random.choice(SAMPLE_ENTITIES['area'][lang])
                        text = template.format(area=entity)
                    elif '{time}' in template:
                        entity = random.choice(SAMPLE_ENTITIES['time'][lang])
                        text = template.format(time=entity)
                    else:
                        text = template
                    
                    # Check if this text already exists
                    if not any(d['text'].lower() == text.lower() for d in augmented):
                        augmented.append({
                            'text': text,
                            'intent': intent,
                            'language': lang,
                            'augmented': True
                        })
                        added += 1
    
    # For other intents without templates, do simple variations
    for intent, examples in intent_groups.items():
        current_count = len([d for d in augmented if d['intent'] == intent])
        
        if current_count < target_per_intent:
            needed = target_per_intent - current_count
            print(f"   {intent}: {current_count} → {target_per_intent} (+{needed}) [simple variations]")
            
            # Create variations by slightly modifying existing examples
            added = 0
            while added < needed and examples:
                base = random.choice(examples)
                text = base['text']
                lang = base.get('language', detect_language(text))
                
                # Simple variations
                variations = []
                if lang == 'tr':
                    variations = [
                        f"{text}?",
                        f"Bana {text}",
                        f"{text} lütfen",
                        f"{text} hakkında",
                        text.lower(),
                        text.capitalize()
                    ]
                else:
                    variations = [
                        f"{text}?",
                        f"Tell me {text}",
                        f"{text} please",
                        f"About {text}",
                        text.lower(),
                        text.capitalize()
                    ]
                
                for var in variations:
                    if added >= needed:
                        break
                    if not any(d['text'].lower() == var.lower() for d in augmented):
                        augmented.append({
                            'text': var,
                            'intent': intent,
                            'language': lang,
                            'augmented': True
                        })
                        added += 1
    
    return augmented


def main():
    input_file = 'expanded_intent_training_data.json'
    output_file = 'augmented_intent_training_data.json'
    target_per_intent = 80
    
    print("\n" + "="*80)
    print("🚀 AGGRESSIVE DATA AUGMENTATION")
    print("="*80)
    
    # Load data
    print(f"\n📥 Loading data from: {input_file}")
    data = load_data(input_file)
    print(f"   Loaded {len(data)} examples")
    
    # Augment
    augmented = augment_with_templates(data, target_per_intent)
    
    # Analyze
    intent_counts = Counter(item['intent'] for item in augmented)
    augmented_count = sum(1 for item in augmented if item.get('augmented', False))
    
    print(f"\n📊 Results:")
    print(f"   Original: {len(data)} examples")
    print(f"   Augmented: {len(augmented)} examples")
    print(f"   Added: {len(augmented) - len(data)} new examples")
    print(f"   Intents: {len(intent_counts)}")
    
    print(f"\n📈 Intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * min(40, count // 2)
        print(f"   {intent:25s} {count:3d} {bar}")
    
    # Save
    output_data = {
        'training_data': augmented,
        'metadata': {
            'total_examples': len(augmented),
            'num_intents': len(intent_counts),
            'intents': sorted(list(set(item['intent'] for item in augmented))),
            'augmentation_target': target_per_intent,
            'augmented_examples': augmented_count
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print("\n" + "="*80)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*80)
    print(f"\n💡 Next step: Train model with:")
    print(f"   python3 train_distilbert_intent_classifier.py \\")
    print(f"     --data-file {output_file} \\")
    print(f"     --epochs 20 \\")
    print(f"     --batch-size 16")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
