"""
Comprehensive Training Data Augmentation for AI Istanbul
Expands the training dataset from 194 to 800+ examples with variations
"""

import json
from pathlib import Path
import random
from typing import List, Dict
import re

# Set seed for reproducibility
random.seed(42)


class IstanbulDataAugmentor:
    """Augment training data with multiple strategies"""
    
    def __init__(self):
        self.turkish_english_map = {
            # Common words
            "Best": ["En iyi", "Best", "Top", "İyi"],
            "Good": ["İyi", "Good", "Güzel"],
            "restaurants": ["restoranlar", "restaurants", "lokantalar", "restoranları"],
            "restaurant": ["restoran", "restaurant", "lokanta"],
            "Where": ["Nerede", "Where", "Neresi", "Nere"],
            "How": ["Nasıl", "How", "Ne şekilde"],
            "What": ["Ne", "What", "Nedir", "Hangi"],
            "Tell me": ["Söyle", "Tell me", "Anlat", "Bana söyle"],
            "show me": ["göster", "show me", "göster bana"],
            "I need": ["ihtiyacım var", "I need", "lazım"],
            "I want": ["istiyorum", "I want", "isterim"],
            "Can you": ["yapabilir misin", "Can you", "yapar mısın"],
            "find": ["bul", "find", "bulabilir misin"],
            "search": ["ara", "search", "arama"],
            "looking for": ["arıyorum", "looking for", "aramaktayım"],
            "help": ["yardım", "help", "yardım et"],
            "please": ["lütfen", "please", "rica etsem"],
            "cheap": ["ucuz", "cheap", "ekonomik"],
            "expensive": ["pahalı", "expensive", "lüks"],
            "near": ["yakın", "near", "yakınında"],
            "in": ["da", "in", "içinde"],
            "to": ["e", "to", "ye"],
            "from": ["dan", "from", "den"],
        }
        
        self.question_templates = {
            'restaurant_search': [
                "Best {} restaurants in {}",
                "Where can I find {} restaurants in {}",
                "I'm looking for {} restaurants near {}",
                "Show me {} restaurants in {}",
                "Recommend {} restaurants in {}",
                "{} restaurants in {} area",
                "Good {} places to eat in {}",
                "Top {} dining options in {}",
                "Any {} restaurants around {}",
                "Looking for {} food in {}",
            ],
            'attraction_search': [
                "{} in Istanbul",
                "Where are the {} in Istanbul",
                "Show me {} in {}",
                "I want to visit {} in {}",
                "What {} should I see in {}",
                "Top {} in {}",
                "Best {} to visit in {}",
                "Looking for {} in {}",
                "Any good {} in {}",
                "Must-see {} in {}",
            ],
            'transport_route': [
                "How to get from {} to {}",
                "Route from {} to {}",
                "Best way from {} to {}",
                "How do I go from {} to {}",
                "Transportation from {} to {}",
                "Getting from {} to {}",
                "Travel from {} to {}",
                "Navigate from {} to {}",
                "Directions from {} to {}",
                "Journey from {} to {}",
            ],
            'weather_query': [
                "What's the weather like {}",
                "Weather forecast for {}",
                "How's the weather {}",
                "Weather in Istanbul {}",
                "Is it {} in Istanbul",
                "Will it {} in Istanbul",
                "Temperature {}",
                "Climate {}",
                "Weather conditions {}",
                "Forecast for {}",
            ],
            'event_search': [
                "Events {} in Istanbul",
                "What's happening {}",
                "{} events in Istanbul",
                "Concerts {}",
                "Festivals {}",
                "Shows {}",
                "Activities {}",
                "Things to do {}",
                "Entertainment {}",
                "Happenings {}",
            ],
            'daily_greeting': [
                "Hello",
                "Hi",
                "Hey",
                "Good morning",
                "Good afternoon",
                "Good evening",
                "Merhaba",
                "Selam",
                "Greetings",
                "Hi there",
            ],
            'daily_help': [
                "I need help",
                "Can you help me",
                "Help me with Istanbul",
                "I'm planning a trip",
                "First time in Istanbul",
                "Visiting Istanbul",
                "Tourist information",
                "Travel advice",
                "Istanbul tips",
                "Guide me",
            ],
        }
        
        # Istanbul locations for templates
        self.locations = [
            "Sultanahmet", "Beyoğlu", "Kadıköy", "Taksim", "Beşiktaş",
            "Üsküdar", "Ortaköy", "Bebek", "Karaköy", "Galata",
            "Fatih", "Balat", "Cihangir", "Nişantaşı", "Şişli"
        ]
        
        self.cuisines = [
            "seafood", "kebab", "Turkish", "Ottoman", "Italian",
            "Asian", "vegetarian", "breakfast", "street food", "dessert"
        ]
        
        self.place_types = [
            "museums", "mosques", "palaces", "parks", "monuments",
            "bazaars", "galleries", "landmarks", "historical sites"
        ]
        
        self.time_expressions = [
            "today", "tomorrow", "this weekend", "next week",
            "this month", "tonight", "this evening", "now"
        ]
    
    def augment_with_templates(self, data: List[Dict]) -> List[Dict]:
        """Generate new examples using templates"""
        augmented = []
        
        for intent, templates in self.question_templates.items():
            # Find examples with this intent
            intent_examples = [ex for ex in data if ex['intent'] == intent]
            
            if intent == 'restaurant_search':
                for cuisine in self.cuisines:
                    for location in self.locations[:10]:  # Use 10 locations
                        for template in templates[:5]:  # Use 5 templates
                            text = template.format(cuisine, location)
                            augmented.append({'text': text, 'intent': intent})
            
            elif intent == 'attraction_search':
                for place_type in self.place_types:
                    for location in self.locations[:8]:
                        for template in templates[:4]:
                            text = template.format(place_type, location)
                            augmented.append({'text': text, 'intent': intent})
            
            elif intent == 'transport_route':
                # Generate routes between locations
                for i, loc1 in enumerate(self.locations[:8]):
                    for loc2 in self.locations[i+1:8]:
                        for template in templates[:3]:
                            text = template.format(loc1, loc2)
                            augmented.append({'text': text, 'intent': intent})
            
            elif intent in ['weather_query', 'event_search']:
                for time_expr in self.time_expressions:
                    for template in templates[:5]:
                        text = template.format(time_expr)
                        augmented.append({'text': text, 'intent': intent})
            
            elif intent in ['daily_greeting', 'daily_help']:
                for template in templates:
                    augmented.append({'text': template, 'intent': intent})
        
        return augmented
    
    def augment_with_paraphrasing(self, data: List[Dict]) -> List[Dict]:
        """Create paraphrased versions of existing examples"""
        augmented = []
        
        paraphrase_patterns = [
            # Pattern 1: Add "I'm looking for"
            (r"^(Best|Good|Top)", r"I'm looking for \1"),
            # Pattern 2: Add "Can you show me"
            (r"^(.*restaurants.*)", r"Can you show me \1"),
            # Pattern 3: Add "Where can I find"
            (r"^(.*in Istanbul.*)", r"Where can I find \1"),
            # Pattern 4: Question form
            (r"^(Museums|Mosques|Palaces)", r"What \1 should I visit"),
        ]
        
        for example in data:
            text = example['text']
            intent = example['intent']
            
            for pattern, replacement in paraphrase_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    if new_text != text:  # Only add if different
                        augmented.append({'text': new_text, 'intent': intent})
        
        return augmented
    
    def augment_with_translations(self, data: List[Dict]) -> List[Dict]:
        """Add Turkish/English variations"""
        augmented = []
        
        for example in data[:100]:  # Augment first 100 examples
            text = example['text']
            intent = example['intent']
            
            # Try to translate key words
            for eng, tur_list in self.turkish_english_map.items():
                if eng in text:
                    for tur in tur_list[1:3]:  # Use 2 variations
                        new_text = text.replace(eng, tur, 1)  # Replace first occurrence
                        if new_text != text:
                            augmented.append({'text': new_text, 'intent': intent})
        
        return augmented
    
    def augment_with_typos(self, data: List[Dict], num_samples: int = 50) -> List[Dict]:
        """Add common typo variations"""
        augmented = []
        
        common_typos = {
            'restaurant': ['resturant', 'restraunt', 'restarant'],
            'museums': ['museams', 'musems', 'musuems'],
            'Istanbul': ['istambul', 'istanbu', 'istanbull'],
            'weather': ['wheather', 'wether', 'wheater'],
            'Sultanahmet': ['sultanhamet', 'sultanamet', 'sultanahmed'],
            'Beyoğlu': ['beyoglu', 'beyogul', 'beyoglue'],
        }
        
        sample = random.sample(data, min(num_samples, len(data)))
        
        for example in sample:
            text = example['text']
            intent = example['intent']
            
            for correct, typos in common_typos.items():
                if correct in text:
                    for typo in typos[:1]:  # Use 1 typo variation
                        new_text = text.replace(correct, typo, 1)
                        augmented.append({'text': new_text, 'intent': intent})
        
        return augmented
    
    def balance_dataset(self, data: List[Dict], target_per_intent: int = 50) -> List[Dict]:
        """Balance the dataset by oversampling minority classes"""
        from collections import Counter
        
        intent_counts = Counter(ex['intent'] for ex in data)
        print(f"\n📊 Original intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {intent:30s}: {count:3d}")
        
        balanced = data.copy()
        
        for intent, count in intent_counts.items():
            if count < target_per_intent:
                # Oversample this intent
                intent_examples = [ex for ex in data if ex['intent'] == intent]
                needed = target_per_intent - count
                
                # Randomly sample with replacement
                oversampled = random.choices(intent_examples, k=needed)
                balanced.extend(oversampled)
        
        # Shuffle
        random.shuffle(balanced)
        
        final_counts = Counter(ex['intent'] for ex in balanced)
        print(f"\n📊 Balanced intent distribution:")
        for intent, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {intent:30s}: {count:3d}")
        
        return balanced
    
    def run_augmentation(
        self,
        input_file: str,
        output_file: str,
        target_total: int = 800,
        balance: bool = True
    ) -> Path:
        """Run full augmentation pipeline"""
        
        print("\n" + "="*70)
        print("🚀 COMPREHENSIVE DATA AUGMENTATION")
        print("="*70)
        
        # Load original data
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        print(f"\n📊 Original data: {len(original_data)} examples")
        
        augmented_data = original_data.copy()
        
        # Step 1: Template-based generation
        print(f"\n1️⃣  Generating template-based examples...")
        template_examples = self.augment_with_templates(original_data)
        print(f"   Generated: {len(template_examples)} examples")
        augmented_data.extend(template_examples)
        
        # Step 2: Paraphrasing
        print(f"\n2️⃣  Creating paraphrased versions...")
        paraphrased = self.augment_with_paraphrasing(original_data)
        print(f"   Generated: {len(paraphrased)} examples")
        augmented_data.extend(paraphrased)
        
        # Step 3: Translations
        print(f"\n3️⃣  Adding Turkish/English variations...")
        translated = self.augment_with_translations(original_data)
        print(f"   Generated: {len(translated)} examples")
        augmented_data.extend(translated)
        
        # Step 4: Typos
        print(f"\n4️⃣  Adding common typo variations...")
        typos = self.augment_with_typos(original_data, num_samples=100)
        print(f"   Generated: {len(typos)} examples")
        augmented_data.extend(typos)
        
        # Remove duplicates
        seen = set()
        unique_data = []
        for ex in augmented_data:
            text_lower = ex['text'].lower().strip()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_data.append(ex)
        
        print(f"\n📊 After removing duplicates: {len(unique_data)} examples")
        
        # Step 5: Balance dataset
        if balance:
            print(f"\n5️⃣  Balancing dataset...")
            unique_data = self.balance_dataset(unique_data, target_per_intent=50)
        
        print(f"\n📊 Final dataset: {len(unique_data)} examples")
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Augmented data saved to: {output_path}")
        
        # Summary
        print("\n" + "="*70)
        print("✅ AUGMENTATION COMPLETE")
        print("="*70)
        print(f"📊 Original: {len(original_data)} examples")
        print(f"📊 Final: {len(unique_data)} examples")
        print(f"📈 Growth: {len(unique_data) - len(original_data)} examples ({(len(unique_data)/len(original_data) - 1)*100:.1f}%)")
        print("="*70 + "\n")
        
        return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment training data")
    parser.add_argument("--input", type=str, default="data/intent_training_data.json",
                       help="Input training data file")
    parser.add_argument("--output", type=str, default="data/intent_training_data_augmented.json",
                       help="Output augmented data file")
    parser.add_argument("--target", type=int, default=800,
                       help="Target number of examples")
    parser.add_argument("--no-balance", action="store_true",
                       help="Don't balance the dataset")
    
    args = parser.parse_args()
    
    augmentor = IstanbulDataAugmentor()
    augmentor.run_augmentation(
        input_file=args.input,
        output_file=args.output,
        target_total=args.target,
        balance=not args.no_balance
    )
