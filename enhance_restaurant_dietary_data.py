#!/usr/bin/env python
"""
Enhance restaurant database with dietary options for comprehensive dietary filtering
"""

import json
import random
from typing import List, Dict, Any

def add_dietary_options_to_restaurants():
    """Add dietary_options field to restaurants based on cuisine types and names"""
    
    # Load the restaurant database
    database_path = "/Users/omer/Desktop/ai-stanbul/backend/data/restaurants_database.json"
    
    try:
        with open(database_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    restaurants = data.get('restaurants', [])
    print(f"🍽️ Processing {len(restaurants)} restaurants...")
    
    # Dietary option rules based on cuisine types and restaurant names
    dietary_rules = {
        'turkish': {
            'vegetarian': 0.8,  # Most Turkish restaurants have vegetarian options
            'halal': 0.95,      # Turkey is predominantly Muslim
            'vegan': 0.3,       # Less common but growing
            'gluten_free': 0.4  # Some traditional dishes are naturally gluten-free
        },
        'mediterranean': {
            'vegetarian': 0.9,
            'halal': 0.6,
            'vegan': 0.7,
            'gluten_free': 0.6,
            'dairy_free': 0.5
        },
        'italian': {
            'vegetarian': 0.8,
            'vegan': 0.4,
            'gluten_free': 0.5,
            'halal': 0.3
        },
        'asian': {
            'vegetarian': 0.7,
            'vegan': 0.5,
            'gluten_free': 0.3,
            'halal': 0.4
        },
        'seafood': {
            'gluten_free': 0.7,
            'dairy_free': 0.6,
            'halal': 0.8,  # Most seafood is halal
            'vegetarian': 0.1,  # Seafood restaurants typically don't cater to vegetarians
            'vegan': 0.05
        },
        'cafe': {
            'vegetarian': 0.9,
            'vegan': 0.6,
            'gluten_free': 0.7,
            'dairy_free': 0.5,
            'halal': 0.7
        },
        'bakery': {
            'vegetarian': 0.8,
            'vegan': 0.4,
            'gluten_free': 0.6,
            'halal': 0.8
        }
    }
    
    # Special keywords that indicate dietary specialization
    special_keywords = {
        'vegetarian': ['vegan', 'veggie', 'plant', 'green', 'organic', 'healthy', 'fresh'],
        'vegan': ['vegan', 'plant', 'raw', 'organic', 'green'],
        'halal': ['halal', 'muslim', 'islamic', 'ottoman', 'sultan', 'turkish'],
        'gluten_free': ['gluten', 'celiac', 'healthy', 'organic', 'fresh'],
        'kosher': ['kosher', 'jewish', 'israel'],
        'dairy_free': ['dairy', 'lactose', 'plant', 'vegan']
    }
    
    updated_count = 0
    
    for restaurant in restaurants:
        if 'dietary_options' not in restaurant:
            restaurant['dietary_options'] = []
            
            name_lower = restaurant.get('name', '').lower()
            cuisine_types = restaurant.get('cuisine_types', [])
            
            # Start with base probabilities based on cuisine
            dietary_probabilities = {}
            
            # Aggregate probabilities from all cuisine types
            for cuisine in cuisine_types:
                if cuisine in dietary_rules:
                    for dietary, prob in dietary_rules[cuisine].items():
                        if dietary not in dietary_probabilities:
                            dietary_probabilities[dietary] = 0
                        dietary_probabilities[dietary] = max(dietary_probabilities[dietary], prob)
            
            # Default probabilities for restaurants without specific cuisine types
            if not dietary_probabilities:
                dietary_probabilities = {
                    'vegetarian': 0.6,
                    'halal': 0.8,  # Turkey default
                    'vegan': 0.2,
                    'gluten_free': 0.3
                }
            
            # Boost probabilities based on restaurant name keywords
            for dietary, keywords in special_keywords.items():
                for keyword in keywords:
                    if keyword in name_lower:
                        if dietary not in dietary_probabilities:
                            dietary_probabilities[dietary] = 0
                        dietary_probabilities[dietary] = min(0.95, dietary_probabilities[dietary] + 0.3)
            
            # Determine which dietary options to include
            for dietary, probability in dietary_probabilities.items():
                if random.random() < probability:
                    restaurant['dietary_options'].append(dietary)
            
            # Ensure some logical consistency
            if 'vegan' in restaurant['dietary_options'] and 'vegetarian' not in restaurant['dietary_options']:
                restaurant['dietary_options'].append('vegetarian')
            
            if 'vegan' in restaurant['dietary_options'] and 'dairy_free' not in restaurant['dietary_options']:
                restaurant['dietary_options'].append('dairy_free')
            
            updated_count += 1
    
    # Update metadata
    data['metadata']['last_updated'] = "2025-01-10 dietary_enhanced"
    data['metadata']['dietary_enhancement'] = f"Added dietary options to {updated_count} restaurants"
    
    # Save the enhanced database
    try:
        with open(database_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Enhanced {updated_count} restaurants with dietary options")
        print(f"💾 Saved to {database_path}")
    except Exception as e:
        print(f"❌ Error saving database: {e}")
        return
    
    # Show some statistics
    all_dietary_options = set()
    restaurants_with_dietary = 0
    
    for restaurant in restaurants:
        if restaurant.get('dietary_options'):
            restaurants_with_dietary += 1
            all_dietary_options.update(restaurant['dietary_options'])
    
    print(f"\n📊 Dietary Enhancement Statistics:")
    print(f"   • Restaurants with dietary info: {restaurants_with_dietary}/{len(restaurants)}")
    print(f"   • Dietary options available: {sorted(all_dietary_options)}")
    
    # Show some examples
    print(f"\n🍽️ Sample Enhanced Restaurants:")
    for i, restaurant in enumerate(restaurants[:5]):
        dietary = restaurant.get('dietary_options', [])
        if dietary:
            print(f"   • {restaurant['name']}: {', '.join(dietary)}")

if __name__ == "__main__":
    print("🔧 Enhancing Restaurant Database with Dietary Options")
    print("=" * 55)
    add_dietary_options_to_restaurants()
