"""
Istanbul Entity Recognition
Advanced entity recognition for Istanbul-specific terms and cultural context.
"""

import re
from typing import Dict, List


class IstanbulEntityRecognizer:
    """Advanced entity recognition for Istanbul-specific terms"""
    
    def __init__(self):
        self.load_istanbul_knowledge_base()
    
    def load_istanbul_knowledge_base(self):
        """Load Istanbul-specific entities and embeddings"""
        
        # Istanbul neighborhoods with variants and local names
        self.neighborhoods = {
            'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula', 'eski şehir'],
            'beyoğlu': ['beyoğlu', 'beyoglu', 'pera', 'galata', 'taksim area'],
            'kadıköy': ['kadıköy', 'kadikoy', 'asian side', 'moda', 'bagdat street'],
            'beşiktaş': ['beşiktaş', 'besiktas', 'ortaköy', 'bebek', 'arnavutköy'],
            'üsküdar': ['üsküdar', 'uskudar', 'çamlıca', 'beylerbeyi'],
            'şişli': ['şişli', 'sisli', 'nişantaşı', 'osmanbey', 'pangaltı'],
            'fatih': ['fatih', 'fener', 'balat', 'eminönü', 'karaköy']
        }
        
        # Landmarks with cultural context
        self.landmarks = {
            'hagia_sophia': {
                'names': ['hagia sophia', 'ayasofya', 'holy wisdom'],
                'type': 'museum',
                'cultural_context': 'Byzantine and Ottoman architectural marvel',
                'nearby_food': 'traditional Ottoman cuisine'
            },
            'blue_mosque': {
                'names': ['blue mosque', 'sultan ahmed mosque', 'sultanahmet camii'],
                'type': 'mosque',
                'cultural_context': 'Active place of worship with stunning blue tiles',
                'nearby_food': 'traditional Turkish breakfast spots'
            },
            'galata_tower': {
                'names': ['galata tower', 'galata kulesi'],
                'type': 'tower',
                'cultural_context': 'Medieval Genoese tower with panoramic views',
                'nearby_food': 'trendy cafes and rooftop restaurants'
            },
            'topkapi_palace': {
                'names': ['topkapi palace', 'topkapı sarayı', 'topkapi'],
                'type': 'palace',
                'cultural_context': 'Former Ottoman imperial palace',
                'nearby_food': 'historic Ottoman recipes'
            },
            'grand_bazaar': {
                'names': ['grand bazaar', 'kapalıçarşı', 'covered bazaar'],
                'type': 'market',
                'cultural_context': 'Historic covered market with traditional crafts',
                'nearby_food': 'traditional Turkish delights and tea'
            },
            'bosphorus': {
                'names': ['bosphorus', 'bosphorus strait', 'boğaz', 'istanbul strait'],
                'type': 'waterway',
                'cultural_context': 'Strategic waterway connecting Europe and Asia',
                'nearby_food': 'fresh seafood restaurants'
            }
        }
        
        # Cuisine types with local nuances
        self.cuisine_entities = {
            'turkish_traditional': ['turkish', 'ottoman', 'traditional', 'lokanta', 'ev yemeği'],
            'street_food': ['street food', 'sokak lezzetleri', 'döner', 'simit', 'balık ekmek'],
            'meze_culture': ['meze', 'meyhane', 'rakı', 'small plates', 'tapas style'],
            'breakfast_culture': ['kahvaltı', 'turkish breakfast', 'serpme kahvaltı', 'village breakfast'],
            'seafood': ['seafood', 'balık', 'fish', 'marine', 'bosphorus fish'],
            'desserts': ['baklava', 'turkish delight', 'lokum', 'künefe', 'sutlaç']
        }
        
        # Time expressions with cultural context
        self.time_entities = {
            'meal_times': {
                'turkish_breakfast': '08:00-12:00',
                'lunch': '12:00-15:00',
                'afternoon_tea': '15:00-17:00',
                'dinner': '19:00-23:00',
                'late_night': '23:00-02:00'
            },
            'cultural_times': {
                'friday_prayer': 'avoid 12:00-14:00 near mosques',
                'ramadan_iftar': 'special evening hours during Ramadan',
                'weekend_brunch': 'extended breakfast hours on weekends'
            }
        }
        
        # Transportation entities
        self.transport_entities = {
            'metro': ['metro', 'subway', 'underground', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'],
            'tram': ['tram', 'tramvay', 'T1', 'T4', 'T5'],
            'bus': ['bus', 'otobüs', 'metrobus', 'BRT'],
            'ferry': ['ferry', 'vapur', 'sea bus', 'deniz otobüsü'],
            'dolmuş': ['dolmuş', 'shared taxi', 'minibus'],
            'taxi': ['taxi', 'taksi', 'uber', 'bitaksi']
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract Istanbul-specific entities from text"""
        
        entities = {
            'neighborhoods': [],
            'landmarks': [],
            'cuisines': [],
            'time_references': [],
            'cultural_context': [],
            'transportation': []
        }
        
        text_lower = text.lower()
        
        # Neighborhood detection with fuzzy matching
        for neighborhood, variants in self.neighborhoods.items():
            if any(variant in text_lower for variant in variants):
                entities['neighborhoods'].append(neighborhood)
        
        # Landmark detection with cultural context
        for landmark_id, landmark_data in self.landmarks.items():
            if any(name in text_lower for name in landmark_data['names']):
                entities['landmarks'].append(landmark_id)
                entities['cultural_context'].append(landmark_data['cultural_context'])
        
        # Cuisine detection
        for cuisine_type, keywords in self.cuisine_entities.items():
            if any(keyword in text_lower for keyword in keywords):
                entities['cuisines'].append(cuisine_type)
        
        # Transportation detection
        for transport_type, keywords in self.transport_entities.items():
            if any(keyword in text_lower for keyword in keywords):
                entities['transportation'].append(transport_type)
        
        # Time reference detection
        time_patterns = [
            r'\b(morning|sabah)\b', r'\b(afternoon|öğleden sonra)\b',
            r'\b(evening|akşam)\b', r'\b(night|gece)\b',
            r'\b(breakfast|kahvaltı)\b', r'\b(lunch|öğle)\b', r'\b(dinner|akşam yemeği)\b'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                entities['time_references'].append(re.search(pattern, text_lower).group())
        
        return entities
