#!/usr/bin/env python3
"""
Enhanced Input Processor for AI Istanbul Chatbot
Handles fuzzy matching, typo correction, context understanding, and query enhancement
"""

import re
from typing import Dict, List, Tuple, Optional
from thefuzz import fuzz, process

class EnhancedInputProcessor:
    def __init__(self):
        # Istanbul locations and landmarks with variants
        self.locations = {
            'sultanahmet': ['sultanahmet', 'sultanamet', 'sultanhmet', 'sultan ahmet'],
            'beyoglu': ['beyoglu', 'beyoğlu', 'beyoglu', 'beyogul', 'beyoğul'],
            'kadikoy': ['kadikoy', 'kadıköy', 'kadikoy', 'kadiköy', 'kadikoy'],
            'taksim': ['taksim', 'takism', 'taksim square'],
            'galata': ['galata', 'galata tower', 'galata köprüsü', 'galata bridge'],
            'uskudar': ['uskudar', 'üsküdar', 'uskudar', 'üskudar'],
            'besiktas': ['besiktas', 'beşiktaş', 'besiktas', 'beşiktaş'],
            'sisli': ['sisli', 'şişli', 'sisli', 'şişli'],
            'fatih': ['fatih'],
            'eminonu': ['eminonu', 'eminönü', 'eminonu'],
            'karakoy': ['karakoy', 'karaköy', 'karakoy'],
            'ortakoy': ['ortakoy', 'ortaköy', 'ortakoy']
        }
        
        # Landmarks and attractions
        self.landmarks = {
            'hagia sophia': ['hagia sophia', 'hagya sofya', 'aya sofya', 'ayasofya', 'agia sophia'],
            'topkapi palace': ['topkapi palace', 'topkapı palace', 'topkapi palase', 'topkapı sarayı'],
            'blue mosque': ['blue mosque', 'sultanahmet mosque', 'sultanahmet camii'],
            'galata tower': ['galata tower', 'galata towar', 'galata kulesi'],
            'grand bazaar': ['grand bazaar', 'grand bazar', 'kapalıçarşı', 'kapali carsi'],
            'basilica cistern': ['basilica cistern', 'bazilika sistern', 'yerebatan sarnici']
        }
        
        # Query types and their indicators
        self.query_types = {
            'restaurant': ['restaurant', 'food', 'eat', 'dining', 'lunch', 'dinner', 'breakfast', 'cafe', 'resturant', 'restrnt'],
            'transportation': ['how to go', 'transport', 'metro', 'bus', 'ferry', 'taxi', 'get to', 'route', 'travel'],
            'museum': ['museum', 'palace', 'gallery', 'exhibition', 'history', 'culture', 'musem', 'musems'],
            'general': ['hello', 'hi', 'help', 'about', 'what can you do', 'weather', 'time']
        }
        
        # Response guidance for different query types
        self.response_guidance = {
            'restaurant_near_landmark': "Focus on restaurants and dining options near {landmark}. Include location-specific restaurant recommendations.",
            'food_at_attraction': "Provide restaurant and food options near {attraction}. Include nearby cafes and dining establishments.",
            'transport_route': "Provide detailed transportation options from {origin} to {destination}. Include metro, bus, ferry, and taxi options with approximate times and costs.",
            'museum_info': "Provide information about museums, opening hours, tickets, and cultural attractions.",
            'capability_inquiry': "Explain chatbot capabilities: helping with restaurants, transportation, museums, attractions, and general Istanbul travel advice."
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra spaces
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Handle common character variations
        text = text.replace('ğ', 'g').replace('ü', 'u').replace('ş', 's')
        text = text.replace('ı', 'i').replace('ö', 'o').replace('ç', 'c')
        
        return text
    
    def detect_locations(self, text: str) -> List[str]:
        """Detect locations mentioned in the text"""
        normalized_text = self.normalize_text(text)
        found_locations = []
        
        for location, variants in self.locations.items():
            for variant in variants:
                if variant.lower() in normalized_text:
                    found_locations.append(location)
                    break
                # Fuzzy matching for typos
                if fuzz.ratio(variant.lower(), normalized_text) > 80:
                    found_locations.append(location)
                    break
        
        return list(set(found_locations))
    
    def detect_landmarks(self, text: str) -> List[str]:
        """Detect landmarks mentioned in the text"""
        normalized_text = self.normalize_text(text)
        found_landmarks = []
        
        for landmark, variants in self.landmarks.items():
            for variant in variants:
                if variant.lower() in normalized_text:
                    found_landmarks.append(landmark)
                    break
                # Fuzzy matching for typos
                if fuzz.ratio(variant.lower(), normalized_text) > 75:
                    found_landmarks.append(landmark)
                    break
        
        return list(set(found_landmarks))
    
    def detect_query_type(self, text: str) -> str:
        """Detect the main type of query"""
        normalized_text = self.normalize_text(text)
        
        type_scores = {}
        for query_type, indicators in self.query_types.items():
            score = 0
            for indicator in indicators:
                if indicator in normalized_text:
                    score += 1
                # Fuzzy matching for typos
                words = normalized_text.split()
                for word in words:
                    if fuzz.ratio(indicator, word) > 85:
                        score += 0.5
            type_scores[query_type] = score
        
        if max(type_scores.values()) > 0:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        
        return 'general'
    
    def fix_typos(self, text: str) -> str:
        """Fix common typos using fuzzy matching"""
        words = text.split()
        corrected_words = []
        
        # Create a comprehensive word list for correction
        all_words = []
        for variants in list(self.locations.values()) + list(self.landmarks.values()):
            all_words.extend(variants)
        for indicators in self.query_types.values():
            all_words.extend(indicators)
        
        for word in words:
            # Skip very short words
            if len(word) <= 2:
                corrected_words.append(word)
                continue
            
            # Find best match
            best_match = process.extractOne(word.lower(), all_words, scorer=fuzz.ratio)
            if best_match and best_match[1] > 85:  # High confidence threshold
                corrected_words.append(best_match[0])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def enhance_query_context(self, text: str) -> Dict:
        """Enhance query with context understanding"""
        # Fix typos first
        corrected_text = self.fix_typos(text)
        
        # Detect components
        locations = self.detect_locations(corrected_text)
        landmarks = self.detect_landmarks(corrected_text)
        query_type = self.detect_query_type(corrected_text)
        
        # Context analysis
        context = {
            'original_query': text,
            'corrected_query': corrected_text,
            'detected_locations': locations,
            'detected_landmarks': landmarks,
            'query_type': query_type,
            'enhancement_applied': corrected_text != text.lower()
        }
        
        # Generate enhanced query
        enhanced_query = corrected_text
        
        # Add context-specific enhancements
        if query_type == 'restaurant' and landmarks:
            context['specific_guidance'] = self.response_guidance['restaurant_near_landmark'].format(
                landmark=landmarks[0]
            )
            enhanced_query = f"restaurants near {landmarks[0]} in istanbul"
            
        elif 'food' in corrected_text and (landmarks or 'palace' in corrected_text):
            context['specific_guidance'] = self.response_guidance['food_at_attraction'].format(
                attraction=landmarks[0] if landmarks else 'attraction'
            )
            enhanced_query = f"restaurants and food options near {landmarks[0] if landmarks else 'attraction'}"
            
        elif query_type == 'transportation' and locations:
            if len(locations) >= 2:
                context['specific_guidance'] = self.response_guidance['transport_route'].format(
                    origin=locations[0], destination=locations[1]
                )
                enhanced_query = f"transportation from {locations[0]} to {locations[1]}"
            elif landmarks:
                context['specific_guidance'] = self.response_guidance['transport_route'].format(
                    origin='current location', destination=landmarks[0]
                )
                enhanced_query = f"how to get to {landmarks[0]}"
                
        elif 'what can you do' in corrected_text or 'capabilities' in corrected_text:
            context['specific_guidance'] = self.response_guidance['capability_inquiry']
            enhanced_query = "chatbot capabilities and features for istanbul travel"
        
        context['enhanced_query'] = enhanced_query
        
        return context
    
    def validate_response_relevance(self, query: str, response: str, expected_type: str) -> Dict:
        """Validate if response is relevant to the query type"""
        issues = []
        
        normalized_response = self.normalize_text(response)
        
        # Check for correct content type
        if expected_type == 'restaurant':
            if 'restaurant' not in normalized_response and 'food' not in normalized_response:
                issues.append("Response should focus on restaurants and food")
            if 'transport' in normalized_response and len(normalized_response.split()) > 50:
                issues.append("Response contains irrelevant transportation information")
                
        elif expected_type == 'transportation':
            transport_keywords = ['metro', 'bus', 'ferry', 'taxi', 'transport', 'route']
            if not any(keyword in normalized_response for keyword in transport_keywords):
                issues.append("Response should focus on transportation options")
                
        elif expected_type == 'museum':
            if 'museum' not in normalized_response and 'palace' not in normalized_response:
                issues.append("Response should focus on museums and cultural sites")
        
        # Check for minimum quality
        if len(response) < 50:
            issues.append("Response too short for meaningful assistance")
        
        return {
            'is_relevant': len(issues) == 0,
            'issues': issues,
            'relevance_score': max(0, 100 - len(issues) * 25)
        }

# Global instance for use in main application
input_processor = EnhancedInputProcessor()

def enhance_query_understanding(user_input: str) -> str:
    """Main function to enhance user query understanding"""
    if not user_input or len(user_input.strip()) == 0:
        return user_input
    
    try:
        enhancement_result = input_processor.enhance_query_context(user_input)
        
        # Return enhanced query if significant improvement was made
        if enhancement_result.get('enhancement_applied') or enhancement_result.get('specific_guidance'):
            return enhancement_result['enhanced_query']
        
        return enhancement_result['corrected_query']
    
    except Exception as e:
        # Fallback to original input if enhancement fails
        print(f"Enhancement error: {e}")
        return user_input

def get_response_guidance(user_input: str) -> Optional[str]:
    """Get specific guidance for response generation"""
    try:
        enhancement_result = input_processor.enhance_query_context(user_input)
        return enhancement_result.get('specific_guidance')
    except:
        return None

if __name__ == "__main__":
    # Test the enhancement system
    test_queries = [
        "restaurants near hagia sophia",
        "food at topkapi palace", 
        "how can I go kadikoy from beyoglu",
        "transport museum istanbul",
        "what can you do",
        "weather in istanbul",
        "how to go galata towar",
        "resturants in galata",
        "musems in istanbul"
    ]
    
    print("🧠 Testing Enhanced Input Processor")
    print("=" * 50)
    
    for query in test_queries:
        result = input_processor.enhance_query_context(query)
        print(f"\nQuery: '{query}'")
        print(f"Enhanced: '{result['enhanced_query']}'")
        print(f"Type: {result['query_type']}")
        print(f"Locations: {result['detected_locations']}")
        print(f"Landmarks: {result['detected_landmarks']}")
        if result.get('specific_guidance'):
            print(f"Guidance: {result['specific_guidance'][:100]}...")
