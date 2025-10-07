#!/usr/bin/env python3
"""
Restaurant Response Formatter
Enhanced with ultra-specialized local guide templates for natural Istanbul travel responses
"""

import re
from typing import Optional

def format_restaurant_response(response: str, user_query: str) -> str:
    """
    Format restaurant responses using enhanced templates for natural local guide style
    
    Args:
        response: The AI-generated response
        user_query: The original user query
        
    Returns:
        Enhanced formatted response with conversational local guide style
    """
    
    # Import enhanced templates
    try:
        from enhanced_response_templates import istanbul_templates, apply_enhanced_formatting
        
        # Check if this is a restaurant/food query
        food_keywords = [
            'restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal', 'breakfast', 
            'lunch', 'dinner', 'where to eat', 'places to eat', 'best food', 
            'good food', 'seafood', 'kebab', 'turkish cuisine', 'coffee', 'cafe',
            'vegan', 'vegetarian', 'expensive', 'cheap', 'budget', 'luxury'
        ]
        
        user_lower = user_query.lower()
        is_food_query = any(keyword in user_lower for keyword in food_keywords)
        
        if not is_food_query:
            return response
        
        # Apply enhanced formatting first
        enhanced_response = apply_enhanced_formatting(response, user_query)
        
        # If response doesn't start with "Here are", ensure proper format
        if not enhanced_response.lower().startswith('here are') or 'restaurant' not in enhanced_response.lower()[:100]:
            location = extract_location_from_query(user_query)
            restaurant_count = count_restaurants_in_response(enhanced_response)
            
            if restaurant_count > 0:
                # Use enhanced template formatting
                opening = istanbul_templates._get_restaurant_opening(location, restaurant_count, user_query)
                cleaned_response = clean_response_content(enhanced_response)
                enhanced_response = f"{opening}\n\n{cleaned_response}"
        
        return enhanced_response
        
    except ImportError:
        # Fallback to basic formatting if enhanced templates not available
        return format_basic_restaurant_response(response, user_query)

def format_basic_restaurant_response(response: str, user_query: str) -> str:
    """Fallback basic formatting when enhanced templates unavailable"""
    food_keywords = [
        'restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal', 'breakfast', 
        'lunch', 'dinner', 'where to eat', 'places to eat', 'best food', 
        'good food', 'seafood', 'kebab', 'turkish cuisine', 'coffee', 'cafe'
    ]
    
    user_lower = user_query.lower()
    is_food_query = any(keyword in user_lower for keyword in food_keywords)
    
    if not is_food_query:
        return response
    
    # If response already starts correctly, return as-is
    if response.lower().startswith('here are') and 'restaurant' in response.lower()[:100]:
        return response
    
    # Extract location from user query
    location = extract_location_from_query(user_query)
    
    # Try to find restaurant list in the response
    restaurant_count = count_restaurants_in_response(response)
    
    # Generate the correct opening
    if restaurant_count > 0:
        if location:
            correct_opening = f"Here are {restaurant_count} restaurants in {location}:"
        else:
            correct_opening = f"Here are {restaurant_count} restaurants:"
    else:
        if location:
            correct_opening = f"Here are some restaurants in {location}:"
        else:
            correct_opening = "Here are some restaurants:"
    
    # Remove any existing incorrect openings
    cleaned_response = clean_response_opening(response)
    
    # Combine correct opening with cleaned response
    formatted_response = f"{correct_opening}\n\n{cleaned_response}"
    
    return formatted_response

def extract_location_from_query(query: str) -> Optional[str]:
    """Extract location from user query."""
    
    # Common Istanbul districts
    districts = [
        'Sultanahmet', 'Beyoğlu', 'Beyoglu', 'Kadıköy', 'Kadikoy', 'Galata',
        'Beşiktaş', 'Besiktas', 'Üsküdar', 'Uskudar', 'Sarıyer', 'Sariyer',
        'Fatih', 'Ortaköy', 'Ortakoy', 'Şişli', 'Sisli', 'Bakırköy', 'Bakirkoy',
        'Levent', 'Maslak', 'Taksim', 'Eminönü', 'Eminonu'
    ]
    
    query_lower = query.lower()
    
    # Look for "in [location]" pattern
    for district in districts:
        if district.lower() in query_lower:
            return district
    
    # Look for location patterns
    location_patterns = [
        r'in\s+(\w+)',
        r'(\w+)\s+restaurants?',
        r'restaurants?\s+in\s+(\w+)'
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, query_lower)
        if match:
            potential_location = match.group(1).title()
            # Check if it's a valid Istanbul district
            for district in districts:
                if district.lower() == potential_location.lower():
                    return district
    
    return None

def count_restaurants_in_response(response: str) -> int:
    """Count the number of restaurants mentioned in the response."""
    
    # Look for numbered lists
    numbered_pattern = r'^\s*\d+\.\s+[A-Z]'
    numbered_matches = re.findall(numbered_pattern, response, re.MULTILINE)
    
    if numbered_matches:
        return len(numbered_matches)
    
    # Look for bullet points
    bullet_pattern = r'^\s*[-•]\s+[A-Z]'
    bullet_matches = re.findall(bullet_pattern, response, re.MULTILINE)
    
    if bullet_matches:
        return len(bullet_matches)
    
    # Look for restaurant names (capitalized words followed by Restaurant/Cafe/etc.)
    restaurant_pattern = r'[A-Z][a-z]+\s+(Restaurant|Cafe|Lokantası|House)'
    restaurant_matches = re.findall(restaurant_pattern, response)
    
    if restaurant_matches:
        return len(set(restaurant_matches))  # Remove duplicates
    
    return 0

def clean_response_opening(response: str) -> str:
    """Remove problematic opening phrases from response."""
    
    # Common problematic openings to remove
    problematic_openings = [
        r'^For the best food.*?:',
        r'^Certainly! When exploring.*?:',
        r'^Sure, I\'d be delighted.*?:',
        r'^[A-Z][a-z]+, located on.*?:',
        r'^Immediate Practical Answer:',
        r'^Here are some.*?options in.*?:',  # Wrong format variations
        r'^Here are.*?dinner options.*?:',
    ]
    
    cleaned = response
    
    for pattern in problematic_openings:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove leading whitespace and newlines
    cleaned = cleaned.lstrip().strip()
    
    return cleaned

def clean_response_content(response: str) -> str:
    """Clean response content removing redundant openings"""
    cleaned_response = response.strip()
    
    # Remove any existing similar openings
    patterns_to_remove = [
        r'^Here are \d+ restaurants? in \w+:?\s*',
        r'^I found \d+ restaurants? in \w+:?\s*',
        r'^Top \d+ restaurants? in \w+:?\s*',
        r'^Let me recommend \d+ restaurants? in \w+:?\s*'
    ]
    
    for pattern in patterns_to_remove:
        cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE | re.MULTILINE)
    
    return cleaned_response.strip()

# Test the formatter
if __name__ == "__main__":
    print("🧪 Testing Restaurant Response Formatter")
    print("=" * 50)
    
    test_cases = [
        {
            "query": "restaurants in Sultanahmet",
            "response": "For the best food experiences in Sultanahmet, explore these culinary delights:\n\n1. Pandeli Restaurant\n2. Asitane Restaurant",
            "expected_start": "Here are 2 restaurants in Sultanahmet:"
        },
        {
            "query": "best food in Beyoğlu", 
            "response": "Certainly! When exploring Beyoğlu, you'll find some exceptional Turkish restaurants:\n\n1. Mikla Restaurant\n2. Karakoy Lokantasi\n3. Neolokal",
            "expected_start": "Here are 3 restaurants in Beyoğlu:"
        },
        {
            "query": "seafood restaurants in Sarıyer",
            "response": "Sarıyer, located on the European side of Istanbul, offers a variety of seafood restaurants due to its proximity to the Bosphorus. Here are some seafood restaurants in Sarıyer:\n\n1. Uskumru Restaurant\n2. Yeniköy Balık Restaurant",
            "expected_start": "Here are 2 restaurants in Sarıyer:"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: '{test['query']}'")
        
        formatted = format_restaurant_response(test["response"], test["query"])
        first_line = formatted.split('\n')[0]
        
        if test["expected_start"] in first_line:
            print(f"   ✅ CORRECT: {first_line}")
        else:
            print(f"   ❌ INCORRECT: {first_line}")
            print(f"   Expected: {test['expected_start']}")
    
    print("\n🎉 Restaurant response formatter ready to integrate!")
