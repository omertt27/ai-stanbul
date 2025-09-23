#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_intelligence import intent_recognizer

def test_entities():
    query = "en iyi restoranlar beyoğlunda"
    print(f"Testing query: '{query}'")
    
    entities = intent_recognizer.extract_entities(query)
    print(f"Extracted entities: {entities}")
    
    locations = entities.get('locations', [])
    print(f"Locations: {locations}")
    print(f"Number of locations: {len(locations)}")
    
    stripped_query = query.lower().strip()
    print(f"Stripped query: '{stripped_query}'")
    print(f"Query in locations: {stripped_query in locations}")
    
    is_single_district_query = len(locations) == 1 and stripped_query in locations
    print(f"is_single_district_query: {is_single_district_query}")

if __name__ == "__main__":
    test_entities()
