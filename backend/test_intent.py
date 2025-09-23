#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_intelligence import intent_recognizer

def test_turkish_query():
    query = "en iyi restoranlar beyoğlunda"
    print(f"Testing query: '{query}'")
    
    detected_intent, confidence = intent_recognizer.recognize_intent(query)
    print(f"Detected intent: {detected_intent}")
    print(f"Confidence: {confidence:.2f}")
    
    entities = intent_recognizer.extract_entities(query)
    print(f"Extracted entities: {entities}")

if __name__ == "__main__":
    test_turkish_query()
