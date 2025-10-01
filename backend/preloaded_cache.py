#!/usr/bin/env python3
"""
Advanced Caching Optimizations
Implements additional optimizations to reach 22.5% cost reduction target
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PreloadedCache:
    """Pre-populated cache with common tourist queries"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.preloaded_responses = self._load_preloaded_responses()
        
    def _load_preloaded_responses(self) -> Dict[str, str]:
        """Load pre-computed responses for common queries"""
        return {
            # Restaurant queries
            'restaurant:turkish_restaurants:sultanahmet': """Certainly! When exploring Sultanahmet, you'll find some exceptional Turkish restaurants that offer authentic local cuisine:

**TOP TURKISH RESTAURANTS IN SULTANAHMET:**

🍽️ **Pandeli** (Historic Ottoman Restaurant)
- Location: Eminönü (5 min from Sultanahmet)  
- Specialty: Traditional Ottoman cuisine in beautiful historic setting
- Must-try: Lamb stew, traditional meze platter
- Price: $$$ | Rating: 4.7/5

🍽️ **Hamdi Restaurant**
- Location: Near Galata Bridge
- Specialty: Famous for lamb dishes and grilled meats
- Must-try: Pirzola (lamb chops), Turkish kebabs
- Price: $$ | Rating: 4.6/5

🍽️ **Sultanahmet Köftecisi**
- Location: Heart of Sultanahmet
- Specialty: Traditional Turkish meatballs (köfte)
- Must-try: İnegöl köfte with pilav
- Price: $ | Rating: 4.5/5

🍽️ **Deraliye Ottoman Cuisine**
- Location: Near Topkapi Palace
- Specialty: Authentic Ottoman palace recipes
- Must-try: Ottoman palace dishes, traditional desserts
- Price: $$$ | Rating: 4.8/5

**PRACTICAL TIPS:**
- Most restaurants open at 11:30 AM
- Dinner is typically served after 7:00 PM
- Reservations recommended for dinner
- Try Turkish breakfast at local cafes

Would you like specific recommendations for any particular type of Turkish cuisine or budget range?""",

            'museum:museum_recommendations:istanbul': """Absolutely! Istanbul has incredible museums that showcase its rich Byzantine and Ottoman heritage. Here are the must-visit museums:

**ESSENTIAL MUSEUMS:**

🏛️ **Hagia Sophia** (UNESCO World Heritage)
- Hours: 9:00 AM - 7:00 PM (closed Mondays)
- Entry: 100 TL | Duration: 1-2 hours
- Highlights: Byzantine mosaics, Ottoman calligraphy, architectural marvel

🏛️ **Topkapi Palace Museum**
- Hours: 9:00 AM - 4:45 PM (closed Tuesdays)  
- Entry: 100 TL | Duration: 2-3 hours
- Highlights: Ottoman sultans' palace, treasury, sacred relics

🏛️ **Basilica Cistern**
- Hours: 9:00 AM - 6:30 PM (daily)
- Entry: 30 TL | Duration: 45 minutes
- Highlights: Ancient underground cistern, Medusa columns

🏛️ **Istanbul Archaeological Museums**
- Hours: 9:00 AM - 4:30 PM (closed Mondays)
- Entry: 30 TL | Duration: 2 hours
- Highlights: Ancient artifacts, Alexander Sarcophagus

🏛️ **Turkish and Islamic Arts Museum**
- Hours: 9:00 AM - 5:00 PM (closed Mondays)
- Entry: 25 TL | Duration: 1.5 hours
- Highlights: Carpets, calligraphy, Ottoman artifacts

**VISITING STRATEGY:**
- Start early (9 AM) to avoid crowds
- Buy Museum Pass Istanbul (375 TL) for multiple museums
- Allow full day for Sultanahmet museums
- Combine with Blue Mosque visit (free entry)

Which museums interest you most? I can provide detailed visiting strategies!""",

            'transport:transportation_route:taksim_sultanahmet': """Here are the best ways to travel from Taksim to Sultanahmet:

**🚇 METRO + TRAM (RECOMMENDED - 25 minutes)**
1. Take M2 Metro from Taksim → Vezneciler (8 stops, 15 min)
2. Walk 10 minutes to Sultanahmet, OR
3. Take T1 Tram from Eminönü → Sultanahmet (2 stops, 5 min)
- **Cost:** 15 TL with Istanbulkart
- **Best for:** Quick, reliable, avoiding traffic

**🚇 METRO + WALKING (20-30 minutes)**
- M2 Metro: Taksim → Vezneciler station
- Walk 15 minutes downhill to Sultanahmet
- **Cost:** 7.5 TL
- **Best for:** Budget-friendly, seeing neighborhoods

**🚌 BUS (25-40 minutes)**
- Route: 28, 30M, 36KE buses
- From Taksim Square → Eminönü/Sultanahmet
- **Cost:** 7.5 TL with Istanbulkart
- **Best for:** Direct route (traffic dependent)

**🚕 TAXI (15-35 minutes)**
- Direct door-to-door service
- **Cost:** 80-120 TL (depending on traffic)
- **Best for:** Comfort, luggage, late hours

**PRACTICAL TIPS:**
- Get Istanbulkart for public transport discounts
- Rush hours: 7-9 AM, 5-7 PM (avoid if possible)
- Metro runs 6 AM - 12 AM daily
- Taksim-Sultanahmet is very walkable (45 min scenic walk)

Would you like specific directions for any of these options?""",

            'general:hagia_sophia:opening_hours': """**HAGIA SOPHIA VISITING INFORMATION:**

🕐 **OPENING HOURS:**
- **Daily:** 9:00 AM - 7:00 PM
- **Closed:** Mondays (for maintenance)
- **Last Entry:** 6:00 PM
- **Prayer Times:** Brief closures during Islamic prayer times (5 times daily, 15-20 min each)

**TICKET INFORMATION:**
- **Entry Fee:** 100 Turkish Lira
- **Children:** Under 8 free
- **Students:** 50% discount with valid ID
- **Museum Pass Istanbul:** Includes Hagia Sophia (375 TL for multiple museums)

**VISITING TIPS:**
- **Best Time:** Early morning (9-10 AM) or late afternoon (5-6 PM)
- **Avoid:** Weekends and holidays (very crowded)
- **Duration:** Allow 1-2 hours for full visit
- **Audio Guide:** Available for 25 TL (highly recommended)

**IMPORTANT NOTES:**
- Hagia Sophia functions as both mosque and museum
- Respectful dress required (covers for shoulders/knees)
- Free head scarves available for women
- Photography allowed (no flash)
- Remove shoes in prayer areas

**NEARBY ATTRACTIONS:**
- Blue Mosque (5 min walk, free entry)  
- Topkapi Palace (10 min walk)
- Basilica Cistern (7 min walk)

Would you like specific guidance on what to see inside Hagia Sophia?"""
        }
    
    def get_preloaded_response(self, normalized_query: str, location: str) -> Optional[str]:
        """Get preloaded response if available"""
        # Try exact match first
        key = f"{normalized_query}:{location.lower()}"
        if key in self.preloaded_responses:
            logger.info(f"🎯 Using preloaded response for: {key}")
            return self.preloaded_responses[key]
        
        # Try category match
        for preloaded_key, response in self.preloaded_responses.items():
            if normalized_query in preloaded_key:
                logger.info(f"🎯 Using category preloaded response for: {normalized_query}")
                return response
        
        return None

# Global preloaded cache instance
_preloaded_cache = None

def get_preloaded_cache():
    """Get global preloaded cache instance"""
    global _preloaded_cache
    if _preloaded_cache is None:
        _preloaded_cache = PreloadedCache()
    return _preloaded_cache

def get_preloaded_response(normalized_query: str, location: str) -> Optional[str]:
    """Get preloaded response for common queries"""
    cache = get_preloaded_cache()
    return cache.get_preloaded_response(normalized_query, location)

if __name__ == "__main__":
    # Test preloaded cache
    cache = PreloadedCache()
    
    # Test queries
    test_queries = [
        ("restaurant:turkish_restaurants", "sultanahmet"),
        ("museum:museum_recommendations", "istanbul"),
        ("transport:transportation_route", "taksim_sultanahmet"),
        ("general:hagia_sophia", "opening_hours")
    ]
    
    print("🧪 Testing Preloaded Cache...")
    for query, location in test_queries:
        response = cache.get_preloaded_response(query, location)
        if response:
            print(f"✅ Found preloaded response for: {query}:{location}")
            print(f"   Length: {len(response)} chars")
        else:
            print(f"❌ No preloaded response for: {query}:{location}")
    
    print("✅ Preloaded cache system ready!")
