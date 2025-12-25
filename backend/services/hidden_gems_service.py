#!/usr/bin/env python
"""
Hidden Gems Database Service for Istanbul AI System
Provides access to local, lesser-known attractions and experiences
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class HiddenGemQuery:
    """Structured query for hidden gems search"""
    district: Optional[str] = None
    category: Optional[str] = None
    keywords: List[str] = None
    difficulty: Optional[str] = None
    cost_preference: Optional[str] = None

class HiddenGemsService:
    """Service for searching and providing hidden gems recommendations"""
    
    def __init__(self, database_path: str = None):
        if database_path is None:
            database_path = "/Users/omer/Desktop/ai-stanbul/backend/data/hidden_gems_database.json"
        
        self.database_path = database_path
        self.gems = []
        self.templates = self._load_response_templates()
        self.load_database()
    
    def _load_response_templates(self) -> Dict[str, Any]:
        """Load response templates for different scenarios"""
        return {
            'intro_phrases': [
                "Here are some hidden gems I'd love to share with you:",
                "Let me reveal some of Istanbul's best-kept secrets:",
                "Here are some local favorites that most tourists never discover:",
                "I know some special places that will make your Istanbul experience unique:"
            ],
            'gem_template': """💎 **{name}** ({district})
🏷️ {category} • 🚶 {difficulty} • 💰 {cost}
📍 {address}
✨ {description}

🤫 Why it's hidden: {why_hidden}
⏰ Best time: {best_time}
💡 Insider tip: {insider_tip}""",
            
            'category_intros': {
                'historical': "These historical gems offer authentic glimpses into Istanbul's layered past:",
                'cultural': "Discover Istanbul's vibrant cultural scene beyond the obvious:",
                'culinary': "Taste authentic Istanbul at these local culinary secrets:",
                'nature': "Find peace in these natural hideaways within the city:",
                'shopping': "Shop like a local at these insider treasure troves:",
                'nightlife': "Experience Istanbul's authentic nightlife scene:"
            },
            
            'no_results': [
                "I don't have specific hidden gems for that area yet, but let me suggest some nearby secrets:",
                "That's a great question! While I'm still building my knowledge of that area, here are some gems from neighboring districts:",
                "I'm always discovering new places! Here are some similar hidden gems you might enjoy:"
            ]
        }
    
    def load_database(self):
        """Load hidden gems database from JSON file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.gems = data.get('gems', [])
                print(f"✅ Loaded {len(self.gems)} hidden gems from database")
            else:
                print(f"⚠️  Hidden gems database not found at {self.database_path}")
                self.gems = []
        except Exception as e:
            print(f"❌ Error loading hidden gems database: {e}")
            self.gems = []
    
    def parse_hidden_gems_query(self, query: str) -> HiddenGemQuery:
        """Parse user query to extract hidden gems search criteria"""
        query_lower = query.lower()
        
        # Extract district
        district_keywords = {
            'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula'],
            'beyoğlu': ['beyoglu', 'beyoğlu', 'istiklal', 'galata', 'taksim'],
            'kadıköy': ['kadikoy', 'kadıköy', 'asian side', 'moda'],
            'beşiktaş': ['besiktas', 'beşiktaş', 'ortakoy', 'ortaköy'],
            'üsküdar': ['uskudar', 'üsküdar'],
            'fatih': ['fatih', 'balat', 'fener'],
            'sarıyer': ['sariyer', 'sarıyer', 'emirgan', 'belgrado']
        }
        
        district = None
        for dist, keywords in district_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                district = dist
                break
        
        # Extract category (check order matters - more specific first)
        category_keywords = {
            'nature': ['nature', 'park', 'garden', 'outdoor', 'hiking', 'forest', 'green', 'natural'],
            'culinary': ['food', 'restaurant', 'cafe', 'coffee', 'eat', 'dining', 'culinary', 'kitchen'],
            'historical': ['historical', 'historic', 'history', 'ancient', 'old', 'mosque', 'church', 'monument'],
            'cultural': ['cultural', 'culture', 'art', 'museum', 'gallery', 'exhibition'],
            'shopping': ['shopping', 'shop', 'market', 'bazaar', 'antique', 'vintage', 'buy'],
            'nightlife': ['nightlife', 'bar', 'club', 'music', 'night', 'drink']
        }
        
        category = None
        for cat, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                category = cat
                break
        
        # Extract keywords
        keywords = []
        special_keywords = ['hidden', 'secret', 'local', 'authentic', 'off-beaten', 'unknown', 'underground']
        for keyword in special_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        # Extract difficulty preference
        difficulty = None
        if any(word in query_lower for word in ['easy', 'accessible', 'simple']):
            difficulty = 'easy'
        elif any(word in query_lower for word in ['challenging', 'difficult', 'adventure']):
            difficulty = 'challenging'
        elif any(word in query_lower for word in ['moderate', 'medium']):
            difficulty = 'moderate'
        
        # Extract cost preference
        cost_preference = None
        if any(word in query_lower for word in ['free', 'budget', 'cheap', 'affordable']):
            cost_preference = 'budget'
        elif any(word in query_lower for word in ['expensive', 'luxury', 'upscale', 'premium']):
            cost_preference = 'expensive'
        
        return HiddenGemQuery(
            district=district,
            category=category,
            keywords=keywords,
            difficulty=difficulty,
            cost_preference=cost_preference
        )
    
    def filter_gems(self, query: HiddenGemQuery, limit: int = 5) -> List[Dict]:
        """Filter hidden gems based on query criteria"""
        if not self.gems:
            return []
        
        filtered_gems = []
        
        for gem in self.gems:
            matches = True
            
            # District filter
            if query.district and gem.get('district', '').lower() != query.district.lower():
                matches = False
            
            # Category filter
            if query.category and gem.get('category', '').lower() != query.category.lower():
                matches = False
            
            # Difficulty filter
            if query.difficulty and gem.get('access_difficulty', '').lower() != query.difficulty.lower():
                matches = False
            
            # Cost filter
            if query.cost_preference:
                gem_cost = gem.get('cost', '').lower()
                if query.cost_preference == 'budget' and gem_cost not in ['free', 'budget-friendly', 'very budget-friendly']:
                    matches = False
                elif query.cost_preference == 'expensive' and gem_cost not in ['upscale', 'luxury', 'expensive']:
                    matches = False
                 # Keyword matching in name, description, and tags
        if query.keywords:
            gem_text = f"{gem.get('name', '')} {gem.get('description', '')} {' '.join(gem.get('tags', []))}".lower()
            keyword_match = any(keyword in gem_text for keyword in query.keywords)
            if not keyword_match:
                matches = False
            
            if matches:
                filtered_gems.append(gem)
        
        # If no exact matches but we have basic criteria, try simpler filter
        if not filtered_gems and (query.district or query.category):
            # Try with just category or district
            filtered_gems = [gem for gem in self.gems 
                           if (not query.category or gem.get('category', '').lower() == query.category.lower()) and
                              (not query.district or gem.get('district', '').lower() == query.district.lower())][:limit]
        
        return filtered_gems[:limit]
    
    def format_gems_response(self, gems: List[Dict], query: HiddenGemQuery) -> str:
        """Format hidden gems into a readable response"""
        if not gems:
            fallback_gems = self.gems[:3] if self.gems else []
            if not fallback_gems:
                return "I'm still building my collection of hidden gems for that area. Check back soon for local secrets!"
            
            intro = "I don't have specific gems for that area yet, but here are some amazing hidden spots to discover:"
        else:
            intro = "Here are some hidden gems I'd love to share with you:"
        
        gems_to_show = gems if gems else fallback_gems
        
        response = intro + "\n\n"
        
        for i, gem in enumerate(gems_to_show, 1):
            gem_text = f"{i}. 💎 **{gem.get('name', 'Unknown Gem')}** ({gem.get('district', 'Istanbul')})\n"
            gem_text += f"🏷️ {gem.get('category', 'attraction').title()} • 🚶 {gem.get('access_difficulty', 'easy').title()} access • 💰 {gem.get('cost', 'varies').title()}\n"
            gem_text += f"📍 {gem.get('location', {}).get('address', 'Location details available')}\n\n"
            gem_text += f"✨ {gem.get('description', 'A wonderful hidden spot in Istanbul.')}\n\n"
            gem_text += f"🤫 **Why it's hidden**: {gem.get('why_hidden', 'Often overlooked by tourists')}\n"
            gem_text += f"⏰ **Best time**: {gem.get('best_time', 'Anytime')}\n"
            gem_text += f"💡 **Insider tip**: {gem.get('insider_tip', 'Ask locals for more details')}\n"
            
            response += gem_text + "\n\n"
        
        return response.strip()
    
    def search_hidden_gems(self, query: str) -> str:
        """Main method to search hidden gems and return formatted response"""
        try:
            parsed_query = self.parse_hidden_gems_query(query)
            gems = self.filter_gems(parsed_query)
            return self.format_gems_response(gems, parsed_query)
        except Exception as e:
            print(f"❌ Error searching hidden gems: {e}")
            return "I'm having trouble accessing my hidden gems collection right now. Try asking about specific areas or types of places you're interested in!"
    
    def search(self, query: str, **kwargs) -> str:
        """Alias for search_hidden_gems - provides consistent interface"""
        return self.search_hidden_gems(query)
    
    def get_gems_by_category(self, category: str, limit: int = 3) -> List[Dict]:
        """Get hidden gems by category"""
        return [gem for gem in self.gems if gem.get('category', '').lower() == category.lower()][:limit]
    
    def get_gems_by_district(self, district: str, limit: int = 3) -> List[Dict]:
        """Get hidden gems by district"""
        return [gem for gem in self.gems if gem.get('district', '').lower() == district.lower()][:limit]
    
    def get_random_gems(self, count: int = 3) -> List[Dict]:
        """Get random selection of hidden gems"""
        import random
        if len(self.gems) <= count:
            return self.gems
        return random.sample(self.gems, count)

# Example usage for testing
if __name__ == "__main__":
    print("💎 Testing Hidden Gems Service")
    print("=" * 40)
    
    service = HiddenGemsService()
    
    test_queries = [
        "hidden places in Sultanahmet",
        "secret cultural spots in Beyoğlu",
        "free historical gems",
        "local food secrets",
        "nature spots off the beaten path"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 30)
        result = service.search_hidden_gems(query)
        print(result[:200] + "..." if len(result) > 200 else result)
        print()
