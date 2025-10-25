"""
Hidden Gems Handler
Specialized handler for hidden gems and secret spots queries
"""

from typing import List, Dict, Optional
import random


class HiddenGemsHandler:
    """Specialized handler for hidden gems and secret spots"""
    
    def __init__(self):
        self.hidden_gems_db = self._load_hidden_gems_database()
        
    def _load_hidden_gems_database(self) -> Dict:
        """Load comprehensive hidden gems database"""
        return {
            'sarıyer': [
                {
                    'name': 'Kilyos Hidden Beach',
                    'type': 'nature',
                    'description': 'Secluded beach known only to locals',
                    'how_to_find': 'Take bus 151 from Sarıyer, ask driver for Kilyos Gizli Plaj',
                    'local_tip': 'Visit on weekdays to avoid crowds',
                    'best_time': 'Early morning or sunset',
                    'cost': 'Free'
                },
                {
                    'name': 'Rumelifeneri Lighthouse Cafe',
                    'type': 'cafe',
                    'description': 'Coffee with lighthouse views at Europe\'s edge',
                    'location': 'End of Sarıyer coastal road',
                    'how_to_find': 'Take 150 bus to last stop, walk 5 minutes',
                    'local_tip': 'Try the Turkish coffee while watching the Black Sea',
                    'best_time': 'Afternoon for best light',
                    'cost': '₺₺'
                },
                {
                    'name': 'Büyükdere Grove',
                    'type': 'nature',
                    'description': 'Ancient forest area with streams and picnic spots',
                    'how_to_find': 'From Sarıyer center, follow signs to Belgrad Ormanı',
                    'local_tip': 'Perfect for a quiet escape from city noise',
                    'best_time': 'Spring and autumn',
                    'cost': 'Free'
                },
                {
                    'name': 'Tarabya Fish Market',
                    'type': 'food',
                    'description': 'Authentic local fish market with small eateries',
                    'how_to_find': 'Near Tarabya Bay, ask locals for "Balık Pazarı"',
                    'local_tip': 'Buy fresh fish and have it cooked at nearby restaurants',
                    'best_time': 'Early morning for freshest catch',
                    'cost': '₺₺'
                }
            ],
            'beşiktaş': [
                {
                    'name': 'Akaretler Row Houses',
                    'type': 'historical',
                    'description': 'Restored Ottoman-era houses with galleries and boutiques',
                    'location': 'Behind Çırağan Palace',
                    'how_to_find': 'From Beşiktaş, walk towards Dolmabahçe, turn right at Çırağan',
                    'hidden_fact': 'Most tourists miss it while rushing to the palace',
                    'local_tip': 'Great for upscale shopping and design stores',
                    'best_time': 'Afternoon',
                    'cost': 'Free to walk around'
                },
                {
                    'name': 'Maçka Democracy Park Secret Paths',
                    'type': 'nature',
                    'description': 'Hidden trails with city views away from main paths',
                    'how_to_find': 'Enter from Maçka cable car station, take left fork',
                    'local_tip': 'Upper trails offer incredible Bosphorus views',
                    'best_time': 'Early morning or sunset',
                    'cost': 'Free'
                },
                {
                    'name': 'Yıldız Palace Hidden Pavilions',
                    'type': 'historical',
                    'description': 'Lesser-known pavilions in Yıldız Park complex',
                    'how_to_find': 'Enter Yıldız Park, follow signs to Malta and Çadır Köşkleri',
                    'local_tip': 'Much quieter than main palace building',
                    'best_time': 'Weekday mornings',
                    'cost': '₺'
                },
                {
                    'name': 'Abbasağa Park Underground Cistern',
                    'type': 'historical',
                    'description': 'Small Byzantine cistern rarely visited',
                    'how_to_find': 'In Abbasağa Park, near the sports center',
                    'hidden_fact': 'Similar to Basilica Cistern but without crowds',
                    'local_tip': 'Ask park guards to unlock it',
                    'best_time': 'Any time',
                    'cost': 'Free'
                }
            ],
            'beyoğlu': [
                {
                    'name': 'Çukurcuma Antique District',
                    'type': 'shopping',
                    'description': 'Maze of vintage shops and antique stores locals treasure',
                    'access': 'Walk from Cihangir down the hill, follow antique shop signs',
                    'how_to_find': 'Between Cihangir and Tophane, winding streets',
                    'local_tip': 'Bargaining is expected and part of the fun',
                    'best_time': 'Afternoon (shops open late)',
                    'cost': 'Free to browse'
                },
                {
                    'name': 'Asmalımescit Secret Courtyards',
                    'type': 'nightlife',
                    'description': 'Hidden courtyard bars known only to locals',
                    'how_to_find': 'Behind main Asmalımescit street, look for unmarked doors',
                    'local_tip': 'Follow locals through mysterious entrances',
                    'best_time': 'After 10 PM',
                    'cost': '₺₺-₺₺₺'
                },
                {
                    'name': 'St. Antoine Church Italian Street',
                    'type': 'historical',
                    'description': 'Quiet street with Italian heritage and cafes',
                    'how_to_find': 'Off İstiklal, near Galatasaray',
                    'hidden_fact': 'Former Italian quarter with hidden trattorias',
                    'local_tip': 'Visit on Sunday for mass and Italian pastries',
                    'best_time': 'Sunday morning',
                    'cost': 'Free'
                },
                {
                    'name': 'Pera Museum Rooftop Terrace',
                    'type': 'view',
                    'description': 'Secret terrace with Golden Horn views',
                    'how_to_find': 'Inside Pera Museum, take elevator to top floor cafe',
                    'local_tip': 'Museum ticket gives access to terrace cafe',
                    'best_time': 'Late afternoon for sunset',
                    'cost': '₺ (museum entrance)'
                }
            ],
            'kadıköy': [
                {
                    'name': 'Moda Hidden Coves',
                    'type': 'nature',
                    'description': 'Small rocky coves for swimming away from crowds',
                    'how_to_find': 'Walk south from Moda pier, past the tea garden',
                    'local_tip': 'Locals swim here instead of busy beaches',
                    'best_time': 'Summer weekdays',
                    'cost': 'Free'
                },
                {
                    'name': 'Yeldeğirmeni Street Art',
                    'type': 'art',
                    'description': 'Entire neighborhood covered in murals',
                    'how_to_find': 'Between Kadıköy and Kızıltoprak, walk the side streets',
                    'hidden_fact': 'Living gallery created by local artists',
                    'local_tip': 'Join a free walking tour on Saturdays',
                    'best_time': 'Morning light for photos',
                    'cost': 'Free'
                },
                {
                    'name': 'Karaköy Güllüoğlu Secret Branch',
                    'type': 'food',
                    'description': 'Original baklava shop without tourist lines',
                    'how_to_find': 'Kadıköy branch near market, ask for "eski şube"',
                    'local_tip': 'Same quality, zero wait time',
                    'best_time': 'Afternoon',
                    'cost': '₺'
                },
                {
                    'name': 'Fenerbahçe Lighthouse Path',
                    'type': 'nature',
                    'description': 'Coastal walk to historic lighthouse',
                    'how_to_find': 'From Kadıköy, take bus to Fenerbahçe, walk to coast',
                    'local_tip': 'Continue walk to Kalamış for full coastal experience',
                    'best_time': 'Sunset',
                    'cost': 'Free'
                }
            ],
            'sultanahmet': [
                {
                    'name': 'Soğukçeşme Sokağı',
                    'type': 'historical',
                    'description': 'Cobblestone street with Ottoman wooden houses',
                    'how_to_find': 'Between Hagia Sophia and Topkapı Palace',
                    'hidden_fact': 'Most tourists rush past without noticing',
                    'local_tip': 'Perfect photo spot early morning',
                    'best_time': 'Before 9 AM',
                    'cost': 'Free'
                },
                {
                    'name': 'Gülhane Park Secret Tea Garden',
                    'type': 'cafe',
                    'description': 'Hidden tea house in park\'s far corner',
                    'how_to_find': 'Enter Gülhane Park, walk to back left corner',
                    'local_tip': 'Locals\' secret lunch spot',
                    'best_time': 'Lunch or afternoon',
                    'cost': '₺'
                },
                {
                    'name': 'Bukoleon Palace Ruins',
                    'type': 'historical',
                    'description': 'Byzantine palace ruins by the sea',
                    'how_to_find': 'Kennedy Caddesi near Cankurtaran, look for ruins in wall',
                    'hidden_fact': 'Hidden in plain sight on main road',
                    'local_tip': 'Free ancient history without crowds',
                    'best_time': 'Any time',
                    'cost': 'Free'
                }
            ],
            'üsküdar': [
                {
                    'name': 'Nakkaştepe Secret Viewpoint',
                    'type': 'view',
                    'description': 'Best Bosphorus view without tourists',
                    'how_to_find': 'Take bus to Nakkaştepe, walk to hilltop',
                    'local_tip': 'Better than Çamlıca Hill, far less crowded',
                    'best_time': 'Sunset',
                    'cost': 'Free'
                },
                {
                    'name': 'Kuzguncuk Painted Stairs',
                    'type': 'neighborhood',
                    'description': 'Colorful stairs in charming village-like area',
                    'how_to_find': 'Ferry to Kuzguncuk, walk uphill',
                    'local_tip': 'Explore the multicultural neighborhood history',
                    'best_time': 'Morning or afternoon',
                    'cost': 'Free'
                },
                {
                    'name': 'Şemsi Paşa Mosque Courtyard',
                    'type': 'peaceful',
                    'description': 'Tiny mosque right on the Bosphorus',
                    'how_to_find': 'Üsküdar waterfront, near ferry dock',
                    'local_tip': 'Most peaceful spot to watch ferries go by',
                    'best_time': 'Late afternoon',
                    'cost': 'Free'
                }
            ]
        }
    
    def get_hidden_gems(
        self, 
        location: Optional[str] = None, 
        gem_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Return curated hidden gems with local insights
        
        Args:
            location: Neighborhood name (optional)
            gem_type: Type of gem (nature, food, historical, etc.)
            limit: Maximum number of results
            
        Returns:
            List of hidden gem recommendations
        """
        if location:
            # Normalize location name
            location_lower = location.lower().replace('i̇', 'i')
            gems = self.hidden_gems_db.get(location_lower, [])
        else:
            # Get gems from all neighborhoods
            gems = []
            for neighborhood_gems in self.hidden_gems_db.values():
                gems.extend(neighborhood_gems)
        
        # Filter by type if specified
        if gem_type:
            gems = [g for g in gems if g.get('type') == gem_type]
        
        # Shuffle for variety and limit
        random.shuffle(gems)
        return gems[:limit]
    
    def format_hidden_gem_response(self, gems: List[Dict], query_location: Optional[str] = None) -> str:
        """Format hidden gems into actionable response"""
        if not gems:
            return self._get_fallback_response(query_location)
        
        location_header = f" in {query_location.title()}" if query_location else ""
        response = f"🔍 **Hidden Gems{location_header}** - Local Secrets Revealed!\n\n"
        response += "Here are some amazing spots that most tourists never find:\n\n"
        
        for i, gem in enumerate(gems, 1):
            response += f"**{i}. {gem['name']}** {'✨' if i == 1 else ''}\n"
            response += f"📍 **Type:** {gem['type'].title()}\n"
            response += f"💡 **What It Is:** {gem['description']}\n"
            
            if gem.get('how_to_find'):
                response += f"🗺️ **How to Find:** {gem['how_to_find']}\n"
            
            if gem.get('local_tip'):
                response += f"🎯 **Local Tip:** {gem['local_tip']}\n"
            
            if gem.get('best_time'):
                response += f"⏰ **Best Time:** {gem['best_time']}\n"
            
            if gem.get('cost'):
                response += f"💰 **Cost:** {gem['cost']}\n"
            
            if gem.get('hidden_fact'):
                response += f"🤫 **Secret:** {gem['hidden_fact']}\n"
            
            response += "\n"
        
        response += "💬 **Pro Tip:** These spots are authentic local favorites. "
        response += "Don't be shy to ask locals for directions - they'll be happy to help!\n\n"
        response += "Want more hidden gems? Ask me about a specific neighborhood! 🗺️"
        
        return response
    
    def _get_fallback_response(self, location: Optional[str]) -> str:
        """Fallback response when no gems found"""
        response = "🔍 Looking for hidden gems"
        if location:
            response += f" in {location.title()}"
        response += "!\n\n"
        
        response += "I have secret spots in these neighborhoods:\n"
        response += "• **Sarıyer** - Hidden beaches and cafes\n"
        response += "• **Beşiktaş** - Secret parks and historical spots\n"
        response += "• **Beyoğlu** - Underground bars and vintage shops\n"
        response += "• **Kadıköy** - Local coves and street art\n"
        response += "• **Sultanahmet** - Off-the-beaten-path historical sites\n"
        response += "• **Üsküdar** - Secret viewpoints and peaceful mosques\n\n"
        response += "Which neighborhood interests you? 😊"
        
        return response
    
    def detect_hidden_gems_query(self, query: str) -> bool:
        """Detect if query is asking for hidden gems"""
        query_lower = query.lower()
        keywords = [
            'hidden', 'secret', 'gem', 'gems', 'off-the-beaten',
            'off the beaten', 'local', 'locals only', 'insider',
            'unknown', 'discover', 'undiscovered', 'authentic',
            'gizli', 'saklı', 'bilinmeyen'  # Turkish keywords
        ]
        return any(keyword in query_lower for keyword in keywords)


# Singleton instance
_hidden_gems_handler = None

def get_hidden_gems_handler() -> HiddenGemsHandler:
    """Get or create hidden gems handler instance"""
    global _hidden_gems_handler
    if _hidden_gems_handler is None:
        _hidden_gems_handler = HiddenGemsHandler()
    return _hidden_gems_handler
