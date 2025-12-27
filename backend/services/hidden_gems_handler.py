"""
Hidden Gems Handler
Specialized handler for hidden gems and secret spots queries
Enhanced with comprehensive database and intelligent filtering
NOW WITH REAL-TIME PERSONALIZATION via Online Learning
"""

from typing import List, Dict, Optional, Tuple
import random
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the comprehensive database
try:
    from data.hidden_gems_database import (
        HIDDEN_GEMS_DATABASE,
        get_all_hidden_gems,
        get_gems_by_neighborhood,
        get_gems_by_type,
        get_gems_by_category,
        get_top_hidden_gems,
        get_all_neighborhoods
    )
    DATABASE_LOADED = True
except ImportError:
    # Try alternate path
    try:
        from backend.data.hidden_gems_database import (
            HIDDEN_GEMS_DATABASE,
            get_all_hidden_gems,
            get_gems_by_neighborhood,
            get_gems_by_type,
            get_gems_by_category,
            get_top_hidden_gems,
            get_all_neighborhoods
        )
        DATABASE_LOADED = True
    except ImportError:
        DATABASE_LOADED = False
        HIDDEN_GEMS_DATABASE = {}

# Import real-time feedback loop for personalization
try:
    from services.realtime_feedback_loop import get_realtime_feedback_loop
    REALTIME_LEARNING_ENABLED = True
except ImportError:
    try:
        from backend.services.realtime_feedback_loop import get_realtime_feedback_loop
        REALTIME_LEARNING_ENABLED = True
    except ImportError:
        REALTIME_LEARNING_ENABLED = False
        print("⚠️ Real-time learning not available - using static recommendations")


class HiddenGemsHandler:
    """Enhanced handler for hidden gems and secret spots with intelligent filtering and real-time personalization"""
    
    # Neighborhood mapping: sub-districts -> parent neighborhoods
    NEIGHBORHOOD_MAPPING = {
        # Beyoğlu sub-districts
        'balat': ['beyoğlu', 'fatih'],  # Balat straddles both areas
        'fener': ['beyoğlu', 'fatih'],
        'cihangir': 'beyoğlu',
        'çukurcuma': 'beyoğlu',
        'galata': 'beyoğlu',
        'karaköy': 'beyoğlu',
        'tophane': 'beyoğlu',
        'asmalımescit': 'beyoğlu',
        'tünel': 'beyoğlu',
        # Beşiktaş sub-districts
        'ortaköy': 'beşiktaş',
        'arnavutköy': 'beşiktaş',
        'bebek': 'beşiktaş',
        'etiler': 'beşiktaş',
        'levent': 'beşiktaş',
        # Sarıyer sub-districts
        'tarabya': 'sarıyer',
        'yeniköy': 'sarıyer',
        'istinye': 'sarıyer',
        'emirgan': 'sarıyer',
        'kilyos': 'sarıyer',
        'bahçeköy': 'sarıyer',
        'zekeriyaköy': 'sarıyer',
        'garipçe': 'sarıyer',
        # Kadıköy sub-districts
        'moda': 'kadıköy',
        'yeldeğirmeni': 'kadıköy',
        'fenerbahçe': 'kadıköy',
        'kalamış': 'kadıköy',
        # Üsküdar sub-districts
        'kuzguncuk': 'üsküdar',
        'beylerbeyi': 'üsküdar',
        'çengelköy': 'üsküdar',
        'nakkaştepe': 'üsküdar',
        # Fatih/Sultanahmet sub-districts
        'sultanahmet': 'sultanahmet',
        'eminönü': 'fatih',
        'sirkeci': 'fatih',
        'gülhane': 'fatih',
    }
    
    def __init__(self, enable_realtime_learning: bool = True):
        self.database_available = DATABASE_LOADED
        if not self.database_available:
            # Fallback to inline database
            self.hidden_gems_db = self._load_fallback_database()
        else:
            self.hidden_gems_db = HIDDEN_GEMS_DATABASE
        
        # Initialize real-time learning
        self.realtime_learning_enabled = enable_realtime_learning and REALTIME_LEARNING_ENABLED
        if self.realtime_learning_enabled:
            try:
                self.feedback_loop = get_realtime_feedback_loop()
                print("✅ Real-time personalization enabled for Hidden Gems")
            except Exception as e:
                print(f"⚠️ Failed to initialize real-time learning: {e}")
                self.realtime_learning_enabled = False
        else:
            self.feedback_loop = None
    
    def _load_fallback_database(self) -> Dict:
        """Fallback database if main database not available"""
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
        category: Optional[str] = None,
        budget: Optional[str] = None,
        limit: int = 5,
        min_hidden_factor: Optional[int] = None
    ) -> List[Dict]:
        """
        Return curated hidden gems with intelligent filtering and ranking
        
        Args:
            location: Neighborhood name (optional)
            gem_type: Type of gem (nature, food, historical, cafe, etc.)
            category: Category filter (beach, hiking, restaurant, etc.)
            budget: Budget filter ('free', 'cheap', 'moderate', 'expensive')
            limit: Maximum number of results
            min_hidden_factor: Minimum hidden factor (1-10 scale)
            
        Returns:
            List of hidden gem recommendations sorted by relevance
        """
        if self.database_available:
            # Use comprehensive database with helper functions
            if location:
                location_lower = location.lower().replace('i̇', 'i')
                
                # First try exact neighborhood match
                gems = get_gems_by_neighborhood(location_lower)
                
                # If no results and it's a sub-district, try parent neighborhood(s)
                # BUT filter results to only include gems mentioning the sub-district
                if not gems and location_lower in self.NEIGHBORHOOD_MAPPING:
                    parent_neighborhoods = self.NEIGHBORHOOD_MAPPING[location_lower]
                    
                    # Handle both single parent and multiple parents
                    if isinstance(parent_neighborhoods, str):
                        parent_neighborhoods = [parent_neighborhoods]
                    
                    # Search all parent neighborhoods
                    gems = []
                    for parent_neighborhood in parent_neighborhoods:
                        parent_gems = get_gems_by_neighborhood(parent_neighborhood)
                        
                        # Filter to only gems that mention the specific sub-district
                        for gem in parent_gems:
                            searchable_text = ' '.join([
                                gem.get('name', ''),
                                gem.get('description', ''),
                                gem.get('how_to_find', ''),
                                gem.get('local_tip', '')
                            ]).lower()
                            
                            if location_lower in searchable_text:
                                gems.append(gem)
                
                # If still no results, search all gems for location name
                if not gems:
                    all_gems = get_all_hidden_gems()
                    gems = []
                    for gem in all_gems:
                        searchable_text = ' '.join([
                            gem.get('name', ''),
                            gem.get('description', ''),
                            gem.get('how_to_find', ''),
                            gem.get('local_tip', '')
                        ]).lower()
                        
                        if location_lower in searchable_text:
                            gems.append(gem)
            else:
                gems = get_all_hidden_gems()
            
            # Apply filters
            if gem_type:
                gems = [g for g in gems if g.get('type', '').lower() == gem_type.lower()]
            
            if category:
                gems = [g for g in gems if g.get('category', '').lower() == category.lower()]
            
            if budget:
                gems = self._filter_by_budget(gems, budget)
            
            if min_hidden_factor:
                gems = [g for g in gems if g.get('hidden_factor', 0) >= min_hidden_factor]
            
            # Sort by hidden_factor for best results
            gems = sorted(gems, key=lambda x: x.get('hidden_factor', 0), reverse=True)
        else:
            # Fallback to old method
            if location:
                location_lower = location.lower().replace('i̇', 'i')
                gems = self.hidden_gems_db.get(location_lower, [])
            else:
                gems = []
                for neighborhood_gems in self.hidden_gems_db.values():
                    gems.extend(neighborhood_gems)
            
            if gem_type:
                gems = [g for g in gems if g.get('type') == gem_type]
            
            # Add variety with shuffle
            random.shuffle(gems)
        
        return gems[:limit]
    
    def _filter_by_budget(self, gems: List[Dict], budget: str) -> List[Dict]:
        """Filter gems by budget constraint"""
        budget_map = {
            'free': ['Free', 'free'],
            'cheap': ['Free', 'free', '50-100 TL', '50-150 TL'],
            'moderate': ['Free', 'free', '50-100 TL', '50-150 TL', '100-200 TL', '150-250 TL'],
            'expensive': ['200-400 TL', '300-500 TL', '150-250 TL']
        }
        
        allowed_costs = budget_map.get(budget.lower(), [])
        if not allowed_costs:
            return gems
        
        return [g for g in gems if any(cost in g.get('cost', '') for cost in allowed_costs)]
    
    def search_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict]:
        """Search gems by keywords in name, description, or tags"""
        if self.database_available:
            all_gems = get_all_hidden_gems()
        else:
            all_gems = []
            for gems in self.hidden_gems_db.values():
                all_gems.extend(gems)
        
        results = []
        for gem in all_gems:
            score = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check in name
                if keyword_lower in gem.get('name', '').lower():
                    score += 3
                # Check in description
                if keyword_lower in gem.get('description', '').lower():
                    score += 2
                # Check in tags
                if 'tags' in gem and keyword_lower in ' '.join(gem['tags']).lower():
                    score += 2
                # Check in type
                if keyword_lower in gem.get('type', '').lower():
                    score += 1
            
            if score > 0:
                gem_copy = gem.copy()
                gem_copy['_relevance_score'] = score
                results.append(gem_copy)
        
        # Sort by relevance and hidden_factor
        results.sort(key=lambda x: (x.get('_relevance_score', 0), x.get('hidden_factor', 0)), reverse=True)
        return results[:limit]
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        location: Optional[str] = None,
        gem_type: Optional[str] = None,
        limit: int = 10,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get personalized hidden gem recommendations using real-time learning
        Falls back to static recommendations if real-time learning is not available
        
        Args:
            user_id: User identifier for personalization
            location: Optional location filter
            gem_type: Optional type filter
            limit: Number of recommendations to return
            session_id: Session ID for tracking
        
        Returns:
            List of personalized gem recommendations with scores
        """
        # Get candidate gems
        candidates = self.get_hidden_gems(
            location=location,
            gem_type=gem_type,
            limit=limit * 3  # Get more candidates for ranking
        )
        
        if not candidates:
            return []
        
        # If real-time learning is enabled, use it for personalization
        if self.realtime_learning_enabled and self.feedback_loop:
            try:
                # Extract item IDs
                candidate_ids = [self._gem_to_item_id(gem) for gem in candidates]
                
                # Get personalized scores from online learning
                personalized = self.feedback_loop.get_recommendations(
                    user_id=user_id,
                    candidate_items=candidate_ids,
                    top_k=limit
                )
                
                # Map back to gems with scores
                id_to_gem = {self._gem_to_item_id(gem): gem for gem in candidates}
                results = []
                for rec in personalized:
                    gem = id_to_gem.get(rec['item_id'])
                    if gem:
                        gem_copy = gem.copy()
                        gem_copy['_personalization_score'] = rec['score']
                        gem_copy['_personalized'] = True
                        results.append(gem_copy)
                
                return results
            
            except Exception as e:
                print(f"⚠️ Personalization failed, falling back to static: {e}")
        
        # Fallback to static recommendations
        return candidates[:limit]
    
    def _gem_to_item_id(self, gem: Dict) -> str:
        """Convert a gem to a unique item ID for the learning system"""
        # Use name as ID (in production, use a proper unique ID)
        return f"hidden_gem:{gem.get('name', '').replace(' ', '_').lower()}"
    
    def get_recommendations_by_time(self, time_of_day: str, limit: int = 5) -> List[Dict]:
        """Get gems recommended for specific time of day"""
        time_keywords = {
            'morning': ['morning', 'breakfast', 'early'],
            'afternoon': ['afternoon', 'lunch'],
            'evening': ['evening', 'sunset', 'late afternoon'],
            'night': ['night', 'nightlife', 'after dark']
        }
        
        keywords = time_keywords.get(time_of_day.lower(), [])
        if self.database_available:
            all_gems = get_all_hidden_gems()
        else:
            all_gems = []
            for gems in self.hidden_gems_db.values():
                all_gems.extend(gems)
        
        filtered = []
        for gem in all_gems:
            best_time = gem.get('best_time', '').lower()
            if any(kw in best_time for kw in keywords):
                filtered.append(gem)
        
        # Sort by hidden_factor
        filtered.sort(key=lambda x: x.get('hidden_factor', 0), reverse=True)
        return filtered[:limit] if filtered else all_gems[:limit]
    
    def format_hidden_gem_response(self, gems: List[Dict], query_location: Optional[str] = None) -> Dict:
        """
        Format hidden gems into structured data for LLM to present naturally.
        Returns dictionary with gems data instead of pre-formatted string.
        This allows the LLM to create contextual, conversational responses.
        """
        if not gems:
            return {
                'gems': [],
                'location': query_location,
                'available_neighborhoods': [
                    'Sarıyer - Hidden beaches and cafes',
                    'Beşiktaş - Secret parks and historical spots',
                    'Beyoğlu - Underground bars and vintage shops',
                    'Kadıköy - Local coves and street art',
                    'Sultanahmet - Off-the-beaten-path historical sites',
                    'Üsküdar - Secret viewpoints and peaceful mosques'
                ],
                'message': 'No gems found for this query'
            }
        
        # Structure gems data for LLM
        formatted_gems = []
        for i, gem in enumerate(gems, 1):
            gem_data = {
                'number': i,
                'name': gem['name'],
                'type': gem.get('type', 'Unknown').title(),
                'category': gem.get('category', '').title() if gem.get('category') else None,
                'description': gem.get('description', 'A hidden gem'),
                'hidden_factor': gem.get('hidden_factor', 5),
                'how_to_find': gem.get('how_to_find'),
                'local_tip': gem.get('local_tip'),
                'best_time': gem.get('best_time'),
                'cost': gem.get('cost'),
                'why_special': gem.get('why_special'),
                'insider_knowledge': gem.get('insider_knowledge'),
                'hidden_fact': gem.get('hidden_fact'),
                'tags': gem.get('tags', [])[:5],
                'coordinates': gem.get('coordinates')
            }
            # Remove None values for cleaner data
            gem_data = {k: v for k, v in gem_data.items() if v is not None}
            formatted_gems.append(gem_data)
        
        return {
            'gems': formatted_gems,
            'location': query_location,
            'count': len(formatted_gems),
            'message': 'success'
        }
    
    def extract_query_parameters(self, query: str) -> Dict[str, any]:
        """Extract parameters from user query for intelligent filtering"""
        query_lower = query.lower()
        params = {
            'location': None,
            'gem_type': None,
            'budget': None,
            'time_of_day': None
        }
        
        # Extract neighborhood (check both top-level and sub-districts)
        if self.database_available:
            neighborhoods = get_all_neighborhoods()
        else:
            neighborhoods = list(self.hidden_gems_db.keys())
        
        # First check for exact neighborhood matches
        for neighborhood in neighborhoods:
            if neighborhood in query_lower:
                params['location'] = neighborhood
                break
        
        # If no match, check sub-districts
        if not params['location']:
            for sub_district, parent in self.NEIGHBORHOOD_MAPPING.items():
                if sub_district in query_lower:
                    params['location'] = sub_district  # Keep original to enable proper search
                    break
        
        # Extract gem type
        type_keywords = {
            'nature': ['nature', 'outdoor', 'park'],
            'cafe': ['cafe', 'coffee'],
            'food': ['food', 'restaurant'],
            'historical': ['historical', 'history']
        }
        
        for type_name, keywords in type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                params['gem_type'] = type_name
                break
        
        return params
    
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
