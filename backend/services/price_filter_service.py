"""
Price Filter Service
Enhanced budget-aware filtering and recommendations
"""

from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PriceFilterService:
    """Enhanced price filtering with budget categories"""
    
    # Budget ranges in Turkish Lira per person
    BUDGET_RANGES = {
        'free': (0, 0),
        'budget': (0, 100),
        'moderate': (100, 300),
        'upscale': (300, 600),
        'luxury': (600, float('inf'))
    }
    
    # Price level symbols
    PRICE_SYMBOLS = {
        'free': '🆓',
        'budget': '₺',
        'moderate': '₺₺',
        'upscale': '₺₺₺',
        'luxury': '₺₺₺₺'
    }
    
    # Free attractions in Istanbul
    FREE_ATTRACTIONS = [
        {
            'name': 'Grand Bazaar',
            'type': 'market',
            'cost': 0,
            'location': 'Beyazıt',
            'note': 'Free to enter and walk around (shopping optional)',
            'hours': 'Mon-Sat 09:00-19:00',
            'tip': 'Best early morning to avoid crowds'
        },
        {
            'name': 'Princes\' Islands Walk',
            'type': 'nature',
            'cost': 'Ferry ticket only (₺15)',
            'location': 'Adalar',
            'note': 'Walking the islands is completely free',
            'hours': 'All day',
            'tip': 'Bring picnic to save on restaurant costs'
        },
        {
            'name': 'Ortaköy Square',
            'type': 'area',
            'cost': 0,
            'location': 'Ortaköy',
            'note': 'Free Bosphorus views and people watching',
            'hours': 'All day, best at sunset',
            'tip': 'Try street food kumpir (₺50-80)'
        },
        {
            'name': 'Taksim Square & Istiklal Street Walk',
            'type': 'area',
            'cost': 0,
            'location': 'Beyoğlu',
            'note': 'Free to walk the famous pedestrian street',
            'hours': 'All day',
            'tip': 'Many free street performances in the evening'
        },
        {
            'name': 'Kadıköy Moda Walk',
            'type': 'area',
            'cost': 0,
            'location': 'Kadıköy',
            'note': 'Beautiful seaside promenade on Asian side',
            'hours': 'All day',
            'tip': 'Free concerts at Moda stage on weekends'
        },
        {
            'name': 'Balat & Fener Walking Tour',
            'type': 'neighborhood',
            'cost': 0,
            'location': 'Balat',
            'note': 'Colorful historic streets, perfect for photos',
            'hours': 'All day, best morning light',
            'tip': 'Free to explore, just bring your camera'
        },
        {
            'name': 'Çamlıca Hill Views',
            'type': 'viewpoint',
            'cost': 0,
            'location': 'Üsküdar',
            'note': 'Panoramic city views from highest point',
            'hours': 'All day',
            'tip': 'Best at sunset, bring tea from nearby cafes'
        },
        {
            'name': 'Gülhane Park',
            'type': 'park',
            'cost': 0,
            'location': 'Sultanahmet',
            'note': 'Historic park next to Topkapı Palace',
            'hours': 'All day',
            'tip': 'Beautiful tulips in spring'
        },
        {
            'name': 'Eminönü Waterfront',
            'type': 'area',
            'cost': 0,
            'location': 'Eminönü',
            'note': 'Watch ferries, feed seagulls, enjoy views',
            'hours': 'All day',
            'tip': 'Try famous fish sandwich (₺40) if hungry'
        },
        {
            'name': 'Beylerbeyi Coastal Walk',
            'type': 'nature',
            'cost': 0,
            'location': 'Beylerbeyi',
            'note': 'Peaceful Bosphorus walk on Asian side',
            'hours': 'All day',
            'tip': 'Less touristy than European side'
        },
        {
            'name': 'Yıldız Park',
            'type': 'park',
            'cost': 0,
            'location': 'Beşiktaş',
            'note': 'Large park with palace grounds',
            'hours': 'All day',
            'tip': 'Free entry, palace museums extra'
        },
        {
            'name': 'Karaköy & Galata Walking',
            'type': 'area',
            'cost': 0,
            'location': 'Karaköy',
            'note': 'Trendy neighborhood exploration',
            'hours': 'All day',
            'tip': 'Street art and vintage shops'
        },
        {
            'name': 'Bebek Bay Walk',
            'type': 'area',
            'cost': 0,
            'location': 'Bebek',
            'note': 'Upscale neighborhood waterfront',
            'hours': 'All day',
            'tip': 'Window shopping and people watching'
        },
        {
            'name': 'Eyüp Sultan Mosque Area',
            'type': 'religious',
            'cost': 0,
            'location': 'Eyüp',
            'note': 'Important mosque, free to visit respectfully',
            'hours': 'Outside prayer times',
            'tip': 'Take cable car to Pierre Loti (₺15)'
        },
        {
            'name': 'Spice Bazaar Browse',
            'type': 'market',
            'cost': 0,
            'location': 'Eminönü',
            'note': 'Free to walk and enjoy aromas',
            'hours': 'Mon-Sat 08:00-19:30',
            'tip': 'Samples are free, buying optional'
        },
        {
            'name': 'Sunday Flee Market Kadıköy',
            'type': 'market',
            'cost': 0,
            'location': 'Kadıköy',
            'note': 'Browse vintage items and antiques',
            'hours': 'Sunday 10:00-19:00',
            'tip': 'Bargaining expected if buying'
        },
        {
            'name': 'Rumeli Fortress Exterior',
            'type': 'historical',
            'cost': 0,
            'location': 'Sarıyer',
            'note': 'View from outside (inside ₺50)',
            'hours': 'Always visible',
            'tip': 'Great photo spot from park below'
        },
        {
            'name': 'Maiden\'s Tower View',
            'type': 'landmark',
            'cost': 0,
            'location': 'Üsküdar',
            'note': 'View from shore (boat ride extra)',
            'hours': 'All day',
            'tip': 'Best photos from Salacak coast'
        },
        {
            'name': 'Bosphorus Bridge View',
            'type': 'landmark',
            'cost': 0,
            'location': 'Ortaköy',
            'note': 'Stunning bridge views from shore',
            'hours': 'All day, magical at night',
            'tip': 'Free light show every evening'
        },
        {
            'name': 'Miniaturk Exterior Walk',
            'type': 'area',
            'cost': 0,
            'location': 'Sütlüce',
            'note': 'Park area around miniatures museum',
            'hours': 'All day',
            'tip': 'Museum entry ₺100, outside free'
        }
    ]
    
    # Budget-friendly restaurants and eateries
    BUDGET_EATS = [
        {
            'name': 'Tarihi Eminönü Balık Ekmek',
            'type': 'street_food',
            'avg_price': 40,
            'dish': 'Fish sandwich (balık ekmek)',
            'location': 'Eminönü waterfront',
            'description': 'Iconic Istanbul experience - fresh fish sandwich on boats',
            'tip': 'Ask for extra onions and lemon'
        },
        {
            'name': 'Lezzet-i Şark',
            'type': 'restaurant',
            'avg_price': 80,
            'specialty': 'Traditional Turkish breakfast',
            'location': 'Kadıköy',
            'description': 'Generous breakfast spread with unlimited tea',
            'tip': 'Go early on weekends to avoid lines'
        },
        {
            'name': 'Kızılkayalar',
            'type': 'restaurant',
            'avg_price': 90,
            'specialty': 'Kebabs and grills',
            'location': 'Sultanahmet',
            'description': 'Family-run, authentic Turkish food',
            'tip': 'Try the Adana kebab, locals\' favorite'
        },
        {
            'name': 'Şekerci Cafer Erol',
            'type': 'cafe',
            'avg_price': 60,
            'specialty': 'Turkish pastries and tea',
            'location': 'Beyoğlu',
            'description': 'Historic patisserie since 1807',
            'tip': 'Try künefe - hot cheese dessert'
        },
        {
            'name': 'Çiya Sofrası',
            'type': 'restaurant',
            'avg_price': 120,
            'specialty': 'Anatolian cuisine',
            'location': 'Kadıköy',
            'description': 'Regional Turkish dishes, rotating menu',
            'tip': 'Ask server for daily specials'
        },
        {
            'name': 'Kumpircim',
            'type': 'street_food',
            'avg_price': 70,
            'dish': 'Kumpir (loaded baked potato)',
            'location': 'Ortaköy',
            'description': 'Turkish street food specialty',
            'tip': 'Don\'t overload with toppings'
        },
        {
            'name': 'Tarihi Karaköy Balıkçısı',
            'type': 'restaurant',
            'avg_price': 150,
            'specialty': 'Seafood',
            'location': 'Karaköy',
            'description': 'Fresh fish at reasonable prices',
            'tip': 'Fish of the day is best value'
        },
        {
            'name': 'Simit Sarayı',
            'type': 'chain_cafe',
            'avg_price': 50,
            'specialty': 'Simit and Turkish tea',
            'location': 'Everywhere (chain)',
            'description': 'Quick, cheap, reliable Turkish snacks',
            'tip': 'Perfect for breakfast on the go'
        },
        {
            'name': 'Hocapaşa Pidecisi',
            'type': 'restaurant',
            'avg_price': 85,
            'specialty': 'Pide (Turkish pizza)',
            'location': 'Sirkeci',
            'description': 'Traditional pide house since 1976',
            'tip': 'Kıymalı (meat) pide is signature'
        },
        {
            'name': 'Dürümzade',
            'type': 'fast_food',
            'avg_price': 60,
            'dish': 'Dürüm wraps',
            'location': 'Beyoğlu',
            'description': 'Famous wrap spot, locals queue here',
            'tip': 'Spicy version is authentic'
        },
        {
            'name': 'Tarihi Sultanahmet Köftecisi',
            'type': 'restaurant',
            'avg_price': 100,
            'specialty': 'Köfte (meatballs)',
            'location': 'Sultanahmet',
            'description': 'Simple menu, perfected over decades',
            'tip': 'Add white beans and rice'
        },
        {
            'name': 'Kanaat Lokantası',
            'type': 'restaurant',
            'avg_price': 95,
            'specialty': 'Home-style Turkish food',
            'location': 'Üsküdar',
            'description': 'Historic lokanta, authentic atmosphere',
            'tip': 'Try their famous rice pudding'
        },
        {
            'name': 'Midpoint',
            'type': 'cafe',
            'avg_price': 80,
            'specialty': 'Sandwiches and coffee',
            'location': 'Multiple locations',
            'description': 'Western-style sandwiches, Turkish prices',
            'tip': 'Student-friendly study spot'
        },
        {
            'name': 'Fıstıkağacı',
            'type': 'restaurant',
            'avg_price': 110,
            'specialty': 'Turkish breakfast and mains',
            'location': 'Cihangir',
            'description': 'Bohemian neighborhood gem',
            'tip': 'Breakfast until 16:00'
        },
        {
            'name': 'Pideci Selim',
            'type': 'restaurant',
            'avg_price': 75,
            'specialty': 'Pide',
            'location': 'Kadıköy',
            'description': 'Local favorite, no frills',
            'tip': 'Mixed pide good for sharing'
        },
        {
            'name': 'Saray Muhallebicisi',
            'type': 'cafe',
            'avg_price': 65,
            'specialty': 'Turkish desserts',
            'location': 'Multiple locations',
            'description': 'Famous for tavuk göğsü (chicken pudding)',
            'tip': 'Try the unusual - it\'s delicious'
        },
        {
            'name': 'Karaköy Güllüoğlu',
            'type': 'bakery',
            'avg_price': 70,
            'specialty': 'Baklava',
            'location': 'Karaköy',
            'description': 'Best baklava in Istanbul',
            'tip': 'Fıstıklı (pistachio) is premium'
        },
        {
            'name': 'Bayramoğlu Döner',
            'type': 'fast_food',
            'avg_price': 65,
            'specialty': 'Döner kebab',
            'location': 'Karaköy',
            'description': 'High-quality döner, not tourist trap',
            'tip': 'Ayran (yogurt drink) essential'
        },
        {
            'name': 'Vefa Bozacısı',
            'type': 'historic_cafe',
            'avg_price': 40,
            'specialty': 'Boza (fermented drink)',
            'location': 'Vefa',
            'description': 'Historic drink shop since 1876',
            'tip': 'Acquired taste, very traditional'
        },
        {
            'name': 'Hafız Mustafa',
            'type': 'patisserie',
            'avg_price': 80,
            'specialty': 'Turkish sweets',
            'location': 'Multiple locations',
            'description': 'Modern Turkish dessert chain',
            'tip': 'Try lokum (Turkish delight)'
        },
        {
            'name': 'Köşkeroğlu',
            'type': 'restaurant',
            'avg_price': 90,
            'specialty': 'Kebabs',
            'location': 'Beşiktaş',
            'description': 'Local kebab house, generous portions',
            'tip': 'İskender kebab recommended'
        },
        {
            'name': 'Namli Gurme',
            'type': 'market_cafe',
            'avg_price': 120,
            'specialty': 'Turkish charcuterie',
            'location': 'Karaköy',
            'description': 'Upscale market with cheap takeaway',
            'tip': 'Get sandwich to go, eat at park'
        },
        {
            'name': 'Cafe Privato',
            'type': 'cafe',
            'avg_price': 85,
            'specialty': 'Breakfast and pastries',
            'location': 'Kadıköy',
            'description': 'Cozy cafe with garden',
            'tip': 'Moda location has best vibe'
        },
        {
            'name': 'Meşhur Adanalı Ciğerci',
            'type': 'restaurant',
            'avg_price': 95,
            'specialty': 'Liver kebabs (ciğer)',
            'location': 'Beyoğlu',
            'description': 'Specialty: grilled lamb liver',
            'tip': 'Not for everyone, very authentic'
        },
        {
            'name': 'Dondurmacı Nazmi',
            'type': 'ice_cream',
            'avg_price': 45,
            'specialty': 'Traditional Turkish ice cream',
            'location': 'Kadıköy',
            'description': 'Local favorite ice cream shop',
            'tip': 'Salep ice cream is signature'
        }
    ]
    
    def __init__(self):
        """Initialize price filter service"""
        logger.info("Price Filter Service initialized")
    
    def detect_budget_query(self, query: str) -> Optional[str]:
        """
        Detect budget-related keywords in query
        Returns: budget level or None
        """
        query_lower = query.lower()
        
        # Free/budget keywords
        if any(word in query_lower for word in ['free', 'ücretsiz', 'bedava', 'no cost', 'without paying']):
            return 'free'
        
        if any(word in query_lower for word in ['cheap', 'budget', 'ucuz', 'ekonomik', 'inexpensive', 'affordable']):
            return 'budget'
        
        # Expensive/luxury keywords
        if any(word in query_lower for word in ['expensive', 'luxury', 'pahalı', 'lüks', 'fine dining', 'upscale', 'high-end']):
            return 'luxury'
        
        if any(word in query_lower for word in ['mid-range', 'moderate', 'orta', 'reasonable']):
            return 'moderate'
        
        return None
    
    def filter_by_budget(self, items: List[Dict], budget_level: str) -> List[Dict]:
        """
        Filter items by budget category
        """
        if budget_level not in self.BUDGET_RANGES:
            return items
        
        min_price, max_price = self.BUDGET_RANGES[budget_level]
        
        filtered = []
        for item in items:
            price = item.get('price', item.get('avg_price', 0))
            if min_price <= price <= max_price:
                filtered.append(item)
        
        return filtered
    
    def get_free_attractions(self, limit: int = 10) -> List[Dict]:
        """Get list of free attractions"""
        return self.FREE_ATTRACTIONS[:limit]
    
    def get_budget_eats(self, limit: int = 10) -> List[Dict]:
        """Get list of budget-friendly restaurants"""
        return self.BUDGET_EATS[:limit]
    
    def format_free_attractions_response(self, attractions: List[Dict] = None) -> str:
        """Format free attractions into readable response"""
        if attractions is None:
            attractions = self.get_free_attractions(15)
        
        response = "🆓 **Free Things to Do in Istanbul**\n\n"
        response += "Here are amazing experiences that won't cost you anything:\n\n"
        
        for i, attr in enumerate(attractions, 1):
            response += f"**{i}. {attr['name']}**\n"
            response += f"📍 {attr['location']}\n"
            response += f"ℹ️ {attr['note']}\n"
            response += f"⏰ {attr['hours']}\n"
            response += f"💡 Tip: {attr['tip']}\n\n"
        
        response += "\n**💰 Saving Money:**\n"
        response += "- Get an Istanbulkart for public transport (save 50%)\n"
        response += "- Many mosques are free (dress respectfully)\n"
        response += "- Parks and coastal walks cost nothing\n"
        response += "- Street food is cheap and delicious\n"
        
        return response
    
    def format_budget_eats_response(self, restaurants: List[Dict] = None) -> str:
        """Format budget restaurants into readable response"""
        if restaurants is None:
            restaurants = self.get_budget_eats(12)
        
        response = "₺ **Budget-Friendly Restaurants in Istanbul**\n\n"
        response += "Delicious authentic food at local prices:\n\n"
        
        for i, rest in enumerate(restaurants, 1):
            symbol = self._get_price_symbol_from_amount(rest['avg_price'])
            response += f"**{i}. {rest['name']}** {symbol}\n"
            response += f"📍 {rest['location']}\n"
            # Handle both 'specialty' and 'dish' fields
            food_item = rest.get('specialty') or rest.get('dish', 'Various dishes')
            response += f"🍽️ {food_item}\n"
            response += f"💰 Avg: ₺{rest['avg_price']} per person\n"
            response += f"ℹ️ {rest['description']}\n"
            response += f"💡 Tip: {rest['tip']}\n\n"
        
        response += "\n**💡 Money-Saving Tips:**\n"
        response += "- Lunch menus (öğle yemeği) are cheaper than dinner\n"
        response += "- 'Lokanta' style restaurants are authentic and affordable\n"
        response += "- Street food (simit, balık ekmek, kumpir) is delicious\n"
        response += "- Avoid tourist traps in Sultanahmet\n"
        response += "- Turkish tea is cheap everywhere (₺5-10)\n"
        
        return response
    
    def _get_price_symbol_from_amount(self, amount: float) -> str:
        """Get price symbol based on amount"""
        if amount == 0:
            return '🆓'
        elif amount < 100:
            return '₺'
        elif amount < 300:
            return '₺₺'
        elif amount < 600:
            return '₺₺₺'
        else:
            return '₺₺₺₺'
    
    def add_price_context_to_response(self, response: str, price_level: str) -> str:
        """Add price level indicator to response"""
        symbol = self.PRICE_SYMBOLS.get(price_level, '')
        
        headers = {
            'free': '🆓 **Free Options**',
            'budget': '₺ **Budget-Friendly Options**',
            'moderate': '₺₺ **Mid-Range Selections**',
            'upscale': '₺₺₺ **Upscale Choices**',
            'luxury': '₺₺₺₺ **Luxury Experiences**'
        }
        
        header = headers.get(price_level, '')
        if header:
            return f"{header}\n\n{response}"
        
        return response
    
    def get_budget_summary(self, price_level: str) -> str:
        """Get budget range summary"""
        min_price, max_price = self.BUDGET_RANGES.get(price_level, (0, 0))
        
        if price_level == 'free':
            return "No cost - completely free!"
        elif price_level == 'budget':
            return f"Under ₺{max_price} per person"
        elif max_price == float('inf'):
            return f"₺{min_price}+ per person"
        else:
            return f"₺{min_price}-₺{max_price} per person"


# Global instance
_price_filter_service = None

def get_price_filter_service() -> PriceFilterService:
    """Get or create price filter service singleton"""
    global _price_filter_service
    if _price_filter_service is None:
        _price_filter_service = PriceFilterService()
    return _price_filter_service
