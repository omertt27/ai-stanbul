"""
Comprehensive Price Database for Istanbul Venues
Detailed pricing information for attractions, restaurants, and activities
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PriceInfo:
    """Price information for a venue"""
    min_price: float
    max_price: float
    currency: str = "TRY"
    notes: str = ""
    price_per: str = "person"  # person, group, item, entrance


# ==================== ATTRACTIONS PRICING ====================

ATTRACTIONS_PRICING = {
    # Museums & Historical Sites
    'Topkapi Palace': {
        'price': 650,
        'budget_level': 'upscale',
        'additional_costs': {
            'Harem Section': 350,
            'Audio Guide': 150
        },
        'notes': 'Museum Pass Istanbul covers entrance',
        'free_days': None,
        'discounts': {
            'students': 200,
            'children_under_12': 'free'
        }
    },
    'Hagia Sophia': {
        'price': 100,
        'budget_level': 'budget',
        'additional_costs': {},
        'notes': 'Entry via Hagia Sophia History & Experience Museum',
        'free_days': None,
        'discounts': {}
    },
    'Blue Mosque': {
        'price': 0,
        'budget_level': 'free',
        'additional_costs': {},
        'notes': 'Free entry, dress modestly',
        'free_days': 'Always free',
        'discounts': {}
    },
    'Basilica Cistern': {
        'price': 450,
        'budget_level': 'moderate',
        'additional_costs': {
            'Audio Guide': 100
        },
        'notes': 'Online booking recommended',
        'free_days': None,
        'discounts': {
            'students': 150
        }
    },
    'Dolmabahce Palace': {
        'price': 650,
        'budget_level': 'upscale',
        'additional_costs': {
            'Harem Section': 450
        },
        'notes': 'Museum Pass covers entrance',
        'free_days': None,
        'discounts': {
            'students': 200
        }
    },
    'Istanbul Modern': {
        'price': 350,
        'budget_level': 'moderate',
        'additional_costs': {},
        'notes': 'Contemporary art museum',
        'free_days': 'Thursday 18:00-20:00',
        'discounts': {
            'students': 175
        }
    },
    'Galata Tower': {
        'price': 450,
        'budget_level': 'moderate',
        'additional_costs': {},
        'notes': 'Book online to skip queues',
        'free_days': None,
        'discounts': {}
    },
    'Rumeli Fortress': {
        'price': 150,
        'budget_level': 'moderate',
        'additional_costs': {},
        'notes': 'Beautiful Bosphorus views',
        'free_days': None,
        'discounts': {}
    },
    'Chora Museum': {
        'price': 450,
        'budget_level': 'moderate',
        'additional_costs': {},
        'notes': 'Byzantine mosaics',
        'free_days': None,
        'discounts': {
            'students': 150
        }
    },
    'Archaeological Museums': {
        'price': 300,
        'budget_level': 'moderate',
        'additional_costs': {},
        'notes': '3 museums in complex',
        'free_days': None,
        'discounts': {
            'students': 100
        }
    },
    'Miniaturk': {
        'price': 200,
        'budget_level': 'moderate',
        'additional_costs': {},
        'notes': 'Miniature park, family friendly',
        'free_days': None,
        'discounts': {
            'students': 100,
            'children': 100
        }
    },
    
    # Experiences & Activities
    'Bosphorus Cruise (Public Ferry)': {
        'price': 50,
        'budget_level': 'budget',
        'additional_costs': {},
        'notes': 'Cheapest way to see Bosphorus',
        'free_days': None,
        'discounts': {
            'istanbulkart': 30
        }
    },
    'Bosphorus Cruise (Tourist)': {
        'price': 800,
        'budget_level': 'upscale',
        'additional_costs': {
            'Dinner Cruise': 1500,
            'Sunset Cruise': 1000
        },
        'notes': 'Private boat tours',
        'free_days': None,
        'discounts': {}
    },
    'Turkish Bath (Hammam) - Budget': {
        'price': 450,
        'budget_level': 'moderate',
        'additional_costs': {
            'Massage': 350,
            'Premium Package': 700
        },
        'notes': 'Local neighborhood hammams',
        'free_days': None,
        'discounts': {}
    },
    'Turkish Bath (Hammam) - Luxury': {
        'price': 1200,
        'budget_level': 'luxury',
        'additional_costs': {
            'Premium Massage': 800,
            'Full Package': 2000
        },
        'notes': 'Historic hammams with spa services',
        'free_days': None,
        'discounts': {}
    },
    'Princes Islands Ferry': {
        'price': 60,
        'budget_level': 'budget',
        'additional_costs': {
            'Bicycle Rental': 250,
            'Electric Cart Tour': 900
        },
        'notes': 'Walking is free on islands',
        'free_days': None,
        'discounts': {
            'istanbulkart': 30
        }
    },
    'Istiklal Street Walking Tour': {
        'price': 0,
        'budget_level': 'free',
        'additional_costs': {
            'Historic Tram': 35
        },
        'notes': 'Self-guided, completely free',
        'free_days': 'Always free',
        'discounts': {}
    },
    'Grand Bazaar Visit': {
        'price': 0,
        'budget_level': 'free',
        'additional_costs': {},
        'notes': 'Shopping optional',
        'free_days': 'Always free',
        'discounts': {}
    },
    'Spice Bazaar Visit': {
        'price': 0,
        'budget_level': 'free',
        'additional_costs': {},
        'notes': 'Free to browse',
        'free_days': 'Always free',
        'discounts': {}
    }
}


# ==================== RESTAURANTS PRICING ====================

RESTAURANTS_PRICING = {
    # Street Food & Budget Eats
    'Simit (Turkish Bagel)': {
        'price': 25,
        'budget_level': 'budget',
        'location': 'Everywhere',
        'type': 'street_food',
        'notes': 'Quick breakfast or snack'
    },
    'Balık Ekmek (Fish Sandwich)': {
        'price': 100,
        'budget_level': 'budget',
        'location': 'Eminönü',
        'type': 'street_food',
        'notes': 'Must-try Istanbul experience'
    },
    'Kumpir (Loaded Potato)': {
        'price': 150,
        'budget_level': 'budget',
        'location': 'Ortaköy',
        'type': 'street_food',
        'notes': 'Filling and customizable'
    },
    'Döner Kebab': {
        'price': 120,
        'budget_level': 'budget',
        'location': 'Everywhere',
        'type': 'fast_food',
        'notes': 'Quick and tasty'
    },
    'Midye Dolma (Stuffed Mussels)': {
        'price': 15,
        'budget_level': 'budget',
        'location': 'Street vendors',
        'type': 'street_food',
        'notes': 'Sold individually, try 5-10'
    },
    
    # Budget Restaurants
    'Çiya Sofrası': {
        'price': 350,
        'budget_level': 'moderate',
        'location': 'Kadıköy',
        'type': 'restaurant',
        'notes': 'Anatolian cuisine, highly rated'
    },
    'Tarihi Sultanahmet Köftecisi': {
        'price': 280,
        'budget_level': 'moderate',
        'location': 'Sultanahmet',
        'type': 'restaurant',
        'notes': 'Famous meatballs since 1920'
    },
    'Kanaat Lokantası': {
        'price': 250,
        'budget_level': 'moderate',
        'location': 'Üsküdar',
        'type': 'lokanta',
        'notes': 'Traditional home-style cooking'
    },
    'Hocapaşa Pidecisi': {
        'price': 200,
        'budget_level': 'moderate',
        'location': 'Sirkeci',
        'type': 'restaurant',
        'notes': 'Turkish pizza specialist'
    },
    'Dürümzade': {
        'price': 150,
        'budget_level': 'moderate',
        'location': 'Beyoğlu',
        'type': 'fast_food',
        'notes': 'Best dürüm wraps, always crowded'
    },
    
    # Mid-Range & Upscale Restaurants
    'Mikla': {
        'price': 1800,
        'budget_level': 'luxury',
        'location': 'Beyoğlu',
        'type': 'fine_dining',
        'notes': 'Rooftop, Michelin Guide, reservations required'
    },
    'Neolokal': {
        'price': 1500,
        'budget_level': 'luxury',
        'location': 'Karaköy',
        'type': 'fine_dining',
        'notes': 'Modern Turkish, Michelin Guide'
    },
    'Karaköy Lokantası': {
        'price': 550,
        'budget_level': 'upscale',
        'location': 'Karaköy',
        'type': 'restaurant',
        'notes': 'Upscale meyhane experience'
    },
    'Asitane': {
        'price': 900,
        'budget_level': 'upscale',
        'location': 'Edirnekapı',
        'type': 'restaurant',
        'notes': 'Ottoman palace cuisine'
    },
    'Balikçi Sabahattin': {
        'price': 800,
        'budget_level': 'upscale',
        'location': 'Sultanahmet',
        'type': 'seafood',
        'notes': 'Fresh seafood, historic setting'
    },
    'Namli Gurme': {
        'price': 450,
        'budget_level': 'moderate',
        'location': 'Karaköy',
        'type': 'deli_cafe',
        'notes': 'Turkish charcuterie and cafe'
    },
    'Beyti': {
        'price': 700,
        'budget_level': 'upscale',
        'location': 'Florya',
        'type': 'restaurant',
        'notes': 'Famous kebab house since 1945'
    },
    
    # Cafes & Desserts
    'Turkish Tea': {
        'price': 30,
        'budget_level': 'budget',
        'location': 'Everywhere',
        'type': 'beverage',
        'notes': 'Standard at any cafe'
    },
    'Turkish Coffee': {
        'price': 75,
        'budget_level': 'budget',
        'location': 'Everywhere',
        'type': 'beverage',
        'notes': 'Traditional brewing'
    },
    'Hafız Mustafa': {
        'price': 200,
        'budget_level': 'moderate',
        'location': 'Multiple locations',
        'type': 'patisserie',
        'notes': 'Turkish sweets and desserts'
    },
    'Karaköy Güllüoğlu': {
        'price': 180,
        'budget_level': 'moderate',
        'location': 'Karaköy',
        'type': 'bakery',
        'notes': 'Best baklava in Istanbul'
    },
    'Saray Muhallebicisi': {
        'price': 150,
        'budget_level': 'moderate',
        'location': 'Multiple locations',
        'type': 'dessert',
        'notes': 'Turkish puddings and desserts'
    },
    'Mandabatmaz': {
        'price': 80,
        'budget_level': 'budget',
        'location': 'Beyoğlu',
        'type': 'cafe',
        'notes': 'Famous for thick Turkish coffee'
    }
}


# ==================== TRANSPORTATION PRICING ====================

TRANSPORTATION_PRICING = {
    'Istanbulkart (Card)': {
        'price': 100,
        'budget_level': 'budget',
        'notes': 'Rechargeable transport card, one-time cost',
        'refundable': False
    },
    'Single Ride (Token)': {
        'price': 35,
        'budget_level': 'budget',
        'notes': 'More expensive than Istanbulkart',
        'refundable': False
    },
    'Metro/Tram/Bus (Istanbulkart)': {
        'price': 17.70,
        'budget_level': 'budget',
        'notes': 'Per ride with Istanbulkart',
        'refundable': False
    },
    'Ferry (Istanbulkart)': {
        'price': 17.70,
        'budget_level': 'budget',
        'notes': 'City ferries with card',
        'refundable': False
    },
    'Taxi (Base Fare)': {
        'price': 100,
        'budget_level': 'moderate',
        'notes': 'Starting fare + per km',
        'refundable': False
    },
    'Airport Shuttle (Havaist)': {
        'price': 180,
        'budget_level': 'moderate',
        'notes': 'Official airport buses',
        'refundable': False
    },
    'Metro to Airport': {
        'price': 350,
        'budget_level': 'upscale',
        'notes': 'M11 line to New Airport',
        'refundable': False
    },
    'Taxi to Airport': {
        'price': 1800,
        'budget_level': 'luxury',
        'notes': 'From Sultanahmet area',
        'refundable': False
    }
}


# ==================== BUDGET CATEGORIES ====================

BUDGET_CATEGORIES = {
    'free': {
        'range': (0, 0),
        'symbol': '🆓',
        'description': 'Completely free',
        'examples': ['Parks', 'Mosques', 'Walking tours', 'Viewpoints']
    },
    'budget': {
        'range': (0, 150),
        'symbol': '₺',
        'description': 'Very affordable',
        'examples': ['Street food', 'Public transport', 'Tea/Coffee', 'Simple meals']
    },
    'moderate': {
        'range': (150, 500),
        'symbol': '₺₺',
        'description': 'Mid-range pricing',
        'examples': ['Museums', 'Local restaurants', 'Casual dining', 'Standard tours']
    },
    'upscale': {
        'range': (500, 1000),
        'symbol': '₺₺₺',
        'description': 'Premium experiences',
        'examples': ['Fine dining', 'Luxury hammams', 'Premium attractions', 'Private tours']
    },
    'luxury': {
        'range': (1000, float('inf')),
        'symbol': '₺₺₺₺',
        'description': 'Top-tier experiences',
        'examples': ['Michelin restaurants', 'Luxury cruises', 'VIP experiences', 'Palace tours']
    }
}


# ==================== UTILITY FUNCTIONS ====================

def get_venue_price(venue_name: str, category: str = 'attractions') -> Optional[Dict]:
    """Get pricing information for a specific venue"""
    databases = {
        'attractions': ATTRACTIONS_PRICING,
        'restaurants': RESTAURANTS_PRICING,
        'transportation': TRANSPORTATION_PRICING
    }
    
    db = databases.get(category, {})
    return db.get(venue_name)


def get_budget_level_venues(budget_level: str, category: str = 'all') -> List[str]:
    """Get all venues matching a budget level"""
    venues = []
    
    databases = {
        'attractions': ATTRACTIONS_PRICING,
        'restaurants': RESTAURANTS_PRICING,
        'transportation': TRANSPORTATION_PRICING
    }
    
    if category == 'all':
        for db in databases.values():
            for name, info in db.items():
                if info.get('budget_level') == budget_level:
                    venues.append(name)
    else:
        db = databases.get(category, {})
        for name, info in db.items():
            if info.get('budget_level') == budget_level:
                venues.append(name)
    
    return venues


def get_price_range_venues(min_price: float, max_price: float, category: str = 'all') -> List[Dict]:
    """Get venues within a specific price range"""
    venues = []
    
    databases = {
        'attractions': ATTRACTIONS_PRICING,
        'restaurants': RESTAURANTS_PRICING,
        'transportation': TRANSPORTATION_PRICING
    }
    
    if category == 'all':
        search_dbs = databases.items()
    else:
        search_dbs = [(category, databases.get(category, {}))]
    
    for cat_name, db in search_dbs:
        for name, info in db.items():
            price = info.get('price', 0)
            if min_price <= price <= max_price:
                venues.append({
                    'name': name,
                    'price': price,
                    'category': cat_name,
                    'budget_level': info.get('budget_level'),
                    'info': info
                })
    
    return venues


def calculate_daily_budget(budget_level: str) -> Dict[str, int]:
    """Calculate suggested daily budget breakdown (2025 prices)"""
    budgets = {
        'budget': {
            'accommodation': 800,
            'food': 500,
            'transport': 150,
            'attractions': 300,
            'total': 1750
        },
        'moderate': {
            'accommodation': 2500,
            'food': 1200,
            'transport': 300,
            'attractions': 800,
            'total': 4800
        },
        'upscale': {
            'accommodation': 5000,
            'food': 2500,
            'transport': 600,
            'attractions': 1500,
            'total': 9600
        },
        'luxury': {
            'accommodation': 10000,
            'food': 5000,
            'transport': 1500,
            'attractions': 3000,
            'total': 19500
        }
    }
    
    return budgets.get(budget_level, budgets['moderate'])


# ==================== EXPORT ====================

__all__ = [
    'ATTRACTIONS_PRICING',
    'RESTAURANTS_PRICING',
    'TRANSPORTATION_PRICING',
    'BUDGET_CATEGORIES',
    'get_venue_price',
    'get_budget_level_venues',
    'get_price_range_venues',
    'calculate_daily_budget'
]
