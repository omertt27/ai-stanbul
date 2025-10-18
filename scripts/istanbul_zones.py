"""Istanbul geographic zones for comprehensive POI coverage"""

ISTANBUL_ZONES = [
    {
        'name': 'Sultanahmet',
        'center': (41.0082, 28.9784),
        'radius': 2000,
        'description': 'Historical peninsula - museums, mosques, palaces'
    },
    {
        'name': 'Taksim/Beyoğlu',
        'center': (41.0369, 28.9850),
        'radius': 2000,
        'description': 'Modern center - shopping, nightlife, restaurants'
    },
    {
        'name': 'Kadıköy',
        'center': (40.9904, 29.0263),
        'radius': 2000,
        'description': 'Asian side - trendy cafes, markets, culture'
    },
    {
        'name': 'Beşiktaş/Ortaköy',
        'center': (41.0428, 29.0060),
        'radius': 2000,
        'description': 'Bosphorus - waterfront, mosques, palaces'
    },
    {
        'name': 'Üsküdar',
        'center': (41.0225, 29.0144),
        'radius': 1500,
        'description': 'Asian Bosphorus - mosques, viewpoints'
    },
    {
        'name': 'Fatih',
        'center': (41.0186, 28.9497),
        'radius': 1500,
        'description': 'Historic markets, churches, mosques'
    },
    {
        'name': 'Galata',
        'center': (41.0255, 28.9742),
        'radius': 1000,
        'description': 'Galata Tower, cafes, art galleries'
    },
    {
        'name': 'Balat/Fener',
        'center': (41.0309, 28.9485),
        'radius': 1000,
        'description': 'Colorful historic neighborhoods'
    }
]

POI_CATEGORIES = [
    'museum',
    'art_gallery',
    'tourist_attraction',
    'restaurant',
    'cafe',
    'park',
    'mosque',
    'church',
    'historical_landmark',
    'shopping_mall',
    'viewpoint',
    'market',
    'bar',
    'night_club',
    'spa'
]
