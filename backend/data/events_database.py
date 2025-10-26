"""
Events Database for Istanbul AI System
Comprehensive database of recurring, seasonal, and cultural events in Istanbul
Part of Phase 2D: Events Service
Supports Turkish and English
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import calendar
import json
from pathlib import Path


# ==================== EVENT CATEGORIES ====================

EVENT_TYPES = {
    'concert': ['Konser', 'Concert'],
    'festival': ['Festival', 'Festival'],
    'exhibition': ['Sergi', 'Exhibition'],
    'theater': ['Tiyatro', 'Theater'],
    'sports': ['Spor', 'Sports'],
    'market': ['Pazar', 'Market'],
    'cultural': ['Kültürel', 'Cultural'],
    'religious': ['Dini', 'Religious'],
    'nightlife': ['Gece Hayatı', 'Nightlife'],
}


# ==================== RECURRING WEEKLY EVENTS ====================

WEEKLY_EVENTS = [
    {
        'id': 'friday_besiktas_market',
        'name': {
            'tr': 'Beşiktaş Cumartesi Pazarı',
            'en': 'Beşiktaş Saturday Market'
        },
        'type': 'market',
        'day_of_week': 5,  # Saturday
        'description': {
            'tr': 'İstanbul\'un en büyük ve en renkli pazarlarından biri. Taze meyve, sebze, giyim ve ev eşyaları.',
            'en': 'One of Istanbul\'s largest and most colorful markets. Fresh produce, clothing, and household items.'
        },
        'location': 'Beşiktaş',
        'time': '08:00-18:00',
        'cost': {
            'tr': 'Ücretsiz giriş',
            'en': 'Free entry'
        },
        'tips': {
            'tr': 'Erken gidin, öğleden sonra çok kalabalık olur. Pazarlık yapın!',
            'en': 'Go early, gets crowded after noon. Bargaining is expected!'
        }
    },
    {
        'id': 'friday_kadikoy_market',
        'name': {
            'tr': 'Kadıköy Salı Pazarı',
            'en': 'Kadıköy Tuesday Market'
        },
        'type': 'market',
        'day_of_week': 1,  # Tuesday
        'description': {
            'tr': 'Asya yakasının popüler pazarı. Organik ürünler ve yerel lezzetler.',
            'en': 'Popular market on the Asian side. Organic products and local delicacies.'
        },
        'location': 'Kadıköy',
        'time': '08:00-17:00',
        'cost': {
            'tr': 'Ücretsiz giriş',
            'en': 'Free entry'
        },
        'tips': {
            'tr': 'Organik ürünler için doğrudan üreticileri arayın',
            'en': 'Look for organic products directly from producers'
        }
    },
    {
        'id': 'friday_live_music',
        'name': {
            'tr': 'Cuma Canlı Müzik - Beyoğlu',
            'en': 'Friday Live Music - Beyoğlu'
        },
        'type': 'nightlife',
        'day_of_week': 4,  # Friday
        'description': {
            'tr': 'Beyoğlu\'ndaki birçok mekan cuma geceleri canlı müzik organizasyonu yapıyor.',
            'en': 'Many venues in Beyoğlu host live music on Friday nights.'
        },
        'location': 'Beyoğlu, İstiklal Caddesi',
        'time': '21:00-02:00',
        'cost': {
            'tr': '₺200-500',
            'en': '₺200-500'
        },
        'tips': {
            'tr': 'Rezervasyon yaptırın, çok popüler!',
            'en': 'Make reservations, very popular!'
        }
    },
    {
        'id': 'sunday_bosphorus_cruise',
        'name': {
            'tr': 'Pazar Boğaz Turu',
            'en': 'Sunday Bosphorus Cruise'
        },
        'type': 'cultural',
        'day_of_week': 6,  # Sunday
        'description': {
            'tr': 'Haftalık halk tekneleri ile Boğaz turu. Yerel halkın favorisi.',
            'en': 'Weekly public ferry Bosphorus tour. Local favorite.'
        },
        'location': 'Eminönü İskelesi',
        'time': '10:00-16:00',
        'cost': {
            'tr': '₺50-100 (İstanbulkart)',
            'en': '₺50-100 (Istanbulkart)'
        },
        'tips': {
            'tr': 'Güzel havada dış güvertede oturun',
            'en': 'Sit on the outdoor deck in good weather'
        }
    },
]


# ==================== SEASONAL/ANNUAL EVENTS ====================

SEASONAL_EVENTS = [
    {
        'id': 'istanbul_music_festival',
        'name': {
            'tr': 'İstanbul Müzik Festivali',
            'en': 'Istanbul Music Festival'
        },
        'type': 'festival',
        'month': 6,  # June
        'duration_days': 30,
        'description': {
            'tr': 'Klasik müzikten caza, dünya müziğinden elektroniğe geniş yelpaze.',
            'en': 'Wide range from classical to jazz, world music to electronic.'
        },
        'location': 'Çeşitli mekanlar / Various venues',
        'cost': {
            'tr': '₺150-800 (konser başına)',
            'en': '₺150-800 (per concert)'
        },
        'website': 'muzik.iksv.org',
        'tips': {
            'tr': 'Biletler erken tükenir, önceden alın',
            'en': 'Tickets sell out early, buy in advance'
        }
    },
    {
        'id': 'istanbul_film_festival',
        'name': {
            'tr': 'İstanbul Film Festivali',
            'en': 'Istanbul Film Festival'
        },
        'type': 'festival',
        'month': 4,  # April
        'duration_days': 15,
        'description': {
            'tr': 'Türkiye\'nin en önemli film festivali. Uluslararası ve yerel filmler.',
            'en': 'Turkey\'s most important film festival. International and local films.'
        },
        'location': 'Çeşitli sinemalar / Various cinemas',
        'cost': {
            'tr': '₺100-200 (film başına)',
            'en': '₺100-200 (per film)'
        },
        'website': 'film.iksv.org',
        'tips': {
            'tr': 'Festival kartı alın, daha ekonomik',
            'en': 'Get a festival pass, more economical'
        }
    },
    {
        'id': 'istanbul_biennial',
        'name': {
            'tr': 'İstanbul Bienali',
            'en': 'Istanbul Biennial'
        },
        'type': 'exhibition',
        'month': 9,  # September (odd years)
        'duration_days': 90,
        'description': {
            'tr': 'Çağdaş sanat bienali. İki yılda bir düzenlenir.',
            'en': 'Contemporary art biennial. Held every two years.'
        },
        'location': 'Çeşitli müzeler ve galeriler',
        'cost': {
            'tr': '₺200-400',
            'en': '₺200-400'
        },
        'website': 'bienal.iksv.org',
        'tips': {
            'tr': 'Kombine bilet alın, tüm mekanları gezin',
            'en': 'Get a combined ticket, visit all venues'
        }
    },
    {
        'id': 'ramadan',
        'name': {
            'tr': 'Ramazan Etkinlikleri',
            'en': 'Ramadan Events'
        },
        'type': 'religious',
        'month': None,  # Varies (lunar calendar)
        'duration_days': 30,
        'description': {
            'tr': 'Ramazan ayı boyunca özel etkinlikler, iftar çadırları, mahya gösterileri.',
            'en': 'Special events during Ramadan, iftar tents, mahya displays.'
        },
        'location': 'Sultanahmet, Eyüp, Fatih',
        'cost': {
            'tr': 'Çoğunlukla ücretsiz',
            'en': 'Mostly free'
        },
        'tips': {
            'tr': 'İftar saatinde Sultanahmet meydanına gidin',
            'en': 'Visit Sultanahmet square at iftar time'
        }
    },
    {
        'id': 'new_year',
        'name': {
            'tr': 'Yılbaşı Kutlamaları',
            'en': 'New Year Celebrations'
        },
        'type': 'cultural',
        'month': 12,
        'duration_days': 1,
        'description': {
            'tr': 'İstanbul\'un her yerinde yılbaşı kutlamaları, havai fişekler.',
            'en': 'New Year celebrations throughout Istanbul, fireworks.'
        },
        'location': 'Taksim, Ortaköy, Kadıköy',
        'cost': {
            'tr': 'Ücretsiz (sokak kutlamaları)',
            'en': 'Free (street celebrations)'
        },
        'tips': {
            'tr': 'Ortaköy veya Taksim\'e erken gidin, çok kalabalık olur',
            'en': 'Go to Ortaköy or Taksim early, gets very crowded'
        }
    },
    {
        'id': 'tulip_festival',
        'name': {
            'tr': 'Lale Festivali',
            'en': 'Tulip Festival'
        },
        'type': 'festival',
        'month': 4,  # April
        'duration_days': 30,
        'description': {
            'tr': 'İstanbul parklarında milyonlarca lale açıyor. Rengarenk görüntü.',
            'en': 'Millions of tulips bloom in Istanbul parks. Colorful spectacle.'
        },
        'location': 'Emirgan Korusu, Gülhane Parkı',
        'cost': {
            'tr': 'Ücretsiz',
            'en': 'Free'
        },
        'tips': {
            'tr': 'Emirgan Korusu\'na sabah erken gidin, en güzel fotoğraflar için',
            'en': 'Go to Emirgan Grove early morning for best photos'
        }
    },
    {
        'id': 'istanbul_jazz_festival',
        'name': {
            'tr': 'İstanbul Caz Festivali',
            'en': 'Istanbul Jazz Festival'
        },
        'type': 'festival',
        'month': 7,  # July
        'duration_days': 20,
        'description': {
            'tr': 'Dünya çapında caz sanatçıları İstanbul\'da. Açık hava konserleri.',
            'en': 'World-class jazz artists in Istanbul. Open-air concerts.'
        },
        'location': 'Harbiye, Zorlu PSM, açık hava mekanları',
        'cost': {
            'tr': '₺200-1,000',
            'en': '₺200-1,000'
        },
        'website': 'caz.iksv.org',
        'tips': {
            'tr': 'Açık hava konserlerine erkenden gidin, yerler dolup taşıyor',
            'en': 'Arrive early for open-air concerts, venues fill up'
        }
    },
]


# ==================== CULTURAL VENUES WITH REGULAR EVENTS ====================

CULTURAL_VENUES = [
    {
        'id': 'zorlu_psm',
        'name': 'Zorlu PSM',
        'type': 'theater',
        'description': {
            'tr': 'Modern sanat merkezi. Konserler, tiyatro, dans gösterileri.',
            'en': 'Modern arts center. Concerts, theater, dance performances.'
        },
        'location': 'Beşiktaş, Zorlu Center',
        'regular_events': {
            'tr': 'Haftanın her günü farklı etkinlikler',
            'en': 'Different events every day of the week'
        },
        'cost': {
            'tr': '₺150-800',
            'en': '₺150-800'
        },
        'website': 'zorlupsm.com',
        'tips': {
            'tr': 'Online rezervasyon yapın, gişe fiyatları daha yüksek',
            'en': 'Book online, box office prices are higher'
        }
    },
    {
        'id': 'cemal_resit_rey',
        'name': 'Cemal Reşit Rey Konser Salonu',
        'type': 'concert',
        'description': {
            'tr': 'Klasik müzik ve senfonik konserler için başlıca mekan.',
            'en': 'Main venue for classical music and symphonic concerts.'
        },
        'location': 'Harbiye',
        'regular_events': {
            'tr': 'Haftalık senfonik konserler (Cuma-Pazar)',
            'en': 'Weekly symphonic concerts (Friday-Sunday)'
        },
        'cost': {
            'tr': '₺100-500',
            'en': '₺100-500'
        },
        'website': 'crr.istanbul',
        'tips': {
            'tr': 'Öğrenci indirimleri var',
            'en': 'Student discounts available'
        }
    },
    {
        'id': 'istanbul_modern',
        'name': 'Istanbul Modern',
        'type': 'exhibition',
        'description': {
            'tr': 'Çağdaş sanat müzesi. Dönem dönem özel sergiler.',
            'en': 'Contemporary art museum. Special exhibitions periodically.'
        },
        'location': 'Karaköy',
        'regular_events': {
            'tr': 'Sürekli değişen sergiler, workshop\'lar',
            'en': 'Rotating exhibitions, workshops'
        },
        'cost': {
            'tr': '₺200',
            'en': '₺200'
        },
        'website': 'istanbulmodern.org',
        'tips': {
            'tr': 'Perşembe günleri 18:00-20:00 ücretsiz',
            'en': 'Free on Thursdays 18:00-20:00'
        }
    },
    {
        'id': 'garajistanbul',
        'name': 'Garaj Istanbul',
        'type': 'theater',
        'description': {
            'tr': 'Alternatif tiyatro ve performans sanatları merkezi.',
            'en': 'Alternative theater and performance arts center.'
        },
        'location': 'Beyoğlu',
        'regular_events': {
            'tr': 'Deneysel tiyatro, dans gösterileri',
            'en': 'Experimental theater, dance performances'
        },
        'cost': {
            'tr': '₺150-350',
            'en': '₺150-350'
        },
        'website': 'garajistanbul.org',
        'tips': {
            'tr': 'İleri tarihli rezervasyon yapın',
            'en': 'Make advance reservations'
        }
    },
]


# ==================== SPORTS EVENTS ====================

SPORTS_EVENTS = [
    {
        'id': 'besiktas_football',
        'name': {
            'tr': 'Beşiktaş Futbol Maçları',
            'en': 'Beşiktaş Football Matches'
        },
        'type': 'sports',
        'description': {
            'tr': 'Vodafone Park\'ta iç saha maçları. Sezon boyunca her hafta.',
            'en': 'Home matches at Vodafone Park. Weekly during season.'
        },
        'location': 'Vodafone Park, Beşiktaş',
        'season': 'August-May',
        'cost': {
            'tr': '₺300-1,500',
            'en': '₺300-1,500'
        },
        'tips': {
            'tr': 'Online bilet alın, stat önünde satıcılardan almayın',
            'en': 'Buy tickets online, don\'t buy from scalpers'
        }
    },
    {
        'id': 'galatasaray_football',
        'name': {
            'tr': 'Galatasaray Futbol Maçları',
            'en': 'Galatasaray Football Matches'
        },
        'type': 'sports',
        'description': {
            'tr': 'Rams Park (Türk Telekom Stadyumu) maçları.',
            'en': 'Matches at Rams Park (Türk Telekom Stadium).'
        },
        'location': 'Rams Park, Seyrantepe',
        'season': 'August-May',
        'cost': {
            'tr': '₺350-2,000',
            'en': '₺350-2,000'
        },
        'tips': {
            'tr': 'Metro ile ulaşım en kolay',
            'en': 'Metro is the easiest way to get there'
        }
    },
    {
        'id': 'fenerbahce_football',
        'name': {
            'tr': 'Fenerbahçe Futbol Maçları',
            'en': 'Fenerbahçe Football Matches'
        },
        'type': 'sports',
        'description': {
            'tr': 'Ülker Stadyumu (Şükrü Saracoğlu) maçları.',
            'en': 'Matches at Ülker Stadium (Şükrü Saracoğlu).'
        },
        'location': 'Kadıköy',
        'season': 'August-May',
        'cost': {
            'tr': '₺300-1,800',
            'en': '₺300-1,800'
        },
        'tips': {
            'tr': 'Maç günü Kadıköy çok kalabalık, erken çıkın',
            'en': 'Kadıköy is very crowded on match days, leave early'
        }
    },
]


# ==================== MONTHLY RECURRING EVENTS ====================

MONTHLY_EVENTS = [
    {
        'id': 'first_friday_art_walk',
        'name': {
            'tr': 'İlk Cuma Sanat Yürüyüşü',
            'en': 'First Friday Art Walk'
        },
        'type': 'cultural',
        'occurrence': 'first_friday',  # First Friday of month
        'description': {
            'tr': 'Karaköy ve Galata galerilerinde özel açılışlar ve etkinlikler.',
            'en': 'Special openings and events at Karaköy and Galata galleries.'
        },
        'location': 'Karaköy, Galata',
        'time': '18:00-22:00',
        'cost': {
            'tr': 'Ücretsiz',
            'en': 'Free'
        },
        'tips': {
            'tr': 'Galeri haritasını önceden edinin',
            'en': 'Get a gallery map beforehand'
        }
    },
    {
        'id': 'antique_market',
        'name': {
            'tr': 'Antika Pazarı',
            'en': 'Antique Market'
        },
        'type': 'market',
        'occurrence': 'last_sunday',  # Last Sunday of month
        'description': {
            'tr': 'Horhor ve Çukurcuma\'da aylık antika pazarı.',
            'en': 'Monthly antique market in Horhor and Çukurcuma.'
        },
        'location': 'Horhor, Çukurcuma',
        'time': '10:00-18:00',
        'cost': {
            'tr': 'Ücretsiz giriş',
            'en': 'Free entry'
        },
        'tips': {
            'tr': 'Sabah erken gidin, en iyi parçalar hızlı tükenir',
            'en': 'Go early morning, best pieces sell out fast'
        }
    },
]


# ==================== UTILITY FUNCTIONS ====================

def load_live_iksv_events() -> List[Dict]:
    """
    Load live İKSV events from multiple sources:
    1. Manually curated events from iksv_manual_events.json (highest priority)
    2. Auto-scraped events from current_events.json (fallback)
    
    Returns a list of events in a normalized format compatible with static events.
    """
    all_live_events = []
    
    # 1. Load manually curated İKSV events (PRIMARY SOURCE)
    manual_events_file = Path(__file__).parent.parent.parent / "data" / "events" / "iksv_manual_events.json"
    if manual_events_file.exists():
        try:
            with open(manual_events_file, 'r', encoding='utf-8') as f:
                manual_data = json.load(f)
            
            for event in manual_data.get('events', []):
                # Enhanced normalization with full event details
                normalized_event = {
                    'id': f"manual_iksv_{event.get('event_number', '')}",
                    'name': {
                        'tr': event.get('title', 'Başlıksız Etkinlik'),
                        'en': event.get('title', 'Untitled Event')
                    },
                    'type': _map_iksv_category_to_type(event.get('category', 'cultural')),
                    'description': event.get('description', {
                        'tr': f"{event.get('title', '')} - {event.get('venue', '')}",
                        'en': f"{event.get('title', '')} - {event.get('venue', '')}"
                    }),
                    'location': event.get('venue', 'Mekan Belirtilmemiş / Venue Not Specified'),
                    'date_str': event.get('date_str', ''),
                    'time': event.get('time', ''),
                    'cost': event.get('price', {
                        'tr': 'Bilet fiyatı için İKSV web sitesine bakınız',
                        'en': 'Check İKSV website for ticket prices'
                    }),
                    'source': 'İKSV Manual',
                    'fetched_at': event.get('fetched_at', ''),
                    'is_live': True,
                    'ticket_url': event.get('ticket_url', 'https://www.iksv.org/tr/bilet'),
                    'image_url': event.get('image_url', ''),
                    'tags': event.get('tags', []),
                    'website': 'www.iksv.org',
                    'tips': {
                        'tr': 'Güncel bilgi ve biletler için İKSV web sitesini ziyaret edin',
                        'en': 'Visit İKSV website for current information and tickets'
                    }
                }
                all_live_events.append(normalized_event)
            
            print(f"✅ Loaded {len(all_live_events)} manually curated İKSV events")
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not load manual İKSV events: {e}")
    
    # 2. Load auto-scraped events (FALLBACK)
    events_file = Path(__file__).parent.parent.parent / "data" / "events" / "current_events.json"
    
    if events_file.exists():
        try:
            with open(events_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scraped_count = 0
            for event in data.get('events', []):
                # Normalize live event to match static event structure
                normalized_event = {
                    'id': f"live_iksv_{event.get('event_number', '')}_{event.get('fetched_at', '').split('T')[0]}",
                    'name': {
                        'tr': event.get('title', 'Başlıksız Etkinlik'),
                        'en': event.get('title', 'Untitled Event')
                    },
                    'type': _map_iksv_category_to_type(event.get('category', 'cultural')),
                    'description': {
                        'tr': f"{event.get('title', '')} - {event.get('venue', '')}",
                        'en': f"{event.get('title', '')} - {event.get('venue', '')}"
                    },
                    'location': event.get('venue', 'Mekan Belirtilmemiş / Venue Not Specified'),
                    'date_str': event.get('date_str', ''),
                    'cost': {
                        'tr': 'Bilet fiyatı için İKSV web sitesine bakınız',
                        'en': 'Check İKSV website for ticket prices'
                    },
                    'source': 'İKSV Live',
                    'fetched_at': event.get('fetched_at', ''),
                    'is_live': True,
                    'website': 'www.iksv.org',
                    'tips': {
                        'tr': 'Güncel bilgi ve biletler için İKSV web sitesini ziyaret edin',
                        'en': 'Visit İKSV website for current information and tickets'
                    }
                }
                all_live_events.append(normalized_event)
                scraped_count += 1
            
            if scraped_count > 0:
                print(f"✅ Loaded {scraped_count} auto-scraped İKSV events")
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not load auto-scraped İKSV events: {e}")
    
    return all_live_events


def _map_iksv_category_to_type(category: str) -> str:
    """Map İKSV event categories to standard event types"""
    category_lower = category.lower()
    
    mapping = {
        'theatre': 'theater',
        'theater': 'theater',
        'tiyatro': 'theater',
        'music': 'concert',
        'müzik': 'concert',
        'concert': 'concert',
        'konser': 'concert',
        'art': 'exhibition',
        'sanat': 'exhibition',
        'sergi': 'exhibition',
        'exhibition': 'exhibition',
        'film': 'cultural',
        'cinema': 'cultural',
        'sinema': 'cultural',
        'dance': 'theater',
        'dans': 'theater',
        'festival': 'festival',
        'salon': 'concert',
    }
    
    for key, value in mapping.items():
        if key in category_lower:
            return value
    
    return 'cultural'  # Default fallback


def get_all_events(include_live: bool = True) -> List[Dict]:
    """
    Get all events from both static database and live İKSV events.
    
    Args:
        include_live: If True, includes live İKSV events from current_events.json
    
    Returns:
        List of all events (static + live if enabled)
    """
    static_events = WEEKLY_EVENTS + SEASONAL_EVENTS + CULTURAL_VENUES + SPORTS_EVENTS + MONTHLY_EVENTS
    
    if include_live:
        live_events = load_live_iksv_events()
        return static_events + live_events
    
    return static_events


def get_event_by_id(event_id: str, include_live: bool = True) -> Optional[Dict]:
    """
    Get event by ID from all event collections.
    
    Args:
        event_id: The unique event identifier
        include_live: If True, searches in live İKSV events as well
    
    Returns:
        Event dictionary or None if not found
    """
    all_events = get_all_events(include_live=include_live)
    for event in all_events:
        if event.get('id') == event_id:
            return event
    return None


def get_events_by_type(event_type: str, include_live: bool = True) -> List[Dict]:
    """
    Get all events of a specific type.
    
    Args:
        event_type: Type of event (concert, festival, exhibition, etc.)
        include_live: If True, includes live İKSV events in search
    
    Returns:
        List of events matching the type
    """
    all_events = get_all_events(include_live=include_live)
    return [e for e in all_events if e.get('type') == event_type]


def get_events_by_month(month: int, include_live: bool = True) -> List[Dict]:
    """
    Get events that occur in a specific month.
    
    Args:
        month: Month number (1-12)
        include_live: If True, attempts to match live events by date string
    
    Returns:
        List of events in the specified month
    """
    month_events = [e for e in SEASONAL_EVENTS if e.get('month') == month]
    
    if include_live:
        # Try to match live events by parsing their date strings
        live_events = load_live_iksv_events()
        for event in live_events:
            if _event_matches_month(event.get('date_str', ''), month):
                month_events.append(event)
    
    return month_events


def _event_matches_month(date_str: str, target_month: int) -> bool:
    """Helper to check if a date string matches a target month"""
    if not date_str:
        return False
    
    month_names = {
        'january': 1, 'ocak': 1,
        'february': 2, 'şubat': 2,
        'march': 3, 'mart': 3,
        'april': 4, 'nisan': 4,
        'may': 5, 'mayıs': 5,
        'june': 6, 'haziran': 6,
        'july': 7, 'temmuz': 7,
        'august': 8, 'ağustos': 8,
        'september': 9, 'eylül': 9,
        'october': 10, 'ekim': 10,
        'november': 11, 'kasım': 11,
        'december': 12, 'aralık': 12,
    }
    
    date_lower = date_str.lower()
    for month_name, month_num in month_names.items():
        if month_name in date_lower and month_num == target_month:
            return True
    
    return False


def get_events_by_day_of_week(day: int, include_live: bool = False) -> List[Dict]:
    """
    Get weekly events that occur on a specific day (0=Monday, 6=Sunday).
    
    Args:
        day: Day of week (0=Monday, 6=Sunday)
        include_live: If True, attempts to match live events (less reliable for day of week)
    
    Returns:
        List of weekly recurring events on that day
    """
    # Weekly recurring events are only in WEEKLY_EVENTS
    return [e for e in WEEKLY_EVENTS if e.get('day_of_week') == day]


def get_current_and_upcoming_events(days_ahead: int = 30, include_live: bool = True) -> List[Dict]:
    """
    Get events happening now and in the near future.
    
    Args:
        days_ahead: Number of days to look ahead (default 30)
        include_live: If True, includes live İKSV events
    
    Returns:
        List of current and upcoming events, prioritizing live events
    """
    current_events = []
    now = datetime.now()
    target_date = now + timedelta(days=days_ahead)
    
    # Add seasonal events in the current/upcoming timeframe
    for event in SEASONAL_EVENTS:
        if event.get('month'):
            if now.month <= event['month'] <= target_date.month:
                current_events.append(event)
    
    # Add live events (they are current by definition)
    if include_live:
        live_events = load_live_iksv_events()
        current_events.extend(live_events)
    
    # Add weekly recurring events
    current_events.extend(WEEKLY_EVENTS)
    
    return current_events


def search_events(query: str, language: str = 'en', include_live: bool = True) -> List[Dict]:
    """
    Search events by keyword in name or description.
    
    Args:
        query: Search query string
        language: Language code ('en' or 'tr')
        include_live: If True, includes live İKSV events in search
    
    Returns:
        List of events matching the search query
    """
    query_lower = query.lower()
    all_events = get_all_events(include_live=include_live)
    results = []
    
    for event in all_events:
        try:
            # Search in name
            name = event.get('name', '')
            if isinstance(name, dict):
                name_text = name.get(language, name.get('en', ''))
            else:
                name_text = str(name) if name else ''
            
            # Search in description
            description = event.get('description', '')
            if isinstance(description, dict):
                desc_text = description.get(language, description.get('en', ''))
            else:
                desc_text = str(description) if description else ''
            
            # Search in location
            location = str(event.get('location', ''))
            
            # Check if query matches any field
            if (query_lower in name_text.lower() or 
                query_lower in desc_text.lower() or 
                query_lower in location.lower()):
                results.append(event)
        except (AttributeError, TypeError) as e:
            # Skip events with malformed data
            continue
    
    return results


def get_iksv_events_only() -> List[Dict]:
    """
    Get only İKSV-related events (both static major festivals and live events).
    
    Returns:
        List of all İKSV events
    """
    iksv_events = []
    
    # Get static İKSV events (major festivals)
    for event in SEASONAL_EVENTS:
        website = event.get('website', '')
        if 'iksv' in website.lower():
            iksv_events.append(event)
    
    # Add live İKSV events
    live_events = load_live_iksv_events()
    iksv_events.extend(live_events)
    
    return iksv_events


def get_live_events_metadata() -> Dict:
    """
    Get metadata about the live events cache.
    
    Returns:
        Dictionary with fetch date, count, and freshness info
    """
    events_file = Path(__file__).parent.parent.parent / "data" / "events" / "current_events.json"
    
    if not events_file.exists():
        return {
            'available': False,
            'message': 'Live events cache not found'
        }
    
    try:
        with open(events_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fetch_date = data.get('fetch_date', '')
        next_fetch = data.get('next_fetch_due', '')
        total_count = data.get('total_count', 0)
        
        return {
            'available': True,
            'total_count': total_count,
            'fetch_date': fetch_date,
            'next_fetch_due': next_fetch,
            'sources': data.get('sources', []),
            'is_fresh': _is_cache_fresh(fetch_date)
        }
    
    except (json.JSONDecodeError, IOError) as e:
        return {
            'available': False,
            'message': f'Error reading cache: {str(e)}'
        }


def _is_cache_fresh(fetch_date_str: str, max_age_days: int = 30) -> bool:
    """Check if the cache is still fresh (within max_age_days)"""
    if not fetch_date_str:
        return False
    
    try:
        fetch_date = datetime.fromisoformat(fetch_date_str.replace('Z', '+00:00'))
        age = datetime.now() - fetch_date.replace(tzinfo=None)
        return age.days <= max_age_days
    except (ValueError, AttributeError):
        return False


# ==================== EXPORT ====================

__all__ = [
    'EVENT_TYPES',
    'WEEKLY_EVENTS',
    'SEASONAL_EVENTS',
    'CULTURAL_VENUES',
    'SPORTS_EVENTS',
    'MONTHLY_EVENTS',
    'get_event_by_id',
    'get_events_by_type',
    'get_events_by_month',
    'get_events_by_day_of_week',
    'get_all_events',
    'get_current_and_upcoming_events',
    'search_events',
    'get_iksv_events_only',
    'load_live_iksv_events',
    'get_live_events_metadata',
]

