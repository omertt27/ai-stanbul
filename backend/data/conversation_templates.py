"""
Conversation Templates for Istanbul AI System
Natural language responses for greetings, thanks, planning, and farewells
Supports Turkish and English
"""

from typing import List, Dict
import random


# ==================== GREETING TEMPLATES ====================

GREETINGS_TURKISH = [
    "Merhaba! 👋 İstanbul hakkında size nasıl yardımcı olabilirim?",
    "Selam! 🌟 İstanbul'u keşfetmenize yardımcı olmak için buradayım. Ne öğrenmek istersiniz?",
    "Hoş geldiniz! 🎉 İstanbul'daki en iyi yerleri bulmak için size yardımcı olabilirim.",
    "Merhaba! İstanbul için harika bir seçim! Size nasıl yardımcı olabilirim?",
    "Günaydın! ☀️ İstanbul maceranız için size rehberlik edebilirim.",
    "İyi günler! 🏙️ İstanbul'un tüm güzelliklerini keşfetmenize yardımcı olabilirim.",
]

GREETINGS_ENGLISH = [
    "Hello! 👋 How can I help you discover Istanbul?",
    "Hi there! 🌟 I'm here to help you explore Istanbul. What would you like to know?",
    "Welcome! 🎉 I can help you find the best spots in Istanbul.",
    "Hey! Great choice visiting Istanbul! How can I assist you?",
    "Good day! ☀️ I'm your guide to everything Istanbul. What are you looking for?",
    "Hi! 🏙️ Ready to explore the wonders of Istanbul? Let me know what you need!",
    "Hello friend! 🎭 Istanbul is amazing, and I'm here to make your visit unforgettable.",
    "Greetings! 🌉 Let me help you discover the magic of Istanbul!",
]

# Mixed/Casual greetings
GREETINGS_CASUAL = [
    "Hey! 😊 What brings you to Istanbul?",
    "Hi! Ready to explore this beautiful city?",
    "Hello! Let's plan something amazing for your Istanbul trip!",
]


# ==================== THANK YOU RESPONSES ====================

THANKS_TURKISH = [
    "Rica ederim! 😊 Başka bir konuda yardımcı olabilir miyim?",
    "Ne demek! İstanbul'u keşfetmenize yardımcı olmak benim için bir zevk. Başka sorunuz var mı?",
    "Bir şey değil! 🌟 Daha fazla soru sormaktan çekinmeyin.",
    "Memnuniyetle! İstanbul hakkında başka bilgi ister misiniz?",
    "Seve seve! Size daha fazla yardımcı olabileceğim bir şey var mı?",
]

THANKS_ENGLISH = [
    "You're welcome! 😊 Is there anything else I can help you with?",
    "My pleasure! Happy to help you explore Istanbul. Any other questions?",
    "No problem! 🌟 Feel free to ask me anything else about Istanbul.",
    "Glad I could help! Would you like to know more about Istanbul?",
    "Anytime! Is there anything else you'd like to discover?",
    "You're very welcome! Let me know if you need more recommendations.",
    "Happy to assist! 🎉 What else would you like to explore?",
]


# ==================== FAREWELL RESPONSES ====================

FAREWELLS_TURKISH = [
    "Güle güle! 👋 İstanbul'da harika zaman geçirin!",
    "Hoşça kalın! 🌟 Keyifli bir ziyaret dilerim!",
    "Görüşürüz! İstanbul'da muhteşem anılar biriktirin! ✨",
    "İyi yolculuklar! Seyahatinizin keyfini çıkarın! 🎒",
    "Allaha ısmarladık! İstanbul sizi bekliyor! 🏙️",
]

FAREWELLS_ENGLISH = [
    "Goodbye! 👋 Have an amazing time in Istanbul!",
    "Farewell! 🌟 Enjoy your wonderful visit!",
    "See you! Have a fantastic trip exploring Istanbul! ✨",
    "Safe travels! Enjoy every moment in this beautiful city! 🎒",
    "Take care! Istanbul awaits you with open arms! 🏙️",
    "Bye! Make wonderful memories in Istanbul! 📸",
    "Have a great trip! May Istanbul exceed all your expectations! 🌉",
]


# ==================== PLANNING HELP RESPONSES ====================

PLANNING_HELP = {
    'turkish': {
        'intro': "İstanbul için mükemmel bir plan hazırlayalım! 🗓️",
        'duration_question': "Kaç gün kalacaksınız?",
        'interests_question': "Ne tür yerler görmeyi tercih edersiniz? (tarihi yerler, müzeler, yerel restoranlar, gizli yerler, vb.)",
    },
    'english': {
        'intro': "Let's create the perfect Istanbul itinerary for you! 🗓️",
        'duration_question': "How many days will you be staying?",
        'interests_question': "What type of places would you like to see? (historical sites, museums, local restaurants, hidden gems, etc.)",
    }
}


# ==================== ITINERARY TEMPLATES ====================

ITINERARY_1DAY = {
    'title': '1-Day Istanbul Highlights',
    'description': 'Perfect for a quick visit - the essential Istanbul experience',
    'schedule': [
        {
            'time': '09:00-11:00',
            'activity': 'Blue Mosque & Hagia Sophia',
            'details': 'Start with Istanbul\'s most iconic landmarks in Sultanahmet',
            'cost': '₺100-200',
            'tips': 'Go early to avoid crowds. Dress modestly for Blue Mosque.'
        },
        {
            'time': '11:00-13:00',
            'activity': 'Basilica Cistern & Grand Bazaar',
            'details': 'Ancient underground cistern, then shopping at world\'s oldest bazaar',
            'cost': '₺450-600',
            'tips': 'Bargaining is expected at Grand Bazaar. Start at 50% of asking price.'
        },
        {
            'time': '13:00-14:30',
            'activity': 'Lunch in Sultanahmet',
            'details': 'Traditional Turkish lunch at local restaurant',
            'cost': '₺250-400',
            'tips': 'Try köfte or kebab with ayran (yogurt drink)'
        },
        {
            'time': '14:30-17:00',
            'activity': 'Topkapi Palace',
            'details': 'Explore the Ottoman sultans\' magnificent palace',
            'cost': '₺650',
            'tips': 'Get Harem section ticket too (₺350 extra) - worth it!'
        },
        {
            'time': '17:00-19:00',
            'activity': 'Bosphorus Sunset Cruise',
            'details': 'Public ferry for budget option or tourist cruise',
            'cost': '₺50-800',
            'tips': 'Public ferry (₺50) offers same views as tourist boats!'
        },
        {
            'time': '19:30-21:00',
            'activity': 'Dinner in Eminönü',
            'details': 'Try famous fish sandwich by the water',
            'cost': '₺100-300',
            'tips': 'Balık ekmek (fish sandwich) is a must-try Istanbul experience'
        }
    ],
    'total_cost': {
        'budget': '₺1,200-1,800',
        'moderate': '₺2,000-3,000',
        'comfort': '₺3,500-5,000'
    }
}

ITINERARY_2DAY = {
    'title': '2-Day Istanbul Essentials',
    'description': 'Perfect balance of history, culture, and local life',
    'day1': {
        'title': 'Day 1: Historic Peninsula',
        'schedule': [
            {
                'time': '09:00-12:00',
                'activity': 'Sultanahmet Square, Blue Mosque, Hagia Sophia',
                'cost': '₺100-200',
                'tips': 'Start early for best photos and smaller crowds'
            },
            {
                'time': '12:00-14:00',
                'activity': 'Lunch & Grand Bazaar',
                'cost': '₺400-600',
                'tips': 'Eat at a lokanta (local restaurant) inside the bazaar'
            },
            {
                'time': '14:00-17:00',
                'activity': 'Topkapi Palace & Gulhane Park',
                'cost': '₺650-1,000',
                'tips': 'Visit Harem section, then relax in beautiful park'
            },
            {
                'time': '17:00-19:00',
                'activity': 'Bosphorus Ferry Ride',
                'cost': '₺50-100',
                'tips': 'Take public ferry from Eminönü - spectacular sunset views'
            },
            {
                'time': '19:30+',
                'activity': 'Dinner in Sultanahmet or Sirkeci',
                'cost': '₺250-500',
                'tips': 'Try traditional meyhane for authentic experience'
            }
        ]
    },
    'day2': {
        'title': 'Day 2: Modern Istanbul & Bosphorus',
        'schedule': [
            {
                'time': '09:00-11:00',
                'activity': 'Spice Bazaar & Morning Tea',
                'cost': '₺100-200',
                'tips': 'Sample spices and Turkish delight (free samples!)'
            },
            {
                'time': '11:00-13:00',
                'activity': 'Galata Tower & Karaköy',
                'cost': '₺450-600',
                'tips': 'Climb tower for panoramic views, explore trendy Karaköy'
            },
            {
                'time': '13:00-15:00',
                'activity': 'Lunch & Istiklal Street',
                'cost': '₺200-400',
                'tips': 'Walk pedestrian street, take historic tram'
            },
            {
                'time': '15:00-17:00',
                'activity': 'Taksim & Cihangir Neighborhoods',
                'cost': 'Free-₺200',
                'tips': 'Explore cafes, vintage shops, local atmosphere'
            },
            {
                'time': '17:00-19:00',
                'activity': 'Ortaköy & Bosphorus Bridge',
                'cost': '₺100-200',
                'tips': 'Try famous kumpir (stuffed potato), watch sunset'
            },
            {
                'time': '19:30+',
                'activity': 'Dinner in Beşiktaş or Ortaköy',
                'cost': '₺300-600',
                'tips': 'Waterfront dining with Bosphorus views'
            }
        ]
    },
    'total_cost': {
        'budget': '₺2,500-4,000',
        'moderate': '₺4,500-7,000',
        'comfort': '₺8,000-12,000'
    }
}

ITINERARY_3DAY = {
    'title': '3-Day Istanbul Complete Experience',
    'description': 'Comprehensive exploration of both sides of Istanbul',
    'summary': 'Day 1: Historic Sultanahmet, Day 2: Modern Beyoğlu & Bosphorus, Day 3: Asian Side & Princes Islands',
    'daily_highlights': [
        'Day 1: Blue Mosque, Hagia Sophia, Topkapi Palace, Grand Bazaar, Bosphorus Cruise',
        'Day 2: Spice Bazaar, Galata Tower, Istiklal Street, Taksim, Ortaköy',
        'Day 3: Üsküdar, Kadıköy, Moda, or Princes Islands ferry trip'
    ],
    'total_cost': {
        'budget': '₺4,000-6,000',
        'moderate': '₺8,000-12,000',
        'comfort': '₺15,000-20,000'
    }
}

ITINERARY_5DAY = {
    'title': '5-Day Istanbul In-Depth Discovery',
    'description': 'Thorough exploration with time for hidden gems and relaxation',
    'summary': 'Days 1-3: All major sites + Day 4: Hidden neighborhoods + Day 5: Bosphorus cruise or museums',
    'highlights': [
        'All major historical sites with no rush',
        'Time to explore local neighborhoods (Balat, Fener, Kadıköy)',
        'Full day Bosphorus tour or museum hopping',
        'Authentic Turkish bath (hammam) experience',
        'Hidden gems and local favorites',
        'Relaxed pace with breaks and flexibility'
    ],
    'total_cost': {
        'budget': '₺8,000-12,000',
        'moderate': '₺15,000-25,000',
        'comfort': '₺30,000-45,000'
    }
}

ITINERARY_7DAY = {
    'title': '7-Day Istanbul Complete Immersion',
    'description': 'Full cultural immersion with day trips and deep exploration',
    'summary': 'Week-long journey covering all Istanbul highlights plus day trips',
    'highlights': [
        'All major attractions at leisurely pace',
        'Multiple neighborhood explorations (both European & Asian sides)',
        'Day trips (Princes Islands, Belgrade Forest, or Polonezköy)',
        'Cultural experiences (hammam, Turkish cooking class, tea garden)',
        'Museum deep-dives (multiple days)',
        'Local markets and authentic dining experiences',
        'Free time for shopping and personal discoveries'
    ],
    'total_cost': {
        'budget': '₺12,000-18,000',
        'moderate': '₺25,000-40,000',
        'comfort': '₺50,000-70,000'
    }
}


# ==================== HELP & CONFUSED RESPONSES ====================

HELP_RESPONSES_TURKISH = [
    "Size nasıl yardımcı olabilirim? 🤔\n\nŞunları sorabilirsiniz:\n- Gezilecek yerler (müzeler, tarihi yerler, parklar)\n- Restoranlar ve yemek önerileri\n- Gizli yerler ve yerel favoriler\n- Ulaşım bilgisi\n- Bütçe dostu aktiviteler\n- Gezi planı oluşturma",
    "İstanbul hakkında ne öğrenmek istersiniz?\n\nPopüler konular:\n✨ Tarihi yerler\n🍽️ Restoran önerileri\n💰 Bütçeye uygun yerler\n🗺️ Mahalle rehberi\n🎭 Kültürel etkinlikler",
]

HELP_RESPONSES_ENGLISH = [
    "How can I assist you with Istanbul? 🤔\n\nYou can ask me about:\n- Places to visit (museums, historic sites, parks)\n- Restaurant and food recommendations\n- Hidden gems and local favorites\n- Transportation information\n- Budget-friendly activities\n- Trip planning and itineraries",
    "What would you like to know about Istanbul?\n\nPopular topics:\n✨ Historical attractions\n🍽️ Restaurant recommendations\n💰 Budget-friendly options\n🗺️ Neighborhood guides\n🎭 Cultural events\n🌤️ Weather-based suggestions",
    "I'm here to help! Tell me:\n- How many days are you staying?\n- What's your budget?\n- What interests you most?\n- Any specific neighborhoods you want to explore?",
]

CONFUSED_RESPONSES = [
    "I didn't quite understand that. Could you rephrase your question about Istanbul? 😊",
    "Hmm, I'm not sure about that. Could you ask in a different way?",
    "I want to help, but I need more clarity. What specifically would you like to know about Istanbul?",
]


# ==================== UTILITY FUNCTIONS ====================

def get_random_greeting(language: str = 'english') -> str:
    """Get a random greeting in specified language"""
    if language.lower() in ['turkish', 'tr', 'türkçe']:
        return random.choice(GREETINGS_TURKISH)
    elif language.lower() in ['casual', 'mixed']:
        return random.choice(GREETINGS_CASUAL)
    else:
        return random.choice(GREETINGS_ENGLISH)


def get_random_thanks_response(language: str = 'english') -> str:
    """Get a random thank you response in specified language"""
    if language.lower() in ['turkish', 'tr', 'türkçe']:
        return random.choice(THANKS_TURKISH)
    else:
        return random.choice(THANKS_ENGLISH)


def get_random_farewell(language: str = 'english') -> str:
    """Get a random farewell in specified language"""
    if language.lower() in ['turkish', 'tr', 'türkçe']:
        return random.choice(FAREWELLS_TURKISH)
    else:
        return random.choice(FAREWELLS_ENGLISH)


def get_help_response(language: str = 'english') -> str:
    """Get a help response in specified language"""
    if language.lower() in ['turkish', 'tr', 'türkçe']:
        return random.choice(HELP_RESPONSES_TURKISH)
    else:
        return random.choice(HELP_RESPONSES_ENGLISH)


def get_itinerary_by_days(days: int) -> Dict:
    """Get itinerary template based on number of days"""
    if days == 1:
        return ITINERARY_1DAY
    elif days == 2:
        return ITINERARY_2DAY
    elif days == 3:
        return ITINERARY_3DAY
    elif days >= 4 and days <= 6:
        return ITINERARY_5DAY
    else:  # 7+ days
        return ITINERARY_7DAY


# ==================== EXPORT ====================

__all__ = [
    'get_random_greeting',
    'get_random_thanks_response',
    'get_random_farewell',
    'get_help_response',
    'get_itinerary_by_days',
    'ITINERARY_1DAY',
    'ITINERARY_2DAY',
    'ITINERARY_3DAY',
    'ITINERARY_5DAY',
    'ITINERARY_7DAY',
    'PLANNING_HELP',
]
