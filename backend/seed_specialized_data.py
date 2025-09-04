#!/usr/bin/env python3
"""
Seed specialized Istanbul data for personalized experiences
"""

from datetime import time
from sqlalchemy.orm import sessionmaker
from database import engine
from specialized_models import TransportRoute, TurkishPhrases, LocalTips
import json

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

def seed_transport_data():
    """Add Istanbul transportation data"""
    print("🚢 Adding ferry and transport data...")
    
    transport_routes = [
        # Ferry routes
        {
            "route_name": "Kadıköy-Eminönü Ferry",
            "transport_type": "ferry",
            "from_location": "Kadıköy",
            "to_location": "Eminönü",
            "duration_minutes": 25,
            "frequency_minutes": 20,
            "first_departure": time(6, 30),
            "last_departure": time(23, 30),
            "price_try": 15.0,
            "notes": "Beautiful Bosphorus views, connects Asian and European sides"
        },
        {
            "route_name": "Üsküdar-Beşiktaş Ferry",
            "transport_type": "ferry",
            "from_location": "Üsküdar",
            "to_location": "Beşiktaş",
            "duration_minutes": 15,
            "frequency_minutes": 30,
            "first_departure": time(6, 45),
            "last_departure": time(22, 45),
            "price_try": 15.0,
            "notes": "Quick connection between Asian side and European side"
        },
        {
            "route_name": "Karaköy-Üsküdar Ferry",
            "transport_type": "ferry",
            "from_location": "Karaköy",
            "to_location": "Üsküdar",
            "duration_minutes": 20,
            "frequency_minutes": 25,
            "first_departure": time(7, 0),
            "last_departure": time(23, 0),
            "price_try": 15.0,
            "notes": "Scenic route passing under historic bridges"
        },
        # Metro routes
        {
            "route_name": "M1 Metro Yenikapi-Atatürk Airport",
            "transport_type": "metro",
            "from_location": "Yenikapi",
            "to_location": "Atatürk Airport",
            "duration_minutes": 45,
            "frequency_minutes": 5,
            "first_departure": time(6, 0),
            "last_departure": time(23, 59),
            "price_try": 5.0,
            "notes": "Direct connection to old airport, connects to M2 line"
        },
        {
            "route_name": "M2 Metro Hacıosman-Vezneciler",
            "transport_type": "metro",
            "from_location": "Hacıosman",
            "to_location": "Vezneciler",
            "duration_minutes": 35,
            "frequency_minutes": 4,
            "first_departure": time(6, 15),
            "last_departure": time(23, 59),
            "price_try": 5.0,
            "notes": "Main metro line connecting north to historic peninsula"
        },
        # Tram routes
        {
            "route_name": "T1 Tram Kabataş-Bağcılar",
            "transport_type": "tram",
            "from_location": "Kabataş",
            "to_location": "Bağcılar",
            "duration_minutes": 55,
            "frequency_minutes": 6,
            "first_departure": time(6, 0),
            "last_departure": time(23, 45),
            "price_try": 5.0,
            "notes": "Passes through Sultanahmet, Grand Bazaar, connects major tourist sites"
        }
    ]
    
    for route_data in transport_routes:
        route = TransportRoute(**route_data)
        session.add(route)
    
    session.commit()
    print(f"✅ Added {len(transport_routes)} transport routes")

def seed_turkish_phrases():
    """Add essential Turkish phrases for tourists"""
    print("🇹🇷 Adding Turkish phrases...")
    
    phrases = [
        # Greetings
        {
            "category": "greetings",
            "english_phrase": "Hello / Good day",
            "turkish_phrase": "Merhaba / İyi günler",
            "pronunciation": "mer-ha-BA / ee-YEE gewn-LER",
            "context": "General greeting, use anytime"
        },
        {
            "category": "greetings", 
            "english_phrase": "Thank you",
            "turkish_phrase": "Teşekkür ederim",
            "pronunciation": "tesh-ek-KEWR ed-er-EEM",
            "context": "Essential phrase for showing appreciation"
        },
        {
            "category": "greetings",
            "english_phrase": "Please / You're welcome",
            "turkish_phrase": "Lütfen / Rica ederim", 
            "pronunciation": "LEWT-fen / ree-JA ed-er-EEM",
            "context": "Politeness essentials"
        },
        
        # Food & Restaurant
        {
            "category": "food",
            "english_phrase": "I am vegetarian",
            "turkish_phrase": "Ben vejetaryenim",
            "pronunciation": "ben veh-jeh-tar-yen-EEM",
            "context": "Essential for dietary restrictions"
        },
        {
            "category": "food",
            "english_phrase": "The bill, please",
            "turkish_phrase": "Hesap, lütfen",
            "pronunciation": "heh-SAP LEWT-fen",
            "context": "When finishing your meal"
        },
        {
            "category": "food",
            "english_phrase": "This is delicious",
            "turkish_phrase": "Bu çok lezzetli",
            "pronunciation": "boo choke lez-ZET-lee",
            "context": "Compliment the food"
        },
        {
            "category": "food",
            "english_phrase": "What do you recommend?",
            "turkish_phrase": "Ne tavsiye edersiniz?",
            "pronunciation": "neh tav-see-YEH ed-er-see-NEEZ",
            "context": "Ask for menu recommendations"
        },
        
        # Directions
        {
            "category": "directions",
            "english_phrase": "Where is...?",
            "turkish_phrase": "... nerede?",
            "pronunciation": "neh-reh-DEH",
            "context": "Ask for locations"
        },
        {
            "category": "directions",
            "english_phrase": "How much does it cost?",
            "turkish_phrase": "Ne kadar?",
            "pronunciation": "neh ka-DAR",
            "context": "Ask for prices"
        },
        {
            "category": "directions",
            "english_phrase": "I don't speak Turkish",
            "turkish_phrase": "Türkçe bilmiyorum",
            "pronunciation": "TEWR-che bil-mee-yor-OOM",
            "context": "When you need help communicating"
        },
        
        # Shopping
        {
            "category": "shopping",
            "english_phrase": "Can you give a discount?",
            "turkish_phrase": "İndirim yapabilir misiniz?",
            "pronunciation": "in-dee-REEM ya-pa-bee-leer mee-see-NEEZ",
            "context": "Bargaining in markets and bazaars"
        },
        {
            "category": "shopping",
            "english_phrase": "Too expensive",
            "turkish_phrase": "Çok pahalı",
            "pronunciation": "choke pa-ha-LU",
            "context": "When negotiating prices"
        },
        
        # Emergencies
        {
            "category": "emergency",
            "english_phrase": "Help!",
            "turkish_phrase": "İmdat!",
            "pronunciation": "im-DAT",
            "context": "Emergency situations"
        },
        {
            "category": "emergency",
            "english_phrase": "Call the police",
            "turkish_phrase": "Polis çağırın",
            "pronunciation": "po-LEES cha-uh-RUN",
            "context": "Emergency situations"
        },
        {
            "category": "emergency",
            "english_phrase": "I need a doctor",
            "turkish_phrase": "Doktora ihtiyacım var",
            "pronunciation": "dok-to-RA ih-tee-ya-JUM var",
            "context": "Medical emergencies"
        }
    ]
    
    for phrase_data in phrases:
        phrase = TurkishPhrases(**phrase_data)
        session.add(phrase)
    
    session.commit()
    print(f"✅ Added {len(phrases)} Turkish phrases")

def seed_local_tips():
    """Add local cultural tips and advice"""
    print("💡 Adding local tips and cultural advice...")
    
    tips = [
        {
            "category": "culture",
            "tip_title": "Mosque Etiquette",
            "tip_content": "Remove shoes before entering, dress modestly (cover shoulders and knees), women should cover hair. Free entry but donations appreciated.",
            "importance_level": "essential",
            "relevant_districts": json.dumps(["Sultanahmet", "Fatih", "Beyoglu"]),
            "is_offline_available": True
        },
        {
            "category": "culture", 
            "tip_title": "Tea Culture",
            "tip_content": "Turkish tea (çay) is served in small glasses. It's polite to accept when offered. Hold the glass by the rim, not the body.",
            "importance_level": "helpful",
            "relevant_districts": json.dumps(["all"]),
            "is_offline_available": True
        },
        {
            "category": "money",
            "tip_title": "Tipping Guidelines", 
            "tip_content": "10-15% tip at restaurants if service charge not included. Round up for taxis. Small tips for hotel staff appreciated.",
            "importance_level": "essential",
            "relevant_districts": json.dumps(["all"]),
            "is_offline_available": True
        },
        {
            "category": "money",
            "tip_title": "Bargaining in Bazaars",
            "tip_content": "Bargaining expected in Grand Bazaar and markets. Start at 50% of asking price. Walk away if not satisfied - they often call you back.",
            "importance_level": "helpful", 
            "relevant_districts": json.dumps(["Sultanahmet", "Fatih"]),
            "is_offline_available": True
        },
        {
            "category": "transportation",
            "tip_title": "Istanbul Card",
            "tip_content": "Get an Istanbul Card for public transport. Works on metro, tram, bus, ferry. Much cheaper than individual tickets. Available at stations.",
            "importance_level": "essential",
            "relevant_districts": json.dumps(["all"]),
            "is_offline_available": True
        },
        {
            "category": "food",
            "tip_title": "Street Food Safety",
            "tip_content": "Look for busy stalls with high turnover. Try simit (Turkish bagel), döner, and balık ekmek. Avoid salads from street vendors.",
            "importance_level": "helpful",
            "relevant_districts": json.dumps(["all"]),
            "is_offline_available": True
        },
        {
            "category": "safety",
            "tip_title": "Tourist Scams to Avoid",
            "tip_content": "Beware of shoe shine scam, fake police, overcharging in bars in Taksim. Always check prices before ordering.",
            "importance_level": "essential",
            "relevant_districts": json.dumps(["Beyoglu", "Sultanahmet"]),
            "is_offline_available": True
        },
        {
            "category": "etiquette",
            "tip_title": "Greeting Customs",
            "tip_content": "Handshakes common in business. Close friends may kiss on both cheeks. Remove shoes when entering homes.",
            "importance_level": "helpful",
            "relevant_districts": json.dumps(["all"]),
            "is_offline_available": True
        }
    ]
    
    for tip_data in tips:
        tip = LocalTips(**tip_data)
        session.add(tip)
    
    session.commit()
    print(f"✅ Added {len(tips)} local tips")

def main():
    try:
        print("🎯 Seeding specialized Istanbul data...")
        
        # Clear existing data
        session.query(TransportRoute).delete()
        session.query(TurkishPhrases).delete()
        session.query(LocalTips).delete()
        session.commit()
        
        # Add new data
        seed_transport_data()
        seed_turkish_phrases()
        seed_local_tips()
        
        print("\n🎉 Specialized data seeding complete!")
        print(f"📊 Summary:")
        print(f"   - Transport routes: {session.query(TransportRoute).count()}")
        print(f"   - Turkish phrases: {session.query(TurkishPhrases).count()}")
        print(f"   - Local tips: {session.query(LocalTips).count()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()
