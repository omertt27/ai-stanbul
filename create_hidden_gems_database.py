#!/usr/bin/env python
"""
Create a comprehensive hidden gems database for Istanbul AI system
"""

import json
import os
from typing import List, Dict, Any

def create_hidden_gems_database():
    """Create a comprehensive hidden gems database for Istanbul"""
    
    hidden_gems = {
        "metadata": {
            "total_gems": 0,
            "districts_covered": 0,
            "categories": ["cultural", "culinary", "nature", "shopping", "nightlife", "artisan", "historical"],
            "last_updated": "2025-01-10",
            "source": "Local knowledge + Istanbul experts"
        },
        "gems": [
            # SULTANAHMET & FATIH (Historic Peninsula) Hidden Gems
            {
                "name": "Sokollu Mehmet Paşa Mosque",
                "district": "Sultanahmet",
                "category": "historical",
                "description": "Intimate 16th-century mosque by Sinan with exquisite İznik tiles, often overlooked by tourists focused on bigger mosques.",
                "location": {
                    "coordinates": [28.971749, 41.004842],
                    "address": "Sokollu Mehmet Paşa, Şehit Mehmet Paşa Yokuşu, 34122 Fatih/İstanbul"
                },
                "why_hidden": "Overshadowed by Blue Mosque and Hagia Sophia nearby",
                "best_time": "Early morning or late afternoon for peaceful atmosphere",
                "insider_tip": "Look for the fragment of the Kaaba's Black Stone embedded in the mihrab",
                "access_difficulty": "easy",
                "cost": "free",
                "tags": ["architecture", "religious", "sinan", "ottoman"]
            },
            {
                "name": "Binbirdirek Cistern",
                "district": "Sultanahmet", 
                "category": "historical",
                "description": "Less crowded Byzantine cistern with 224 columns, offering the same mystical atmosphere as Basilica Cistern without the crowds.",
                "location": {
                    "coordinates": [28.972561, 41.004165],
                    "address": "Binbirdirek, İmran Öktem Cd. No:2, 34122 Fatih/İstanbul"
                },
                "why_hidden": "Most tourists only know about Basilica Cistern",
                "best_time": "Weekday afternoons",
                "insider_tip": "Often hosts art exhibitions and concerts",
                "access_difficulty": "easy",
                "cost": "25 TL (much cheaper than Basilica Cistern)",
                "tags": ["byzantine", "underground", "architecture", "exhibitions"]
            },
            
            # BEYOĞLU Hidden Gems
            {
                "name": "Ara Güler Museum",
                "district": "Beyoğlu",
                "category": "cultural",
                "description": "Intimate museum dedicated to Turkey's most famous photographer, showing Istanbul through his lens across decades.",
                "location": {
                    "coordinates": [28.974134, 41.031842],
                    "address": "Boğazkesen Cd. No:42, 34425 Beyoğlu/İstanbul"
                },
                "why_hidden": "New museum (2018) in a side street off Istiklal",
                "best_time": "Tuesday-Sunday, 10:00-19:00",
                "insider_tip": "Free entry on Thursdays after 16:00",
                "access_difficulty": "easy",
                "cost": "20 TL",
                "tags": ["photography", "art", "istanbul-history", "culture"]
            },
            {
                "name": "Asri Türbe (Tomb of Galip Dede)",
                "district": "Galata",
                "category": "cultural",
                "description": "Historic Mevlevi lodge with a peaceful garden, traditional architecture, and incredible Bosphorus views.",
                "location": {
                    "coordinates": [28.974489, 41.025856],
                    "address": "Galip Dede Cd. No:15, 34421 Beyoğlu/İstanbul"
                },
                "why_hidden": "Hidden behind high walls near Galata Tower",
                "best_time": "Late afternoon for golden hour views",
                "insider_tip": "Sometimes hosts Sufi music performances",
                "access_difficulty": "moderate",
                "cost": "free",
                "tags": ["sufi", "views", "architecture", "peaceful"]
            },
            {
                "name": "Galata Mevlevihanesi (Whirling Dervishes Museum)",
                "district": "Galata",
                "category": "cultural",
                "description": "Active Mevlevi lodge where you can witness authentic whirling dervish ceremonies, not tourist performances.",
                "location": {
                    "coordinates": [28.974134, 41.025642],
                    "address": "Galip Dede Cd. No:15, 34421 Beyoğlu/İstanbul"
                },
                "why_hidden": "Most tourists only see commercialized shows elsewhere",
                "best_time": "Sunday evenings for authentic sema ceremonies",
                "insider_tip": "Book ahead for ceremony attendance",
                "access_difficulty": "easy",
                "cost": "Museum: 15 TL, Ceremony: 40 TL",
                "tags": ["sufi", "authentic", "ceremony", "spiritual"]
            },
            
            # KADIKÖY Hidden Gems
            {
                "name": "Yeldeğirmeni Street Art District",
                "district": "Kadıköy",
                "category": "cultural",
                "description": "Vibrant neighborhood with incredible street art, murals by international artists, and authentic local cafes.",
                "location": {
                    "coordinates": [29.021187, 40.989654],
                    "address": "Yeldeğirmeni Mahallesi, 34734 Kadıköy/İstanbul"
                },
                "why_hidden": "Off the main tourist path in Asian side",
                "best_time": "Weekend afternoons for best light and open cafes",
                "insider_tip": "Follow the Mural Istanbul map for the best pieces",
                "access_difficulty": "easy",
                "cost": "free",
                "tags": ["street-art", "alternative", "hipster", "local"]
            },
            {
                "name": "Modahan",
                "district": "Kadıköy",
                "category": "shopping",
                "description": "Multi-story vintage and second-hand paradise with unique finds from clothing to vinyl records.",
                "location": {
                    "coordinates": [29.014789, 40.987412],
                    "address": "Caferağa, General Asım Gündüz Cd. No:62, 34710 Kadıköy/İstanbul"
                },
                "why_hidden": "Locals' secret shopping spot",
                "best_time": "Weekdays for less crowded browsing",
                "insider_tip": "Haggling is expected and part of the fun",
                "access_difficulty": "easy",
                "cost": "varies",
                "tags": ["vintage", "shopping", "unique", "local"]
            },
            
            # BEŞİKTAŞ Hidden Gems
            {
                "name": "Abdi İpekçi Park Secret Garden",
                "district": "Beşiktaş",
                "category": "nature",
                "description": "Hidden garden behind the main park with traditional Ottoman garden design and peaceful walking paths.",
                "location": {
                    "coordinates": [29.007834, 41.043721],
                    "address": "Abdi İpekçi Parkı arkası, Teşvikiye, 34367 Şişli/İstanbul"
                },
                "why_hidden": "Tucked behind the main park area",
                "best_time": "Early morning or late afternoon",
                "insider_tip": "Perfect spot for a quiet break from shopping in Nişantaşı",
                "access_difficulty": "easy",
                "cost": "free",
                "tags": ["garden", "peaceful", "ottoman-design", "nature"]
            },
            {
                "name": "Beşiktaş Fish Market (Balık Pazarı)",
                "district": "Beşiktaş",
                "category": "culinary",
                "description": "Authentic fish market where locals shop, with tiny restaurants serving the freshest seafood at fraction of tourist prices.",
                "location": {
                    "coordinates": [29.003456, 41.042187],
                    "address": "Sinanpaşa, Beşiktaş Balık Pazarı, 34353 Beşiktaş/İstanbul"
                },
                "why_hidden": "Locals-only spot, no tourist signs",
                "best_time": "Early morning for freshest selection",
                "insider_tip": "Point to the fish you want and they'll cook it for you",
                "access_difficulty": "moderate",
                "cost": "very budget-friendly",
                "tags": ["seafood", "authentic", "local", "budget"]
            },
            
            # ÜSKÜDAR Hidden Gems
            {
                "name": "Şemsi Pasha Mosque",
                "district": "Üsküdar",
                "category": "historical",
                "description": "Tiny waterfront mosque by Sinan, perfectly framed by the Bosphorus with stunning views of the European side.",
                "location": {
                    "coordinates": [29.020431, 41.021842],
                    "address": "Üsküdar İskelesi, 34664 Üsküdar/İstanbul"
                },
                "why_hidden": "Small size means most tourists miss it",
                "best_time": "Sunset for spectacular golden hour photography",
                "insider_tip": "The mosque is one of Sinan's smallest but most perfectly proportioned works",
                "access_difficulty": "easy",
                "cost": "free",
                "tags": ["sinan", "waterfront", "sunset", "photography"]
            },
            {
                "name": "Çinili Mosque (Tiled Mosque)",
                "district": "Üsküdar",
                "category": "historical",
                "description": "17th-century mosque famous for its stunning İznik tiles, considered some of the finest examples of Ottoman ceramic art.",
                "location": {
                    "coordinates": [29.023187, 41.026542],
                    "address": "Çinili, Çinili Cd., 34668 Üsküdar/İstanbul"
                },
                "why_hidden": "Located in residential area away from tourist routes",
                "best_time": "Morning when light streams through windows",
                "insider_tip": "The tiles date from the golden age of İznik pottery",
                "access_difficulty": "moderate",
                "cost": "free",
                "tags": ["tiles", "iznik", "ottoman", "ceramics"]
            },
            
            # BALAT & FENER Hidden Gems
            {
                "name": "Balat Antique Shops",
                "district": "Fatih",
                "category": "shopping",
                "description": "Narrow streets filled with genuine antique shops selling Ottoman-era artifacts, old photographs, and vintage treasures.",
                "location": {
                    "coordinates": [28.946789, 41.026412],
                    "address": "Balat Mahallesi, Vodina Cd., 34087 Fatih/İstanbul"
                },
                "why_hidden": "Requires walking through residential neighborhoods",
                "best_time": "Weekend mornings when all shops are open",
                "insider_tip": "Many items have authentic provenance certificates",
                "access_difficulty": "moderate",
                "cost": "varies widely",
                "tags": ["antiques", "ottoman", "authentic", "treasure-hunting"]
            },
            {
                "name": "Bulgarian St. Stephen Church (Iron Church)",
                "district": "Balat",
                "category": "historical",
                "description": "Unique prefabricated iron church from 1898, transported from Vienna and assembled on the Golden Horn.",
                "location": {
                    "coordinates": [28.947642, 41.027831],
                    "address": "Mursel Paşa Cd. No:85, 34083 Fatih/İstanbul"
                },
                "why_hidden": "In Bulgarian community area, minimal signage",
                "best_time": "Sunday mornings during service for authentic atmosphere",
                "insider_tip": "Only surviving prefabricated iron church in the world",
                "access_difficulty": "moderate",
                "cost": "free",
                "tags": ["unique", "iron", "bulgarian", "engineering"]
            },
            
            # ARNAVUTKÖY & BEBEK Hidden Gems
            {
                "name": "Arnavutköy Wooden Houses",
                "district": "Beşiktaş",
                "category": "historical",
                "description": "Ottoman wooden waterfront mansions (yalı) that survived fires and development, offering glimpse of old Bosphorus life.",
                "location": {
                    "coordinates": [29.042187, 41.063428],
                    "address": "Arnavutköy Mahallesi, Bebek Cd., 34345 Beşiktaş/İstanbul"
                },
                "why_hidden": "Residential area, not promoted as tourist destination",
                "best_time": "Late afternoon walk along the waterfront",
                "insider_tip": "Some are still private residences, others house cafes",
                "access_difficulty": "easy",
                "cost": "free",
                "tags": ["yali", "ottoman", "architecture", "bosphorus"]
            },
            
            # CULINARY Hidden Gems
            {
                "name": "Pandeli Restaurant Upper Floor",
                "district": "Eminönü",
                "category": "culinary",
                "description": "Historic 1901 restaurant above Spice Bazaar with original İznik tiles, serving Ottoman palace cuisine. The upstairs section is less known.",
                "location": {
                    "coordinates": [28.968542, 41.016428],
                    "address": "Rüstem Paşa, Pandeli Sokak No:1, 34116 Fatih/İstanbul"
                },
                "why_hidden": "Most diners stay on ground floor",
                "best_time": "Lunch for traditional Ottoman recipes",
                "insider_tip": "Ask for the historical menu with dishes served to sultans",
                "access_difficulty": "easy",
                "cost": "upscale",
                "tags": ["ottoman-cuisine", "historical", "tiles", "palace-recipes"]
            },
            {
                "name": "Çukur Kahve (The Pit Coffee)",
                "district": "Beyoğlu",
                "category": "culinary",
                "description": "Underground coffee shop in a former cistern, serving specialty coffee in an atmospheric stone chamber.",
                "location": {
                    "coordinates": [28.978134, 41.029642],
                    "address": "Asmalı Mescit, Meşrutiyet Cd. No:99, 34430 Beyoğlu/İstanbul"
                },
                "why_hidden": "Entrance is easy to miss on busy street",
                "best_time": "Afternoon for cozy atmosphere",
                "insider_tip": "Try their signature Turkish coffee with a modern twist",
                "access_difficulty": "moderate",
                "cost": "moderate",
                "tags": ["coffee", "underground", "atmospheric", "specialty"]
            },
            
            # NATURE Hidden Gems
            {
                "name": "Belgrade Forest Secret Trails",
                "district": "Sarıyer",
                "category": "nature",
                "description": "Off-the-beaten-path hiking trails in Belgrade Forest with Ottoman-era aqueducts and pristine nature.",
                "location": {
                    "coordinates": [28.987234, 41.183642],
                    "address": "Bahçeköy, Belgrad Ormanı, 34453 Sarıyer/İstanbul"
                },
                "why_hidden": "Requires local knowledge to find the best trails",
                "best_time": "Early spring and autumn for perfect weather",
                "insider_tip": "Bring a map - trails are not well marked",
                "access_difficulty": "challenging",
                "cost": "free",
                "tags": ["hiking", "nature", "aqueducts", "forest"]
            },
            {
                "name": "Emirgan Park Tulip Hill",
                "district": "Sarıyer",
                "category": "nature",
                "description": "Hidden section of Emirgan Park with the best tulip displays during the festival, away from the main crowds.",
                "location": {
                    "coordinates": [29.056234, 41.108642],
                    "address": "Emirgan, Emirgan Korusu, 34467 Sarıyer/İstanbul"
                },
                "why_hidden": "Located on a hill behind the main park area",
                "best_time": "April during tulip season",
                "insider_tip": "Best photo spot is from the upper terrace at sunset",
                "access_difficulty": "moderate",
                "cost": "free",
                "tags": ["tulips", "photography", "seasonal", "views"]
            }
        ]
    }
    
    # Update metadata
    hidden_gems["metadata"]["total_gems"] = len(hidden_gems["gems"])
    districts = set(gem["district"] for gem in hidden_gems["gems"])
    hidden_gems["metadata"]["districts_covered"] = len(districts)
    
    # Save to file
    data_dir = "/Users/omer/Desktop/ai-stanbul/backend/data"
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, "hidden_gems_database.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(hidden_gems, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Hidden Gems Database Created Successfully!")
        print(f"📍 Location: {file_path}")
        print(f"💎 Total gems: {hidden_gems['metadata']['total_gems']}")
        print(f"🏘️ Districts covered: {hidden_gems['metadata']['districts_covered']}")
        print(f"🏷️ Categories: {', '.join(hidden_gems['metadata']['categories'])}")
        
        # Show sample gems by category
        print(f"\n🔍 Sample Gems by Category:")
        categories = {}
        for gem in hidden_gems["gems"]:
            cat = gem["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(gem["name"])
        
        for category, gems in categories.items():
            print(f"   • {category}: {len(gems)} gems")
            print(f"     Examples: {', '.join(gems[:2])}")
            
    except Exception as e:
        print(f"❌ Error creating hidden gems database: {e}")

if __name__ == "__main__":
    print("💎 Creating Hidden Gems Database for Istanbul AI")
    print("=" * 55)
    create_hidden_gems_database()
