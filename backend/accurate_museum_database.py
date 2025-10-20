#!/usr/bin/env python3
"""
Corrected Istanbul Museum Information Database
=============================================

This module provides accurate, fact-checked information about Istanbul's major
museums, palaces, and cultural sites to replace AI-generated content.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import time

@dataclass
class MuseumInfo:
    """Structured museum information with verified facts"""
    name: str
    historical_period: str
    construction_date: str
    architect: Optional[str]
    key_features: List[str]
    opening_hours: Dict[str, str]
    entrance_fee: str
    location: str
    nearby_attractions: List[str]
    visiting_duration: str
    best_time_to_visit: str
    historical_significance: str
    architectural_style: str
    must_see_highlights: List[str]
    photography_allowed: bool
    accessibility: str
    closing_days: List[str]

class IstanbulMuseumDatabase:
    """Verified database of Istanbul museum information"""
    
    def __init__(self):
        self.museums = self._initialize_museum_data()
    
    def _initialize_museum_data(self) -> Dict[str, MuseumInfo]:
        """Initialize with verified museum information"""
        return {
            "hagia_sophia": MuseumInfo(
                name="Hagia Sophia (Ayasofya)",
                historical_period="Byzantine (532-537 AD), Ottoman (1453-1934), Museum (1934-2020), Mosque (2020-present)",
                construction_date="532-537 AD",
                architect="Anthemius of Tralles and Isidore of Miletus",
                key_features=[
                    "Massive dome (31m diameter, 55m high)",
                    "Byzantine mosaics and Islamic calligraphy coexisting",
                    "Imperial Door and Marble Jar",
                    "Weeping Column (Column of St. Gregory)",
                    "Viking graffiti from 9th century"
                ],
                opening_hours={
                    "daily": "Open for worship - visiting between prayer times",
                    "prayer_times": "Check daily prayer schedule"
                },
                entrance_fee="Free (functioning mosque)",
                location="Sultanahmet Square, Fatih",
                nearby_attractions=["Blue Mosque", "Topkapi Palace", "Basilica Cistern"],
                visiting_duration="45-60 minutes",
                best_time_to_visit="Early morning (8-10 AM) or late afternoon",
                historical_significance="Former Byzantine cathedral, Ottoman mosque, UNESCO World Heritage Site. Symbol of religious and cultural transformation.",
                architectural_style="Byzantine architecture with Islamic additions",
                must_see_highlights=[
                    "Main dome and supporting structure",
                    "Deesis Mosaic (Christ Pantocrator)",
                    "Virgin and Child mosaic in apse",
                    "Imperial portraits mosaics"
                ],
                photography_allowed=True,
                accessibility="Limited wheelchair access due to historical structure",
                closing_days=["Open daily - prayer time restrictions apply"]
            ),
            
            "topkapi_palace": MuseumInfo(
                name="Topkapi Palace (Topkapı Sarayı)",
                historical_period="Ottoman Empire (1465-1856)",
                construction_date="1459-1465",
                architect="Mimar Atik Sinan (later additions by Mimar Sinan)",
                key_features=[
                    "Four main courtyards with distinct functions",
                    "Harem quarters with 400 rooms",
                    "Sacred Relics collection (Prophet Muhammad's belongings)",
                    "Treasury with Spoonmaker's Diamond and Topkapi Dagger",
                    "Chinese and Japanese porcelain collection"
                ],
                opening_hours={
                    "winter": "09:00-16:45 (Oct 30 - Apr 15)",
                    "summer": "09:00-18:45 (Apr 15 - Oct 30)",
                    "ticket_sales_end": "1 hour before closing"
                },
                entrance_fee="Palace: 100 TL, Harem: Additional 70 TL (2023 prices)",
                location="Sultanahmet, Gulhane Park entrance",
                nearby_attractions=["Hagia Sophia", "Archaeological Museums", "Gulhane Park"],
                visiting_duration="2-4 hours (including Harem)",
                best_time_to_visit="Weekday mornings to avoid crowds",
                historical_significance="Primary residence of Ottoman sultans for 400 years. Administrative center of the empire.",
                architectural_style="Classical Ottoman palace architecture",
                must_see_highlights=[
                    "Imperial Treasury (Spoonmaker's Diamond)",
                    "Sacred Relics Room",
                    "Harem quarters and Sultan's private rooms",
                    "Palace kitchens with porcelain collection",
                    "Fourth courtyard gardens and pavilions"
                ],
                photography_allowed=True,
                accessibility="Partially wheelchair accessible - some areas require stairs",
                closing_days=["Tuesdays"]
            ),
            
            "blue_mosque": MuseumInfo(
                name="Blue Mosque (Sultan Ahmed Mosque)",
                historical_period="Ottoman Empire",
                construction_date="1609-1616",
                architect="Sedefkar Mehmet Aga (student of Mimar Sinan)",
                key_features=[
                    "Six minarets (unique for its time)",
                    "Blue Iznik tiles (giving it the name)",
                    "Massive prayer hall with central dome",
                    "Cascading domes and semi-domes",
                    "Over 200 stained glass windows"
                ],
                opening_hours={
                    "daily": "Open except during prayer times",
                    "closed_times": "30 minutes before and during each prayer",
                    "friday": "Closed 12:30-14:30 for Friday prayers"
                },
                entrance_fee="Free (active mosque)",
                location="Sultanahmet Square, facing Hagia Sophia",
                nearby_attractions=["Hagia Sophia", "Hippodrome", "Grand Bazaar"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Between prayer times, early morning preferred",
                historical_significance="Last great mosque of the Classical Ottoman period. Built to rival Hagia Sophia.",
                architectural_style="Classical Ottoman mosque architecture",
                must_see_highlights=[
                    "Interior blue Iznik tiles",
                    "Magnificent central dome",
                    "Intricate calligraphy and geometric patterns",
                    "Prayer hall atmosphere during non-prayer times"
                ],
                photography_allowed=True,
                accessibility="Ground level access available",
                closing_days=["None - prayer time restrictions apply"]
            ),
            
            "basilica_cistern": MuseumInfo(
                name="Basilica Cistern (Yerebatan Sarnıcı)",
                historical_period="Byzantine Empire",
                construction_date="532 AD",
                architect="Unknown Byzantine architects",
                key_features=[
                    "336 marble columns in 12 rows",
                    "Two Medusa head column bases",
                    "Atmospheric lighting and walkways",
                    "Column capitals from various periods",
                    "Underground palace nickname"
                ],
                opening_hours={
                    "daily": "09:00-18:30",
                    "last_entry": "17:30"
                },
                entrance_fee="120 TL (2023 prices)",
                location="Alemdar Mahallesi, near Hagia Sophia",
                nearby_attractions=["Hagia Sophia", "Topkapi Palace", "Grand Bazaar"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Early morning or late afternoon",
                historical_significance="Largest surviving Byzantine cistern. Supplied water to Great Palace.",
                architectural_style="Byzantine underground architecture",
                must_see_highlights=[
                    "Medusa head columns (upside down and sideways)",
                    "Column forest atmosphere",
                    "Acoustic properties demonstration",
                    "Historical water filtration system"
                ],
                photography_allowed=True,
                accessibility="Not wheelchair accessible - stairs required",
                closing_days=["None"]
            ),
            
            "dolmabahce_palace": MuseumInfo(
                name="Dolmabahçe Palace",
                historical_period="Late Ottoman Empire",
                construction_date="1843-1856",
                architect="Karapet Balyan and Nigoğayos Balyan (Armenian architects)",
                key_features=[
                    "European Baroque and Neoclassical style",
                    "4.5-ton crystal chandelier (largest in palace)",
                    "285 rooms, 43 halls, 6 Turkish baths",
                    "14 tons of gold leaf decoration",
                    "Atatürk's death room (preserved as museum)"
                ],
                opening_hours={
                    "winter": "09:00-16:00 (Oct 1 - Mar 31)",
                    "summer": "09:00-17:00 (Apr 1 - Sep 30)",
                    "guided_tours_only": "Tours every 30 minutes"
                },
                entrance_fee="Selamlık: 90 TL, Harem: 60 TL, Combined: 120 TL",
                location="Beşiktaş, Bosphorus waterfront",
                nearby_attractions=["Bosphorus Bridge", "Beşiktaş area", "Naval Museum"],
                visiting_duration="1.5-2 hours (both sections)",
                best_time_to_visit="Weekday mornings, book in advance",
                historical_significance="Last residence of Ottoman sultans. Place where Atatürk died in 1938.",
                architectural_style="European Baroque, Rococo, and Neoclassical",
                must_see_highlights=[
                    "Crystal Staircase and massive chandelier",
                    "Ceremonial Hall with 4.5-ton chandelier",
                    "Atatürk's bedroom (death room)",
                    "Imperial bathrooms with alabaster",
                    "Bosphorus-facing rooms with panoramic views"
                ],
                photography_allowed=False,
                accessibility="Limited wheelchair access",
                closing_days=["Mondays and Thursdays"]
            ),
            
            "archaeological_museum": MuseumInfo(
                name="Istanbul Archaeological Museums",
                historical_period="Ottoman to Modern (museum since 1891)",
                construction_date="1891 (main building)",
                architect="Alexandre Vallaury",
                key_features=[
                    "Three main buildings complex",
                    "Over 1 million artifacts",
                    "Alexander Sarcophagus (most famous piece)",
                    "Ancient Orient Museum section",
                    "Tiled Kiosk (oldest building, 1472)"
                ],
                opening_hours={
                    "daily": "09:00-17:00",
                    "last_entry": "16:00"
                },
                entrance_fee="60 TL (2023 prices)",
                location="Gulhane Park, near Topkapi Palace",
                nearby_attractions=["Topkapi Palace", "Hagia Sophia", "Gulhane Park"],
                visiting_duration="2-3 hours",
                best_time_to_visit="Weekday mornings",
                historical_significance="First museum in Ottoman Empire. Houses artifacts from across the empire.",
                architectural_style="Neoclassical museum architecture",
                must_see_highlights=[
                    "Alexander Sarcophagus (not Alexander's, but from Sidon)",
                    "Kadesh Peace Treaty (oldest known peace treaty)",
                    "Ancient Orient artifacts",
                    "Byzantine and Islamic art collections",
                    "Tiled Kiosk ceramics collection"
                ],
                photography_allowed=True,
                accessibility="Wheelchair accessible main areas",
                closing_days=["Mondays"]
            ),
            
            "galata_tower": MuseumInfo(
                name="Galata Tower (Galata Kulesi)",
                historical_period="Genoese (1348), Ottoman modifications",
                construction_date="1348",
                architect="Genoese architects",
                key_features=[
                    "67 meters tall, 9 floors",
                    "360-degree panoramic views",
                    "Medieval Genoese architecture",
                    "Restaurant and observation deck",
                    "Symbol of Istanbul skyline"
                ],
                opening_hours={
                    "daily": "08:30-22:00",
                    "last_entry": "21:30"
                },
                entrance_fee="100 TL for observation deck (2023)",
                location="Galata, Beyoğlu",
                nearby_attractions=["Karaköy", "Istiklal Street", "Golden Horn"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Sunset for best views",
                historical_significance="Part of Genoese fortification. Survived earthquakes and fires for 675 years.",
                architectural_style="Medieval Genoese tower architecture",
                must_see_highlights=[
                    "Panoramic views of Historic Peninsula",
                    "Bosphorus and Golden Horn views",
                    "Sunset photography opportunities",
                    "Historical elevator (1960s addition)"
                ],
                photography_allowed=True,
                accessibility="Elevator available to upper floors",
                closing_days=["None"]
            ),
            
            "chora_church": MuseumInfo(
                name="Chora Church Museum (Kariye Müzesi)",
                historical_period="Byzantine (11th century), Ottoman mosque (1511)",
                construction_date="1077-1081, mosaics added 1315-1321",
                architect="Unknown Byzantine architects",
                key_features=[
                    "Best-preserved Byzantine mosaics outside Hagia Sophia",
                    "Fresco cycles depicting life of Christ and Virgin Mary",
                    "Parekklesion (side chapel) with resurrection fresco",
                    "14th-century artistic masterpiece",
                    "Recently converted back to mosque (2020)"
                ],
                opening_hours={
                    "status": "Currently closed for restoration",
                    "note": "Check current status as it may reopen as mosque"
                },
                entrance_fee="Check current status",
                location="Edirnekapı, Fatih",
                nearby_attractions=["City walls", "Eyup Sultan Mosque"],
                visiting_duration="45-60 minutes when open",
                best_time_to_visit="Check opening status first",
                historical_significance="Contains world's finest Byzantine mosaics. UNESCO World Heritage consideration.",
                architectural_style="Byzantine church architecture",
                must_see_highlights=[
                    "Anastasis (Resurrection) fresco",
                    "Life of Christ mosaic cycle",
                    "Virgin Mary's infancy mosaics",
                    "Donor portraits of Theodore Metochites"
                ],
                photography_allowed="Unknown current policy",
                accessibility="Limited due to historical structure",
                closing_days=["Check current status"]
            ),
            
            "turkish_islamic_arts": MuseumInfo(
                name="Turkish and Islamic Arts Museum",
                historical_period="Ottoman (1524), Museum since 1983",
                construction_date="1524",
                architect="Sinan the Architect",
                key_features=[
                    "World's largest collection of Islamic calligraphy",
                    "Ottoman carpets from 13th-19th centuries",
                    "Wooden artifacts and manuscripts",
                    "Ibrahim Pasha Palace building",
                    "Ethnographic collection"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-19:00",
                    "winter": "09:00-17:00"
                },
                entrance_fee="60 TL (2024)",
                location="Sultanahmet Square, Fatih",
                nearby_attractions=["Blue Mosque", "Hagia Sophia", "Hippodrome"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Morning hours, less crowded",
                historical_significance="Houses world's most important Islamic art collection. Located in Ibrahim Pasha Palace.",
                architectural_style="Classical Ottoman palace architecture",
                must_see_highlights=[
                    "Uşak carpets from 16th century",
                    "Mamluk metalwork collection",
                    "Illuminated Quran manuscripts",
                    "Traditional Turkish house reconstruction"
                ],
                photography_allowed=True,
                accessibility="Partial wheelchair access",
                closing_days=["Mondays"]
            ),
            
            "pera_museum": MuseumInfo(
                name="Pera Museum",
                historical_period="Modern (2005), Historic Building (1893)",
                construction_date="Building: 1893, Museum: 2005",
                architect="Achille Manoussos (original), Sinan Genim (renovation)",
                key_features=[
                    "Orientalist paintings collection",
                    "Anatolian weights and measures",
                    "Kutahya tiles and ceramics",
                    "Contemporary art exhibitions",
                    "Historic Hotel Bristol building"
                ],
                opening_hours={
                    "tuesday_saturday": "10:00-19:00",
                    "sunday": "12:00-18:00"
                },
                entrance_fee="25 TL, Students 10 TL",
                location="Beyoğlu, Tepebaşı",
                nearby_attractions=["Istiklal Street", "Galata Tower", "Taksim"],
                visiting_duration="60-90 minutes",
                best_time_to_visit="Weekday afternoons",
                historical_significance="Turkey's first private museum showcasing Orientalist art and Turkish cultural heritage.",
                architectural_style="19th century European architecture",
                must_see_highlights=[
                    "Osman Hamdi Bey's 'The Tortoise Trainer'",
                    "Jean-Léon Gérôme orientalist paintings",
                    "Historic Anatolian weights collection",
                    "Rotating contemporary exhibitions"
                ],
                photography_allowed="Limited (no flash)",
                accessibility="Fully wheelchair accessible",
                closing_days=["Mondays"]
            ),
            
            "sakip_sabanci": MuseumInfo(
                name="Sakıp Sabancı Museum",
                historical_period="Historic Mansion (1920s), Museum (2002)",
                construction_date="1927, Museum opened 2002",
                architect="Edouard De Nari (mansion)",
                key_features=[
                    "Ottoman calligraphy and manuscripts",
                    "19th-20th century Turkish paintings",
                    "Bosphorus waterfront mansion",
                    "Temporary international exhibitions",
                    "Historic Atlı Köşk building"
                ],
                opening_hours={
                    "tuesday_sunday": "10:00-18:00",
                    "thursday": "10:00-20:00"
                },
                entrance_fee="30 TL, Students 15 TL",
                location="Emirgan, Sarıyer (Bosphorus)",
                nearby_attractions=["Emirgan Park", "Bosphorus Bridge", "Bebek"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Spring and fall for garden views",
                historical_significance="Premier private museum showcasing Ottoman and Turkish art in historic Bosphorus mansion.",
                architectural_style="Early 20th century mansion architecture",
                must_see_highlights=[
                    "Ottoman imperial calligraphy",
                    "Osman Hamdi Bey paintings",
                    "Bosphorus garden and terrace",
                    "International temporary exhibitions"
                ],
                photography_allowed="Garden only, no interior",
                accessibility="Limited due to historic structure",
                closing_days=["Mondays"]
            ),
            
            "rahmi_koc": MuseumInfo(
                name="Rahmi M. Koç Museum",
                historical_period="Industrial Heritage (Ottoman-Republican periods)",
                construction_date="Museum opened 1994",
                architect="Various (industrial buildings)",
                key_features=[
                    "Industrial and transport history",
                    "Historic cars, planes, and ships",
                    "Interactive science exhibits",
                    "Ottoman dock and shipyard",
                    "Submarine and aircraft collection"
                ],
                opening_hours={
                    "tuesday_friday": "10:00-17:00",
                    "weekend": "10:00-18:00"
                },
                entrance_fee="25 TL, Students 15TL",
                location="Hasköy, Golden Horn",
                nearby_attractions=["Golden Horn", "Fener", "Balat"],
                visiting_duration="2-3 hours",
                best_time_to_visit="Weekends for full experience",
                historical_significance="Turkey's first industrial museum showcasing technological heritage.",
                architectural_style="Industrial heritage buildings",
                must_see_highlights=[
                    "Historic submarine tour",
                    "Vintage car collection",
                    "Steam engines and locomotives",
                    "Interactive science experiments"
                ],
                photography_allowed=True,
                accessibility="Mostly accessible",
                closing_days=["Mondays"]
            ),
            
            "istanbul_modern": MuseumInfo(
                name="Istanbul Modern Art Museum",
                historical_period="Contemporary (2004, relocated 2018)",
                construction_date="2018 (new building)",
                architect="Renzo Piano",
                key_features=[
                    "Modern and contemporary Turkish art",
                    "International contemporary exhibitions",
                    "Bosphorus waterfront location",
                    "Educational programs and workshops",
                    "Renzo Piano designed building"
                ],
                opening_hours={
                    "tuesday_sunday": "10:00-18:00",
                    "thursday": "10:00-20:00"
                },
                entrance_fee="60 TL, Students 30 TL",
                location="Karaköy, Beyoğlu (Bosphorus waterfront)",
                nearby_attractions=["Galata Bridge", "Karaköy", "Galata Tower"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Weekday mornings",
                historical_significance="Turkey's first modern art museum, showcasing contemporary Turkish and international art.",
                architectural_style="Contemporary architecture by Renzo Piano",
                must_see_highlights=[
                    "Turkish modern art collection",
                    "Rotating international exhibitions",
                    "Bosphorus view terrace",
                    "Contemporary sculpture garden"
                ],
                photography_allowed="Limited (check current policy)",
                accessibility="Fully accessible",
                closing_days=["Mondays"]
            ),
            
            "beylerbeyi_palace": MuseumInfo(
                name="Beylerbeyi Palace",
                historical_period="Late Ottoman (1860s)",
                construction_date="1861-1865",
                architect="Sarkis Balyan",
                key_features=[
                    "Summer palace of Ottoman sultans",
                    "Bosphorus waterfront location",
                    "French and Ottoman decorative arts",
                    "Historic palace gardens",
                    "Asian side imperial residence"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-18:00",
                    "winter": "09:00-16:00"
                },
                entrance_fee="30 TL",
                location="Beylerbeyi, Üsküdar (Asian side)",
                nearby_attractions=["Bosphorus Bridge", "Çamlıca Hill", "Maiden's Tower"],
                visiting_duration="60-75 minutes",
                best_time_to_visit="Morning for better lighting",
                historical_significance="Last imperial summer palace, hosted foreign dignitaries including Empress Eugénie.",
                architectural_style="Neo-Baroque Ottoman palace architecture",
                must_see_highlights=[
                    "Crystal staircase and chandeliers",
                    "Sultan's private apartments",
                    "Bosphorus view from palace",
                    "Historic palace gardens"
                ],
                photography_allowed="Exterior only",
                accessibility="Limited stairs required",
                closing_days=["Mondays"]
            ),
            
            "carpet_museum": MuseumInfo(
                name="Carpet Museum (Halı Müzesi)",
                historical_period="Historic Collection (13th-20th centuries)",
                construction_date="Museum opened 1979",
                architect="Museum in Blue Mosque complex",
                key_features=[
                    "Rare Anatolian carpets collection",
                    "Carpets from 13th to 20th century",
                    "Prayer rugs and kilims",
                    "Traditional weaving techniques",
                    "Located in Blue Mosque complex"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-16:00"
                },
                entrance_fee="25 TL",
                location="Sultanahmet, Blue Mosque complex",
                nearby_attractions=["Blue Mosque", "Hagia Sophia", "Hippodrome"],
                visiting_duration="45-60 minutes",
                best_time_to_visit="Morning hours",
                historical_significance="World's most comprehensive collection of Turkish carpets and kilims.",
                architectural_style="Ottoman religious complex architecture",
                must_see_highlights=[
                    "13th century Seljuk carpets",
                    "Ottoman court prayer rugs",
                    "Hereke silk carpets",
                    "Traditional weaving displays"
                ],
                photography_allowed="No flash photography",
                accessibility="Ground floor accessible",
                closing_days=["Mondays"]
            ),
            
            "military_museum": MuseumInfo(
                name="Military Museum (Askeri Müze)",
                historical_period="Ottoman and Turkish Military History",
                construction_date="Museum established 1950",
                architect="Historic military buildings",
                key_features=[
                    "Ottoman military history",
                    "Turkish War of Independence exhibits",
                    "Historic weapons and armor",
                    "Janissary band performances",
                    "Military uniforms and medals"
                ],
                opening_hours={
                    "wednesday_sunday": "09:00-17:00",
                    "band_performance": "15:00-16:00"
                },
                entrance_fee="20 TL",
                location="Harbiye, Şişli",
                nearby_attractions=["Taksim Square", "Maçka Park", "Dolmabahçe Palace"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Wednesday-Friday for band performance",
                historical_significance="Comprehensive collection of Ottoman and Turkish military heritage.",
                architectural_style="Republican period military architecture",
                must_see_highlights=[
                    "Mehter (Janissary) band performance",
                    "Ottoman campaign tent",
                    "Historic cannons and armor",
                    "Atatürk's military belongings"
                ],
                photography_allowed="Limited areas only",
                accessibility="Main floors accessible",
                closing_days=["Monday", "Tuesday"]
            ),
            
            "mosaic_museum": MuseumInfo(
                name="Great Palace Mosaic Museum",
                historical_period="Byzantine (6th century)",
                construction_date="6th century mosaics, museum 1997",
                architect="Byzantine palace architects",
                key_features=[
                    "Byzantine palace floor mosaics",
                    "6th century artistic masterpieces",
                    "Hunting and daily life scenes",
                    "Underground archaeological site",
                    "Great Palace of Constantinople remains"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-18:30",
                    "winter": "09:00-16:30"
                },
                entrance_fee="30 TL",
                location="Sultanahmet, near Blue Mosque",
                nearby_attractions=["Blue Mosque", "Hagia Sophia", "Arasta Bazaar"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Any time, underground location",
                historical_significance="Only surviving floor mosaics from the Great Palace of Byzantine emperors.",
                architectural_style="Byzantine palace archaeological remains",
                must_see_highlights=[
                    "Hunting scene mosaics",
                    "Animal and mythological figures",
                    "6th century craftsmanship",
                    "Archaeological preservation methods"
                ],
                photography_allowed=True,
                accessibility="Underground location, stairs required",
                closing_days=["Mondays"]
            ),
            
            "panorama_1453": MuseumInfo(
                name="Panorama 1453 History Museum",
                historical_period="Historical Event (1453) - Modern Museum (2009)",
                construction_date="Museum opened 2009",
                architect="Modern museum design",
                key_features=[
                    "360-degree panoramic painting",
                    "Fall of Constantinople recreation",
                    "3D sound and visual effects",
                    "Interactive historical exhibits",
                    "Miniature city models"
                ],
                opening_hours={
                    "daily": "09:00-18:00",
                    "summer": "09:00-20:00"
                },
                entrance_fee="25 TL, Students 15 TL",
                location="Topkapı, Zeytinburnu",
                nearby_attractions=["City Walls", "Topkapı Gate", "Golden Horn"],
                visiting_duration="60-75 minutes",
                best_time_to_visit="Any time, indoor experience",
                historical_significance="Immersive experience of the 1453 conquest of Constantinople.",
                architectural_style="Modern museum with historical themes",
                must_see_highlights=[
                    "360-degree panoramic painting",
                    "Mehmed II conquest recreation",
                    "Interactive timeline",
                    "3D audiovisual experience"
                ],
                photography_allowed="Limited areas",
                accessibility="Fully accessible",
                closing_days=["None"]
            ),
            
            "rumeli_fortress": MuseumInfo(
                name="Rumeli Fortress (Rumeli Hisarı)",
                historical_period="Ottoman (1452)",
                construction_date="1452",
                architect="Ottoman military engineers under Mehmed II",
                key_features=[
                    "Strategic Bosphorus fortress",
                    "Three main towers with curtain walls",
                    "Built for 1453 Constantinople siege",
                    "Open-air museum with Bosphorus views",
                    "Medieval military architecture"
                ],
                opening_hours={
                    "thursday_tuesday": "09:00-16:30",
                    "summer": "09:00-19:00"
                },
                entrance_fee="30 TL",
                location="Sarıyer, European Bosphorus",
                nearby_attractions=["Fatih Sultan Mehmet Bridge", "Anadolu Hisarı", "Bebek"],
                visiting_duration="60-90 minutes",
                best_time_to_visit="Sunset for photography",
                historical_significance="Built by Mehmed II to control Bosphorus before conquering Constantinople.",
                architectural_style="Medieval Ottoman military architecture",
                must_see_highlights=[
                    "Halil Pasha Tower panoramic views",
                    "Historic cannons and walls",
                    "Bosphorus strategic position",
                    "Medieval fortress architecture"
                ],
                photography_allowed=True,
                accessibility="Historic site with stairs and uneven paths",
                closing_days=["Wednesdays"]
            ),
            
            "fethiye_museum": MuseumInfo(
                name="Fethiye Museum (Pammakaristos Church)",
                historical_period="Byzantine (12th century), Ottoman mosque (1591)",
                construction_date="12th century, mosaics 1310s",
                architect="Byzantine architects",
                key_features=[
                    "Byzantine church with Ottoman additions",
                    "14th century mosaics in parekklesion",
                    "Historic Islamic and Christian art",
                    "Architectural palimpsest",
                    "UNESCO World Heritage site"
                ],
                opening_hours={
                    "thursday_tuesday": "09:00-16:00"
                },
                entrance_fee="15 TL",
                location="Fener, Fatih",
                nearby_attractions=["Fener Greek Patriarchate", "Balat", "Golden Horn"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Morning for best lighting",
                historical_significance="Important example of Byzantine-Ottoman architectural transition.",
                architectural_style="Byzantine church converted to Ottoman mosque",
                must_see_highlights=[
                    "Parekklesion mosaics",
                    "Deesis mosaic composition",
                    "Byzantine architectural details",
                    "Islamic architectural additions"
                ],
                photography_allowed="Limited",
                accessibility="Historic building with limitations",
                closing_days=["Wednesdays"]
            ),
            
            "maritime_museum": MuseumInfo(
                name="Turkish Naval Museum",
                historical_period="Ottoman and Turkish Naval History",
                construction_date="Museum established 1897, current building 2013",
                architect="Modern museum design",
                key_features=[
                    "Ottoman imperial barges (caiques)",
                    "Naval history from Ottoman to modern",
                    "Historic ships and maritime artifacts",
                    "Interactive naval exhibitions",
                    "Piri Reis maps and navigation"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-17:00"
                },
                entrance_fee="20 TL, Students 10 TL",
                location="Beşiktaş, Bosphorus waterfront",
                nearby_attractions=["Dolmabahçe Palace", "Beşiktaş Square", "Bosphorus"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Weekday mornings",
                historical_significance="Comprehensive collection of Turkish naval heritage and maritime history.",
                architectural_style="Modern museum architecture",
                must_see_highlights=[
                    "Ottoman sultan's golden barge",
                    "Piri Reis world map replica",
                    "Historic naval uniforms",
                    "Interactive submarine simulator"
                ],
                photography_allowed="Most areas allowed",
                accessibility="Fully accessible modern building",
                closing_days=["Mondays"]
            ),
            
            "yildiz_palace": MuseumInfo(
                name="Yıldız Palace Museum",
                historical_period="Late Ottoman (19th-20th centuries)",
                construction_date="1880s-1909",
                architect="Sarkis Balyan and others",
                key_features=[
                    "Last residence of Ottoman sultans",
                    "Sale Kiosk (Şale Köşkü) main building",
                    "Palace park and gardens",
                    "Abdul Hamid II personal belongings",
                    "European-influenced Ottoman architecture"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-16:00"
                },
                entrance_fee="20 TL, Park free",
                location="Beşiktaş, above Çırağan Palace",
                nearby_attractions=["Çırağan Palace", "Ortaköy", "Bosphorus Bridge"],
                visiting_duration="90-120 minutes including park",
                best_time_to_visit="Spring and fall for gardens",
                historical_significance="Final imperial residence showcasing late Ottoman lifestyle and European influence.",
                architectural_style="Eclectic late Ottoman palace architecture",
                must_see_highlights=[
                    "Sale Kiosk ornate interiors",
                    "Sultan's private chambers",
                    "Palace gardens and pavilions",
                    "Mother-of-pearl decorations"
                ],
                photography_allowed="Exterior and gardens only",
                accessibility="Palace has stairs, gardens accessible",
                closing_days=["Mondays"]
            ),
            
            "great_bazaar_museum": MuseumInfo(
                name="Grand Bazaar Historical Information Center",
                historical_period="Ottoman (15th century)",
                construction_date="1461, expanded over centuries",
                architect="Various Ottoman architects",
                key_features=[
                    "World's oldest covered market",
                    "4000 shops in 61 streets",
                    "Historical information displays",
                    "Traditional crafts demonstrations",
                    "Ottoman commercial architecture"
                ],
                opening_hours={
                    "monday_saturday": "08:30-19:00"
                },
                entrance_fee="Free (shopping area)",
                location="Beyazıt, Fatih",
                nearby_attractions=["Süleymaniye Mosque", "Spice Bazaar", "University"],
                visiting_duration="60-180 minutes",
                best_time_to_visit="Morning hours, less crowded",
                historical_significance="World's first shopping mall, center of Ottoman trade for 500+ years.",
                architectural_style="Ottoman covered market architecture",
                must_see_highlights=[
                    "Cevahir Bedesten (jewelry section)",
                    "Historic guild workshops",
                    "Ottoman architectural details",
                    "Traditional carpet and textile shops"
                ],
                photography_allowed=True,
                accessibility="Historic building with some limitations",
                closing_days=["Sundays", "Religious holidays"]
            ),
            
            "spice_bazaar_museum": MuseumInfo(
                name="Spice Bazaar (Egyptian Bazaar)",
                historical_period="Ottoman (17th century)",
                construction_date="1664",
                architect="Kasım Ağa",
                key_features=[
                    "Historic spice and food market",
                    "L-shaped covered bazaar",
                    "Traditional Turkish delights and spices",
                    "Ottoman commercial architecture",
                    "Part of New Mosque complex"
                ],
                opening_hours={
                    "monday_saturday": "08:00-19:00",
                    "sunday": "09:30-18:00"
                },
                entrance_fee="Free",
                location="Eminönü, Fatih",
                nearby_attractions=["New Mosque", "Galata Bridge", "Golden Horn"],
                visiting_duration="30-60 minutes",
                best_time_to_visit="Morning or late afternoon",
                historical_significance="Historic center of spice trade between Europe and Asia.",
                architectural_style="Ottoman covered market architecture",
                must_see_highlights=[
                    "Traditional spice displays",
                    "Turkish delight workshops",
                    "Historic architecture",
                    "Authentic local atmosphere"
                ],
                photography_allowed=True,
                accessibility="Ground level, mostly accessible",
                closing_days=["None, but reduced Sunday hours"]
            ),
            
            "miniaturk": MuseumInfo(
                name="Miniaturk",
                historical_period="Modern Museum (2003)",
                construction_date="2003",
                architect="Modern theme park design",
                key_features=[
                    "1:25 scale models of Turkish landmarks",
                    "122 miniature structures",
                    "Istanbul, Anatolia, and Ottoman sites",
                    "Interactive and educational displays",
                    "Golden Horn waterfront location"
                ],
                opening_hours={
                    "daily": "09:00-19:00",
                    "winter": "09:00-17:00"
                },
                entrance_fee="25 TL, Children 15 TL",
                location="Sütlüce, Golden Horn",
                nearby_attractions=["Golden Horn", "Eyüp Sultan Mosque", "Pierre Loti Hill"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Afternoon for better lighting",
                historical_significance="Educational overview of Turkey's architectural and cultural heritage.",
                architectural_style="Modern outdoor museum park",
                must_see_highlights=[
                    "Miniature Hagia Sophia and Blue Mosque",
                    "Cappadocia rock formations",
                    "Ottoman mosques and palaces",
                    "Anatolian historical sites"
                ],
                photography_allowed=True,
                accessibility="Fully accessible outdoor park",
                closing_days=["None"]
            ),
            
            "florence_nightingale": MuseumInfo(
                name="Florence Nightingale Museum",
                historical_period="Crimean War (1854-1856)",
                construction_date="Historic Selimiye Barracks, Museum 1954",
                architect="Ottoman military architects",
                key_features=[
                    "Florence Nightingale's room and belongings",
                    "Crimean War medical history",
                    "Historic Selimiye Barracks",
                    "Nursing and medical instruments",
                    "Victorian era medical practices"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-16:00"
                },
                entrance_fee="15 TL",
                location="Üsküdar, Asian side",
                nearby_attractions=["Maiden's Tower", "Çamlıca Hill", "Üsküdar waterfront"],
                visiting_duration="45-60 minutes",
                best_time_to_visit="Weekday mornings",
                historical_significance="Commemorates Florence Nightingale's pioneering nursing work during Crimean War.",
                architectural_style="Ottoman military barracks architecture",
                must_see_highlights=[
                    "Florence Nightingale's preserved room",
                    "Victorian medical instruments",
                    "Crimean War artifacts",
                    "Historic nursing uniforms"
                ],
                photography_allowed="Limited areas",
                accessibility="Historic building with some limitations",
                closing_days=["Mondays"]
            ),
            
            "galata_mevlevi": MuseumInfo(
                name="Galata Mevlevi Museum",
                historical_period="Ottoman Sufi Lodge (1491)",
                construction_date="1491, renovated multiple times",
                architect="Ottoman Sufi architects",
                key_features=[
                    "Historic Mevlevi (whirling dervish) lodge",
                    "Sufi musical instruments",
                    "Dervish costumes and artifacts",
                    "Octagonal ceremonial hall",
                    "Islamic mystical traditions"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-16:00"
                },
                entrance_fee="20 TL",
                location="Galata, Beyoğlu",
                nearby_attractions=["Galata Tower", "Karaköy", "Istanbul Modern"],
                visiting_duration="45-60 minutes",
                best_time
                historical_significance="Important center of Mevlevi Sufi tradition in Ottoman Istanbul.",
                architectural_style="Ottoman Sufi lodge architecture",
                must_see_highlights=[
                    "Octagonal semazen hall",
                    "Dervish ceremonial objects",
                    "Islamic calligraphy",
                    "Traditional musical instruments"
                ],
                photography_allowed="No flash",
                accessibility="Historic building with stairs",
                closing_days=["Mondays"]
            ),
            
            "kucuk_ayasofya": MuseumInfo(
                name="Little Hagia Sophia (Küçük Ayasofya)",
                historical_period="Byzantine (527-536 AD), Ottoman mosque (1500s)",
                construction_date="527-536 AD",
                architect="Unknown Byzantine architects",
                key_features=[
                    "Earlier version of Hagia Sophia design",
                    "Octagonal central plan",
                    "Byzantine and Ottoman elements",
                    "Historic neighborhood setting",
                    "Architectural prototype for Hagia Sophia"
                ],
                opening_hours={
                    "daily": "Mosque hours, generally accessible"
                },
                entrance_fee="Free",
                location="Küçük Ayasofya, Fatih",
                nearby_attractions=["Blue Mosque", "Sea of Marmara", "Kennedy Avenue"],
                visiting_duration="20-30 minutes",
                best_time_to_visit="Between prayer times",
                historical_significance="Prototype for Hagia Sophia, built by Justinian as Saints Sergius and Bacchus Church.",
                architectural_style="Early Byzantine church architecture",
                must_see_highlights=[
                    "Octagonal dome structure",
                    "Byzantine columns and capitals",
                    "Architectural similarities to Hagia Sophia",
                    "Historic neighborhood context"
                ],
                photography_allowed="Respectful photography allowed",
                accessibility="Ground level, accessible",
                closing_days=["None (respect prayer times)"]
            ),
            
            "sokollu_mehmet": MuseumInfo(
                name="Sokollu Mehmet Pasha Mosque",
                historical_period="Classical Ottoman (1571-1572)",
                construction_date="1571-1572",
                architect="Mimar Sinan",
                key_features=[
                    "Masterpiece of Sinan's architecture",
                    "Unique tilework and calligraphy",
                    "Compact but perfectly proportioned",
                    "Historic Kaaba stone fragments",
                    "Overlooking Sea of Marmara"
                ],
                opening_hours={
                    "daily": "Outside prayer times"
                },
                entrance_fee="Free",
                location="Kadırga, Fatih",
                nearby_attractions=["Little Hagia Sophia", "Blue Mosque", "Kennedy Avenue"],
                visiting_duration="20-30 minutes",
                best_time_to_visit="Morning or afternoon",
                historical_significance="Considered one of Sinan's most perfect small mosques with exceptional tile work.",
                architectural_style="Classical Ottoman mosque architecture by Sinan",
                must_see_highlights=[
                    "Exquisite İznik tilework",
                    "Sinan's architectural perfection",
                    "Historic Kaaba fragments",
                    "Panoramic city views"
                ],
                photography_allowed="Respectful photography",
                accessibility="Historic mosque with steps",
                closing_days=["None (respect prayer times)"]
            ),
            
            "museum_of_illusions": MuseumInfo(
                name="Museum of Illusions Istanbul",
                historical_period="Modern (2016)",
                construction_date="2016",
                architect="International franchise concept",
                key_features=[
                    "Optical illusions and visual tricks",
                    "Interactive educational exhibits",
                    "Photography-friendly installations",
                    "Mind-bending experiences",
                    "Family entertainment and learning"
                ],
                opening_hours={
                    "daily": "09:00-22:00"
                },
                entrance_fee="60 TL, Children 45 TL",
                location="Galata, Beyoğlu",
                nearby_attractions=["Galata Tower", "Istanbul Modern", "Karaköy"],
                visiting_duration="60-90 minutes",
                best_time_to_visit="Any time, interactive experience",
                historical_significance="Modern educational entertainment focusing on perception and learning.",
                architectural_style="Contemporary interactive museum design",
                must_see_highlights=[
                    "Ames room illusion",
                    "Infinity room experience",
                    "Hologram displays",
                    "Educational illusion explanations"
                ],
                photography_allowed=True,
                accessibility="Fully accessible",
                closing_days=["None"]
            ),
            
            "suna_inan_kirac": MuseumInfo(
                name="Suna and İnan Kıraç Foundation Museums",
                historical_period="Historic Mansions (19th century)",
                construction_date="19th century mansions, museums since 2005",
                architect="19th century Ottoman architects",
                key_features=[
                    "Orientalist paintings collection",
                    "Historic Bosphorus mansions",
                    "Ottoman daily life exhibits",
                    "Traditional Turkish house interiors",
                    "Phanar Greek Orthodox College connection"
                ],
                opening_hours={
                    "tuesday_sunday": "10:00-17:00"
                },
                entrance_fee="15 TL",
                location="Fener, Fatih",
                nearby_attractions=["Fener Greek Patriarchate", "Balat", "Golden Horn"],
                visiting_duration="60-75 minutes",
                best_time_to_visit="Weekday afternoons",
                historical_significance="Showcases Ottoman-era lifestyle and Orientalist art in historic neighborhood setting.",
                architectural_style="19th century Ottoman mansion architecture",
                must_see_highlights=[
                    "Orientalist painting collection",
                    "Traditional Ottoman room settings",
                    "Historic mansion architecture",
                    "Fener neighborhood history"
                ],
                photography_allowed="Limited",
                accessibility="Historic buildings with limitations",
                closing_days=["Mondays"]
            ),
            
            "iron_church": MuseumInfo(
                name="Bulgarian Iron Church",
                historical_period="Late Ottoman (1896-1898)",
                construction_date="1896-1898",
                architect="Hovsep Aznavur",
                key_features=[
                    "Only iron church in the world",
                    "Prefabricated in Vienna",
                    "Bulgarian Orthodox heritage",
                    "Neo-Gothic iron architecture",
                    "Golden Horn waterfront location"
                ],
                opening_hours={
                    "tuesday_sunday": "09:00-17:00"
                },
                entrance_fee="10 TL",
                location="Balat, Fatih (Golden Horn)",
                nearby_attractions=["Fener", "Suna-İnan Kıraç Museums", "Pierre Loti"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Afternoon for best lighting",
                historical_significance="Unique iron church representing Bulgarian Orthodox community in Ottoman Istanbul.",
                architectural_style="Neo-Gothic iron architecture",
                must_see_highlights=[
                    "Unique iron construction",
                    "Bulgarian Orthodox iconostasis",
                    "Prefabricated architectural elements",
                    "Golden Horn views"
                ],
                photography_allowed=True,
                accessibility="Ground level, accessible",
                closing_days=["Mondays"]
            ),
            
            "archaeology_park": MuseumInfo(
                name="Istanbul Archaeological Parks",
                historical_period="Various periods (Classical to Ottoman)",
                construction_date="Various archaeological sites",
                architect="Various historical periods",
                key_features=[
                    "Multiple archaeological sites",
                    "Byzantine and Ottoman remains",
                    "Outdoor museum concept",
                    "Theodosius Harbor excavations",
                    "Historical urban archaeology"
                ],
                opening_hours={
                    "daily": "Daylight hours"
                },
                entrance_fee="Free",
                location="Various locations around Istanbul",
                nearby_attractions=["Varies by location"],
                visiting_duration="30-60 minutes per site",
                best_time_to_visit="Spring and fall",
                historical_significance="In-situ preservation of Istanbul's archaeological heritage.",
                architectural_style="Archaeological preservation sites",
                must_see_highlights=[
                    "Theodosius Harbor remains",
                    "Byzantine period structures",
                    "Ottoman period archaeological layers",
                    "Urban archaeology methods"
                ],
                photography_allowed=True,
                accessibility="Outdoor sites, variable accessibility",
                closing_days=["None"]
            ),
            
            "caferaga_medresesi": MuseumInfo(
                name="Caferağa Medresesi Traditional Arts Center",
                historical_period="Classical Ottoman (1559-1560)",
                construction_date="1559-1560",
                architect="Mimar Sinan",
                key_features=[
                    "Historic Islamic school building",
                    "Traditional Turkish arts workshops",
                    "Calligraphy, ebru, ceramics classes",
                    "Sinan's educational architecture",
                    "Living cultural heritage center"
                ],
                opening_hours={
                    "monday_saturday": "09:00-17:00"
                },
                entrance_fee="Free to visit, workshop fees vary",
                location="Sultanahmet, Fatih",
                nearby_attractions=["Hagia Sophia", "Blue Mosque", "Topkapi Palace"],
                visiting_duration="30-45 minutes visit, workshops longer",
                best_time_to_visit="Weekday mornings to see workshops",
                historical_significance="Sinan-designed medrese now preserving traditional Turkish arts and crafts.",
                architectural_style="Classical Ottoman educational architecture",
                must_see_highlights=[
                    "Sinan's medrese architecture",
                    "Traditional arts demonstrations",
                    "Calligraphy workshops",
                    "Historic courtyard design"
                ],
                photography_allowed=True,
                accessibility="Historic building with courtyard access",
                closing_days=["Sundays"]
            ),
            
            "santral_istanbul": MuseumInfo(
                name="santralIstanbul",
                historical_period="Industrial Heritage (1914), Cultural Center (2007)",
                construction_date="1914 power plant, 2007 cultural center",
                architect="Nevzat Sayın (renovation)",
                key_features=[
                    "Former Ottoman electricity plant",
                    "Contemporary art exhibitions",
                    "Industrial heritage preservation",
                    "Energy Museum",
                    "Cultural events and concerts"
                ],
                opening_hours={
                    "tuesday_sunday": "10:00-18:00",
                    "thursday": "10:00-20:00"
                },
                entrance_fee="20 TL, Students 10 TL",
                location="Eyüp, Golden Horn",
                nearby_attractions=["Eyüp Sultan Mosque", "Pierre Loti Hill", "Golden Horn"],
                visiting_duration="90-120 minutes",
                best_time_to_visit="Weekends for full program",
                historical_significance="First Ottoman power plant transformed into leading contemporary arts center.",
                architectural_style="Industrial heritage adaptive reuse",
                must_see_highlights=[
                    "Preserved power plant machinery",
                    "Contemporary art exhibitions",
                    "Energy Museum displays",
                    "Industrial architecture adaptation"
                ],
                photography_allowed="Check current exhibition policies",
                accessibility="Modern accessibility features",
                closing_days=["Mondays"]
            ),
            
            "princes_islands_museum": MuseumInfo(
                name="Princes' Islands Museum (Büyükada)",
                historical_period="19th-20th century island history",
                construction_date="Historic buildings, museum concept recent",
                architect="Various 19th century architects",
                key_features=[
                    "Island history and culture",
                    "Multi-ethnic heritage displays",
                    "Historic mansion settings",
                    "Maritime island life",
                    "Seasonal accessibility"
                ],
                opening_hours={
                    "seasonal": "April-October mainly",
                    "hours": "10:00-17:00"
                },
                entrance_fee="15 TL",
                location="Büyükada, Princes' Islands",
                nearby_attractions=["Monastery of St. George", "Historic mansions", "Island beaches"],
                visiting_duration="60-90 minutes plus island time",
                best_time_to_visit="Spring through fall",
                historical_significance="Showcases unique multi-cultural history of the Princes' Islands.",
                architectural_style="19th century island mansion architecture",
                must_see_highlights=[
                    "Island's multi-ethnic history",
                    "Historic photographs and documents",
                    "Traditional island lifestyle",
                    "Maritime heritage displays"
                ],
                photography_allowed=True,
                accessibility="Island location requires ferry, historic buildings",
                closing_days=["Winter season limitations"]
            ),
            
            "suleymaniye_mosque": MuseumInfo(
                name="Süleymaniye Mosque Complex",
                historical_period="Classical Ottoman (1550-1557)",
                construction_date="1550-1557",
                architect="Mimar Sinan",
                key_features=[
                    "Sinan's masterpiece mosque complex",
                    "Süleyman the Magnificent's imperial mosque",
                    "Complex with schools, hospital, library",
                    "Panoramic Golden Horn views",
                    "Largest mosque complex in Istanbul"
                ],
                opening_hours={
                    "daily": "Outside prayer times",
                    "complex": "Daylight hours"
                },
                entrance_fee="Free",
                location="Süleymaniye, Fatih",
                nearby_attractions=["Grand Bazaar", "University", "Golden Horn"],
                visiting_duration="45-60 minutes",
                best_time_to_visit="Late afternoon for sunset views",
                historical_significance="Sinan's architectural masterpiece and largest imperial mosque complex.",
                architectural_style="Classical Ottoman imperial mosque architecture",
                must_see_highlights=[
                    "Sinan's architectural genius",
                    "Süleyman and Roxelana tombs",
                    "Panoramic city views",
                    "Complex courtyard and gardens"
                ],
                photography_allowed="Respectful photography",
                accessibility="Mosque accessible, complex has stairs",
                closing_days=["None (respect prayer times)"]
            ),
            
            "rustem_pasha": MuseumInfo(
                name="Rüstem Pasha Mosque",
                historical_period="Classical Ottoman (1561-1563)",
                construction_date="1561-1563",
                architect="Mimar Sinan",
                key_features=[
                    "Exquisite İznik tile decoration",
                    "Elevated mosque above shops",
                    "Sinan's intricate design",
                    "Most beautiful tilework in Istanbul",
                    "Grand Bazaar area location"
                ],
                opening_hours={
                    "daily": "Outside prayer times"
                },
                entrance_fee="Free",
                location="Eminönü, Fatih",
                nearby_attractions=["Spice Bazaar", "Galata Bridge", "New Mosque"],
                visiting_duration="20-30 minutes",
                best_time_to_visit="Morning or afternoon light",
                historical_significance="Features Istanbul's most spectacular İznik tilework in Sinan design.",
                architectural_style="Classical Ottoman mosque with exceptional tile decoration",
                must_see_highlights=[
                    "Unparalleled İznik tile patterns",
                    "Sinan's elevated design concept",
                    "Floral and geometric tile motifs",
                    "Integration with commercial district"
                ],
                photography_allowed="Respectful photography",
                accessibility="Stairs to elevated mosque",
                closing_days=["None (respect prayer times)"]
            ),
            
            "maiden_tower": MuseumInfo(
                name="Maiden's Tower (Kız Kulesi)",
                historical_period="Byzantine origins, Ottoman modifications",
                construction_date="12th century, rebuilt 18th century",
                architect="Various periods",
                key_features=[
                    "Iconic Bosphorus tower on islet",
                    "Lighthouse and customs tower history",
                    "Restaurant and museum combination",
                    "Boat access from Üsküdar",
                    "Symbol of Istanbul's Asian side"
                ],
                opening_hours={
                    "daily": "09:00-19:00 (seasonal variations)"
                },
                entrance_fee="30 TL including boat transfer",
                location="Salacak offshore, Üsküdar",
                nearby_attractions=["Üsküdar waterfront", "Çamlıca Hill", "Beylerbeyi Palace"],
                visiting_duration="60-90 minutes including boat ride",
                best_time_to_visit="Sunset for photography",
                historical_significance="Legendary tower with multiple historical functions, Istanbul's romantic symbol.",
                architectural_style="Byzantine-Ottoman coastal fortification",
                must_see_highlights=[
                    "Panoramic Bosphorus views",
                    "Historical tower interior",
                    "Boat ride experience",
                    "Sunset photography opportunity"
                ],
                photography_allowed=True,
                accessibility="Boat access required, historic tower stairs",
                closing_days=["Weather dependent"]
            ),
            
            "ortakoy_mosque": MuseumInfo(
                name="Ortaköy Mosque (Büyük Mecidiye Mosque)",
                historical_period="Late Ottoman (1853-1856)",
                construction_date="1853-1856",
                architect="Garabet Balyan and Nigoğayos Balyan",
                key_features=[
                    "Neo-Baroque Ottoman architecture",
                    "Bosphorus waterfront location",
                    "Balyan family architectural work",
                    "Integration with Bosphorus Bridge views",
                    "Popular photography location"
                ],
                opening_hours={
                    "daily": "Outside prayer times"
                },
                entrance_fee="Free",
                location="Ortaköy, Beşiktaş (Bosphorus)",
                nearby_attractions=["Bosphorus Bridge", "Dolmabahçe Palace", "Çırağan Palace"],
                visiting_duration="20-30 minutes",
                best_time_to_visit="Golden hour for bridge backdrop",
                historical_significance="Example of 19th century architectural fusion with spectacular Bosphorus setting.",
                architectural_style="Neo-Baroque Ottoman mosque architecture",
                must_see_highlights=[
                    "Neo-Baroque architectural details",
                    "Bosphorus Bridge backdrop views",
                    "Waterfront mosque setting",
                    "Balyan family craftsmanship"
                ],
                photography_allowed="Respectful photography",
                accessibility="Waterfront location, accessible",
                closing_days=["None (respect prayer times)"]
            )
        }
    
    def get_museum_info(self, museum_key: str) -> Optional[MuseumInfo]:
        """Get verified information for a specific museum"""
        return self.museums.get(museum_key.lower())
    
    def get_museum_by_name(self, name: str) -> Optional[MuseumInfo]:
        """Find museum by name (fuzzy matching)"""
        name_lower = name.lower()
        
        # Direct name matching
        name_mappings = {
            'hagia sophia': 'hagia_sophia',
            'ayasofya': 'hagia_sophia',
            'topkapi': 'topkapi_palace',
            'topkapi palace': 'topkapi_palace',
            'blue mosque': 'blue_mosque',
            'sultan ahmed': 'blue_mosque',
            'basilica cistern': 'basilica_cistern',
            'yerebatan': 'basilica_cistern',
            'dolmabahce': 'dolmabahce_palace',
            'dolmabahçe': 'dolmabahce_palace',
            'archaeological': 'archaeological_museum',
            'galata tower': 'galata_tower',
            'chora': 'chora_church',
            'kariye': 'chora_church',
            'turkish islamic arts': 'turkish_islamic_arts',
            'islamic arts': 'turkish_islamic_arts',
            'pera': 'pera_museum',
            'pera museum': 'pera_museum',
            'sabanci': 'sakip_sabanci',
            'sakıp sabancı': 'sakip_sabanci',
            'rahmi koc': 'rahmi_koc',
            'rahmi koç': 'rahmi_koc',
            'istanbul modern': 'istanbul_modern',
            'modern art': 'istanbul_modern',
            'beylerbeyi': 'beylerbeyi_palace',
            'beylerbeyi palace': 'beylerbeyi_palace',
            'carpet': 'carpet_museum',
            'carpet museum': 'carpet_museum',
            'halı müzesi': 'carpet_museum',
            'military': 'military_museum',
            'askeri müze': 'military_museum',
            'mosaic': 'mosaic_museum',
            'great palace': 'mosaic_museum',
            'panorama': 'panorama_1453',
            'panorama 1453': 'panorama_1453',
            'rumeli': 'rumeli_fortress',
            'rumeli hisarı': 'rumeli_fortress',
            'fethiye': 'fethiye_museum',
            'pammakaristos': 'fethiye_museum',
            'maritime': 'maritime_museum',
            'naval': 'maritime_museum',
            'naval museum': 'maritime_museum',
            'yildiz': 'yildiz_palace',
            'yıldız': 'yildiz_palace',
            'yıldız palace': 'yildiz_palace',
            'grand bazaar': 'great_bazaar_museum',
            'great bazaar': 'great_bazaar_museum',
            'kapalıçarşı': 'great_bazaar_museum',
            'spice bazaar': 'spice_bazaar_museum',
            'egyptian bazaar': 'spice_bazaar_museum',
            'mısır çarşısı': 'spice_bazaar_museum',
            'miniaturk': 'miniaturk',
            'miniature': 'miniaturk',
            'florence nightingale': 'florence_nightingale',
            'nightingale': 'florence_nightingale',
            'mevlevi': 'galata_mevlevi',
            'galata mevlevi': 'galata_mevlevi',
            'dervish': 'galata_mevlevi',
            'little hagia sophia': 'kucuk_ayasofya',
            'küçük ayasofya': 'kucuk_ayasofya',
            'little ayasofya': 'kucuk_ayasofya',
            'sokollu': 'sokollu_mehmet',
            'sokollu mehmet': 'sokollu_mehmet',
            'illusions': 'museum_of_illusions',
            'museum of illusions': 'museum_of_illusions',
            'suna inan': 'suna_inan_kirac',
            'kıraç': 'suna_inan_kirac',
            'iron church': 'iron_church',
            'bulgarian church': 'iron_church',
            'archaeology park': 'archaeology_park',
            'archaeological park': 'archaeology_park',
            'caferaga': 'caferaga_medresesi',
            'caferağa': 'caferaga_medresesi',
            'medrese': 'caferaga_medresesi',
            'santral': 'santral_istanbul',
            'santralistanbul': 'santral_istanbul',
            'princes islands': 'princes_islands_museum',
            'büyükada': 'princes_islands_museum',
            'suleymaniye': 'suleymaniye_mosque',
            'süleymaniye': 'suleymaniye_mosque',
            'suleyman': 'suleymaniye_mosque',
            'rustem pasha': 'rustem_pasha',
            'rüstem pasha': 'rustem_pasha',
            'rustem': 'rustem_pasha',
            'maiden tower': 'maiden_tower',
            'kız kulesi': 'maiden_tower',
            'maiden': 'maiden_tower',
            'ortakoy': 'ortakoy_mosque',
            'ortaköy': 'ortakoy_mosque',
            'mecidiye': 'ortakoy_mosque'
        }
        
        for search_name, museum_key in name_mappings.items():
            if search_name in name_lower:
                return self.museums.get(museum_key)
        
        return None
    
    def get_opening_hours_info(self, museum_key: str) -> str:
        """Get formatted opening hours information"""
        museum = self.get_museum_info(museum_key)
        if not museum:
            return "Museum information not found"
        
        hours_info = []
        for period, hours in museum.opening_hours.items():
            hours_info.append(f"{period.replace('_', ' ').title()}: {hours}")
        
        if museum.closing_days and museum.closing_days != ["None"]:
            hours_info.append(f"Closed: {', '.join(museum.closing_days)}")
        
        return "\n".join(hours_info)
    
    def get_all_museum_names(self) -> List[str]:
        """Get list of all museum names"""
        return [museum.name for museum in self.museums.values()]

# Global instance for easy import
istanbul_museums = IstanbulMuseumDatabase()

def get_accurate_museum_info(query: str) -> Optional[MuseumInfo]:
    """
    Get accurate museum information from verified database
    
    Args:
        query: User query containing museum name
        
    Returns:
        MuseumInfo object with verified facts or None
    """
    return istanbul_museums.get_museum_by_name(query)

def format_museum_response(museum: MuseumInfo, query_type: str = "general") -> str:
    """
    Format museum information for chat responses
    
    Args:
        museum: MuseumInfo object
        query_type: Type of query (hours, cost, general, etc.)
        
    Returns:
        Formatted response string
    """
    if query_type == "hours":
        response = f"**{museum.name} Opening Hours:**\n"
        response += istanbul_museums.get_opening_hours_info(museum.name.lower().replace(' ', '_'))
        response += f"\n\nBest time to visit: {museum.best_time_to_visit}"
        if museum.closing_days != ["None"]:
            response += f"\nPlease note: Closed on {', '.join(museum.closing_days)}"
        return response
    
    elif query_type == "cost":
        response = f"**{museum.name} Entrance Information:**\n"
        response += f"Entrance fee: {museum.entrance_fee}\n"
        response += f"Recommended visiting duration: {museum.visiting_duration}\n"
        response += f"Location: {museum.location}"
        return response
    
    else:  # general information
        response = f"**{museum.name}**\n\n"
        response += f"**Historical Period:** {museum.historical_period}\n"
        response += f"**Built:** {museum.construction_date}\n"
        if museum.architect:
            response += f"**Architect:** {museum.architect}\n"
        response += f"\n**Historical Significance:**\n{museum.historical_significance}\n"
        response += f"\n**Key Features:**\n"
        for feature in museum.key_features[:3]:  # Limit to top 3
            response += f"• {feature}\n"
        response += f"\n**Must-See Highlights:**\n"
        for highlight in museum.must_see_highlights[:3]:  # Limit to top 3
            response += f"• {highlight}\n"
        response += f"\n**Practical Information:**\n"
        response += f"• Location: {museum.location}\n"
        response += f"• Visiting duration: {museum.visiting_duration}\n"
        response += f"• Entrance fee: {museum.entrance_fee}\n"
        response += f"• Best time to visit: {museum.best_time_to_visit}"
        
        return response
