"""
Simple Istanbul Tourism Data Collection Demo
Week 1-2 Implementation - Basic working version with sample data
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SimpleIstanbulCollector:
    """Simple data collector with sample data for demonstration"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_sample_data(self) -> List[Dict]:
        """Generate comprehensive sample Istanbul tourism data"""
        return [
            # Historical Attractions
            {
                'id': 'hagia_sophia',
                'name': 'Hagia Sophia',
                'category': 'historical_sites',
                'location': 'Sultanahmet',
                'description': 'A former Byzantine church and Ottoman mosque, Hagia Sophia is now a museum that showcases Istanbul\'s rich multicultural history. Built in 537 AD, it was the world\'s largest cathedral for nearly 1000 years.',
                'rating': 4.7,
                'coordinates': {'lat': 41.0086, 'lng': 28.9802},
                'opening_hours': '09:00-17:00',
                'entrance_fee': 'Free',
                'source': 'istanbul_tourism_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'blue_mosque',
                'name': 'Blue Mosque (Sultan Ahmed Mosque)',
                'category': 'historical_sites',
                'location': 'Sultanahmet',
                'description': 'The Blue Mosque is famous for its distinctive blue tiles and six minarets. Built in the early 17th century, it remains an active place of worship while welcoming visitors outside prayer times.',
                'rating': 4.6,
                'coordinates': {'lat': 41.0054, 'lng': 28.9768},
                'opening_hours': '08:30-18:00',
                'entrance_fee': 'Free',
                'source': 'istanbul_tourism_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'topkapi_palace',
                'name': 'Topkapi Palace',
                'category': 'museums',
                'location': 'Sultanahmet',
                'description': 'Former residence of Ottoman sultans for over 400 years, Topkapi Palace is now a museum housing imperial treasures, including the famous Topkapi Dagger and sacred Islamic relics.',
                'rating': 4.5,
                'coordinates': {'lat': 41.0115, 'lng': 28.9833},
                'opening_hours': '09:00-18:00',
                'entrance_fee': '320 TL',
                'source': 'istanbul_tourism_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'grand_bazaar',
                'name': 'Grand Bazaar',
                'category': 'shopping',
                'location': 'Beyazıt',
                'description': 'One of the oldest and largest covered markets in the world, the Grand Bazaar features 4,000 shops selling everything from carpets and jewelry to spices and traditional crafts.',
                'rating': 4.3,
                'coordinates': {'lat': 41.0108, 'lng': 28.9681},
                'opening_hours': '09:00-19:00',
                'entrance_fee': 'Free',
                'source': 'istanbul_tourism_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'galata_tower',
                'name': 'Galata Tower',
                'category': 'landmarks',
                'location': 'Galata',
                'description': 'Built by the Genoese in 1348, Galata Tower offers panoramic views of Istanbul. The medieval stone tower has served various purposes throughout history and is now a popular tourist attraction.',
                'rating': 4.4,
                'coordinates': {'lat': 41.0256, 'lng': 28.9742},
                'opening_hours': '09:00-20:00',
                'entrance_fee': '150 TL',
                'source': 'istanbul_tourism_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            
            # Dining and Restaurants
            {
                'id': 'pandeli_restaurant',
                'name': 'Pandeli Restaurant',
                'category': 'restaurants',
                'location': 'Eminönü',
                'description': 'Historic Ottoman restaurant established in 1901, famous for traditional Turkish cuisine and its beautiful blue Iznik tile interior. Located above the Spice Bazaar.',
                'rating': 4.2,
                'cuisine_type': 'Turkish',
                'price_level': 'expensive',
                'coordinates': {'lat': 41.0168, 'lng': 28.9731},
                'opening_hours': '12:00-22:00',
                'source': 'dining_guide',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'ciya_sofrasi',
                'name': 'Çiya Sofrası',
                'category': 'restaurants',
                'location': 'Kadıköy',
                'description': 'Renowned restaurant specializing in authentic Anatolian cuisine, featuring forgotten regional dishes from across Turkey. A favorite among food enthusiasts and locals.',
                'rating': 4.6,
                'cuisine_type': 'Anatolian',
                'price_level': 'moderate',
                'coordinates': {'lat': 40.9834, 'lng': 29.0297},
                'opening_hours': '11:00-22:00',
                'source': 'dining_guide',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'hamdi_restaurant',
                'name': 'Hamdi Restaurant',
                'category': 'restaurants',
                'location': 'Eminönü',
                'description': 'Famous for its kebabs and traditional Turkish dishes, Hamdi offers stunning views of the Golden Horn. Established in 1960, it\'s known for high-quality meat dishes.',
                'rating': 4.4,
                'cuisine_type': 'Turkish Kebab',
                'price_level': 'moderate',
                'coordinates': {'lat': 41.0175, 'lng': 28.9726},
                'opening_hours': '11:30-23:00',
                'source': 'dining_guide',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            
            # Transportation
            {
                'id': 'metro_m1',
                'name': 'M1 Metro Line (Yenikapı-Atatürk Airport)',
                'category': 'transportation',
                'type': 'metro',
                'description': 'The M1 metro line connects central Istanbul to Atatürk Airport, serving major stops including Aksaray, Emniyet-Fatih, and Bayrampaşa-Maltepe.',
                'route_details': {
                    'line_color': 'red',
                    'total_stations': 23,
                    'journey_time': '45 minutes end-to-end',
                    'frequency': 'Every 5-10 minutes'
                },
                'fare': '15 TL',
                'operating_hours': '06:00-00:30',
                'accessibility': 'Wheelchair accessible',
                'source': 'iett_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'ferry_eminonu_uskudar',
                'name': 'Eminönü-Üsküdar Ferry',
                'category': 'transportation',
                'type': 'ferry',
                'description': 'Scenic ferry route across the Bosphorus connecting the European and Asian sides of Istanbul. Offers beautiful views of the city skyline.',
                'route_details': {
                    'departure': 'Eminönü',
                    'arrival': 'Üsküdar',
                    'journey_time': '15 minutes',
                    'frequency': 'Every 15-20 minutes'
                },
                'fare': '7 TL',
                'operating_hours': '06:30-23:30',
                'scenic_value': 'High',
                'source': 'ido_ferry',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'tram_t1',
                'name': 'T1 Tram Line (Kabataş-Bağcılar)',
                'category': 'transportation',
                'type': 'tram',
                'description': 'Modern tram line serving major tourist areas including Sultanahmet, Eminönü, and Galata Bridge. Essential for visitors exploring historic districts.',
                'route_details': {
                    'key_stops': ['Kabataş', 'Karaköy', 'Eminönü', 'Sultanahmet', 'Beyazıt'],
                    'journey_time': '25 minutes to historic area',
                    'frequency': 'Every 5-7 minutes'
                },
                'fare': '15 TL',
                'operating_hours': '06:00-00:00',
                'tourist_value': 'Very High',
                'source': 'iett_official',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            
            # Cultural Heritage
            {
                'id': 'ottoman_architecture',
                'title': 'Ottoman Architecture in Istanbul',
                'category': 'cultural_heritage',
                'content': 'Istanbul showcases the finest examples of Ottoman architecture, characterized by large central domes, smaller surrounding domes, and slender minarets. The Süleymaniye Mosque, built by architect Mimar Sinan, represents the peak of Ottoman architectural achievement.',
                'period': 'Ottoman Empire (1299-1922)',
                'significance': 'UNESCO World Heritage elements',
                'examples': ['Süleymaniye Mosque', 'Blue Mosque', 'Topkapi Palace'],
                'source': 'cultural_ministry',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'byzantine_heritage',
                'title': 'Byzantine Heritage of Constantinople',
                'category': 'cultural_heritage',
                'content': 'As the former Constantinople, Istanbul preserves significant Byzantine heritage including Hagia Sophia, the Basilica Cistern, and remnants of the Great Palace. These monuments reflect the city\'s role as the Eastern Roman Empire\'s capital.',
                'period': 'Byzantine Empire (330-1453)',
                'significance': 'Former capital of Eastern Roman Empire',
                'examples': ['Hagia Sophia', 'Basilica Cistern', 'Chora Church'],
                'source': 'cultural_ministry',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            
            # User Reviews (Sample)
            {
                'id': 'review_hagia_sophia_1',
                'place_name': 'Hagia Sophia',
                'place_id': 'hagia_sophia',
                'reviewer_type': 'tourist',
                'rating': 5,
                'review_text': 'Absolutely breathtaking! The architectural beauty and historical significance make this a must-visit. The blend of Christian and Islamic elements tells the story of Istanbul perfectly. Allow at least 2 hours to fully appreciate it.',
                'visit_date': '2024-10-01',
                'helpful_votes': 47,
                'verified_visit': True,
                'source': 'review_aggregator',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'review_grand_bazaar_1',
                'place_name': 'Grand Bazaar',
                'place_id': 'grand_bazaar',
                'reviewer_type': 'experienced_traveler',
                'rating': 4,
                'review_text': 'Amazing experience but be prepared to bargain! The variety of goods is incredible - carpets, jewelry, spices, and souvenirs. Start your negotiation at half the asking price. Can get crowded, so visit early morning for a better experience.',
                'visit_date': '2024-09-28',
                'helpful_votes': 23,
                'verified_visit': True,
                'source': 'review_aggregator',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            },
            
            # Practical Information
            {
                'id': 'istanbul_card_info',
                'title': 'Istanbul Tourist Transportation Card',
                'category': 'practical_info',
                'content': 'The Istanbul Card (Istanbulkart) is essential for public transportation. It works on metro, bus, tram, ferry, and funicular. You can buy and top up the card at machines in stations. Transfers between different transport modes offer discounts.',
                'tips': [
                    'Available at all metro stations and major stops',
                    'Minimum load: 20 TL',
                    'Transfers within 2 hours get discounted rates',
                    'Works for multiple passengers (tap for each person)'
                ],
                'cost_savings': 'Up to 50% compared to single tickets',
                'source': 'transportation_guide',
                'collected_at': datetime.now().isoformat(),
                'language': 'en'
            }
        ]
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """Simulate data collection and return summary"""
        logger.info("🚀 Starting sample data collection...")
        
        # Generate sample data
        sample_data = self.generate_sample_data()
        
        # Organize by category
        data_by_category = {}
        for item in sample_data:
            category = item.get('category', 'general')
            if category not in data_by_category:
                data_by_category[category] = []
            data_by_category[category].append(item)
        
        # Save data files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all data as one file
        all_data_file = self.output_dir / f"istanbul_sample_data_{timestamp}.json"
        with open(all_data_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        # Save by category
        for category, items in data_by_category.items():
            category_file = self.output_dir / f"istanbul_{category}_{timestamp}.json"
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
        
        # Create collection summary
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_records': len(sample_data),
            'records_by_source': {
                'sample_data_generator': len(sample_data)
            },
            'records_by_category': {cat: len(items) for cat, items in data_by_category.items()},
            'successful_sources': 1,
            'data_files_created': [
                str(all_data_file),
                *[str(self.output_dir / f"istanbul_{cat}_{timestamp}.json") for cat in data_by_category.keys()]
            ],
            'next_steps': [
                'Run data quality validation',
                'Format data for training',
                'Begin model training pipeline'
            ]
        }
        
        # Save summary
        summary_file = self.output_dir / f"collection_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Collection completed. Generated {len(sample_data)} sample records")
        logger.info(f"📁 Data saved to: {self.output_dir}")
        
        return summary

async def main():
    """Test the simple collector"""
    collector = SimpleIstanbulCollector()
    summary = await collector.collect_all_data()
    
    print("\n" + "="*60)
    print("ISTANBUL TOURISM SAMPLE DATA COLLECTION")
    print("="*60)
    print(f"Total Records: {summary['total_records']}")
    print(f"Categories: {', '.join(summary['records_by_category'].keys())}")
    print("\nRecords by Category:")
    for category, count in summary['records_by_category'].items():
        print(f"  {category}: {count}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
