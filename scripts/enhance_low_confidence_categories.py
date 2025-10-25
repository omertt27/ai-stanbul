"""
Enhancement Script for Low-Confidence Intent Categories
Adds targeted training data for:
- Local Tips / Hidden Gems
- Recommendations
- Events
- Neighborhoods
- Attractions
- Practical Info
- Price Inquiries
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

class LowConfidenceCategoryEnhancer:
    """Generate additional training data for low-confidence categories"""
    
    def __init__(self, base_data_path: str = "data/intent_training_data.json"):
        self.base_data_path = Path(base_data_path)
        self.enhancement_data = []
        
    def generate_local_tips_data(self) -> List[Dict]:
        """Generate 150+ training samples for local tips / hidden gems"""
        samples = []
        
        # Local tips patterns
        local_tips_queries = [
            # Direct local tips requests
            "What do locals do in Istanbul?",
            "Give me some insider tips for Istanbul",
            "Hidden gems in Istanbul",
            "Local secrets in Istanbul",
            "What should tourists avoid in Istanbul?",
            "Best local experiences in Istanbul",
            "Authentic Turkish culture experiences",
            "Where do Istanbul locals eat?",
            "Local hangout spots in Istanbul",
            "Off the beaten path Istanbul",
            
            # Neighborhood-specific local tips
            "Local tips for Kadıköy",
            "Hidden gems in Beyoğlu",
            "Where do locals go in Beşiktaş?",
            "Secret spots in Sultanahmet",
            "Authentic places in Fatih",
            "Local favorites in Üsküdar",
            "Hidden cafes in Şişli",
            "Secret viewpoints in Istanbul",
            "Underground art scene Istanbul",
            "Local markets off the tourist trail",
            
            # Specific local experiences
            "Best local breakfast spots",
            "Where to get authentic Turkish tea",
            "Local fish markets",
            "Neighborhood bakeries Istanbul",
            "Family-run restaurants Istanbul",
            "Traditional Turkish hammams locals use",
            "Local barber shops Istanbul",
            "Where locals buy baklava",
            "Authentic street food spots",
            "Local grocery stores Istanbul",
            
            # Cultural experiences
            "How to experience Istanbul like a local",
            "Local traditions in Istanbul",
            "Turkish coffee culture",
            "Local tea gardens",
            "Neighborhood life in Istanbul",
            "Daily routine of Istanbul locals",
            "Turkish breakfast traditions",
            "Local shopping habits",
            "How locals use public transport",
            "Istanbul local etiquette",
            
            # Hidden gems
            "Secret rooftop bars Istanbul",
            "Hidden bookshops Istanbul",
            "Underground music venues",
            "Secret gardens Istanbul",
            "Hidden historical sites",
            "Off-radar museums",
            "Secret beaches near Istanbul",
            "Hidden courtyards",
            "Secret photography spots",
            "Unknown viewpoints Istanbul",
            
            # Authentic experiences
            "Authentic Turkish music venues",
            "Traditional carpet shops locals visit",
            "Local artisan workshops",
            "Authentic Turkish baths",
            "Traditional Turkish restaurants",
            "Local craftsmen Istanbul",
            "Authentic spice shops",
            "Traditional tea houses",
            "Local pottery studios",
            "Authentic Turkish cooking classes",
            
            # Variations with different phrasings
            "Show me Istanbul's hidden treasures",
            "I want to experience Istanbul authentically",
            "Take me off the tourist path",
            "Where can I find real Istanbul?",
            "Undiscovered places in Istanbul",
            "Local insider knowledge",
            "What am I missing in Istanbul?",
            "Hidden Istanbul experiences",
            "Secret local favorites",
            "Unknown Istanbul spots",
            
            # More specific local tips
            "Best local fish sandwich spots",
            "Where locals buy fresh produce",
            "Authentic Turkish dessert shops",
            "Local cheese shops",
            "Traditional Turkish delicatessen",
            "Where to find homemade Turkish food",
            "Local wine shops Istanbul",
            "Authentic Turkish ceramics shops",
            "Local antique markets",
            "Neighborhood festivals Istanbul",
            
            # Time-specific local tips
            "What do locals do on Sunday mornings?",
            "Where do locals go after work?",
            "Local evening activities",
            "Weekend local spots",
            "Early morning local activities",
            "Late night local hangouts",
            "Rainy day local favorites",
            "Summer evening local spots",
            "Winter local activities",
            "Local New Year traditions",
            
            # More hidden gems
            "Hidden libraries Istanbul",
            "Secret art galleries",
            "Underground theaters",
            "Hidden jazz clubs",
            "Secret picnic spots",
            "Unknown historical fountains",
            "Hidden Ottoman houses",
            "Secret walking paths",
            "Unknown Byzantine ruins",
            "Hidden neighborhood parks",
            
            # Cultural immersion
            "How to blend in with locals",
            "Local shopping districts",
            "Turkish neighborhood culture",
            "Local food traditions",
            "Authentic Istanbul lifestyle",
            "Traditional Turkish customs",
            "Local celebration traditions",
            "Neighborhood character Istanbul",
            "Local community centers",
            "Traditional Turkish social life",
            
            # Additional authentic experiences
            "Traditional Turkish barbers",
            "Local tailors Istanbul",
            "Authentic Turkish shoe makers",
            "Traditional bookbinders",
            "Local leather craftsmen",
            "Authentic Turkish jewelry makers",
            "Traditional Turkish calligraphers",
            "Local musical instrument makers",
            "Authentic Turkish textile shops",
            "Traditional Turkish glass blowers",
            
            # More local favorites
            "Best local meze restaurants",
            "Where locals buy olives",
            "Authentic Turkish pickle shops",
            "Local honey vendors",
            "Traditional Turkish jam makers",
            "Where locals buy Turkish coffee",
            "Authentic spice blenders",
            "Local halva shops",
            "Traditional Turkish candy makers",
            "Where locals buy fresh pasta",
            
            # Hidden cultural spots
            "Hidden mosque courtyards",
            "Secret historical wells",
            "Unknown Ottoman fountains",
            "Hidden Byzantine churches",
            "Secret historical baths",
            "Unknown historical mansions",
            "Hidden cemetery gardens",
            "Secret historical libraries",
            "Unknown historical bridges",
            "Hidden waterfront spots",
        ]
        
        for query in local_tips_queries:
            samples.append({
                "text": query,
                "intent": "practical_info",  # Map to existing intent
                "sub_intent": "local_tips",
                "entities": {"type": "local_knowledge"}
            })
        
        return samples
    
    def generate_recommendation_data(self) -> List[Dict]:
        """Generate 100+ training samples for recommendations"""
        samples = []
        
        recommendation_queries = [
            # General recommendations
            "What do you recommend for first-time visitors?",
            "Best places for photography in Istanbul",
            "Recommend activities for families",
            "What's worth visiting in Istanbul?",
            "Suggest romantic spots in Istanbul",
            "Best things to do in winter",
            "Summer activities in Istanbul",
            "Recommend places for couples",
            "Family-friendly activities Istanbul",
            "Best experiences in Istanbul",
            
            # Time-based recommendations
            "What to do in Istanbul in 3 days?",
            "Weekend recommendations Istanbul",
            "One day in Istanbul recommendations",
            "Week-long Istanbul itinerary",
            "Evening recommendations Istanbul",
            "Morning activities Istanbul",
            "Rainy day recommendations",
            "Sunny day recommendations",
            "Winter weekend activities",
            "Summer evening suggestions",
            
            # Interest-based recommendations
            "Recommend art galleries",
            "Best shopping recommendations",
            "Food tour recommendations",
            "Historical sites recommendations",
            "Architecture recommendations",
            "Cultural recommendations",
            "Nightlife recommendations",
            "Nature recommendations Istanbul",
            "Outdoor activity recommendations",
            "Indoor activity recommendations",
            
            # Demographic-specific
            "Recommendations for solo travelers",
            "Couple recommendations Istanbul",
            "Family recommendations",
            "Senior-friendly recommendations",
            "Kid-friendly recommendations",
            "Budget recommendations",
            "Luxury recommendations",
            "Youth hostel recommendations",
            "Business traveler recommendations",
            "Digital nomad recommendations",
            
            # Seasonal recommendations
            "Spring recommendations Istanbul",
            "Summer recommendations Istanbul",
            "Fall recommendations Istanbul",
            "Winter recommendations Istanbul",
            "Holiday season recommendations",
            "Ramadan activities recommendations",
            "New Year recommendations",
            "Valentine's Day recommendations",
            "Wedding anniversary recommendations",
            "Birthday celebration recommendations",
            
            # Activity-specific
            "Photography spot recommendations",
            "Sunset viewing recommendations",
            "Bosphorus cruise recommendations",
            "Walking tour recommendations",
            "Food tasting recommendations",
            "Museum recommendations",
            "Park recommendations",
            "Beach recommendations near Istanbul",
            "Hiking recommendations around Istanbul",
            "Cycling route recommendations",
            
            # Budget-conscious
            "Free activity recommendations",
            "Budget-friendly recommendations",
            "Affordable recommendations Istanbul",
            "Cheap eats recommendations",
            "Value recommendations Istanbul",
            "Cost-effective recommendations",
            "Money-saving recommendations",
            "Economic recommendations",
            "Inexpensive recommendations",
            "Low-cost recommendations",
            
            # Experience-based
            "Unique experience recommendations",
            "Memorable recommendations Istanbul",
            "Must-do recommendations",
            "Once-in-a-lifetime recommendations",
            "Bucket list recommendations",
            "Authentic experience recommendations",
            "Cultural immersion recommendations",
            "Adventure recommendations",
            "Relaxation recommendations",
            "Wellness recommendations Istanbul",
            
            # More specific recommendations
            "Best brunch recommendations",
            "Rooftop bar recommendations",
            "Turkish bath recommendations",
            "Seafood restaurant recommendations",
            "Street food recommendations",
            "Dessert shop recommendations",
            "Coffee shop recommendations",
            "Bookstore recommendations",
            "Antique shop recommendations",
            "Souvenir shop recommendations",
        ]
        
        for query in recommendation_queries:
            samples.append({
                "text": query,
                "intent": "recommendation_request",
                "entities": {}
            })
        
        return samples
    
    def generate_events_data(self) -> List[Dict]:
        """Generate 150+ training samples for events"""
        samples = []
        
        event_queries = [
            # General event searches
            "What events are happening this weekend?",
            "Any concerts in Istanbul?",
            "Cultural festivals this month",
            "Where to find live music?",
            "Art exhibitions happening now",
            "Events near me",
            "What's on in Istanbul tonight?",
            "Events this week Istanbul",
            "Upcoming events Istanbul",
            "Current events Istanbul",
            
            # Specific event types
            "Jazz concerts Istanbul",
            "Rock concerts Istanbul",
            "Classical music events",
            "Traditional Turkish music events",
            "Dance performances Istanbul",
            "Theater shows Istanbul",
            "Opera performances Istanbul",
            "Ballet performances Istanbul",
            "Contemporary dance events",
            "Folk music events",
            
            # Cultural events
            "Cultural festivals Istanbul",
            "Film festivals Istanbul",
            "Art festivals Istanbul",
            "Music festivals Istanbul",
            "Food festivals Istanbul",
            "Literature festivals Istanbul",
            "Poetry readings Istanbul",
            "Book fairs Istanbul",
            "Craft fairs Istanbul",
            "Design festivals Istanbul",
            
            # Time-specific
            "Events this weekend",
            "Events tonight Istanbul",
            "Events tomorrow Istanbul",
            "Events next week",
            "Events this month",
            "Events in December",
            "New Year events Istanbul",
            "Christmas events Istanbul",
            "Summer festival events",
            "Winter cultural events",
            
            # Venue-specific
            "Events at Zorlu Center",
            "Events at İş Sanat",
            "Events at Cemal Reşit Rey",
            "Events at Borusan Sanat",
            "Events at Akbank Sanat",
            "Events at Istanbul Modern",
            "Events at Pera Museum",
            "Events at Sakıp Sabancı Museum",
            "Outdoor events Istanbul",
            "Indoor events Istanbul",
            
            # Art exhibitions
            "Current art exhibitions",
            "Contemporary art exhibitions",
            "Photography exhibitions",
            "Sculpture exhibitions",
            "Modern art exhibitions",
            "Traditional art exhibitions",
            "Gallery openings Istanbul",
            "Museum exhibitions Istanbul",
            "Art gallery events",
            "Installation art events",
            
            # Performance arts
            "Stand-up comedy shows",
            "Improv comedy Istanbul",
            "Comedy events Istanbul",
            "Circus performances",
            "Magic shows Istanbul",
            "Cabaret shows Istanbul",
            "Musical theater Istanbul",
            "Performance art events",
            "Live entertainment Istanbul",
            "Variety shows Istanbul",
            
            # Sports events
            "Football matches Istanbul",
            "Basketball games Istanbul",
            "Volleyball matches Istanbul",
            "Sports events Istanbul",
            "Marathon events Istanbul",
            "Cycling events Istanbul",
            "Water sports events",
            "Extreme sports events",
            "Traditional sports events",
            "Athletic events Istanbul",
            
            # Special interest events
            "Technology conferences Istanbul",
            "Startup events Istanbul",
            "Business networking events",
            "Educational workshops Istanbul",
            "Language exchange events",
            "Cooking classes Istanbul",
            "Art workshops Istanbul",
            "Photography workshops",
            "Dance classes Istanbul",
            "Yoga events Istanbul",
            
            # Family events
            "Family-friendly events",
            "Kids events Istanbul",
            "Children's theater Istanbul",
            "Puppet shows Istanbul",
            "Family concerts",
            "Children's workshops",
            "Family festivals",
            "Educational events for kids",
            "Interactive exhibitions kids",
            "Children's art activities",
            
            # Night events
            "Nightlife events Istanbul",
            "Club events Istanbul",
            "DJ performances Istanbul",
            "Electronic music events",
            "Late night events",
            "After-hours events",
            "Night market events",
            "Moonlight concerts",
            "Evening cultural events",
            "Night walking tours",
            
            # Free events
            "Free events Istanbul",
            "Free concerts Istanbul",
            "Free exhibitions",
            "Free workshops Istanbul",
            "Free cultural events",
            "Community events Istanbul",
            "Public events Istanbul",
            "Open-air events Istanbul",
            "Street festivals Istanbul",
            "Public performances",
            
            # Seasonal events
            "Spring festivals Istanbul",
            "Summer concerts Istanbul",
            "Autumn cultural events",
            "Winter festivals Istanbul",
            "Holiday season events",
            "Ramadan events Istanbul",
            "İftar events Istanbul",
            "Religious festivals",
            "National holiday events",
            "Seasonal celebrations",
        ]
        
        for query in event_queries:
            samples.append({
                "text": query,
                "intent": "event_search",
                "entities": {}
            })
        
        return samples
    
    def generate_neighborhood_data(self) -> List[Dict]:
        """Generate 100+ training samples for neighborhoods"""
        samples = []
        
        neighborhoods = [
            "Beyoğlu", "Kadıköy", "Beşiktaş", "Sultanahmet", "Fatih",
            "Üsküdar", "Şişli", "Sarıyer", "Ortaköy", "Balat",
            "Karaköy", "Galata", "Cihangir", "Bebek", "Arnavutköy"
        ]
        
        patterns = [
            "Tell me about {} neighborhood",
            "What is {} like?",
            "Describe the {} area",
            "What's {} known for?",
            "Is {} safe?",
            "Best things to do in {}",
            "Where to eat in {}",
            "Nightlife in {}",
            "Shopping in {}",
            "History of {}",
            "Character of {}",
            "Atmosphere in {}",
            "What to see in {}",
            "Walking tour of {}",
            "Best time to visit {}",
        ]
        
        for neighborhood in neighborhoods:
            for pattern in patterns:
                samples.append({
                    "text": pattern.format(neighborhood),
                    "intent": "neighborhood_info",
                    "entities": {"neighborhood": neighborhood}
                })
        
        # Add comparison queries
        for i, n1 in enumerate(neighborhoods):
            for n2 in neighborhoods[i+1:i+3]:  # Compare with next 2
                samples.append({
                    "text": f"Compare {n1} and {n2}",
                    "intent": "comparison_request",
                    "entities": {"neighborhoods": [n1, n2]}
                })
        
        return samples
    
    def generate_attraction_data(self) -> List[Dict]:
        """Generate 100+ training samples for attractions"""
        samples = []
        
        attraction_patterns = [
            # Museum queries
            "Best museums in Istanbul",
            "Contemporary art museums",
            "Historical museums Istanbul",
            "Archaeological museums",
            "Modern art museums Istanbul",
            "Science museums Istanbul",
            "Military museums Istanbul",
            "Maritime museums Istanbul",
            "Islamic art museums",
            "Ottoman museums Istanbul",
            
            # Monument queries
            "Historical monuments Istanbul",
            "Byzantine monuments",
            "Ottoman monuments",
            "Ancient monuments Istanbul",
            "Religious monuments",
            "Architectural monuments",
            "UNESCO sites Istanbul",
            "Heritage sites Istanbul",
            "Historical landmarks",
            "Ancient ruins Istanbul",
            
            # Palace queries
            "Palaces in Istanbul",
            "Ottoman palaces",
            "Royal palaces Istanbul",
            "Historical palaces",
            "Palace museums Istanbul",
            "Imperial palaces",
            "Sultan palaces",
            "Bosphorus palaces",
            "Waterfront palaces",
            "Palace gardens Istanbul",
            
            # Mosque queries
            "Famous mosques Istanbul",
            "Historic mosques",
            "Imperial mosques",
            "Ottoman mosques",
            "Byzantine churches converted to mosques",
            "Blue Mosque information",
            "Süleymaniye Mosque info",
            "Eyüp Sultan Mosque",
            "Ortaköy Mosque",
            "Mosque architecture Istanbul",
            
            # Park queries
            "Parks in Istanbul",
            "Green spaces Istanbul",
            "Public gardens Istanbul",
            "Botanical gardens",
            "Picnic spots Istanbul",
            "Outdoor spaces Istanbul",
            "Nature parks Istanbul",
            "Walking parks Istanbul",
            "Dog-friendly parks",
            "Children's playgrounds Istanbul",
            
            # Tower queries
            "Galata Tower information",
            "Maiden's Tower Istanbul",
            "Historic towers Istanbul",
            "Observation towers",
            "Byzantine towers",
            "Ottoman towers",
            "City walls Istanbul",
            "Fortifications Istanbul",
            "Castle remains Istanbul",
            "Defensive structures",
            
            # Bazaar queries
            "Grand Bazaar information",
            "Spice Bazaar Istanbul",
            "Historic bazaars",
            "Covered markets Istanbul",
            "Traditional markets",
            "Shopping bazaars",
            "Antique bazaars",
            "Book bazaars Istanbul",
            "Fish markets Istanbul",
            "Local markets Istanbul",
        ]
        
        for query in attraction_patterns:
            # Determine specific intent
            if any(word in query.lower() for word in ["museum", "gallery"]):
                intent = "attraction_info"
                sub_type = "museum"
            elif any(word in query.lower() for word in ["palace"]):
                intent = "attraction_info"
                sub_type = "palace"
            elif any(word in query.lower() for word in ["mosque", "church"]):
                intent = "attraction_info"
                sub_type = "religious"
            elif any(word in query.lower() for word in ["park", "garden"]):
                intent = "attraction_info"
                sub_type = "park"
            else:
                intent = "attraction_search"
                sub_type = "general"
            
            samples.append({
                "text": query,
                "intent": intent,
                "entities": {"type": sub_type}
            })
        
        return samples
    
    def generate_practical_info_data(self) -> List[Dict]:
        """Generate training samples for practical information"""
        samples = []
        
        practical_queries = [
            # Money & payments
            "Where can I exchange money?",
            "Best currency exchange in Istanbul",
            "Are credit cards accepted?",
            "ATM locations Istanbul",
            "Bank hours Istanbul",
            "Exchange rates Istanbul",
            "Money changers Istanbul",
            "Cash or card in Istanbul?",
            "Foreign currency exchange",
            "Bank services for tourists",
            
            # Emergency
            "Emergency numbers in Istanbul",
            "Police station locations",
            "Hospital emergency rooms",
            "Ambulance number Turkey",
            "Tourist police Istanbul",
            "Lost passport procedure",
            "Emergency medical services",
            "Fire department number",
            "Coast guard number",
            "Emergency dentist Istanbul",
            
            # Health services
            "Where is the nearest pharmacy?",
            "24-hour pharmacies Istanbul",
            "International hospitals Istanbul",
            "English-speaking doctors",
            "Dental clinics Istanbul",
            "Medical insurance Istanbul",
            "Vaccination requirements Turkey",
            "Health services for tourists",
            "Private hospitals Istanbul",
            "Medical emergency procedure",
            
            # Practical necessities
            "What's the voltage in Turkey?",
            "Power adapter type Turkey",
            "Internet cafes Istanbul",
            "WiFi availability Istanbul",
            "SIM card for tourists",
            "Mobile data plans Turkey",
            "Public restrooms Istanbul",
            "Luggage storage Istanbul",
            "Laundromat services",
            "Dry cleaning Istanbul",
            
            # Documentation
            "Do I need a visa for Turkey?",
            "Visa requirements Turkey",
            "E-visa Turkey application",
            "Passport requirements",
            "Travel insurance Turkey",
            "Registration requirements",
            "Residence permit Turkey",
            "Tourist registration",
            "Embassy locations Istanbul",
            "Consulate services Istanbul",
            
            # Customs & regulations
            "Customs regulations Turkey",
            "Duty-free allowance",
            "What can I bring to Turkey?",
            "Prohibited items Turkey",
            "Pet travel regulations",
            "Medication import rules",
            "Alcohol import limits",
            "Tobacco import limits",
            "Currency declaration",
            "Export restrictions Turkey",
            
            # Tips & etiquette
            "How much to tip in restaurants?",
            "Tipping customs Turkey",
            "Service charge included?",
            "Tipping taxi drivers",
            "Tipping hotel staff",
            "Tipping tour guides",
            "Bargaining etiquette Turkey",
            "Cultural etiquette Istanbul",
            "Dress code mosques",
            "Social customs Turkey",
            
            # Communication
            "Do people speak English in Istanbul?",
            "Translation services Istanbul",
            "Language barriers Turkey",
            "Common Turkish phrases",
            "Learning basic Turkish",
            "English-speaking services",
            "Tourist information centers",
            "Communication tips Turkey",
            "Body language Turkey",
            "Cultural communication",
        ]
        
        for query in practical_queries:
            samples.append({
                "text": query,
                "intent": "practical_info",
                "entities": {}
            })
        
        return samples
    
    def generate_price_inquiry_data(self) -> List[Dict]:
        """Generate training samples for price inquiries"""
        samples = []
        
        price_queries = [
            # Museum prices
            "How much does a museum ticket cost?",
            "Museum entrance fees Istanbul",
            "Topkapı Palace ticket price",
            "Hagia Sophia entrance fee",
            "Istanbul Museum Pass price",
            "Student discount museums",
            "Free museum days Istanbul",
            "Museum ticket deals",
            "Combined ticket prices",
            "Annual museum pass Istanbul",
            
            # Transportation prices
            "Price of metro tickets",
            "How much is an Istanbulkart?",
            "Taxi fares Istanbul",
            "Ferry ticket prices",
            "Airport transfer cost",
            "Bus ticket prices",
            "Tram fare Istanbul",
            "Metro card price",
            "Daily transport pass",
            "Tourist transport card",
            
            # Accommodation prices
            "What's the price range for hotels?",
            "Average hotel cost Istanbul",
            "Budget hotel prices",
            "Luxury hotel rates",
            "Hostel prices Istanbul",
            "Airbnb costs Istanbul",
            "Hotel room rates",
            "Accommodation budget",
            "Nightly rates hotels",
            "Hotel price comparison",
            
            # Food & dining prices
            "Average meal price Istanbul",
            "Restaurant price range",
            "Street food prices",
            "Fine dining costs",
            "Breakfast prices Istanbul",
            "Lunch menu prices",
            "Dinner costs Istanbul",
            "Coffee prices Istanbul",
            "Turkish tea price",
            "Alcohol prices Turkey",
            
            # Activities & tours
            "Bosphorus cruise prices",
            "Tour package costs",
            "Walking tour prices",
            "Food tour costs Istanbul",
            "Turkish bath prices",
            "Cooking class prices",
            "Day trip costs",
            "Private guide rates",
            "Group tour prices",
            "Excursion costs Istanbul",
            
            # Shopping
            "Souvenir prices Istanbul",
            "Carpet prices Istanbul",
            "Turkish delight prices",
            "Spice prices",
            "Jewelry prices Turkey",
            "Clothing costs Istanbul",
            "Bargaining prices bazaar",
            "Market prices Istanbul",
            "Antique prices",
            "Handicraft costs",
            
            # General budget
            "Is Istanbul expensive?",
            "Daily budget Istanbul",
            "How much money should I bring?",
            "Cost of living Istanbul",
            "Tourist expenses Istanbul",
            "Average costs Turkey",
            "Budget for 3 days",
            "Spending money Istanbul",
            "Travel budget Turkey",
            "Price level Istanbul",
        ]
        
        for query in price_queries:
            samples.append({
                "text": query,
                "intent": "price_inquiry",
                "entities": {}
            })
        
        return samples
    
    def enhance_training_data(self):
        """Generate all enhancement data and save to file"""
        print("🚀 Generating enhancement data for low-confidence categories...")
        print("=" * 80)
        
        # Load existing data
        if self.base_data_path.exists():
            with open(self.base_data_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"✅ Loaded {len(existing_data)} existing samples")
        else:
            existing_data = []
            print("⚠️  No existing data found, starting fresh")
        
        # Generate new samples
        print("\n📝 Generating new training samples...")
        
        local_tips = self.generate_local_tips_data()
        print(f"  ├─ Local Tips / Hidden Gems: {len(local_tips)} samples")
        
        recommendations = self.generate_recommendation_data()
        print(f"  ├─ Recommendations: {len(recommendations)} samples")
        
        events = self.generate_events_data()
        print(f"  ├─ Events: {len(events)} samples")
        
        neighborhoods = self.generate_neighborhood_data()
        print(f"  ├─ Neighborhoods: {len(neighborhoods)} samples")
        
        attractions = self.generate_attraction_data()
        print(f"  ├─ Attractions: {len(attractions)} samples")
        
        practical = self.generate_practical_info_data()
        print(f"  ├─ Practical Info: {len(practical)} samples")
        
        prices = self.generate_price_inquiry_data()
        print(f"  └─ Price Inquiries: {len(prices)} samples")
        
        # Combine all new samples
        all_new_samples = (
            local_tips + recommendations + events + neighborhoods + 
            attractions + practical + prices
        )
        
        # Combine with existing data
        enhanced_data = existing_data + all_new_samples
        
        print(f"\n📊 Total samples: {len(enhanced_data)}")
        print(f"  ├─ Original: {len(existing_data)}")
        print(f"  └─ New: {len(all_new_samples)}")
        
        # Save enhanced dataset
        output_path = self.base_data_path.parent / "intent_training_data_enhanced.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Enhanced dataset saved to: {output_path}")
        
        # Generate intent distribution report
        self._generate_distribution_report(enhanced_data)
        
        return output_path
    
    def _generate_distribution_report(self, data: List[Dict]):
        """Generate and display intent distribution"""
        from collections import Counter
        
        intent_counts = Counter(item['intent'] for item in data)
        
        print("\n📊 Intent Distribution:")
        print("-" * 80)
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(data)) * 100
            bar = "█" * int(percentage / 2)
            print(f"{intent:30} | {count:4} ({percentage:5.1f}%) {bar}")
        print("-" * 80)


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("LOW-CONFIDENCE CATEGORY ENHANCEMENT")
    print("=" * 80)
    
    enhancer = LowConfidenceCategoryEnhancer()
    output_path = enhancer.enhance_training_data()
    
    print("\n" + "=" * 80)
    print("✅ ENHANCEMENT COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Enhanced dataset: {output_path}")
    print("\n🔄 Next steps:")
    print("  1. Review the enhanced dataset")
    print("  2. Run fine-tuning script with the new data:")
    print(f"     python scripts/finetune_intent_classifier.py")
    print("  3. Test the improved model")
    print("  4. Deploy to production")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
