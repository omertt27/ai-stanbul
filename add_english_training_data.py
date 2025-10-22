#!/usr/bin/env python3
"""
Add More English Training Data
Focus on intents with low English accuracy:
- accommodation (failed)
- transportation (failed)  
- romantic (failed)
- family_activities (failed)
"""

import json
import random

# Additional English training data (50+ samples per weak intent)
ADDITIONAL_ENGLISH_DATA = {
    "accommodation": [
        "I need a hotel", "Where can I stay?", "Hotel near Sultanahmet",
        "Cheap accommodation", "Budget hotel", "Hostel recommendations",
        "5 star hotel", "Luxury accommodation", "Book a hotel",
        "Hotel with breakfast", "Pet friendly hotel", "Hotel near airport",
        "Where to sleep?", "I need a place to stay", "Accommodation options",
        "Hotel with pool", "Boutique hotel", "Guest house",
        "Hotel booking", "Reserve a room", "Check hotel prices",
        "Hotel availability", "Double room", "Single room",
        "Suite reservation", "Hotel deals", "Best hotels",
        "Hotel near beach", "City center hotel", "Affordable stay",
        "Hotel with spa", "Family hotel", "Business hotel",
        "Hotel amenities", "Hotel facilities", "Room service",
        "Hotel check-in", "Hotel check-out", "Late checkout",
        "Early check-in", "Hotel cancellation", "Modify booking",
        "Hotel reviews", "Recommended hotels", "Top rated hotels",
        "Hotel comparison", "Hotel search", "Find accommodation",
        "Lodging options", "Where to book hotel", "Hotel website",
        "Hotel phone number", "Hotel address", "Hotel location",
    ],
    
    "transportation": [
        "How to get to airport?", "Metro map", "Bus schedule",
        "Tram route", "Public transport", "Transportation card",
        "How to use metro?", "Ferry times", "Taxi fare",
        "Airport shuttle", "Metro ticket", "Bus fare",
        "How to get around?", "City transport", "Metro line",
        "Tram stops", "Bus route", "Ferry schedule",
        "Istanbul card", "Transport pass", "Daily ticket",
        "Night bus", "Airport bus", "Metro station",
        "Nearest metro", "How to reach?", "Getting there",
        "Transport options", "Cheapest transport", "Fastest route",
        "Metro hours", "Last metro", "First bus",
        "Transfer point", "Metro transfer", "Bus connection",
        "Tram connection", "Ferry pier", "Boat schedule",
        "Marmaray", "Metrobus", "Dolmus",
        "Private transfer", "Car rental", "Uber",
        "Taxi stand", "Call taxi", "Transport app",
        "Route planner", "Journey planner", "Trip calculator",
        "Transport map", "Metro app", "Bus tracker",
        "Real-time schedule", "Transport delays", "Service updates",
    ],
    
    "romantic": [
        "Romantic places", "Date ideas", "Romantic dinner",
        "Couples activities", "Romantic spots", "Sunset viewing",
        "Romantic restaurant", "Special dinner", "Anniversary dinner",
        "Proposal places", "Honeymoon spots", "Romantic walk",
        "Couple activities", "Date night", "Romantic views",
        "Candlelit dinner", "Rooftop dinner", "Private dinner",
        "Romantic cruise", "Couple massage", "Spa for couples",
        "Romantic hotel", "Couples getaway", "Valentine's ideas",
        "Romantic experience", "Love locks", "Romantic bridge",
        "Couple photos", "Romantic scenery", "Best sunset spots",
        "Romantic boat", "Private tour", "VIP experience",
        "Romantic gift", "Flower shop", "Jewelry store",
        "Romantic surprise", "Special occasion", "Engagement places",
        "Wedding venues", "Honeymoon activities", "Romantic gardens",
        "Couples spa", "Wine tasting", "Romantic cafe",
        "Love themed", "Couple friendly", "Romantic atmosphere",
        "Intimate dining", "Cozy restaurant", "Quiet places",
        "Peaceful spots", "Scenic views", "Beautiful locations",
    ],
    
    "family_activities": [
        "Things to do with kids", "Family activities", "Kid friendly places",
        "Children attractions", "Family fun", "Kids activities",
        "Where to go with children?", "Family outing", "Kids entertainment",
        "Playground", "Children museum", "Kids park",
        "Family restaurant", "Kids menu", "High chair available",
        "Stroller friendly", "Baby changing room", "Family facilities",
        "Aquarium", "Zoo", "Theme park",
        "Water park", "Indoor playground", "Outdoor activities",
        "Educational activities", "Kids workshop", "Family tour",
        "Child friendly tour", "Kids program", "Family package",
        "Children discount", "Family ticket", "Kids free",
        "Age appropriate", "Toddler friendly", "Teen activities",
        "Family beach", "Safe for kids", "Supervised activities",
        "Kids club", "Babysitter", "Childcare",
        "Family show", "Kids movie", "Children theater",
        "Puppet show", "Magic show", "Circus",
        "Kids party", "Birthday venues", "Party entertainment",
        "Family picnic", "Kids cafe", "Play area",
        "Trampoline park", "Bounce house", "Kids gym",
        "Science center", "Discovery center", "Interactive exhibits",
        "Hands-on museum", "Kids learning", "Educational games",
    ],
    
    # Also boost other intents with more English examples
    "restaurant": [
        "Where to eat?", "Good restaurants", "Best food",
        "Dining options", "Places to eat", "Food recommendations",
        "Local cuisine", "Turkish food", "Seafood restaurant",
        "Vegetarian options", "Vegan restaurant", "Halal food",
        "Fine dining", "Casual dining", "Fast food",
        "Street food", "Food court", "Buffet restaurant",
        "Breakfast place", "Brunch spot", "Lunch restaurant",
        "Dinner reservation", "Late night food", "24 hour restaurant",
    ],
    
    "attraction": [
        "Tourist attractions", "Sightseeing", "Must see places",
        "Popular spots", "Famous landmarks", "Historical sites",
        "Cultural sites", "City tour", "Guided tour",
        "Walking tour", "Bus tour", "Hop on hop off",
        "Tourist spots", "Photo spots", "Instagram locations",
        "Main attractions", "Top sites", "Best views",
    ],
    
    "museum": [
        "Art museum", "History museum", "Modern art",
        "Contemporary art", "Museum hours", "Museum entrance",
        "Museum collection", "Special exhibition", "Museum cafe",
        "Audio guide", "Museum tour", "Free museum",
        "Museum pass", "Museum shop", "Photography allowed",
    ],
    
    "shopping": [
        "Shopping areas", "Where to shop?", "Best shops",
        "Shopping mall", "Outlet stores", "Local market",
        "Souvenir shop", "Gift shop", "Shopping street",
        "Bazaar", "Craft market", "Antique shops",
        "Designer stores", "Luxury brands", "Budget shopping",
    ],
    
    "emergency": [
        "Call police", "Hospital emergency", "Ambulance",
        "I need help", "Lost passport", "Stolen wallet",
        "Medical emergency", "Pharmacy", "Doctor",
        "Police station", "Embassy", "Consulate",
        "Emergency number", "Help me", "Urgent",
    ],
    
    "weather": [
        "Weather today", "Temperature", "Will it rain?",
        "Weather forecast", "Is it sunny?", "Climate",
        "What to wear?", "Weather tomorrow", "Weekly forecast",
        "Hot or cold?", "Humidity", "Wind speed",
    ],
    
    "booking": [
        "Make reservation", "Book tickets", "Reserve table",
        "Online booking", "Cancel booking", "Modify reservation",
        "Booking confirmation", "Reservation fee", "Advance booking",
        "Group booking", "Last minute booking", "Book now",
    ],
    
    "budget": [
        "Cheap options", "Budget friendly", "Save money",
        "Free things to do", "Affordable", "Low cost",
        "Student discount", "Budget travel", "Economical",
        "Best deals", "Money saving tips", "Cheap eats",
    ],
}


def add_english_training_data():
    """Add more English training data"""
    print("=" * 80)
    print("ADDING ENGLISH TRAINING DATA")
    print("=" * 80)
    print()
    
    # Load existing bilingual dataset
    with open('bilingual_training_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    original_count = len(dataset)
    
    # Add new English data
    for intent, queries in ADDITIONAL_ENGLISH_DATA.items():
        for query in queries:
            dataset.append({
                "text": query,
                "intent": intent
            })
    
    # Shuffle
    random.shuffle(dataset)
    
    new_count = len(dataset)
    added_count = new_count - original_count
    
    print(f"📊 Dataset Statistics:")
    print(f"   Original: {original_count} samples")
    print(f"   Added: {added_count} English samples")
    print(f"   Total: {new_count} samples")
    print()
    
    # Count samples per intent
    intent_counts = {}
    for item in dataset:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("📈 Samples per intent:")
    for intent in sorted(intent_counts.keys()):
        count = intent_counts[intent]
        print(f"   {intent:20s}: {count:3d} samples")
    print()
    
    # Save enhanced dataset
    output_file = "enhanced_bilingual_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    print()
    print("=" * 80)
    print("✅ ENGLISH DATA ENHANCED!")
    print("=" * 80)
    print()
    print("Next: python3 train_enhanced_bilingual.py")
    print()


if __name__ == "__main__":
    add_english_training_data()
