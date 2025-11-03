"""
Training Data Generator for Istanbul AI Intent Classifier
Generates comprehensive training dataset for all 9 intent categories
"""

import json
import random
from datetime import datetime

# Training data with 100+ examples per intent
TRAINING_DATA = {
    'transportation': [
        # GPS-based queries
        "how can i go to taksim from my location",
        "how do i get to sultanahmet from here",
        "navigate me to galata tower",
        "directions from my location to blue mosque",
        "take me to topkapi palace from where i am",
        "route from here to kadikoy",
        "show me the way to besiktas from my current location",
        "how to reach istiklal street from my position",
        
        # Metro/public transport
        "metro from sultanahmet to besiktas",
        "which metro line goes to taksim",
        "metro route to kadikoy",
        "how to use istanbul metro",
        "metro map istanbul",
        "metro station near hagia sophia",
        "m2 metro line stops",
        "metro to the airport",
        
        # Bus queries
        "bus to taksim square",
        "which bus goes to sultanahmet",
        "bus number to blue mosque",
        "bus route to besiktas",
        "how to take the bus in istanbul",
        "bus schedule to kadikoy",
        
        # Ferry/maritime
        "ferry to kadikoy",
        "bosphorus ferry times",
        "ferry from eminonu to uskudar",
        "maritime transport istanbul",
        "how to take the ferry",
        "ferry schedule",
        
        # Airport transfers
        "airport transfer to city center",
        "how to get from ist airport to sultanahmet",
        "taxi from saw airport",
        "havaist bus to taksim",
        "airport shuttle",
        "cheapest way to airport",
        
        # İstanbulkart
        "how to use istanbulkart",
        "where to buy istanbulkart",
        "istanbulkart price",
        "how to top up istanbulkart",
        "istanbulkart locations",
        
        # Walking directions
        "walking distance to blue mosque",
        "can i walk from sultanahmet to galata tower",
        "how far is taksim from here",
        "walking route to topkapi",
        "is it walkable to hagia sophia",
        
        # Tram queries
        "tram to sultanahmet",
        "t1 tram line",
        "tram stops in istanbul",
        "how to take the tram",
        
        # Distance/time queries
        "how far is taksim from sultanahmet",
        "distance to blue mosque",
        "how long to get to galata tower",
        "travel time to kadikoy",
        
        # General transportation
        "public transport in istanbul",
        "best way to get around istanbul",
        "transportation options",
        "how to travel in istanbul",
        "getting around the city",
        
        # Specific routes
        "from sultanahmet to besiktas",
        "from taksim to kadikoy",
        "from galata to topkapi",
        "from eminonu to ortakoy",
        
        # Multiple destinations
        "how to visit sultanahmet and galata in one day",
        "transportation between major attractions",
        
        # Traffic and timing
        "best time to avoid traffic",
        "rush hour in istanbul",
        "fastest way to taksim",
        
        # Accessibility
        "wheelchair accessible transport",
        "disabled access metro",
        
        # Variations with typos (common user errors)
        "how can i go taksim",
        "metro to sultanamet",  # typo
        "bus kadikoy",
        "ferry bosphorous",  # typo
        
        # More natural language variations
        "i need to get to taksim",
        "can you help me reach sultanahmet",
        "looking for directions to blue mosque",
        "want to go to galata tower",
        "need route to kadikoy",
        "trying to get to besiktas",
        
        # Turkish-English mix
        "nasıl gidebilirim taksim",
        "metro nerede sultanahmet",
        "otobus galata tower",
        
        # Casual/colloquial
        "what's the fastest way to taksim",
        "quickest route to sultanahmet",
        "easiest way to get to blue mosque",
        
        # Question variations
        "is there a metro to taksim",
        "are there buses to sultanahmet",
        "can i take the tram to galata",
        "which transport goes to kadikoy",
        
        # More GPS-based
        "directions from my gps location",
        "route starting from where i am",
        "navigate from current position",
        "from my coordinates to taksim",
        
        # Additional transport modes
        "funicular to taksim",
        "cable car istanbul",
        "dolmus to ortakoy",
        "minibus to besiktas",
        
        # Practical questions
        "cost of metro ticket",
        "price to taksim",
        "how much is the bus fare",
        "transport passes available",
    ],
    
    'restaurants': [
        # Location-based
        "best restaurants in beyoglu",
        "restaurants near me",
        "where to eat in sultanahmet",
        "good restaurants in taksim",
        "dining in kadikoy",
        "restaurants in besiktas",
        "where can i eat in galata",
        "food places near blue mosque",
        
        # Cuisine-specific
        "turkish restaurants",
        "where can i eat seafood",
        "best kebab place",
        "traditional turkish food",
        "ottoman cuisine restaurants",
        "mediterranean restaurants",
        "italian restaurants istanbul",
        "asian food istanbul",
        "chinese restaurants",
        "japanese restaurants",
        "sushi in istanbul",
        
        # Dietary restrictions
        "vegetarian restaurants",
        "vegan restaurants istanbul",
        "halal restaurants",
        "kosher food istanbul",
        "gluten free restaurants",
        "restaurants for allergies",
        "lactose free dining",
        
        # Price-based
        "cheap restaurants in istanbul",
        "budget friendly dining",
        "affordable restaurants",
        "expensive restaurants",
        "fine dining istanbul",
        "luxury restaurants",
        "mid-range restaurants",
        
        # Meal times
        "breakfast in sultanahmet",
        "brunch places",
        "lunch restaurants",
        "dinner in beyoglu",
        "late night food",
        "24 hour restaurants",
        
        # Special occasions
        "romantic restaurants",
        "restaurants with view",
        "rooftop restaurants",
        "bosphorus view dining",
        "restaurants for anniversary",
        "date night restaurants",
        
        # Specific dishes
        "where to eat baklava",
        "best turkish breakfast",
        "mezes restaurants",
        "fish restaurants",
        "meze bars",
        "street food istanbul",
        
        # Quick searches
        "nearby restaurants",
        "restaurants open now",
        "delivery restaurants",
        "takeaway food",
        
        # Family-friendly
        "kid friendly restaurants",
        "family restaurants",
        "restaurants with play area",
        
        # Reviews/quality
        "highly rated restaurants",
        "best reviewed restaurants",
        "top restaurants istanbul",
        "popular restaurants",
        "famous restaurants",
        
        # Specific requests
        "outdoor seating restaurants",
        "garden restaurants",
        "terrace dining",
        "pet friendly restaurants",
        "restaurants with wifi",
        
        # Neighborhood specific
        "restaurants in sisli",
        "dining in uskudar",
        "food in ortakoy",
        "restaurants in eminonu",
        
        # Street food
        "street food near me",
        "balik ekmek where",
        "simit vendors",
        "doner kebab",
        
        # Casual/colloquial
        "where should i eat",
        "good food nearby",
        "hungry, where to go",
        "food recommendations",
        "where do locals eat",
        
        # With typos
        "restrants in taksim",  # typo
        "resturant beyoglu",  # typo
        "sea food istanbul",  # spacing
        
        # Turkish-English mix
        "lokanta sultanahmet",
        "kahvalti restaurant",
        "balik restaurant",
        "kebap house",
        
        # Time-based
        "restaurants open late",
        "breakfast places open early",
        "lunch specials",
        
        # Authentic experience
        "authentic turkish restaurants",
        "traditional dining",
        "local food spots",
        "non-touristy restaurants",
    ],
    
    'attractions': [
        # General attraction queries
        "what to see in istanbul",
        "best places to visit",
        "top attractions",
        "must see places",
        "tourist spots",
        "things to do in istanbul",
        "sightseeing in istanbul",
        
        # Museums
        "museums in istanbul",
        "best museums",
        "hagia sophia",
        "topkapi palace",
        "archaeological museum",
        "istanbul modern",
        "museum tickets",
        "free museums",
        
        # Historical sites
        "blue mosque",
        "basilica cistern",
        "galata tower",
        "maiden tower",
        "dolmabahce palace",
        "rumeli fortress",
        "theodosian walls",
        
        # Religious sites
        "mosques in istanbul",
        "suleymaniye mosque",
        "fatih mosque",
        "eyup sultan mosque",
        "churches in istanbul",
        "synagogues",
        
        # Markets/bazaars
        "grand bazaar",
        "spice bazaar",
        "egyptian bazaar",
        "flea markets",
        "book bazaar",
        
        # Natural attractions
        "parks in istanbul",
        "gulhane park",
        "emirgan park",
        "yildiz park",
        "belgrade forest",
        "princes islands",
        
        # Waterfront
        "bosphorus cruise",
        "golden horn",
        "ortakoy",
        "bebek",
        "waterfront walks",
        
        # Modern attractions
        "istiklal street",
        "taksim square",
        "zorlu center",
        "istanbul shopping",
        
        # Family-friendly
        "family attractions",
        "kids activities",
        "playgrounds",
        "theme parks",
        "aquarium istanbul",
        
        # Budget-conscious
        "free things to do",
        "cheap attractions",
        "free entry days",
        "budget activities",
        
        # Weather-based
        "indoor attractions",
        "rainy day activities",
        "outdoor activities",
        "summer activities",
        "winter things to do",
        
        # Time-based
        "one day in istanbul",
        "weekend activities",
        "evening attractions",
        "night activities",
        
        # Photography
        "best photo spots",
        "instagram locations",
        "scenic views",
        "sunset viewpoints",
        
        # Cultural
        "cultural sites",
        "heritage sites",
        "unesco sites",
        "historical places",
        
        # District-specific
        "attractions in sultanahmet",
        "things to do in beyoglu",
        "kadikoy attractions",
        "besiktas places to visit",
        
        # Romantic
        "romantic spots",
        "couple activities",
        "date ideas",
        
        # Adventure
        "adventure activities",
        "water sports",
        "climbing",
        
        # Art & culture
        "art galleries",
        "contemporary art",
        "exhibitions",
        "cultural centers",
        
        # Casual queries
        "what should i visit",
        "where to go in istanbul",
        "tourist attractions",
        "places to see",
        "what's worth visiting",
        
        # Specific interests
        "architectural sites",
        "ottoman architecture",
        "byzantine sites",
        "archaeological sites",
    ],
    
    'neighborhoods': [
        # General neighborhood queries
        "tell me about beyoglu",
        "what's special about sultanahmet",
        "describe kadikoy",
        "besiktas neighborhood",
        "sisli area",
        "uskudar district",
        
        # Character/vibe
        "bohemian neighborhoods",
        "hip areas istanbul",
        "trendy districts",
        "historic neighborhoods",
        "modern areas",
        "traditional districts",
        
        # Specific neighborhoods
        "fatih neighborhood",
        "galata area",
        "balat neighborhood",
        "fener district",
        "ortakoy area",
        "nisantasi neighborhood",
        "cihangir area",
        "karakoy district",
        
        # Asian side
        "asian side neighborhoods",
        "moda neighborhood",
        "caddebostan area",
        "bostanci district",
        
        # What to do
        "what to do in kadikoy",
        "activities in beyoglu",
        "things to see in sultanahmet",
        "where to go in besiktas",
        
        # Best time to visit
        "when to visit sultanahmet",
        "best time for kadikoy",
        "weekend in beyoglu",
        
        # Local insights
        "where do locals hang out",
        "authentic neighborhoods",
        "non-touristy areas",
        "local favorites",
        
        # Comparison
        "beyoglu vs sultanahmet",
        "kadikoy or besiktas",
        "best neighborhood for tourists",
        "safest neighborhood",
        
        # Living/staying
        "best area to stay",
        "where to stay in istanbul",
        "neighborhoods for tourists",
        "safe areas for tourists",
        
        # Nightlife
        "neighborhoods with nightlife",
        "party districts",
        "bar areas",
        "club districts",
        
        # Shopping
        "shopping neighborhoods",
        "best areas for shopping",
        "market districts",
        
        # Food scene
        "foodie neighborhoods",
        "best area for restaurants",
        "culinary districts",
        
        # Waterfront neighborhoods
        "bosphorus neighborhoods",
        "seaside areas",
        "waterfront districts",
        
        # Historic areas
        "old istanbul neighborhoods",
        "historic districts",
        "heritage areas",
        
        # Specific questions
        "is beyoglu safe",
        "is sultanahmet touristy",
        "kadikoy atmosphere",
        "besiktas character",
        
        # Casual queries
        "neighborhoods to explore",
        "areas worth visiting",
        "district recommendations",
        "where should i go",
    ],
    
    'daily_talks': [
        # Greetings
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        
        # Getting to know
        "who are you",
        "what is your name",
        "tell me about yourself",
        "what can you do",
        "how can you help",
        "what do you know",
        
        # General Istanbul questions
        "tell me about istanbul",
        "what is istanbul like",
        "describe istanbul",
        "istanbul facts",
        "istanbul history",
        
        # Personal recommendations
        "what do you recommend",
        "what should i do",
        "give me suggestions",
        "help me plan",
        "what's good in istanbul",
        
        # Small talk
        "how are you",
        "what's up",
        "nice to meet you",
        "thank you",
        "thanks",
        "you're helpful",
        "great job",
        
        # Questions about the system
        "how do you work",
        "are you ai",
        "are you human",
        "what's your purpose",
        
        # Casual conversation
        "i love istanbul",
        "istanbul is beautiful",
        "amazing city",
        "enjoying my visit",
        
        # Planning
        "first time in istanbul",
        "visiting istanbul",
        "tourist in istanbul",
        "new to istanbul",
        
        # General help
        "i need help",
        "can you assist me",
        "i have a question",
        "confused",
        
        # Farewell
        "goodbye",
        "bye",
        "see you",
        "talk later",
        
        # Feedback
        "you're great",
        "very helpful",
        "not helpful",
        "wrong information",
        
        # Casual questions
        "what's the best thing about istanbul",
        "why visit istanbul",
        "istanbul vs other cities",
        
        # Language
        "do you speak turkish",
        "can you help in english",
        "language assistance",
        
        # Weather small talk
        "nice weather today",
        "it's raining",
        "beautiful day",
        
        # Personal context
        "i'm from america",
        "visiting with family",
        "solo traveler",
        "on business trip",
        
        # Time-related
        "how long to visit istanbul",
        "spending 3 days",
        "weekend trip",
        
        # General queries
        "tell me something interesting",
        "fun facts",
        "local customs",
        "turkish culture",
    ],
    
    'local_tips': [
        # Hidden gems
        "hidden gems in istanbul",
        "secret spots",
        "off the beaten path",
        "undiscovered places",
        "locals only spots",
        
        # Authentic experiences
        "authentic istanbul",
        "real istanbul experience",
        "non-touristy things to do",
        "where locals go",
        "local favorites",
        
        # Insider tips
        "insider tips istanbul",
        "local secrets",
        "tips from locals",
        "what locals recommend",
        
        # Specific hidden gems
        "balat neighborhood secrets",
        "hidden cafes",
        "secret viewpoints",
        "unknown museums",
        
        # Food secrets
        "where locals eat",
        "local food spots",
        "hidden restaurants",
        "best street food locals know",
        
        # Shopping secrets
        "local markets",
        "where to shop like a local",
        "best deals",
        "hidden shops",
        
        # Cultural insights
        "local customs",
        "cultural tips",
        "etiquette in istanbul",
        "things to know",
        
        # Money-saving tips
        "how to save money",
        "budget tips",
        "free activities locals know",
        "cheap eats",
        
        # Timing tips
        "best time to visit attractions",
        "avoid crowds",
        "when locals visit",
        
        # Safety tips
        "safe areas",
        "areas to avoid",
        "safety tips",
        "scams to avoid",
        
        # Local habits
        "how locals live",
        "daily life istanbul",
        "local lifestyle",
        
        # Neighborhoods locals love
        "neighborhoods locals prefer",
        "where should i live",
        "authentic neighborhoods",
        
        # Practical tips
        "using turkish lira",
        "tipping in turkey",
        "haggling tips",
        "sim card turkey",
        
        # Transportation secrets
        "transport tricks",
        "how locals travel",
        "avoid tourist traps",
        
        # Seasonal tips
        "summer in istanbul",
        "winter tips",
        "spring activities",
        "autumn recommendations",
        
        # Weekend activities
        "what locals do on weekends",
        "weekend spots",
        "sunday activities",
        
        # Evening/night
        "where locals go at night",
        "evening activities",
        "sunset spots locals know",
        
        # Coffee/tea culture
        "best local cafes",
        "tea gardens locals visit",
        "turkish coffee spots",
        
        # Lesser-known attractions
        "underrated places",
        "overlooked spots",
        "places tourists miss",
        
        # Local events
        "local festivals",
        "neighborhood events",
        "community activities",
    ],
    
    'weather': [
        # Current weather
        "what's the weather like",
        "weather in istanbul",
        "is it raining",
        "is it sunny",
        "temperature istanbul",
        "how hot is it",
        "how cold is it",
        
        # Weather forecast
        "weather forecast",
        "will it rain tomorrow",
        "weather this week",
        "weekend weather",
        
        # Activity planning
        "is it good weather for sightseeing",
        "should i bring umbrella",
        "what to wear today",
        "outfit recommendations",
        
        # Season-specific
        "weather in summer",
        "winter weather istanbul",
        "spring temperature",
        "autumn weather",
        
        # Rain queries
        "is it going to rain",
        "rainy season istanbul",
        "rain forecast",
        "chances of rain",
        
        # Temperature
        "how warm is istanbul",
        "coldest months",
        "hottest time of year",
        "average temperature",
        
        # Activities based on weather
        "what to do in rain",
        "rainy day activities",
        "sunny day activities",
        "indoor activities for bad weather",
        
        # Practical questions
        "do i need jacket",
        "is it humid",
        "windy today",
        "weather conditions",
        
        # Comparison
        "weather compared to home",
        "temperature vs new york",
        "istanbul climate",
        
        # Best time to visit
        "best weather for visiting",
        "when is good weather",
        "avoid bad weather",
        
        # Specific weather events
        "snowfall istanbul",
        "heatwave",
        "storms",
        "fog in istanbul",
        
        # Bosphorus-related
        "weather on bosphorus",
        "boat tour weather",
        "ferry weather",
        
        # Photography
        "good weather for photos",
        "sunset time",
        "sunrise time",
        "golden hour",
        
        # Casual queries
        "nice weather today",
        "beautiful day",
        "terrible weather",
        "perfect weather",
    ],
    
    'events': [
        # General event queries
        "what events are happening",
        "events in istanbul",
        "things happening now",
        "upcoming events",
        "this weekend events",
        
        # Music
        "concerts in istanbul",
        "live music",
        "music festivals",
        "classical concerts",
        "jazz concerts",
        
        # Art
        "art exhibitions",
        "gallery openings",
        "art festivals",
        "contemporary art shows",
        
        # Cultural
        "cultural events",
        "cultural festivals",
        "traditional events",
        "heritage celebrations",
        
        # İKSV events
        "istanbul festival",
        "film festival",
        "jazz festival",
        "theater festival",
        "biennial",
        
        # Theater/performance
        "theater shows",
        "plays in istanbul",
        "opera",
        "ballet",
        "dance performances",
        
        # Festivals
        "festivals this month",
        "food festivals",
        "street festivals",
        "neighborhood festivals",
        
        # Sports events
        "football matches",
        "basketball games",
        "sports events",
        "marathon istanbul",
        
        # Markets/fairs
        "craft fairs",
        "antique fairs",
        "book fairs",
        "food markets",
        
        # Holiday events
        "ramadan events",
        "new year events",
        "christmas activities",
        "easter events",
        
        # Kids events
        "events for children",
        "family events",
        "kids activities this weekend",
        
        # Free events
        "free events",
        "free concerts",
        "free exhibitions",
        
        # Nightlife events
        "club events",
        "party tonight",
        "dj sets",
        "nightclub events",
        
        # Specific venues
        "events at zorlu",
        "hagia sophia events",
        "topkapi events",
        
        # Time-specific
        "events tonight",
        "today's events",
        "next week events",
        "monthly calendar",
        
        # Genre-specific
        "rock concerts",
        "electronic music",
        "traditional music",
        "world music",
        
        # Tickets
        "event tickets",
        "where to buy tickets",
        "ticket prices",
        
        # Casual queries
        "what's on",
        "anything happening",
        "entertainment options",
        "what to do tonight",
    ],
    
    'route_planning': [
        # Multi-stop itineraries
        "plan my day in istanbul",
        "one day itinerary",
        "3 day itinerary",
        "weekend plan",
        
        # Multiple attractions
        "visit 3 museums in one day",
        "best route for multiple attractions",
        "how to see sultanahmet and beyoglu",
        "combine hagia sophia and blue mosque",
        
        # Time-optimized
        "most efficient route",
        "quickest way to see attractions",
        "time-saving itinerary",
        "optimize my route",
        
        # Theme-based
        "ottoman heritage route",
        "byzantine tour",
        "food tour route",
        "photography route",
        
        # District tours
        "sultanahmet walking tour",
        "beyoglu route",
        "kadikoy itinerary",
        "explore besiktas",
        
        # Duration-based
        "half day plan",
        "morning itinerary",
        "afternoon activities",
        "evening route",
        
        # Custom routes
        "create custom tour",
        "personalized itinerary",
        "plan according to interests",
        
        # Family routes
        "family itinerary",
        "route for kids",
        "elderly-friendly route",
        
        # Budget routes
        "budget itinerary",
        "free attractions route",
        "cheap day out",
        
        # Bosphorus routes
        "bosphorus tour plan",
        "waterfront itinerary",
        "ferry route plan",
        
        # Walking tours
        "walking itinerary",
        "pedestrian routes",
        "walking distance attractions",
        
        # Start point specific
        "itinerary from taksim",
        "route starting sultanahmet",
        "plan from hotel",
        
        # Interest-based
        "history tour route",
        "art gallery itinerary",
        "shopping route",
        "food tour plan",
        
        # Multi-day plans
        "2 days in istanbul",
        "week long itinerary",
        "5 day plan",
        
        # Combination tours
        "museums and restaurants route",
        "attractions and shopping",
        "culture and food tour",
        
        # Practical planning
        "plan with metro",
        "walking and public transport",
        "minimize travel time",
        
        # Seasonal routes
        "summer itinerary",
        "winter tour plan",
        "rainy day route",
        
        # Casual queries
        "how to spend my day",
        "best way to see istanbul",
        "where should i go first",
        "plan my visit",
    ]
}

def generate_training_file():
    """Generate training data in JSON format"""
    
    # Prepare data in the format needed for training
    training_samples = []
    
    for intent, queries in TRAINING_DATA.items():
        for query in queries:
            training_samples.append({
                'text': query,
                'intent': intent,  # Changed from 'label' to 'intent'
                'timestamp': datetime.now().isoformat()
            })
    
    # Shuffle to avoid bias
    random.shuffle(training_samples)
    
    # Split into train/test (80/20)
    split_idx = int(len(training_samples) * 0.8)
    train_data = training_samples[:split_idx]
    test_data = training_samples[split_idx:]
    
    # Save to files
    output = {
        'metadata': {
            'total_samples': len(training_samples),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'num_intents': len(TRAINING_DATA),
            'intents': list(TRAINING_DATA.keys()),
            'samples_per_intent': {
                intent: len(queries) 
                for intent, queries in TRAINING_DATA.items()
            },
            'generated_at': datetime.now().isoformat()
        },
        'train_data': train_data,
        'test_data': test_data
    }
    
    # Save to file
    with open('intent_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("✅ Training data generated successfully!")
    print(f"\n📊 Statistics:")
    print(f"Total samples: {len(training_samples)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Number of intents: {len(TRAINING_DATA)}")
    print(f"\n📋 Samples per intent:")
    for intent, queries in TRAINING_DATA.items():
        print(f"  {intent}: {len(queries)} samples")
    
    return output

if __name__ == "__main__":
    generate_training_file()
