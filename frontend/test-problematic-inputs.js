// Comprehensive test script to test chatbot with problematic inputs
const API_BASE = 'http://localhost:8001';

async function testProblematicInputs() {
    console.log('🧪 Testing Chatbot with Problematic Inputs...\n');

    const testCases = [
        {
            category: '🔥 Edge Cases & Typos',
            inputs: [
                "restaurnats in istanbul",  // typo in "restaurants"
                "retaurants in istanbull", // typos in both words
                "restorans in beyoglu",     // Turkish spelling
                "where eat in fatih",       // missing "to"
                "hungry in",                // incomplete query
                "food in ",                 // trailing space
                "  restaurants  in  taksim  ", // extra spaces
                "RESTAURANTS IN GALATA",    // all caps
                "restaurants\nin\ngalata",  // line breaks
            ]
        },
        {
            category: '🌍 Non-Istanbul Locations (Should NOT trigger API)',
            inputs: [
                "restaurants in paris",
                "restaurants in new york",
                "places to visit in london",
                "food in tokyo",
                "best restaurants in madrid",
                "attractions in rome",
                "where to eat in berlin",
                "show me places in barcelona",
            ]
        },
        {
            category: '❓ Ambiguous Queries',
            inputs: [
                "restaurants",              // no location
                "places",                   // no location
                "food",                     // too generic
                "where to eat",            // no location
                "show me attractions",     // no location
                "good food",               // too vague
                "i want food",             // casual, no location
                "hungry",                  // single word
            ]
        },
        {
            category: '💬 Personal/Greeting Queries (Should NOT trigger API)',
            inputs: [
                "hello",
                "how are you",
                "how r u",
                "what's your name",
                "tell me about yourself",
                "hi there",
                "good morning",
                "thank you",
                "bye",
                "help",
            ]
        },
        {
            category: '🔍 Mixed Content Queries',
            inputs: [
                "i want to visit restaurants in istanbul but also need hotel recommendations",
                "restaurants in istanbul and also tell me about weather",
                "hello, can you show me places in istanbul please?",
                "good morning! where are good restaurants in beyoglu?",
                "thanks for the help. now show me attractions in fatih",
            ]
        },
        {
            category: '🚨 Potential Security/Injection Attempts',
            inputs: [
                "restaurants'; DROP TABLE places; --",
                "places in <script>alert('xss')</script>",
                "restaurants in istanbul'; DELETE FROM restaurants; --",
                "places to visit in istanbul OR 1=1",
                "restaurants in istanbul UNION SELECT * FROM users",
                "places in istanbul' AND '1'='1",
            ]
        },
        {
            category: '📝 Long Queries',
            inputs: [
                "i am planning a trip to istanbul and would really love to find some good restaurants in the galata area that serve traditional turkish food and have good atmosphere",
                "my family is visiting istanbul next month and we need recommendations for places to visit in sultanahmet district, especially historical sites that kids would enjoy",
            ]
        },
        {
            category: '🔤 Different Languages & Scripts',
            inputs: [
                "beyoğlu'nda restoranlar",  // Turkish
                "галата башня ресторани",    // Cyrillic
                "레스토랑 이스탄불",          // Korean
                "مطاعم في اسطنبول",         // Arabic
                "restaurants σε κωνσταντινούπολη", // Greek
            ]
        }
    ];

    for (const testCategory of testCases) {
        console.log(`\n${'='.repeat(50)}`);
        console.log(`${testCategory.category}`);
        console.log(`${'='.repeat(50)}\n`);

        for (const input of testCategory.inputs) {
            console.log(`Testing: "${input}"`);
            
            try {
                // Test the detection logic
                const shouldTriggerRestaurant = testRestaurantDetection(input);
                const shouldTriggerPlaces = testPlacesDetection(input);
                
                console.log(`  🍽️  Restaurant detection: ${shouldTriggerRestaurant ? '✅ YES' : '❌ NO'}`);
                console.log(`  🏛️  Places detection: ${shouldTriggerPlaces ? '✅ YES' : '❌ NO'}`);
                
                // Check if it should fallback to GPT
                const shouldFallbackToGPT = !shouldTriggerRestaurant && !shouldTriggerPlaces;
                console.log(`  🤖 Fallback to GPT: ${shouldFallbackToGPT ? '✅ YES' : '❌ NO'}`);
                
                // Test actual API call if detected
                if (shouldTriggerRestaurant) {
                    console.log(`  🔄 Would call: /restaurants/search`);
                }
                if (shouldTriggerPlaces) {
                    console.log(`  🔄 Would call: /places/`);
                }
                
                console.log('');
                
            } catch (error) {
                console.log(`  ❌ Error: ${error.message}\n`);
            }
        }
    }
    
    console.log(`\n${'='.repeat(50)}`);
    console.log('🎯 SUMMARY OF EXPECTATIONS');
    console.log(`${'='.repeat(50)}`);
    console.log('✅ Should trigger Restaurant API: Queries with location + food keywords');
    console.log('✅ Should trigger Places API: Queries with location + attraction keywords');  
    console.log('❌ Should NOT trigger APIs: Non-Istanbul locations, greetings, ambiguous queries');
    console.log('🤖 Should fallback to GPT: Everything that doesn\'t match specific patterns');
}

// Mock detection functions (simplified versions of the actual logic)
function testRestaurantDetection(userInput) {
    const input = userInput.toLowerCase();
    
    const explicitRestaurantRequests = [
        'restaurants in', 'where to eat in', 'food in', 'eat in', 'dining in',
        'kebab in', 'seafood in', 'cafe in', 'breakfast in', 'lunch in', 'dinner in'
    ];
    
    const istanbulDistricts = [
        'istanbul', 'beyoglu', 'galata', 'taksim', 'sultanahmet', 'fatih',
        'kadikoy', 'besiktas', 'uskudar', 'sisli', 'ortakoy', 'karakoy'
    ];
    
    const hasRestaurantKeyword = explicitRestaurantRequests.some(keyword => input.includes(keyword));
    if (!hasRestaurantKeyword) return false;
    
    // Simple location extraction (just check if any district is mentioned)
    const hasIstanbulLocation = istanbulDistricts.some(district => input.includes(district));
    
    return hasIstanbulLocation;
}

function testPlacesDetection(userInput) {
    const input = userInput.toLowerCase();
    
    const explicitPlacesRequests = [
        'attractions in', 'places to visit', 'sights in', 'landmarks in',
        'museums in', 'galleries in', 'churches in', 'mosques in',
        'what to see in', 'tourist attractions', 'historic sites',
        'best attractions', 'show me attractions', 'top attractions'
    ];
    
    const istanbulDistricts = [
        'istanbul', 'beyoglu', 'galata', 'taksim', 'sultanahmet', 'fatih',
        'kadikoy', 'besiktas', 'uskudar', 'sisli', 'ortakoy', 'karakoy'
    ];
    
    const hasPlacesKeyword = explicitPlacesRequests.some(keyword => input.includes(keyword));
    if (!hasPlacesKeyword) return false;
    
    // Check for Istanbul or general attractions query
    const hasIstanbulLocation = istanbulDistricts.some(district => input.includes(district));
    
    return hasIstanbulLocation;
}

// Run the test
testProblematicInputs();
