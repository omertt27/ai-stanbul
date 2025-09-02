#!/usr/bin/env node

/**
 * Final test script to verify the complete restaurant recommendation flow
 * 
 * SYSTEM OVERVIEW:
 * - Uses GPT-powered smart routing (simplified from complex detection logic)
 * - Only intercepts VERY explicit restaurant requests with location
 * - Everything else goes to OpenAI GPT for intelligent conversation
 * - Backend returns actual restaurants (not hotels) via Google Places API
 * 
 * FLOW:
 * 1. User input -> Frontend detection
 * 2. If explicit restaurant+location -> Restaurant API (4 real restaurants)
 * 3. If anything else -> OpenAI GPT chat (contextual responses)
 * 
 * EXAMPLES:
 * - "restaurants in Beyoglu" -> API returns 4 restaurants
 * - "I'm Turkish, coming tomorrow" -> GPT provides travel advice
 * - "tell me about good food" -> GPT offers to get live restaurant data
 */

const apiUrl = 'http://localhost:8001';

// Simplified test detection function - matches new frontend logic
function detectRestaurantQuery(message) {
    const input = message.toLowerCase();
    
    // Only intercept very specific restaurant requests with location
    // These are the ONLY phrases that trigger the restaurant API
    const explicitRestaurantRequests = [
        'restaurants in',        // "restaurants in Beyoglu"
        'where to eat in',       // "where to eat in Sultanahmet"
        'restaurant recommendations for', // "restaurant recommendations for Taksim"
        'good restaurants in',   // "good restaurants in Galata"
        'best restaurants in',   // "best restaurants in Kadikoy"
        'restaurants near',      // "restaurants near Taksim Square"
        'where to eat near',     // "where to eat near Galata Tower"
        'dining in',            // "dining in Beyoglu"
        'food in',              // "food in Sultanahmet"
        'eat kebab in',         // "i want eat kebab in fatih"
        'want kebab in',        // "i want kebab in beyoglu"
        'eat turkish food in',  // "eat turkish food in taksim"
        'eat in',               // "i want to eat in fatih"
        'want to eat in'        // "i want to eat in sultanahmet"
    ];
    
    return explicitRestaurantRequests.some(keyword => input.includes(keyword));
}

// Test extraction function - extracts location/district from user queries
function extractLocationFromQuery(query) {
    const locationPatterns = [
        /in\s+([a-zA-Zığüşöç\s]+)/i,        // "restaurants in Beyoglu"
        /at\s+([a-zA-Zığüşöç\s]+)/i,        // "restaurants at Taksim"
        /near\s+([a-zA-Zığüşöç\s]+)/i,      // "restaurants near Galata Tower"
        /around\s+([a-zA-Zığüşöç\s]+)/i,    // "restaurants around Sultanahmet"
        /\b([a-zA-Zığüşöç]+)\s+restaurants?/i, // "Beyoglu restaurants"
        /restaurants?\s+in\s+([a-zA-Zığüşöç\s]+)/i // "restaurants in Kadikoy"
    ];
    
    // Try each pattern to extract location name
    for (const pattern of locationPatterns) {
        const match = query.match(pattern);
        if (match && match[1]) {
            return match[1].trim(); // Return the captured location name
        }
    }
    
    return null; // No location found
}

// Test cuisine extraction function - extracts cuisine type from user queries  
function extractCuisineFromQuery(query) {
    const input = query.toLowerCase();
    const cuisineKeywords = {
        'kebab': 'kebab',           // "i want eat kebab in fatih"
        'turkish': 'turkish',      // "turkish food in beyoglu"
        'ottoman': 'turkish',      // "ottoman cuisine"
        'seafood': 'seafood',      // "seafood in galata"
        'fish': 'seafood',         // "fish restaurants"
        'italian': 'italian',      // "italian food"
        'pizza': 'italian',        // "pizza in taksim"
        'asian': 'asian',          // "asian cuisine"
        'chinese': 'chinese',      // "chinese food"
        'japanese': 'japanese',    // "japanese restaurant"
        'sushi': 'japanese',       // "sushi in beyoglu"
        'mediterranean': 'mediterranean', // "mediterranean food"
        'coffee': 'cafe',          // "coffee in sultanahmet"
        'cafe': 'cafe'            // "cafe recommendations"
    };
    
    // Check for cuisine keywords
    for (const [keyword, cuisine] of Object.entries(cuisineKeywords)) {
        if (input.includes(keyword)) {
            return cuisine; // Return standardized cuisine name
        }
    }
    
    return null; // No cuisine found
}

// Test API call function
async function fetchRestaurantRecommendations(location = null, cuisine = null) {
    try {
        // Build URL parameters for the restaurant API
        const params = new URLSearchParams();
        if (location) params.append('district', location); // Add district filter
        if (cuisine) params.append('keyword', cuisine);   // Add cuisine filter
        params.append('limit', '4'); // Request exactly 4 restaurants

        const response = await fetch(`${apiUrl}/restaurants/search?${params}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ API Response Status:', data.status);
        console.log('✅ Total Found:', data.total_found);
        
        // Display restaurant information
        if (data.restaurants && data.restaurants.length > 0) {
            console.log('✅ Restaurant Names:');
            data.restaurants.forEach((restaurant, index) => {
                console.log(`   ${index + 1}. ${restaurant.name}`);        // Restaurant name
                console.log(`      📍 ${restaurant.address}`);             // Full address
                console.log(`      ⭐ Rating: ${restaurant.rating || 'N/A'}`); // Google rating
                console.log(`      🍽️ Types: ${restaurant.cuisine_types || 'N/A'}`); // Cuisine categories
                console.log('');
            });
        }
        
        return data;
    } catch (error) {
        console.error('❌ Error fetching restaurant recommendations:', error.message);
        return null;
    }
}

async function runTests() {
    console.log('🧪 Testing Restaurant Recommendation System\n');
    
    // Test 1: Detection (Simplified Logic)
    console.log('Test 1: Simplified Restaurant Query Detection');
    const testQueries = [
        'restaurants in Beyoglu',                         // ✅ Should detect (explicit location)
        'where to eat in Sultanahmet',                   // ✅ Should detect (explicit location) 
        'good restaurants in Taksim',                    // ✅ Should detect (explicit location)
        'i want eat kebab in fatih',                     // ✅ Should detect (kebab + location)
        'im an ukrainian',                               // ❌ Should NOT detect (personal info)
        'im turkish, i will come tomorrow to istanbul',  // ❌ Should NOT detect (travel chat)
        'tell me about good food',                       // ❌ Should NOT detect (general question)
        'what should i visit tomorrow',                  // ❌ Should NOT detect (travel question)
        'hello, how are you?',                          // ❌ Should NOT detect (greeting)
        'restaurant recommendations please'              // ❌ Should NOT detect (no location)
    ];
    
    testQueries.forEach(query => {
        const detected = detectRestaurantQuery(query);
        console.log(`   "${query}" -> ${detected ? '✅ DETECTED' : '❌ NOT DETECTED'}`);
    });
    
    console.log('\nTest 2: Location Extraction');
    const locationQueries = [
        'restaurants in Beyoglu',     // Should extract "Beyoglu"
        'where to eat in Sultanahmet', // Should extract "Sultanahmet"
        'Galata restaurants',         // Should extract "Galata"
        'restaurants near Taksim'     // Should extract "Taksim"
    ];
    
    locationQueries.forEach(query => {
        const location = extractLocationFromQuery(query);
        console.log(`   "${query}" -> Location: "${location || 'NONE'}"`);
    });
    
    // Test 3: API Call - Get real restaurant data from backend
    console.log('\nTest 3: API Response for Beyoglu');
    const result = await fetchRestaurantRecommendations('Beyoglu');
    
    if (result && result.restaurants && result.restaurants.length > 0) {
        console.log('✅ Complete flow working! Backend is returning actual restaurants.');
    } else {
        console.log('❌ Issue with API response or backend filtering.');
    }
}

runTests().catch(console.error);
