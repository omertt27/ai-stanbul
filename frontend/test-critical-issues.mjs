// Real-world test using actual chatbot detection functions
import { fetchStreamingResults, fetchRestaurantRecommendations, fetchPlacesRecommendations, extractLocationFromQuery } from './src/api/api.js';

// Copy the actual detection functions from Chatbot.jsx
function isExplicitRestaurantRequest(userInput) {
  console.log('🔍 Checking for explicit restaurant request:', userInput);
  const input = userInput.toLowerCase();

  // Only intercept very specific restaurant requests with location
  const explicitRestaurantRequests = [
    'restaurants in', 'where to eat in', 'restaurant recommendations for',
    'good restaurants in', 'best restaurants in', 'restaurants near',
    'where to eat near', 'dining in', 'food in', 'eat kebab in',
    'want kebab in', 'eat turkish food in', 'eat in', 'want to eat in',
    'find restaurants in', 'show me restaurants in', 'give me restaurants in',
    'best place to eat in', 'good place to eat in', 'recommend restaurants in',
    'suggest restaurants in', 'kebab in', 'seafood in', 'pizza in',
    'cafe in', 'breakfast in', 'brunch in', 'dinner in', 'lunch in',
    'eat something in', 'hungry in', 'where can i eat in',
    'where should i eat in', 'food places in', 'local food in',
    'authentic food in', 'traditional food in', 'vegetarian in',
    'vegan in', 'halal in', 'rooftop in', 'restaurants around',
    'eat around', 'food around', 'dining around', 'places to eat in',
    'best food in', 'good food in', 'find food in', 'find a restaurant in',
    'find me a restaurant in', 'suggest a restaurant in',
    'recommend a restaurant in', 'show restaurants in', 'show me food in',
    'show me places to eat in', 'give me food in', 'give me a restaurant in',
    'give me places to eat in', 'any restaurants in', 'any good restaurants in',
    'any food in', 'any place to eat in', 'any suggestions for food in',
    'any suggestions for restaurants in'
  ];
  
  // Only allow Istanbul or known districts
  const istanbulDistricts = [
    'istanbul', 'beyoglu', 'beyoğlu', 'galata', 'taksim', 'sultanahmet', 'fatih',
    'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'ortakoy',
    'ortaköy', 'sisli', 'şişli', 'karakoy', 'karaköy', 'bebek', 'arnavutkoy',
    'arnavutköy', 'balat', 'fener', 'eminonu', 'eminönü', 'bakirkoy', 'bakırköy', 'maltepe'
  ];

  const isExplicit = explicitRestaurantRequests.some(keyword => input.includes(keyword));
  if (!isExplicit) return false;
  
  // Extract location and check if it's Istanbul or a known district
  const { district, location } = extractLocationFromQuery(userInput);
  if (!district && !location) return false;
  
  // Use either district or location for matching
  const normalized = (district || location || '').trim().toLowerCase();
  // Only allow if normalized exactly matches a known Istanbul district (no partial matches)
  const isIstanbul = istanbulDistricts.includes(normalized);
  if (!isIstanbul) {
    console.log('❌ Location is not Istanbul or a known district:', normalized);
    return false;
  }
  return true;
}

function isExplicitPlacesRequest(userInput) {
  console.log('🔍 Checking for explicit places request:', userInput);
  const input = userInput.toLowerCase();

  const explicitPlacesRequests = [
    'attractions in', 'places to visit in', 'sights in', 'landmarks in',
    'monuments in', 'museums in', 'galleries in', 'churches in',
    'mosques in', 'palaces in', 'towers in', 'what to see in',
    'where to go in', 'tourist attractions in', 'historic sites in',
    'show me places in', 'recommend places in', 'suggest places in',
    'find places in', 'best places in', 'top places in', 'must see in',
    'worth visiting in', 'places around', 'attractions around',
    'sights around', 'things to do in', 'activities in', 'visit in',
    'explore in', 'cultural sites in', 'historical places in',
    'best attractions', 'show me attractions', 'top attractions'
  ];

  const istanbulDistricts = [
    'istanbul', 'beyoglu', 'beyoğlu', 'galata', 'taksim', 'sultanahmet', 'fatih',
    'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'ortakoy',
    'ortaköy', 'sisli', 'şişli', 'karakoy', 'karaköy', 'bebek', 'arnavutkoy',
    'arnavutköy', 'balat', 'fener', 'eminonu', 'eminönü', 'bakirkoy', 'bakırköy', 'maltepe'
  ];

  const isExplicit = explicitPlacesRequests.some(keyword => input.includes(keyword));
  if (!isExplicit) return false;

  const { district, location } = extractLocationFromQuery(userInput);
  if (!district && !location) {
    if (input.includes('istanbul')) return true;
    return false;
  }

  const normalized = (district || location || '').trim().toLowerCase();
  const isIstanbul = istanbulDistricts.includes(normalized);
  if (!isIstanbul) {
    console.log('❌ Location is not Istanbul or a known district:', normalized);
    return false;
  }
  return true;
}

async function testCriticalIssues() {
  console.log('🚨 Testing Critical Issues Found...\n');

  const criticalTests = [
    {
      name: 'Typo Handling',
      tests: [
        'restaurnats in istanbul',
        'retaurants in beyoglu', 
        'restorans in fatih',
        'plases to visit in galata',
        'attrctions in istanbul'
      ]
    },
    {
      name: 'Extra Spaces',
      tests: [
        '  restaurants  in  taksim  ',
        'places   to   visit   in   istanbul',
        ' food in sultanahmet '
      ]
    },
    {
      name: 'Security Injection Tests',
      tests: [
        "restaurants in istanbul'; DROP TABLE places; --",
        "places to visit in istanbul OR 1=1",
        "restaurants in istanbul UNION SELECT * FROM users",
        "places in <script>alert('xss')</script> istanbul"
      ]
    },
    {
      name: 'Mixed Case',
      tests: [
        'RESTAURANTS IN ISTANBUL',
        'Places To Visit In BEYOGLU',
        'Food In TakSiM'
      ]
    }
  ];

  for (const category of criticalTests) {
    console.log(`\n${'='.repeat(40)}`);
    console.log(`🔍 ${category.name}`);
    console.log(`${'='.repeat(40)}`);

    for (const testInput of category.tests) {
      console.log(`\nTesting: "${testInput}"`);
      
      try {
        const restaurantMatch = isExplicitRestaurantRequest(testInput);
        const placesMatch = isExplicitPlacesRequest(testInput);
        
        console.log(`  🍽️  Restaurant: ${restaurantMatch ? '✅ DETECTED' : '❌ NOT DETECTED'}`);
        console.log(`  🏛️  Places: ${placesMatch ? '✅ DETECTED' : '❌ NOT DETECTED'}`);
        
        if (restaurantMatch || placesMatch) {
          console.log('  ⚠️  WOULD CALL API - Check if this is intended!');
        } else {
          console.log('  ✅ Correctly falls back to GPT');
        }
        
        // Test location extraction
        const extracted = extractLocationFromQuery(testInput);
        console.log(`  📍 Extracted: district="${extracted.district}", location="${extracted.location}"`);
        
      } catch (error) {
        console.log(`  ❌ ERROR: ${error.message}`);
      }
    }
  }
  
  console.log('\n🎯 RECOMMENDATIONS:');
  console.log('1. Add fuzzy matching for common typos');
  console.log('2. Normalize whitespace before processing');
  console.log('3. Add input sanitization to prevent injection');
  console.log('4. Consider supporting Turkish language queries');
  console.log('5. Add comprehensive logging for edge cases');
}

testCriticalIssues();
