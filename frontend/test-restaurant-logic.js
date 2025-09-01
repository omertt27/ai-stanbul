// Test script for restaurant recommendation logic
console.log('Testing restaurant detection logic...');

// Helper function to detect if user is asking for restaurant recommendations
const isRestaurantAdviceRequest = (userInput) => {
  const input = userInput.toLowerCase();
  const restaurantKeywords = [
    'restaurant', 'restaurants', 'food', 'eat', 'dining', 'meal',
    'recommend', 'suggestion', 'advice', 'where to eat',
    'good food', 'best restaurant', 'restaurant recommendation',
    'places to eat', 'food recommendation', 'cuisine', 'turkish cuisine',
    'restaurant advice', 'give me restaurant', 'find restaurant',
    'show me restaurant', 'good restaurants', 'best places to eat'
  ];
  
  return restaurantKeywords.some(keyword => input.includes(keyword));
};

// Test cases
const testCases = [
  'Give me restaurant advice - recommend 4 good restaurants',
  'Find authentic Turkish restaurants in Istanbul',
  'Where can I eat good food?',
  'What are the best attractions in Istanbul?',
  'Tell me about history',
  'I need restaurant recommendations',
  'Show me some good restaurants',
  'What about food and dining options?'
];

testCases.forEach(testCase => {
  const result = isRestaurantAdviceRequest(testCase);
  console.log(`"${testCase}" -> ${result ? '✅ RESTAURANT' : '❌ NOT RESTAURANT'}`);
});

// Helper function to format restaurant recommendations
const formatRestaurantRecommendations = (restaurants) => {
  if (!restaurants || restaurants.length === 0) {
    return "I'm sorry, I couldn't find any restaurant recommendations at the moment. Please try again or be more specific about your preferences.";
  }

  let formattedResponse = "🍽️ **Here are 4 great restaurant recommendations for you:**\n\n";
  
  restaurants.slice(0, 4).forEach((restaurant, index) => {
    const name = restaurant.name || 'Unknown Restaurant';
    const rating = restaurant.rating ? `⭐ ${restaurant.rating}` : '';
    const address = restaurant.address || restaurant.vicinity || '';
    const description = restaurant.description || 'A popular dining spot in Istanbul.';
    
    formattedResponse += `**${index + 1}. ${name}**\n`;
    if (rating) formattedResponse += `${rating}\n`;
    if (address) formattedResponse += `📍 ${address}\n`;
    formattedResponse += `${description}\n\n`;
  });

  formattedResponse += "Would you like more details about any of these restaurants or recommendations for a specific type of cuisine?";
  
  return formattedResponse;
};

// Test formatting with sample data
const sampleRestaurants = [
  {
    name: "Test Restaurant 1",
    rating: 4.5,
    address: "123 Test Street, Istanbul",
    description: "A great Turkish restaurant."
  },
  {
    name: "Test Restaurant 2", 
    rating: 4.2,
    address: "456 Sample Ave, Istanbul",
    description: "Authentic Turkish cuisine."
  }
];

console.log('\n--- Formatted Output Test ---');
console.log(formatRestaurantRecommendations(sampleRestaurants));

console.log('\nTest completed.');
