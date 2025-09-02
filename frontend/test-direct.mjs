// Quick test to verify the restaurant logic
import { fetchRestaurantRecommendations } from './src/api/api.js';

const testQuery = 'give me restrants in beyoglu';

console.log('Testing restaurant functionality...');

// Test the function directly
fetchRestaurantRecommendations(testQuery)
  .then(data => {
    console.log('✅ Success:', data);
    console.log('Restaurants found:', data.restaurants?.length);
    if (data.restaurants?.[0]) {
      console.log('First restaurant:', data.restaurants[0].name);
    }
  })
  .catch(error => {
    console.error('❌ Error:', error);
  });
