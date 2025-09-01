// Test the complete restaurant request flow
const testQuery = 'give me restaurants in beyoglu';

// Simulate the API call
fetch('http://localhost:8001/restaurants/search?district=Beyoglu&limit=4')
  .then(r => r.json())
  .then(data => {
    console.log('✅ API Response for "restaurants in beyoglu":');
    console.log('Total found:', data.total_found);
    console.log('Location searched:', data.location_searched);
    console.log('Restaurants:');
    data.restaurants.forEach((restaurant, index) => {
      console.log(`${index + 1}. ${restaurant.name} (${restaurant.rating}⭐)`);
      console.log(`   📍 ${restaurant.address}`);
    });
  })
  .catch(e => console.error('❌ Error:', e));
