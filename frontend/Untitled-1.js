fetch('http://localhost:8001/restaurants/search?limit=2')
  .then(r => r.json())
  .then(d => console.log('✅ Working:', d.total_found, 'restaurants found'))
  .catch(e => console.error('❌ Error:', e));