// API utility that works for both local and deployed environments
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'; // Use correct port 8001
const API_URL = `${BASE_URL}/ai`;
const STREAM_API_URL = `${BASE_URL}/ai/stream`;
const RESTAURANTS_API_URL = `${BASE_URL}/restaurants/search`;
const PLACES_API_URL = `${BASE_URL}/places/`;

console.log('API Configuration:', {
  BASE_URL,
  API_URL,
  STREAM_API_URL,
  RESTAURANTS_API_URL,
  PLACES_API_URL,
  VITE_API_URL: import.meta.env.VITE_API_URL
});

export const fetchResults = async (query) => {
  try {
    console.log('Making API request to:', API_URL, 'with query:', query);
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_input: query }),
    });
    console.log('Response status:', response.status, response.statusText);
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API error response:', errorText);
      throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    console.log('API response data:', data);
    return data;
  } catch (error) {
    console.error('Fetch error:', error);
    throw error;
  }
};

export const fetchStreamingResults = async (query, onChunk) => {
  const response = await fetch(STREAM_API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_input: query }),
  });
  
  if (!response.ok) throw new Error('API error');
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          
          try {
            const parsed = JSON.parse(data);
            if (parsed.delta && parsed.delta.content) {
              onChunk(parsed.delta.content);
            }
          } catch (e) {
            // Ignore parsing errors for malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
};

// Helper function to extract location/district from user input
export const extractLocationFromQuery = (userInput) => {
  const input = userInput.toLowerCase();
  
  // Istanbul districts (both Turkish and English variants)
  const districts = {
    'beyoğlu': 'Beyoglu',
    'beyoglu': 'Beyoglu',
    'galata': 'Beyoglu',
    'taksim': 'Beyoglu',
    'sultanahmet': 'Sultanahmet',
    'fatih': 'Fatih',
    'kadıköy': 'Kadikoy',
    'kadikoy': 'Kadikoy',
    'beşiktaş': 'Besiktas',
    'besiktas': 'Besiktas',
    'şişli': 'Sisli',
    'sisli': 'Sisli',
    'üsküdar': 'Uskudar',
    'uskudar': 'Uskudar',
    'ortaköy': 'Besiktas',
    'ortakoy': 'Besiktas',
    'karaköy': 'Beyoglu',
    'karakoy': 'Beyoglu',
    'eminönü': 'Fatih',
    'eminonu': 'Fatih',
    'bakırköy': 'Bakirkoy',
    'bakirkoy': 'Bakirkoy'
  };
  
  // Check for district matches
  for (const [key, value] of Object.entries(districts)) {
    if (input.includes(key)) {
      return { district: value, keyword: null };
    }
  }
  
  // Check for cuisine keywords
  const cuisineKeywords = {
    'turkish': 'Turkish',
    'ottoman': 'Turkish',
    'kebab': 'kebab',
    'seafood': 'seafood',
    'fish': 'seafood',
    'italian': 'Italian',
    'pizza': 'Italian',
    'asian': 'Asian',
    'chinese': 'Chinese',
    'japanese': 'Japanese',
    'sushi': 'Japanese',
    'mediterranean': 'Mediterranean',
    'coffee': 'cafe',
    'cafe': 'cafe'
  };
  
  for (const [key, value] of Object.entries(cuisineKeywords)) {
    if (input.includes(key)) {
      return { district: null, keyword: value };
    }
  }
  
  return { district: null, keyword: null };
};

export const fetchRestaurantRecommendations = async (userInput = '', limit = 4) => {
  try {
    console.log('fetchRestaurantRecommendations called with userInput:', userInput);
    const { district, keyword } = extractLocationFromQuery(userInput);
    console.log('Extracted filters - District:', district, 'Keyword:', keyword);
    
    const params = new URLSearchParams();
    if (district) params.append('district', district);
    if (keyword) params.append('keyword', keyword);
    params.append('limit', limit.toString());

    const url = `${RESTAURANTS_API_URL}?${params}`;
    console.log('Making restaurant API request to:', url);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    console.log('Restaurant API response status:', response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Restaurant API error response:', errorText);
      throw new Error(`Restaurant API error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Restaurant API response data:', data);
    console.log('Number of restaurants returned:', data.restaurants?.length);
    return data;
  } catch (error) {
    console.error('Restaurant fetch error:', error);
    throw error;
  }
};

export const fetchPlacesRecommendations = async (userInput = '', limit = 6) => {
  try {
    console.log('fetchPlacesRecommendations called with userInput:', userInput);
    const { district, keyword } = extractLocationFromQuery(userInput);
    console.log('Extracted filters - District:', district, 'Keyword:', keyword);
    
    const params = new URLSearchParams();
    if (district) params.append('district', district);
    if (keyword) params.append('keyword', keyword);
    params.append('limit', limit.toString());

    const url = `${PLACES_API_URL}?${params}`;
    console.log('Making places API request to:', url);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    console.log('Places API response status:', response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Places API error response:', errorText);
      throw new Error(`Places API error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Places API response data:', data);
    console.log('Number of places returned:', data.length);
    return { places: data }; // Wrap in places object to match expected format
  } catch (error) {
    console.error('Places fetch error:', error);
    throw error;
  }
};
