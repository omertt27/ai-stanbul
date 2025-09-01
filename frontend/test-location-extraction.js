// Test the location extraction logic
const extractLocationFromQuery = (userInput) => {
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

// Test cases
const testQueries = [
  'give me restaurants in beyoglu',
  'show me food in Kadıköy',
  'restaurants in taksim',
  'I want Turkish restaurants',
  'find seafood restaurants',
  'good restaurants advice',
  'pizza places in Galata'
];

console.log('Testing location extraction:');
testQueries.forEach(query => {
  const result = extractLocationFromQuery(query);
  console.log(`"${query}" -> District: ${result.district}, Keyword: ${result.keyword}`);
});
