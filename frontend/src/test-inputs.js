// 50 Test Inputs for Istanbul Guide Chatbot
// Categories: Restaurants, Attractions, Districts, Transportation, Culture, Shopping, Nightlife, Hotels, Weather, Events

export const testInputs = [
  // RESTAURANTS & FOOD (10 questions)
  {
    id: 1,
    category: "Restaurants",
    input: "Best traditional Turkish restaurants in Sultanahmet",
    expectedTopics: ["Turkish cuisine", "Sultanahmet", "traditional food", "restaurants"]
  },
  {
    id: 2,
    category: "Restaurants", 
    input: "Where can I find authentic kebab in Beyoğlu?",
    expectedTopics: ["kebab", "Beyoğlu", "authentic", "Turkish food"]
  },
  {
    id: 3,
    category: "Restaurants",
    input: "Seafood restaurants with Bosphorus view",
    expectedTopics: ["seafood", "Bosphorus", "view", "restaurants"]
  },
  {
    id: 4,
    category: "Restaurants",
    input: "Budget-friendly local food in Kadıköy",
    expectedTopics: ["budget", "local food", "Kadıköy", "cheap eats"]
  },
  {
    id: 5,
    category: "Restaurants",
    input: "Best breakfast places in Beşiktaş",
    expectedTopics: ["breakfast", "Beşiktaş", "morning food", "kahvaltı"]
  },
  {
    id: 6,
    category: "Restaurants",
    input: "Vegetarian restaurants near Taksim Square",
    expectedTopics: ["vegetarian", "Taksim", "plant-based", "restaurants"]
  },
  {
    id: 7,
    category: "Restaurants",
    input: "Street food vendors in Eminönü",
    expectedTopics: ["street food", "Eminönü", "vendors", "local snacks"]
  },
  {
    id: 8,
    category: "Restaurants",
    input: "Rooftop restaurants in Galata",
    expectedTopics: ["rooftop", "Galata", "restaurants", "views"]
  },
  {
    id: 9,
    category: "Restaurants",
    input: "Turkish coffee houses in historic areas",
    expectedTopics: ["Turkish coffee", "historic", "coffee houses", "traditional"]
  },
  {
    id: 10,
    category: "Restaurants",
    input: "Fine dining with Ottoman cuisine",
    expectedTopics: ["fine dining", "Ottoman", "cuisine", "upscale"]
  },

  // ATTRACTIONS & LANDMARKS (10 questions)
  {
    id: 11,
    category: "Attractions",
    input: "Must-see historical sites in Istanbul",
    expectedTopics: ["historical sites", "must-see", "landmarks", "history"]
  },
  {
    id: 12,
    category: "Attractions",
    input: "How to visit Hagia Sophia and Blue Mosque in one day",
    expectedTopics: ["Hagia Sophia", "Blue Mosque", "one day", "itinerary"]
  },
  {
    id: 13,
    category: "Attractions",
    input: "Topkapi Palace visiting hours and tickets",
    expectedTopics: ["Topkapi Palace", "hours", "tickets", "visiting info"]
  },
  {
    id: 14,
    category: "Attractions",
    input: "Best viewpoints in Istanbul for photography",
    expectedTopics: ["viewpoints", "photography", "views", "scenic spots"]
  },
  {
    id: 15,
    category: "Attractions",
    input: "Underground Cistern tour information",
    expectedTopics: ["Basilica Cistern", "underground", "tour", "Byzantine"]
  },
  {
    id: 16,
    category: "Attractions",
    input: "Museums to visit on a rainy day",
    expectedTopics: ["museums", "rainy day", "indoor activities", "culture"]
  },
  {
    id: 17,
    category: "Attractions",
    input: "Galata Tower opening times and entrance fee",
    expectedTopics: ["Galata Tower", "opening times", "entrance fee", "panoramic view"]
  },
  {
    id: 18,
    category: "Attractions",
    input: "Dolmabahçe Palace guided tours",
    expectedTopics: ["Dolmabahçe Palace", "guided tours", "Ottoman", "architecture"]
  },
  {
    id: 19,
    category: "Attractions",
    input: "Hidden gems off the beaten path",
    expectedTopics: ["hidden gems", "off beaten path", "secret spots", "local discoveries"]
  },
  {
    id: 20,
    category: "Attractions",
    input: "Istanbul Modern Art Museum exhibitions",
    expectedTopics: ["Istanbul Modern", "art museum", "exhibitions", "contemporary art"]
  },

  // DISTRICTS & NEIGHBORHOODS (8 questions)
  {
    id: 21,
    category: "Districts",
    input: "What to do in Karaköy district",
    expectedTopics: ["Karaköy", "activities", "district guide", "neighborhood"]
  },
  {
    id: 22,
    category: "Districts",
    input: "Walking tour of Balat neighborhood",
    expectedTopics: ["Balat", "walking tour", "colorful houses", "Jewish quarter"]
  },
  {
    id: 23,
    category: "Districts",
    input: "Üsküdar Asian side attractions",
    expectedTopics: ["Üsküdar", "Asian side", "attractions", "peaceful"]
  },
  {
    id: 24,
    category: "Districts",
    input: "Ortaköy market and mosque",
    expectedTopics: ["Ortaköy", "market", "mosque", "Bosphorus"]
  },
  {
    id: 25,
    category: "Districts",
    input: "Şişli shopping and entertainment",
    expectedTopics: ["Şişli", "shopping", "entertainment", "modern district"]
  },
  {
    id: 26,
    category: "Districts",
    input: "Best areas to stay for first-time visitors",
    expectedTopics: ["accommodation", "first-time visitors", "areas to stay", "hotels"]
  },
  {
    id: 27,
    category: "Districts",
    input: "Fatih district historical significance",
    expectedTopics: ["Fatih", "historical", "significance", "Old City"]
  },
  {
    id: 28,
    category: "Districts",
    input: "Arnavutköy waterfront restaurants",
    expectedTopics: ["Arnavutköy", "waterfront", "restaurants", "Bosphorus village"]
  },

  // TRANSPORTATION (6 questions)
  {
    id: 29,
    category: "Transportation",
    input: "How to get from airport to city center",
    expectedTopics: ["airport", "city center", "transportation", "transfer"]
  },
  {
    id: 30,
    category: "Transportation",
    input: "Istanbul public transport card and prices",
    expectedTopics: ["public transport", "Istanbulkart", "prices", "metro"]
  },
  {
    id: 31,
    category: "Transportation",
    input: "Ferry schedules to Princes Islands",
    expectedTopics: ["ferry", "Princes Islands", "schedules", "sea transport"]
  },
  {
    id: 32,
    category: "Transportation",
    input: "Metro lines and connections in Istanbul",
    expectedTopics: ["metro", "lines", "connections", "subway"]
  },
  {
    id: 33,
    category: "Transportation",
    input: "Taxi vs Uber prices and safety",
    expectedTopics: ["taxi", "Uber", "prices", "safety", "transportation"]
  },
  {
    id: 34,
    category: "Transportation",
    input: "Walking distances between major attractions",
    expectedTopics: ["walking", "distances", "attractions", "pedestrian"]
  },

  // CULTURE & HISTORY (6 questions)
  {
    id: 35,
    category: "Culture",
    input: "Turkish bath (hammam) experience guide",
    expectedTopics: ["hammam", "Turkish bath", "experience", "traditional"]
  },
  {
    id: 36,
    category: "Culture",
    input: "Byzantine history in Istanbul",
    expectedTopics: ["Byzantine", "history", "Constantinople", "empire"]
  },
  {
    id: 37,
    category: "Culture",
    input: "Ottoman Empire sites and stories",
    expectedTopics: ["Ottoman", "empire", "sites", "history"]
  },
  {
    id: 38,
    category: "Culture",
    input: "Traditional Turkish music venues",
    expectedTopics: ["Turkish music", "traditional", "venues", "folk music"]
  },
  {
    id: 39,
    category: "Culture",
    input: "Local customs and etiquette tips",
    expectedTopics: ["customs", "etiquette", "local culture", "tips"]
  },
  {
    id: 40,
    category: "Culture",
    input: "Religious sites visiting guidelines",
    expectedTopics: ["religious sites", "guidelines", "mosques", "churches"]
  },

  // SHOPPING (4 questions)
  {
    id: 41,
    category: "Shopping",
    input: "Grand Bazaar shopping tips and bargaining",
    expectedTopics: ["Grand Bazaar", "shopping", "bargaining", "tips"]
  },
  {
    id: 42,
    category: "Shopping",
    input: "Modern shopping malls in Istanbul",
    expectedTopics: ["shopping malls", "modern", "retail", "stores"]
  },
  {
    id: 43,
    category: "Shopping",
    input: "Spice Bazaar best products to buy",
    expectedTopics: ["Spice Bazaar", "products", "spices", "souvenirs"]
  },
  {
    id: 44,
    category: "Shopping",
    input: "Turkish carpet and textile shopping",
    expectedTopics: ["Turkish carpet", "textiles", "authentic", "quality"]
  },

  // NIGHTLIFE & ENTERTAINMENT (3 questions)
  {
    id: 45,
    category: "Nightlife",
    input: "Best nightlife areas and bars",
    expectedTopics: ["nightlife", "bars", "entertainment", "evening"]
  },
  {
    id: 46,
    category: "Nightlife",
    input: "Live music venues and concerts",
    expectedTopics: ["live music", "venues", "concerts", "entertainment"]
  },
  {
    id: 47,
    category: "Nightlife",
    input: "Traditional Turkish entertainment shows",
    expectedTopics: ["Turkish shows", "traditional entertainment", "cultural shows"]
  },

  // PRACTICAL INFORMATION (3 questions)
  {
    id: 48,
    category: "Practical",
    input: "Weather and best time to visit Istanbul",
    expectedTopics: ["weather", "best time", "climate", "seasons"]
  },
  {
    id: 49,
    category: "Practical",
    input: "Currency exchange and tipping customs",
    expectedTopics: ["currency", "exchange", "tipping", "money"]
  },
  {
    id: 50,
    category: "Practical",
    input: "Emergency numbers and healthcare in Istanbul",
    expectedTopics: ["emergency", "healthcare", "medical", "safety"]
  }
];

// Function to test chatbot with all inputs
export const testChatbot = async (sendMessage) => {
  const results = [];
  
  for (let i = 0; i < testInputs.length; i++) {
    const testCase = testInputs[i];
    console.log(`Testing ${i + 1}/50: ${testCase.input}`);
    
    try {
      const response = await sendMessage(testCase.input);
      results.push({
        ...testCase,
        response: response,
        success: true,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      results.push({
        ...testCase,
        response: null,
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
    
    // Add delay between requests to avoid overwhelming the API
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  return results;
};

// Function to analyze test results
export const analyzeResults = (results) => {
  const analysis = {
    totalTests: results.length,
    successful: results.filter(r => r.success).length,
    failed: results.filter(r => !r.success).length,
    categories: {}
  };
  
  // Analyze by category
  testInputs.forEach(test => {
    if (!analysis.categories[test.category]) {
      analysis.categories[test.category] = {
        total: 0,
        successful: 0,
        failed: 0
      };
    }
    analysis.categories[test.category].total++;
  });
  
  results.forEach(result => {
    const category = result.category;
    if (result.success) {
      analysis.categories[category].successful++;
    } else {
      analysis.categories[category].failed++;
    }
  });
  
  return analysis;
};

// Export individual test categories for focused testing
export const testsByCategory = {
  restaurants: testInputs.filter(t => t.category === "Restaurants"),
  attractions: testInputs.filter(t => t.category === "Attractions"),
  districts: testInputs.filter(t => t.category === "Districts"),
  transportation: testInputs.filter(t => t.category === "Transportation"),
  culture: testInputs.filter(t => t.category === "Culture"),
  shopping: testInputs.filter(t => t.category === "Shopping"),
  nightlife: testInputs.filter(t => t.category === "Nightlife"),
  practical: testInputs.filter(t => t.category === "Practical")
};
