/**
 * AI-Powered Content Enhancement Utilities
 * Auto-generates tags, suggests improvements, and provides content scoring
 */

// Common Istanbul-related keywords and their variations
const ISTANBUL_KEYWORDS = {
  districts: [
    'sultanahmet', 'beyoglu', 'galata', 'taksim', 'kadikoy', 'besiktas', 
    'sisli', 'uskudar', 'bakirkoy', 'ortakoy', 'karakoy', 'eminonu',
    'fatih', 'sariyer', 'zeytinburnu', 'avcilar'
  ],
  attractions: [
    'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar',
    'galata tower', 'bosphorus', 'golden horn', 'basilica cistern',
    'dolmabahce palace', 'spice bazaar', 'princes islands', 'chora church'
  ],
  food: [
    'turkish breakfast', 'kebab', 'baklava', 'turkish delight', 'tea',
    'coffee', 'meze', 'raki', 'pide', 'doner', 'simit', 'borek'
  ],
  culture: [
    'ottoman', 'byzantine', 'mosque', 'hammam', 'bazaar', 'ferry',
    'turkish', 'asian side', 'european side', 'call to prayer', 'minaret'
  ],
  activities: [
    'walking tour', 'boat trip', 'sunset', 'photography', 'shopping',
    'nightlife', 'rooftop', 'street food', 'museum', 'art gallery'
  ]
};

/**
 * Auto-generate relevant tags from blog post content
 * @param {string} title - Blog post title
 * @param {string} content - Blog post content
 * @param {string} district - Selected district
 * @returns {Array} Array of suggested tags
 */
export const generateAutoTags = (title, content, district = '') => {
  const text = `${title} ${content} ${district}`.toLowerCase();
  const suggestedTags = new Set();

  // Always add core tags
  suggestedTags.add('istanbul');
  suggestedTags.add('travel');

  // Add district if specified
  if (district && district !== 'General') {
    suggestedTags.add(district.toLowerCase());
  }

  // Check for district mentions
  ISTANBUL_KEYWORDS.districts.forEach(keyword => {
    if (text.includes(keyword)) {
      suggestedTags.add(keyword);
    }
  });

  // Check for attractions
  ISTANBUL_KEYWORDS.attractions.forEach(keyword => {
    if (text.includes(keyword)) {
      suggestedTags.add(keyword.replace(/\s+/g, '_'));
    }
  });

  // Check for food-related content
  ISTANBUL_KEYWORDS.food.forEach(keyword => {
    if (text.includes(keyword)) {
      suggestedTags.add('food');
      suggestedTags.add('turkish_cuisine');
    }
  });

  // Check for cultural content
  ISTANBUL_KEYWORDS.culture.forEach(keyword => {
    if (text.includes(keyword)) {
      suggestedTags.add('culture');
      suggestedTags.add('history');
    }
  });

  // Check for activities
  ISTANBUL_KEYWORDS.activities.forEach(keyword => {
    if (text.includes(keyword)) {
      suggestedTags.add('activities');
      if (keyword.includes('tour')) suggestedTags.add('tours');
      if (keyword.includes('food')) suggestedTags.add('food');
      if (keyword.includes('art') || keyword.includes('museum')) suggestedTags.add('art');
    }
  });

  // Add content type tags based on common patterns
  if (text.includes('hidden') || text.includes('secret') || text.includes('local')) {
    suggestedTags.add('hidden_gems');
  }
  if (text.includes('guide') || text.includes('tips') || text.includes('how to')) {
    suggestedTags.add('travel_guide');
  }
  if (text.includes('best') || text.includes('top') || text.includes('must')) {
    suggestedTags.add('recommendations');
  }
  if (text.includes('budget') || text.includes('cheap') || text.includes('free')) {
    suggestedTags.add('budget_travel');
  }
  if (text.includes('luxury') || text.includes('premium') || text.includes('upscale')) {
    suggestedTags.add('luxury_travel');
  }

  return Array.from(suggestedTags).slice(0, 8); // Limit to 8 tags
};

/**
 * Analyze content quality and provide scoring
 * @param {string} title - Blog post title
 * @param {string} content - Blog post content
 * @returns {Object} Content quality analysis
 */
export const analyzeContentQuality = (title, content) => {
  const wordCount = content.trim().split(/\s+/).length;
  const charCount = content.length;
  const paragraphs = content.split('\n').filter(p => p.trim().length > 0).length;
  
  let score = 0;
  const feedback = [];
  const suggestions = [];

  // Title analysis
  if (title.length >= 10 && title.length <= 60) {
    score += 15;
  } else if (title.length < 10) {
    feedback.push('Title is too short');
    suggestions.push('Consider expanding your title to be more descriptive');
  } else {
    feedback.push('Title is too long for SEO');
    suggestions.push('Shorten your title to under 60 characters');
  }

  // Content length analysis
  if (wordCount >= 300 && wordCount <= 2000) {
    score += 20;
  } else if (wordCount < 300) {
    feedback.push('Content is too short');
    suggestions.push('Add more details and insights to reach at least 300 words');
  } else {
    feedback.push('Content is very long');
    suggestions.push('Consider breaking into multiple posts or adding subheadings');
  }

  // Structure analysis
  if (paragraphs >= 3) {
    score += 15;
  } else {
    feedback.push('Content needs better structure');
    suggestions.push('Break your content into more paragraphs for better readability');
  }

  // Readability indicators
  const hasSubheadings = content.includes('##') || content.includes('**') || content.includes('***');
  if (hasSubheadings) {
    score += 15;
    feedback.push('Good use of formatting');
  } else {
    suggestions.push('Add subheadings or bold text to improve structure');
  }

  // Istanbul relevance
  const istanbulMentions = (content.toLowerCase().match(/istanbul|turkey|ottoman|byzantine/g) || []).length;
  if (istanbulMentions >= 2) {
    score += 15;
    feedback.push('Good Istanbul context');
  } else {
    suggestions.push('Add more specific Istanbul references for better relevance');
  }

  // Practical information
  const hasPracticalInfo = /opening hours|price|cost|how to get|location|address|metro|bus/i.test(content);
  if (hasPracticalInfo) {
    score += 10;
    feedback.push('Contains practical travel information');
  } else {
    suggestions.push('Consider adding practical information like hours, prices, or directions');
  }

  // Personal experience indicators
  const hasPersonalTouch = /i visited|my experience|i recommend|personally|we went/i.test(content);
  if (hasPersonalTouch) {
    score += 10;
    feedback.push('Personal experience adds authenticity');
  } else {
    suggestions.push('Share personal experiences to make the content more engaging');
  }

  // Determine overall quality
  let quality = 'Poor';
  if (score >= 80) quality = 'Excellent';
  else if (score >= 65) quality = 'Good';
  else if (score >= 50) quality = 'Fair';

  return {
    score,
    quality,
    wordCount,
    charCount,
    paragraphs,
    feedback,
    suggestions,
    analysis: {
      titleLength: title.length,
      readabilityScore: hasSubheadings ? 'Good' : 'Needs Improvement',
      istanbulRelevance: istanbulMentions >= 2 ? 'High' : 'Low',
      practicalInfo: hasPracticalInfo ? 'Yes' : 'No',
      personalTouch: hasPersonalTouch ? 'Yes' : 'No'
    }
  };
};

/**
 * Generate content suggestions based on the selected district
 * @param {string} district - Selected district
 * @returns {Array} Array of content suggestions
 */
export const getDistrictContentSuggestions = (district) => {
  const suggestions = {
    'Sultanahmet': [
      'Visit early morning to avoid crowds',
      'Mention the best photo spots around Hagia Sophia',
      'Include tips about mosque visiting etiquette',
      'Suggest nearby traditional restaurants',
      'Add information about the area\'s Byzantine and Ottoman history'
    ],
    'Galata': [
      'Include sunset timing for Galata Tower views',
      'Mention the trendy cafes and art galleries',
      'Talk about the area\'s transformation from merchant district',
      'Suggest the best rooftop bars with Bosphorus views',
      'Include walking route from Karaköy to Taksim'
    ],
    'Beyoğlu': [
      'Mention İstiklal Street and its historic tram',
      'Include nightlife recommendations',
      'Talk about the area\'s European architecture',
      'Suggest vintage shopping spots',
      'Include cultural venues like Istanbul Modern'
    ],
    'Kadıköy': [
      'Highlight the famous Tuesday market',
      'Mention ferry connections from European side',
      'Include street food recommendations',
      'Talk about the alternative culture scene',
      'Suggest bars and meyhanes for evening entertainment'
    ]
  };

  return suggestions[district] || [
    'Include specific location details',
    'Add practical visitor information',
    'Mention local dining recommendations',
    'Share personal experiences and insights',
    'Include transportation tips'
  ];
};

/**
 * SEO optimization suggestions for content
 * @param {string} title - Blog post title
 * @param {string} content - Blog post content
 * @param {Array} tags - Current tags
 * @returns {Object} SEO suggestions
 */
export const getSEOSuggestions = (title, content, tags = []) => {
  const suggestions = [];
  const warnings = [];

  // Title SEO analysis
  if (!title.toLowerCase().includes('istanbul')) {
    suggestions.push('Consider including "Istanbul" in your title for better SEO');
  }

  if (title.length > 60) {
    warnings.push('Title is too long for search results (over 60 characters)');
  }

  // Content SEO analysis
  const metaDescription = content.substring(0, 160).replace(/<[^>]*>/g, '');
  if (metaDescription.length < 120) {
    suggestions.push('Consider adding more introductory content for better meta descriptions');
  }

  // Tags analysis
  if (tags.length < 3) {
    suggestions.push('Add more relevant tags to improve discoverability');
  }

  if (tags.length > 10) {
    warnings.push('Too many tags can dilute SEO impact (consider limiting to 8-10)');
  }

  return {
    suggestions,
    warnings,
    metaDescription,
    titleLength: title.length,
    estimatedReadTime: Math.ceil(content.split(' ').length / 200) // Average reading speed
  };
};

export default {
  generateAutoTags,
  analyzeContentQuality,
  getDistrictContentSuggestions,
  getSEOSuggestions
};
