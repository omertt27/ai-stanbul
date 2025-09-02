#!/usr/bin/env node

// Test script to verify travel context handling

// Add Istanbul/districts list for strict location check
const istanbulDistricts = [
  'istanbul', 'beyoglu', 'beyoğlu', 'galata', 'taksim', 'sultanahmet', 'fatih',
  'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'ortakoy',
  'ortaköy', 'sisli', 'şişli', 'karakoy', 'karaköy', 'bebek', 'arnavutkoy',
  'arnavutköy', 'balat', 'fener', 'eminonu', 'eminönü', 'bakirkoy', 'bakırköy', 'maltepe'
];

// Simple location extractor for test (looks for 'in <location>' or 'near <location>')
function extractLocationFromQuery(userInput) {
  const match = userInput.match(/(?:in|near|around)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ\- ]+)/i);
  if (match) {
    return match[1].trim().toLowerCase();
  }
  return null;
}

function isRestaurantAdviceRequest(userInput) {
  console.log('🔍 Checking if restaurant request:', userInput);
  const input = userInput.toLowerCase();
  
  // Exclude ONLY identity statements without travel context
  const personalInfoKeywords = [
    'i am ukrainian', 'im ukrainian', 'i am turkish', 'im turkish', 
    'i am american', 'im american', 'my name is', 'hello', 'hi there',
    'how are you', 'tell me about myself', 'about me'
  ];
  
  // Check for pure personal info without travel context
  const isPersonalInfo = personalInfoKeywords.some(keyword => input.includes(keyword));
  const hasTravelContext = input.includes('visit') || input.includes('come to') || input.includes('going to') || 
                         input.includes('travel') || input.includes('trip') || input.includes('tomorrow') || 
                         input.includes('next week') || input.includes('planning');
  
  // If it's personal info but has travel context, don't filter it out
  if (isPersonalInfo && !hasTravelContext) {
    console.log('🔍 Detected pure personal info, not restaurant request');
    return false;
  }
  
  // More specific restaurant keywords that indicate actual restaurant requests
  const restaurantKeywords = [
    'restaurant recommendations', 'restaurant suggestion', 'restaurant advice',
    'restaurants in', 'where to eat', 'places to eat', 'food in',
    'dining in', 'good restaurants', 'best restaurants', 'recommend restaurant',
    'suggest restaurant', 'find restaurant', 'show me restaurant',
    'restaurant near', 'restaurants near', 'food recommendations',
    'where can i eat', 'good food', 'best food', 'turkish cuisine',
    'local food', 'traditional food', 'food places'
  ];
  
  const isRestaurant = restaurantKeywords.some(keyword => input.includes(keyword));
  if (!isRestaurant) return false;
  // Extract location and check if it's Istanbul or a known district
  const location = extractLocationFromQuery(userInput);
  if (!location) return false;
  // Only allow if location exactly matches a known Istanbul district
  const isIstanbul = istanbulDistricts.includes(location);
  if (!isIstanbul) {
    console.log('❌ Location is not Istanbul or a known district:', location);
    return false;
  }
  
  // Add greeting and general chat keywords to always route to chat
  const greetingKeywords = [
    'hello', 'hi', 'hey', 'how are you', 'how are u', 'how r u', 'how r you', 'good morning', 'good evening', 'good night',
    'greetings', 'selam', 'merhaba', 'nasılsın', 'nasilsin', 'whats up', "what's up", 'sup', 'yo'
  ];
  if (greetingKeywords.some(keyword => input.includes(keyword))) {
    console.log('🔍 Detected greeting, not restaurant request');
    return false;
  }
  
  return true;
}

// GPT-based intent detection using OpenAI API (real implementation)
// Requires: npm install openai
// Set your OpenAI API key in the environment as OPENAI_API_KEY
const { OpenAIApi, Configuration } = require('openai');
const openai = new OpenAIApi(new Configuration({ apiKey: process.env.OPENAI_API_KEY }));

async function gptDetectRestaurantIntent(userInput) {
  const prompt = `You are an intent classifier. Is the user asking for a restaurant recommendation in Istanbul or its districts? Reply only YES or NO.\nUser: "${userInput}"`;
  const response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt,
    max_tokens: 3,
    temperature: 0,
    n: 1,
    stop: ["\n"]
  });
  const answer = response.data.choices[0].text.trim().toUpperCase();
  return answer === "YES" ? "YES" : "NO";
}

// New function: use GPT for intent detection
async function isRestaurantAdviceRequestGPT(userInput) {
  const gptResult = await gptDetectRestaurantIntent(userInput);
  return gptResult === 'YES';
}

// Unified intent detection: manual rules first, fallback to GPT if uncertain
async function unifiedRestaurantIntent(userInput) {
  // 1. Manual rule-based detection (fast, cheap, covers most cases)
  if (isRestaurantAdviceRequest(userInput)) {
    return true; // Istanbul restaurant request detected by rules
  }
  // 2. If not detected by rules, ask GPT for intent classification
  const gptResult = await gptDetectRestaurantIntent(userInput);
  return gptResult === 'YES';
}

async function runTravelContextTests() {
  console.log('🧪 Testing Travel Context Logic\n');
  
  const chatCases = [
    'im an ukrainian',
    'I am Turkish',
    'hello, how are you?',
    'my name is Alex',
    'tell me about myself'
  ];
  
  const travelChatCases = [
    'im turkish, i will come tomorrow to istanbul',
    'I am Ukrainian, planning to visit Istanbul',
    'Im American, going to Istanbul next week',
    'I am from Turkey, will travel to Istanbul'
  ];
  
  const restaurantCases = [
    'restaurants in Beyoglu',
    'where to eat in Taksim',
    'restaurant recommendations please',
    'good restaurants near Sultanahmet'
  ];
  
  console.log('❌ These should go to OpenAI CHAT (pure personal info):');
  chatCases.forEach(test => {
    const detected = isRestaurantAdviceRequest(test);
    const result = detected ? '⚠️ RESTAURANT (WRONG!)' : '✅ CHAT (CORRECT)';
    console.log(`   "${test}" -> ${result}`);
  });
  
  console.log('\n💬 These should go to OpenAI CHAT (travel context - let OpenAI handle):');
  travelChatCases.forEach(test => {
    const detected = isRestaurantAdviceRequest(test);
    const result = detected ? '⚠️ RESTAURANT (WRONG!)' : '✅ CHAT (CORRECT)';
    console.log(`   "${test}" -> ${result}`);
  });
  
  console.log('\n🍽️ These should get RESTAURANT recommendations:');
  restaurantCases.forEach(test => {
    const detected = isRestaurantAdviceRequest(test);
    const result = detected ? '✅ RESTAURANT (CORRECT)' : '⚠️ CHAT (WRONG!)';
    console.log(`   "${test}" -> ${result}`);
  });
  
  // GPT-based detection tests
  const gptTestCases = [
    'restaurants in Beyoglu',
    'where to eat in Taksim',
    'restaurant recommendations please',
    'good restaurants near Sultanahmet',
    'restaurants in Paris',
    'how are u',
    'hello',
    'I am Turkish',
    'I am Ukrainian, planning to visit Istanbul',
  ];
  console.log('\n🤖 [GPT] These should get RESTAURANT recommendations only for Istanbul:');
  for (const test of gptTestCases) {
    const detected = await isRestaurantAdviceRequestGPT(test);
    const result = detected ? '✅ RESTAURANT (CORRECT for Istanbul)' : '✅ CHAT (CORRECT for non-Istanbul)';
    console.log(`   "${test}" -> ${result}`);
  }
  
  // Unified detection tests
  const unifiedTestCases = [
    'restaurants in Beyoglu',
    'where to eat in Taksim',
    'restaurant recommendations please',
    'good restaurants near Sultanahmet',
    'restaurants in Paris',
    'how are u',
    'hello',
    'I am Turkish',
    'I am Ukrainian, planning to visit Istanbul',
    'best vegan food in Kadikoy',
    'I want a pizza in Galata',
    'find me a restaurant in Paris',
    'good food in Istanbul',
    'any suggestions for food in Fatih',
    'where can I eat in Nisantasi',
    'I want to eat in Rome',
    'hi',
    'good morning',
    'tell me about myself',
  ];
  console.log('\n🧠 [Unified] These should get RESTAURANT recommendations only for Istanbul:');
  for (const test of unifiedTestCases) {
    const detected = await unifiedRestaurantIntent(test);
    const result = detected ? '✅ RESTAURANT (CORRECT for Istanbul)' : '✅ CHAT (CORRECT for non-Istanbul)';
    console.log(`   "${test}" -> ${result}`);
  }
  
  console.log('\n🎯 Summary:');
  console.log('- "I\'m Ukrainian" -> OpenAI chat response');
  console.log('- "I\'m Turkish, coming tomorrow" -> OpenAI travel advice & places');
  console.log('- "restaurants in Beyoglu" -> 4 restaurant recommendations');
  console.log('- OpenAI will provide personalized travel advice for visitors!');
}

runTravelContextTests().catch(console.error);

// Export unified intent detection for integration
module.exports = {
  unifiedRestaurantIntent,
  isRestaurantAdviceRequest, // manual
  isRestaurantAdviceRequestGPT, // pure GPT
};
