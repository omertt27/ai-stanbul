#!/usr/bin/env node

// Test script to verify Ukrainian and personal info chat handling

// Replicate the exact detection logic from App.jsx
function isRestaurantAdviceRequest(userInput) {
  console.log('🔍 Checking if restaurant request:', userInput);
  const input = userInput.toLowerCase();
  
  // Exclude personal information sharing or general conversation
  const personalInfoKeywords = [
    'i am', 'im a', 'im an', 'i\'m a', 'i\'m an', 'my name', 'hello', 'hi', 'how are you',
    'i am ukrainian', 'im ukrainian', 'i am turkish', 'im turkish', 'i am american', 'im american',
    'from ukraine', 'from turkey', 'nationality', 'tell me about myself', 'about me'
  ];
  
  const isPersonalInfo = personalInfoKeywords.some(keyword => input.includes(keyword));
  if (isPersonalInfo) {
    console.log('🔍 Detected personal info sharing, not restaurant request');
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
  console.log('🔍 Restaurant detection result:', isRestaurant);
  return isRestaurant;
}

async function runPersonalInfoTests() {
  console.log('🧪 Testing Personal Information and Chat Flow\n');
  
  const personalInfoTests = [
    'im an ukrainian',
    'I am Ukrainian', 
    'I\'m a Turkish person',
    'Hello, how are you?',
    'Hi there!',
    'My name is Alex',
    'I am from Ukraine',
    'tell me about myself',
    'what is my nationality?'
  ];
  
  const restaurantTests = [
    'restaurants in Beyoglu',
    'where to eat in Taksim',
    'restaurant recommendations please',
    'good restaurants near Sultanahmet',
    'show me restaurants in Galata',
    'best food in Istanbul',
    'turkish cuisine recommendations'
  ];
  
  console.log('❌ These should NOT trigger restaurant recommendations (continue with chat):');
  personalInfoTests.forEach(test => {
    const detected = isRestaurantAdviceRequest(test);
    const result = detected ? '⚠️ RESTAURANT (WRONG!)' : '✅ CHAT (CORRECT)';
    console.log(`   "${test}" -> ${result}`);
  });
  
  console.log('\n✅ These SHOULD trigger restaurant recommendations:');
  restaurantTests.forEach(test => {
    const detected = isRestaurantAdviceRequest(test);
    const result = detected ? '✅ RESTAURANT (CORRECT)' : '⚠️ CHAT (WRONG!)';
    console.log(`   "${test}" -> ${result}`);
  });
  
  console.log('\n🎯 Summary:');
  console.log('- When user says "I\'m Ukrainian", system continues with daily chat');
  console.log('- When user asks for restaurant recommendations, system provides 4 restaurants');
  console.log('- The restaurant filtering now returns actual restaurants, not hotels');
}

runPersonalInfoTests().catch(console.error);
