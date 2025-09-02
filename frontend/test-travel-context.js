#!/usr/bin/env node

// Test script to verify travel context handling

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
  console.log('🔍 Restaurant detection result:', isRestaurant);
  return isRestaurant;
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
  
  console.log('\n🎯 Summary:');
  console.log('- "I\'m Ukrainian" -> OpenAI chat response');
  console.log('- "I\'m Turkish, coming tomorrow" -> OpenAI travel advice & places');
  console.log('- "restaurants in Beyoglu" -> 4 restaurant recommendations');
  console.log('- OpenAI will provide personalized travel advice for visitors!');
}

runTravelContextTests().catch(console.error);
