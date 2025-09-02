#!/usr/bin/env node

// Demonstration: GPT-Powered Smart Routing (Simplified Approach)

// Instead of complex detection logic, use a simple approach:
function shouldUseRestaurantAPI(userInput) {
    // Only use a few very obvious restaurant keywords
    const explicitRestaurantRequests = [
        'restaurants in', 'where to eat in', 'restaurant recommendations for',
        'good restaurants in', 'best restaurants in', 'restaurants near'
    ];
    
    const input = userInput.toLowerCase();
    return explicitRestaurantRequests.some(keyword => input.includes(keyword));
}

// Let GPT handle everything else with enhanced system prompt
function getEnhancedSystemPrompt() {
    return `You are KAM, Istanbul's AI travel guide. You can:

AUTOMATIC SMART ROUTING:
- Detect when users want restaurant recommendations and offer to get live data
- Handle personal introductions warmly and pivot to travel planning
- Provide contextual Istanbul advice based on user's background/travel plans

INTELLIGENT RESPONSES:
For "I'm Ukrainian" → "Nice to meet you! Are you planning to visit Istanbul? I'd love to help!"
For "I'm Turkish, coming tomorrow" → "Welcome back home! What areas of Istanbul are you planning to explore?"
For "restaurants in Beyoglu" → "I can get you live restaurant recommendations! Let me fetch the best options in Beyoglu for you."

CAPABILITIES:
- Live restaurant data via Google Maps API (say "Let me get live restaurant data for you!")
- Istanbul attractions, districts, culture, transport advice
- Personalized recommendations based on user background
- Travel planning and practical tips

Always be conversational, helpful, and Istanbul-focused. If someone asks for specific restaurant recommendations with location, offer to get live data.`;
}

async function testSmartRouting() {
    console.log('🧠 GPT-Powered Smart Routing Test\n');
    
    const testCases = [
        {
            input: 'im ukrainian',
            expected: 'CHAT',
            description: 'Personal introduction'
        },
        {
            input: 'im turkish, coming tomorrow to istanbul',
            expected: 'CHAT',
            description: 'Travel planning conversation'
        },
        {
            input: 'restaurants in Beyoglu',
            expected: 'API',
            description: 'Explicit restaurant request with location'
        },
        {
            input: 'where to eat in Taksim',
            expected: 'API',
            description: 'Explicit dining request with location'
        },
        {
            input: 'tell me about good food',
            expected: 'CHAT',
            description: 'General food question - let GPT handle and offer API'
        },
        {
            input: 'what should i visit tomorrow',
            expected: 'CHAT',
            description: 'General travel question'
        }
    ];
    
    console.log('🎯 Simplified Logic Results:');
    testCases.forEach(testCase => {
        const wouldUseAPI = shouldUseRestaurantAPI(testCase.input);
        const result = wouldUseAPI ? 'API' : 'CHAT';
        const status = result === testCase.expected ? '✅' : '⚠️';
        
        console.log(`${status} "${testCase.input}"`);
        console.log(`   Expected: ${testCase.expected}, Got: ${result}`);
        console.log(`   Strategy: ${testCase.description}`);
        console.log('');
    });
    
    console.log('💡 Key Insight:');
    console.log('- Only intercept VERY explicit restaurant+location requests');
    console.log('- Let GPT handle everything else intelligently');
    console.log('- GPT can offer to fetch live data when appropriate');
    console.log('- Much simpler code, more flexible responses');
}

testSmartRouting().catch(console.error);
