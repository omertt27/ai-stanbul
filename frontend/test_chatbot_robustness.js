/**
 * Automated Chatbot Robustness Test Suite
 * Tests the chatbot API with problematic inputs to ensure robust error handling
 */

const API_BASE_URL = 'http://localhost:8001';

// Problematic test inputs
const testInputs = [
    // Empty and whitespace inputs
    { category: 'Empty/Whitespace', input: '', expected: 'should handle gracefully' },
    { category: 'Empty/Whitespace', input: '   ', expected: 'should handle gracefully' },
    { category: 'Empty/Whitespace', input: '\n\t\r', expected: 'should handle gracefully' },
    
    // Very long inputs
    { category: 'Long Input', input: 'A'.repeat(10000), expected: 'should handle or truncate' },
    { category: 'Long Input', input: 'Tell me about Istanbul restaurants ' + 'very '.repeat(1000), expected: 'should handle or truncate' },
    
    // Special characters and encoding
    { category: 'Special Characters', input: '!@#$%^&*()_+-=[]{}|;:,.<>?', expected: 'should handle gracefully' },
    { category: 'Special Characters', input: '../../etc/passwd', expected: 'should not cause security issues' },
    { category: 'Special Characters', input: '<script>alert("xss")</script>', expected: 'should be sanitized' },
    { category: 'Special Characters', input: 'DROP TABLE users;', expected: 'should not affect database' },
    
    // Unicode and international characters
    { category: 'Unicode', input: '🏛️🍽️🎭 İstanbul müzeleri nerede? 🇹🇷', expected: 'should handle Unicode' },
    { category: 'Unicode', input: '北京餐厅推荐', expected: 'should handle Chinese characters' },
    { category: 'Unicode', input: 'مطاعم في اسطنبول', expected: 'should handle Arabic' },
    { category: 'Unicode', input: 'рестораны в Стамбуле', expected: 'should handle Cyrillic' },
    
    // Rapid-fire and concurrent requests
    { category: 'Rapid Fire', input: 'test1', expected: 'should handle concurrent requests' },
    { category: 'Rapid Fire', input: 'test2', expected: 'should handle concurrent requests' },
    { category: 'Rapid Fire', input: 'test3', expected: 'should handle concurrent requests' },
    
    // Mixed content types
    { category: 'Mixed Content', input: 'What about restaurants 123 !@# مطعم 🍽️', expected: 'should handle mixed content' },
    
    // Nonsensical inputs
    { category: 'Nonsensical', input: 'asdfghjkl qwertyuiop zxcvbnm', expected: 'should provide helpful response' },
    { category: 'Nonsensical', input: '1234567890 !!!!! ????? %%%%', expected: 'should provide helpful response' },
    
    // Boundary cases
    { category: 'Boundary', input: null, expected: 'should handle null gracefully' },
    { category: 'Boundary', input: undefined, expected: 'should handle undefined gracefully' },
    
    // JSON injection attempts
    { category: 'Injection', input: '{"message": "hacked"}', expected: 'should not parse as JSON' },
    { category: 'Injection', input: "'; DROP TABLE messages; --", expected: 'should not affect database' },
];

// Test results storage
const testResults = [];

// Test function
async function testChatbotInput(testCase) {
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_BASE_URL}/ai`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: testCase.input,
                session_id: `test_${Date.now()}_${Math.random()}`
            })
        });

        const responseTime = Date.now() - startTime;
        const responseData = await response.json();

        return {
            ...testCase,
            success: true,
            status: response.status,
            responseTime,
            response: responseData,
            error: null
        };
    } catch (error) {
        const responseTime = Date.now() - startTime;
        return {
            ...testCase,
            success: false,
            status: null,
            responseTime,
            response: null,
            error: error.message
        };
    }
}

// Run all tests
async function runAllTests() {
    console.log('🚀 Starting Chatbot Robustness Test Suite...');
    console.log(`Testing ${testInputs.length} problematic inputs`);
    console.log('═'.repeat(60));

    for (let i = 0; i < testInputs.length; i++) {
        const testCase = testInputs[i];
        console.log(`\n📝 Test ${i + 1}/${testInputs.length}: ${testCase.category}`);
        console.log(`Input: "${testCase.input}"`);
        
        const result = await testChatbotInput(testCase);
        testResults.push(result);
        
        if (result.success) {
            console.log(`✅ Success (${result.responseTime}ms) - Status: ${result.status}`);
            console.log(`Response: ${JSON.stringify(result.response).substring(0, 100)}...`);
        } else {
            console.log(`❌ Failed (${result.responseTime}ms) - Error: ${result.error}`);
        }
        
        // Small delay to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Test rapid-fire requests
    console.log('\n\n🔥 Testing Rapid-Fire Concurrent Requests...');
    const rapidFirePromises = [];
    for (let i = 0; i < 5; i++) {
        rapidFirePromises.push(testChatbotInput({
            category: 'Concurrent',
            input: `Rapid fire test ${i + 1}`,
            expected: 'should handle concurrent requests'
        }));
    }
    
    const rapidFireResults = await Promise.all(rapidFirePromises);
    const successfulConcurrent = rapidFireResults.filter(r => r.success).length;
    console.log(`✅ ${successfulConcurrent}/5 concurrent requests succeeded`);

    // Summary
    console.log('\n\n📊 TEST SUMMARY');
    console.log('═'.repeat(60));
    
    const successful = testResults.filter(r => r.success).length;
    const failed = testResults.filter(r => !r.success).length;
    const avgResponseTime = testResults.reduce((sum, r) => sum + r.responseTime, 0) / testResults.length;
    
    console.log(`Total Tests: ${testResults.length}`);
    console.log(`✅ Successful: ${successful}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📈 Success Rate: ${(successful / testResults.length * 100).toFixed(1)}%`);
    console.log(`⏱️ Average Response Time: ${avgResponseTime.toFixed(0)}ms`);
    
    // Group results by category
    const byCategory = {};
    testResults.forEach(result => {
        if (!byCategory[result.category]) {
            byCategory[result.category] = { total: 0, successful: 0 };
        }
        byCategory[result.category].total++;
        if (result.success) byCategory[result.category].successful++;
    });
    
    console.log('\n📋 Results by Category:');
    Object.entries(byCategory).forEach(([category, stats]) => {
        const rate = (stats.successful / stats.total * 100).toFixed(1);
        console.log(`  ${category}: ${stats.successful}/${stats.total} (${rate}%)`);
    });
    
    // Show failures
    const failures = testResults.filter(r => !r.success);
    if (failures.length > 0) {
        console.log('\n⚠️ Failed Tests:');
        failures.forEach((failure, i) => {
            console.log(`  ${i + 1}. ${failure.category}: "${failure.input}" - ${failure.error}`);
        });
    }
    
    console.log('\n🎯 Test Complete!');
    return testResults;
}

// Export for use in browser or Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { runAllTests, testInputs };
} else {
    // Browser environment
    window.chatbotRobustnessTest = { runAllTests, testInputs };
}
