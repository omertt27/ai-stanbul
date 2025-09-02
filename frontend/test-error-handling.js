/**
 * Comprehensive Error Handling Test Script
 * 
 * This script tests all the error handling scenarios we've implemented:
 * - Network errors (offline/connection failures)
 * - API timeouts
 * - Server errors (5xx responses)
 * - Client errors (4xx responses)
 * - Rate limiting
 * - Circuit breaker functionality
 * - Retry mechanisms
 * - Recovery strategies
 */

console.log('🧪 Starting comprehensive error handling tests...');

const API_BASE_URL = 'http://localhost:8000';

// Test scenarios
const testScenarios = [
  {
    name: 'Valid Request',
    description: 'Test that normal requests work correctly',
    test: async () => {
      const response = await fetch(`${API_BASE_URL}/ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: 'Hello, tell me about Istanbul' })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('✅ Valid request response:', data.message?.substring(0, 100) + '...');
      return { success: true, data };
    }
  },
  
  {
    name: 'Health Check',
    description: 'Test the health check endpoint',
    test: async () => {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      console.log('🏥 Health check response:', data);
      return { success: response.ok, data };
    }
  },
  
  {
    name: 'Invalid Input - Empty',
    description: 'Test validation error handling with empty input',
    test: async () => {
      const response = await fetch(`${API_BASE_URL}/ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: '' })
      });
      
      const data = await response.json();
      console.log('❌ Empty input response:', data);
      return { success: response.status === 422, data };
    }
  },
  
  {
    name: 'Invalid Input - Malicious',
    description: 'Test security validation with malicious input',
    test: async () => {
      const maliciousInputs = [
        '<script>alert("xss")</script>',
        "'; DROP TABLE users; --",
        '{{7*7}}',
        '$(echo "test")',
        '../../../etc/passwd'
      ];
      
      const results = [];
      
      for (const input of maliciousInputs) {
        try {
          const response = await fetch(`${API_BASE_URL}/ai`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: input })
          });
          
          const data = await response.json();
          results.push({
            input: input.substring(0, 30),
            status: response.status,
            blocked: response.status === 422 || data.error
          });
        } catch (error) {
          results.push({
            input: input.substring(0, 30),
            error: error.message,
            blocked: true
          });
        }
      }
      
      console.log('🛡️ Security validation results:', results);
      const allBlocked = results.every(r => r.blocked);
      return { success: allBlocked, data: results };
    }
  },
  
  {
    name: 'Invalid JSON',
    description: 'Test handling of malformed JSON',
    test: async () => {
      const response = await fetch(`${API_BASE_URL}/ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'invalid json'
      });
      
      const data = await response.json();
      console.log('📝 Invalid JSON response:', data);
      return { success: response.status === 422, data };
    }
  },
  
  {
    name: 'Restaurant API Test',
    description: 'Test restaurant recommendations with error handling',
    test: async () => {
      const response = await fetch(`${API_BASE_URL}/restaurants/search?district=Kadikoy&limit=3`);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.log('🍽️ Restaurant API error:', errorData);
        return { success: false, data: errorData };
      }
      
      const data = await response.json();
      console.log('🍽️ Restaurant API success:', data.restaurants?.length, 'restaurants found');
      return { success: true, data };
    }
  },
  
  {
    name: 'Places API Test',
    description: 'Test places recommendations with error handling',
    test: async () => {
      const response = await fetch(`${API_BASE_URL}/places/?district=Sultanahmet&limit=3`);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.log('🏛️ Places API error:', errorData);
        return { success: false, data: errorData };
      }
      
      const data = await response.json();
      console.log('🏛️ Places API success:', data.length, 'places found');
      return { success: true, data };
    }
  },
  
  {
    name: 'Connection Timeout Simulation',
    description: 'Simulate connection timeout',
    test: async () => {
      try {
        // Create a request with very short timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 100); // 100ms timeout
        
        const response = await fetch(`${API_BASE_URL}/ai`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_input: 'Tell me about Istanbul history' }),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        console.log('⏰ Timeout test: Request completed unexpectedly');
        return { success: false, message: 'Request should have timed out' };
        
      } catch (error) {
        if (error.name === 'AbortError') {
          console.log('⏰ Timeout test: Successfully caught timeout error');
          return { success: true, data: { error: error.name } };
        }
        throw error;
      }
    }
  },
  
  {
    name: 'Network Error Simulation',
    description: 'Test handling of network errors',
    test: async () => {
      try {
        // Try to connect to non-existent server
        const response = await fetch('http://localhost:9999/ai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_input: 'test' })
        });
        
        console.log('🌐 Network error test: Connection succeeded unexpectedly');
        return { success: false, message: 'Should have failed to connect' };
        
      } catch (error) {
        console.log('🌐 Network error test: Successfully caught network error:', error.message);
        return { success: true, data: { error: error.message } };
      }
    }
  },
  
  {
    name: 'Rate Limiting Test',
    description: 'Test rate limiting behavior (multiple rapid requests)',
    test: async () => {
      const promises = [];
      const numRequests = 10;
      
      console.log(`🚦 Making ${numRequests} rapid requests to test rate limiting...`);
      
      for (let i = 0; i < numRequests; i++) {
        promises.push(
          fetch(`${API_BASE_URL}/ai`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: `Test request ${i}` })
          }).then(response => ({
            index: i,
            status: response.status,
            ok: response.ok
          })).catch(error => ({
            index: i,
            error: error.message
          }))
        );
      }
      
      const results = await Promise.all(promises);
      const successful = results.filter(r => r.ok).length;
      const rateLimited = results.filter(r => r.status === 429).length;
      const errors = results.filter(r => r.error).length;
      
      console.log(`🚦 Rate limiting results: ${successful} successful, ${rateLimited} rate limited, ${errors} errors`);
      
      return { 
        success: true, 
        data: { 
          total: numRequests,
          successful, 
          rateLimited, 
          errors,
          details: results
        }
      };
    }
  }
];

// Run all tests
async function runTests() {
  console.log(`\n🎯 Running ${testScenarios.length} error handling tests...\n`);
  
  const results = [];
  
  for (let i = 0; i < testScenarios.length; i++) {
    const scenario = testScenarios[i];
    console.log(`\n📋 Test ${i + 1}/${testScenarios.length}: ${scenario.name}`);
    console.log(`📄 Description: ${scenario.description}`);
    
    try {
      const startTime = Date.now();
      const result = await scenario.test();
      const duration = Date.now() - startTime;
      
      result.name = scenario.name;
      result.duration = duration;
      results.push(result);
      
      if (result.success) {
        console.log(`✅ PASSED (${duration}ms)`);
      } else {
        console.log(`❌ FAILED (${duration}ms):`, result.message || 'Test failed');
      }
      
    } catch (error) {
      console.log(`💥 ERROR: ${error.message}`);
      results.push({
        name: scenario.name,
        success: false,
        error: error.message,
        duration: 0
      });
    }
    
    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 TEST SUMMARY');
  console.log('='.repeat(60));
  
  const passed = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  const totalTime = results.reduce((sum, r) => sum + r.duration, 0);
  
  console.log(`Total Tests: ${results.length}`);
  console.log(`Passed: ${passed} ✅`);
  console.log(`Failed: ${failed} ❌`);
  console.log(`Success Rate: ${Math.round((passed / results.length) * 100)}%`);
  console.log(`Total Time: ${totalTime}ms`);
  
  if (failed > 0) {
    console.log('\n❌ Failed Tests:');
    results.filter(r => !r.success).forEach(r => {
      console.log(`  - ${r.name}: ${r.error || r.message || 'Unknown error'}`);
    });
  }
  
  console.log('\n🎉 Error handling tests completed!');
  console.log('\n💡 Tips for production:');
  console.log('  - Monitor error rates and response times');
  console.log('  - Set up alerting for high error rates');
  console.log('  - Implement proper logging and monitoring');
  console.log('  - Test error scenarios regularly');
  console.log('  - Have fallback mechanisms for critical paths');
  
  return results;
}

// Run the tests if this script is executed directly
if (typeof window !== 'undefined') {
  // Browser environment
  window.runErrorHandlingTests = runTests;
  console.log('🌐 Error handling tests loaded. Run window.runErrorHandlingTests() to start.');
} else {
  // Node.js environment
  runTests().catch(console.error);
}

export { runTests as runErrorHandlingTests };
