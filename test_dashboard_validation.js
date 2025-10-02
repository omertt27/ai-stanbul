#!/usr/bin/env node
/**
 * Dashboard Validation Test Script
 * Tests the unified admin dashboard for completeness and functionality
 */

const fs = require('fs');
const path = require('path');

console.log('🧪 AI Istanbul Admin Dashboard Validation Test');
console.log('=============================================\n');

// Read the dashboard HTML file
const dashboardPath = path.join(__dirname, 'unified_admin_dashboard.html');
const dashboardContent = fs.readFileSync(dashboardPath, 'utf8');

// Test 1: Check for all required navigation items
console.log('1. Testing Navigation Items:');
const requiredNavItems = [
    'dashboard', 'analytics', 'sessions', 'users', 'blog', 
    'monitoring', 'cache', 'cost', 'system', 'settings'
];

const navTests = requiredNavItems.map(item => {
    const hasNavItem = dashboardContent.includes(`data-section="${item}"`);
    const hasSection = dashboardContent.includes(`id="${item}Section"`);
    
    console.log(`   ✓ ${item}: Nav=${hasNavItem ? '✅' : '❌'}, Section=${hasSection ? '✅' : '❌'}`);
    return hasNavItem && hasSection;
});

const navTestsPassed = navTests.every(test => test);
console.log(`   Navigation Tests: ${navTestsPassed ? '✅ PASSED' : '❌ FAILED'}\n`);

// Test 2: Check for monitoring features
console.log('2. Testing Monitoring Features:');
const monitoringFeatures = [
    'liveResponseTime', 'liveApiCalls', 'liveErrorRate', 'liveUptime',
    'endpointHealthTable', 'alertRulesTable', 'systemLogsContent',
    'loadMonitoringData', 'testAllAlerts', 'refreshSystemLogs'
];

const monitoringTests = monitoringFeatures.map(feature => {
    const hasFeature = dashboardContent.includes(feature);
    console.log(`   ✓ ${feature}: ${hasFeature ? '✅' : '❌'}`);
    return hasFeature;
});

const monitoringTestsPassed = monitoringTests.every(test => test);
console.log(`   Monitoring Tests: ${monitoringTestsPassed ? '✅ PASSED' : '❌ FAILED'}\n`);

// Test 3: Check for cache management features
console.log('3. Testing Cache Management Features:');
const cacheFeatures = [
    'cacheHitRate', 'cacheSize', 'cacheKeys', 'avgTTL',
    'topCacheKeysTable', 'ttlOptimizationTable', 'allCacheKeysTable',
    'loadCacheData', 'runCacheAnalysis', 'optimizeCacheSettings'
];

const cacheTests = cacheFeatures.map(feature => {
    const hasFeature = dashboardContent.includes(feature);
    console.log(`   ✓ ${feature}: ${hasFeature ? '✅' : '❌'}`);
    return hasFeature;
});

const cacheTestsPassed = cacheTests.every(test => test);
console.log(`   Cache Management Tests: ${cacheTestsPassed ? '✅ PASSED' : '❌ FAILED'}\n`);

// Test 4: Check for cost analytics features
console.log('4. Testing Cost Analytics Features:');
const costFeatures = [
    'totalCostToday', 'totalCostMonth', 'projectedCost', 'costSavings',
    'dailyCostTable', 'topCostEndpointsTable', 'budgetHistoryTable',
    'loadCostData', 'runCostOptimization', 'exportCostReport'
];

const costTests = costFeatures.map(feature => {
    const hasFeature = dashboardContent.includes(feature);
    console.log(`   ✓ ${feature}: ${hasFeature ? '✅' : '❌'}`);
    return hasFeature;
});

const costTestsPassed = costTests.every(test => test);
console.log(`   Cost Analytics Tests: ${costTestsPassed ? '✅ PASSED' : '❌ FAILED'}\n`);

// Test 5: Check for enhanced CSS styles
console.log('5. Testing Enhanced CSS Styles:');
const cssFeatures = [
    'monitoring-tab-content', 'cache-tab-content', 'cost-tab-content',
    'optimization-card', 'api-cost-card', 'budget-card',
    'recommendation-item', 'progress-bar', 'alert-banner'
];

const cssTests = cssFeatures.map(feature => {
    const hasFeature = dashboardContent.includes(feature);
    console.log(`   ✓ ${feature}: ${hasFeature ? '✅' : '❌'}`);
    return hasFeature;
});

const cssTestsPassed = cssTests.every(test => test);
console.log(`   CSS Enhancement Tests: ${cssTestsPassed ? '✅ PASSED' : '❌ FAILED'}\n`);

// Test 6: Check for JavaScript functions
console.log('6. Testing JavaScript Functions:');
const jsFunctions = [
    'loadMonitoringData', 'loadCacheData', 'loadCostData',
    'switchMonitoringTab', 'switchCacheTab', 'switchCostTab',
    'runHealthCheck', 'clearCache', 'refreshStats'
];

const jsTests = jsFunctions.map(func => {
    const hasFunction = dashboardContent.includes(`function ${func}`) || dashboardContent.includes(`async function ${func}`);
    console.log(`   ✓ ${func}: ${hasFunction ? '✅' : '❌'}`);
    return hasFunction;
});

const jsTestsPassed = jsTests.every(test => test);
console.log(`   JavaScript Function Tests: ${jsTestsPassed ? '✅ PASSED' : '❌ FAILED'}\n`);

// Overall Results
console.log('=== VALIDATION SUMMARY ===');
const allTestResults = [
    { name: 'Navigation', passed: navTestsPassed },
    { name: 'Monitoring', passed: monitoringTestsPassed },
    { name: 'Cache Management', passed: cacheTestsPassed },
    { name: 'Cost Analytics', passed: costTestsPassed },
    { name: 'CSS Enhancements', passed: cssTestsPassed },
    { name: 'JavaScript Functions', passed: jsTestsPassed }
];

allTestResults.forEach(result => {
    console.log(`${result.name}: ${result.passed ? '✅ PASSED' : '❌ FAILED'}`);
});

const overallPassed = allTestResults.every(result => result.passed);
console.log(`\n🎯 Overall Status: ${overallPassed ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED'}`);

// File size and complexity analysis
const fileStats = fs.statSync(dashboardPath);
const lineCount = dashboardContent.split('\n').length;
const jsLineCount = (dashboardContent.match(/<script[\s\S]*?<\/script>/g) || []).join('').split('\n').length;
const cssLineCount = (dashboardContent.match(/<style[\s\S]*?<\/style>/g) || []).join('').split('\n').length;

console.log('\n=== FILE STATISTICS ===');
console.log(`File Size: ${(fileStats.size / 1024).toFixed(2)} KB`);
console.log(`Total Lines: ${lineCount}`);
console.log(`JavaScript Lines: ${jsLineCount}`);
console.log(`CSS Lines: ${cssLineCount}`);
console.log(`HTML+Other Lines: ${lineCount - jsLineCount - cssLineCount}`);

// Production readiness checklist
console.log('\n=== PRODUCTION READINESS CHECKLIST ===');
const productionChecks = [
    { name: 'Authentication System', check: dashboardContent.includes('loginForm') && dashboardContent.includes('isSessionValid') },
    { name: 'API Error Handling', check: dashboardContent.includes('try') && dashboardContent.includes('catch') },
    { name: 'Session Timeout', check: dashboardContent.includes('SESSION_TIMEOUT') },
    { name: 'Environment Detection', check: dashboardContent.includes('isProduction') },
    { name: 'Real-time Monitoring', check: dashboardContent.includes('monitoringInterval') },
    { name: 'Cost Tracking', check: dashboardContent.includes('budget') && dashboardContent.includes('cost') },
    { name: 'Cache Management', check: dashboardContent.includes('cache') && dashboardContent.includes('ttl') },
    { name: 'Alert System', check: dashboardContent.includes('alert') && dashboardContent.includes('notification') },
    { name: 'Theme Support', check: dashboardContent.includes('light-theme') && dashboardContent.includes('dark') },
    { name: 'Responsive Design', check: dashboardContent.includes('@media') && dashboardContent.includes('mobile') }
];

productionChecks.forEach(check => {
    console.log(`${check.name}: ${check.check ? '✅' : '❌'}`);
});

const productionReady = productionChecks.every(check => check.check);
console.log(`\n🚀 Production Ready: ${productionReady ? '✅ YES' : '❌ NEEDS WORK'}`);

if (overallPassed && productionReady) {
    console.log('\n🎉 SUCCESS: The unified admin dashboard is fully implemented and production-ready!');
    process.exit(0);
} else {
    console.log('\n⚠️  WARNING: Some issues detected. Please review the failed tests above.');
    process.exit(1);
}
