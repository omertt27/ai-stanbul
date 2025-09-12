#!/usr/bin/env node

// Simple test script to diagnose chatbot connection issues
const http = require('http');
const https = require('https');

const testBackend = () => {
  console.log('🔍 Testing backend connection...\n');
  
  // Test health endpoint
  const healthOptions = {
    hostname: 'localhost',
    port: 8001,
    path: '/health',
    method: 'GET',
    timeout: 5000
  };

  const req = http.request(healthOptions, (res) => {
    let data = '';
    res.on('data', (chunk) => data += chunk);
    res.on('end', () => {
      console.log('✅ Health endpoint response:', res.statusCode, data);
      testAiEndpoint();
    });
  });

  req.on('error', (error) => {
    console.log('❌ Health endpoint error:', error.message);
    console.log('💡 Backend might not be running on port 8001');
    console.log('💡 Try starting it with: cd backend && python start.py');
  });

  req.on('timeout', () => {
    console.log('⏰ Health endpoint timeout - backend may be slow to respond');
    req.destroy();
  });

  req.end();
};

const testAiEndpoint = () => {
  console.log('\n🤖 Testing AI endpoint...\n');
  
  const postData = JSON.stringify({
    user_input: 'Hello, this is a test'
  });

  const options = {
    hostname: 'localhost',
    port: 8001,
    path: '/ai',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(postData)
    },
    timeout: 10000
  };

  const req = http.request(options, (res) => {
    let data = '';
    res.on('data', (chunk) => data += chunk);
    res.on('end', () => {
      console.log('✅ AI endpoint response:', res.statusCode);
      try {
        const parsed = JSON.parse(data);
        console.log('📝 Response data:', parsed);
        console.log('✅ Backend is working correctly!');
      } catch (e) {
        console.log('📝 Raw response:', data);
      }
    });
  });

  req.on('error', (error) => {
    console.log('❌ AI endpoint error:', error.message);
  });

  req.on('timeout', () => {
    console.log('⏰ AI endpoint timeout');
    req.destroy();
  });

  req.write(postData);
  req.end();
};

// Run the test
testBackend();
