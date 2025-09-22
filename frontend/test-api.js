// Test API configuration
const BASE_URL = process.env.VITE_API_URL || 'http://localhost:8000';
const cleanBaseUrl = BASE_URL.replace(/\/ai\/?$/, '');
const BLOG_API_URL = `${cleanBaseUrl}/blog`;

console.log('Environment VITE_API_URL:', process.env.VITE_API_URL);
console.log('BASE_URL:', BASE_URL);
console.log('cleanBaseUrl:', cleanBaseUrl);
console.log('BLOG_API_URL:', BLOG_API_URL);
console.log('Full blog post URL for ID 1:', `${BLOG_API_URL}/1`);

// Test fetch
fetch(`${BLOG_API_URL}/1`)
  .then(response => {
    console.log('Response status:', response.status);
    return response.json();
  })
  .then(data => {
    console.log('Response data:', data);
  })
  .catch(error => {
    console.error('Fetch error:', error);
  });
