import { 
  fetchWithRetry, 
  getUserFriendlyMessage, 
  classifyError, 
  ErrorTypes,
  createCircuitBreaker
} from '../utils/errorHandler.js';

// API configuration
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const BLOG_API_URL = `${BASE_URL}/blog`;

// Circuit breaker for blog API
const blogCircuitBreaker = createCircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 30000
});

// Enhanced error handling wrapper
const handleBlogApiError = (error, response = null, context = '') => {
  const errorType = classifyError(error, response);
  const userMessage = getUserFriendlyMessage(error, response);
  
  console.error(`${context} error:`, {
    message: error.message,
    type: errorType,
    userMessage,
    status: response?.status,
    online: navigator.onLine
  });
  
  const enhancedError = new Error(userMessage);
  enhancedError.originalError = error;
  enhancedError.type = errorType;
  enhancedError.response = response;
  enhancedError.isRetryable = [ErrorTypes.NETWORK, ErrorTypes.TIMEOUT, ErrorTypes.SERVER].includes(errorType);
  
  return enhancedError;
};

// Blog API functions
export const fetchBlogPosts = async (params = {}) => {
  return blogCircuitBreaker.call(async () => {
    try {
      const searchParams = new URLSearchParams();
      Object.keys(params).forEach(key => {
        if (params[key] !== undefined && params[key] !== null) {
          searchParams.append(key, params[key].toString());
        }
      });
      
      const url = `${BLOG_API_URL}/posts${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
      console.log('🔍 Fetching blog posts from:', url);
      
      const response = await fetchWithRetry(url, {
        method: 'GET',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 15000
      }, {
        maxAttempts: 3,
        baseDelay: 1000
      });
      
      const data = await response.json();
      console.log('✅ Blog posts fetched:', data.length, 'posts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Posts');
    }
  });
};

export const fetchBlogPost = async (postId) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('📖 Fetching blog post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/posts/${postId}`, {
        method: 'GET',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 10000
      }, {
        maxAttempts: 2,
        baseDelay: 500
      });
      
      const data = await response.json();
      console.log('✅ Blog post fetched:', data.title);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Post');
    }
  });
};

export const createBlogPost = async (postData) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('✍️ Creating blog post:', postData.title);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/posts`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(postData),
        timeout: 20000
      }, {
        maxAttempts: 2,
        baseDelay: 1000
      });
      
      const data = await response.json();
      console.log('✅ Blog post created:', data.id);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Create Blog Post');
    }
  });
};

export const uploadBlogImage = async (file) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('📸 Uploading blog image:', file.name);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/upload-image`, {
        method: 'POST',
        body: formData,
        timeout: 30000 // Longer timeout for file uploads
      }, {
        maxAttempts: 2,
        baseDelay: 1000
      });
      
      const data = await response.json();
      console.log('✅ Image uploaded:', data.image_url);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Upload Blog Image');
    }
  });
};

export const likeBlogPost = async (postId) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('❤️ Liking blog post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/posts/${postId}/like`, {
        method: 'POST',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 5000
      }, {
        maxAttempts: 2,
        baseDelay: 500
      });
      
      const data = await response.json();
      console.log('✅ Post liked:', data.likes_count, 'total likes');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Like Blog Post');
    }
  });
};

export const checkLikeStatus = async (postId) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('🔍 Checking like status for post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/posts/${postId}/like-status`, {
        method: 'GET',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 5000
      }, {
        maxAttempts: 2,
        baseDelay: 500
      });
      
      const data = await response.json();
      console.log('✅ Like status checked:', data);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Check Like Status');
    }
  });
};

export const fetchBlogDistricts = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('📍 Fetching blog districts');
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/districts`, {
        method: 'GET',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 10000
      }, {
        maxAttempts: 2,
        baseDelay: 500
      });
      
      const data = await response.json();
      console.log('✅ Blog districts fetched:', data.length, 'districts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Districts');
    }
  });
};
