import { 
  fetchWithRetry, 
  getUserFriendlyMessage, 
  classifyError, 
  ErrorTypes,
  createCircuitBreaker
} from '../utils/errorHandler.js';

// API configuration
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const cleanBaseUrl = BASE_URL.replace(/\/ai\/?$/, '');
const BLOG_API_URL = `${cleanBaseUrl}/blog/`;

// Debug logging
console.log('🔧 API Configuration:');
console.log('  VITE_API_URL:', import.meta.env.VITE_API_URL);
console.log('  BASE_URL:', BASE_URL);
console.log('  BLOG_API_URL:', BLOG_API_URL);

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
      
      const url = `${BLOG_API_URL}${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
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
      console.log('✅ Blog posts fetched:', data.posts?.length || 0, 'posts, total:', data.total);
      console.log('📊 Blog API response structure:', Object.keys(data));
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
      console.log('🔗 Blog API URL:', BLOG_API_URL);
      console.log('🎯 Full URL:', `${BLOG_API_URL}${postId}`);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}${postId}`, {
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
      
      const response = await fetchWithRetry(`${BLOG_API_URL}`, {
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
      
      const response = await fetchWithRetry(`${BLOG_API_URL}upload-image`, {
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

export const likeBlogPost = async (postId, userIdentifier = 'default_user') => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('❤️ Liking blog post:', postId, 'Type:', typeof postId);
      console.log('🔗 Full like URL:', `${BLOG_API_URL}${postId}/like`);
      
      // Use the JSON file-based endpoint that now tracks individual users
      const response = await fetchWithRetry(`${BLOG_API_URL}${postId}/like`, {
        method: 'POST',
        headers: { 
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userIdentifier
        }),
        timeout: 5000
      }, {
        maxAttempts: 2,
        baseDelay: 500
      });
      
      const data = await response.json();
      console.log('✅ Post liked:', data.likes_count, 'total likes');
      return data;
      
    } catch (error) {
      console.error('❌ Like failed for post ID:', postId, 'Error:', error.message);
      throw handleBlogApiError(error, null, `Like Blog Post (ID: ${postId})`);
    }
  });
};

export const checkLikeStatus = async (postId, userIdentifier = 'default_user') => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log(`🔍 Checking like status for post ${postId}`);
      
      // Use the JSON file-based endpoint that now tracks individual users
      const response = await fetchWithRetry(`${BLOG_API_URL}${postId}/like-status?user_id=${userIdentifier}`, {
        method: 'GET',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 10000
      });

      if (!response.ok) {
        throw new Error(`Failed to check like status: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ Like status checked:', data);
      return data; // { isLiked: boolean, likes: number }
      
    } catch (error) {
      console.error('❌ Error checking like status:', error);
      // Return default values on error
      return { isLiked: false, likes: 0 };
    }
  });
};

export const fetchBlogDistricts = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('📍 Fetching blog districts');
      
      const response = await fetchWithRetry(`${BLOG_API_URL}districts`, {
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

export const fetchBlogTags = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('📋 Fetching blog tags');
      
      const response = await fetchWithRetry(`${BLOG_API_URL}tags`, {
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
      console.log('✅ Blog tags fetched:', data.length, 'tags');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Tags');
    }
  });
};

export const fetchFeaturedPosts = async (limit = 3) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('⭐ Fetching featured posts');
      
      const response = await fetchWithRetry(`${BLOG_API_URL}featured?limit=${limit}`, {
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
      console.log('✅ Featured posts fetched:', data.featured_posts?.length || 0, 'posts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Featured Posts');
    }
  });
};

export const fetchTrendingPosts = async (limit = 5) => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('🔥 Fetching trending posts');
      
      const response = await fetchWithRetry(`${BLOG_API_URL}trending?limit=${limit}`, {
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
      console.log('✅ Trending posts fetched:', data.trending_posts?.length || 0, 'posts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Trending Posts');
    }
  });
};

export const fetchBlogStats = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      console.log('📊 Fetching blog stats');
      
      const response = await fetchWithRetry(`${BLOG_API_URL}stats`, {
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
      console.log('✅ Blog stats fetched:', data);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Stats');
    }
  });
};

export const fetchRelatedPosts = async (postId, limit = 4) => {
  try {
    console.log('🔗 Fetching related posts for:', postId);
    
    // Get all posts and filter client-side since backend doesn't have related endpoint
    const allPosts = await fetchBlogPosts();
    const currentPost = allPosts.posts?.find(p => p.id === postId);
    
    if (!currentPost) {
      return { related_posts: [] };
    }
    
    // Simple related posts logic: same category or shared tags
    const related = allPosts.posts
      ?.filter(p => p.id !== postId)
      ?.filter(p => 
        p.category === currentPost.category || 
        p.tags?.some(tag => currentPost.tags?.includes(tag))
      )
      ?.slice(0, limit) || [];
    
    console.log('✅ Related posts fetched:', related.length, 'posts');
    return { related_posts: related };
    
  } catch (error) {
    console.error('❌ Failed to fetch related posts:', error);
    return { related_posts: [] };
  }
};

export const seedSamplePosts = async () => {
  // Backend automatically seeds sample posts on startup
  console.log('🌱 Sample posts are automatically seeded by backend');
  return { message: 'Sample posts already available' };
};

// Circuit breaker utilities
export const resetBlogCircuitBreaker = () => {
  blogCircuitBreaker.reset();
  console.log('🔄 Blog circuit breaker has been reset');
};

export const getBlogCircuitBreakerState = () => {
  return {
    state: blogCircuitBreaker.state,
    failureCount: blogCircuitBreaker.failureCount,
    nextAttempt: blogCircuitBreaker.nextAttempt
  };
};

// Validation utilities
export const validatePostExists = async (postId) => {
  try {
    console.log('🔍 Validating post exists:', postId);
    const response = await fetchWithRetry(`${BLOG_API_URL}${postId}`, {
      method: 'GET',
      headers: { 
        'Accept': 'application/json'
      },
      timeout: 5000
    }, {
      maxAttempts: 1
    });
    
    return response.ok;
  } catch (error) {
    console.warn('⚠️ Post validation failed for ID:', postId, error.message);
    return false;
  }
};

export const safeLikeBlogPost = async (postId, userIdentifier = 'default_user') => {
  // First validate the post exists
  const exists = await validatePostExists(postId);
  if (!exists) {
    console.error('❌ Cannot like non-existent post:', postId);
    throw new Error(`Blog post with ID ${postId} does not exist`);
  }
  
  return likeBlogPost(postId, userIdentifier);
};
