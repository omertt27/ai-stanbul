import { 
  fetchWithRetry, 
  getUserFriendlyMessage, 
  classifyError, 
  ErrorTypes,
  createCircuitBreaker
} from '../utils/errorHandler.js';
import { Logger } from '../utils/logger';

const logger = new Logger('BlogAPI');

// API configuration
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const cleanBaseUrl = BASE_URL.replace(/\/ai\/?$/, '');
const BLOG_API_URL = `${cleanBaseUrl}/api/blog/posts`;

// Debug logging
logger.debug('🔧 API Configuration:');
logger.debug('  VITE_API_URL:', import.meta.env.VITE_API_URL);
logger.debug('  BASE_URL:', BASE_URL);
logger.debug('  BLOG_API_URL:', BLOG_API_URL);

// Circuit breaker for blog API
const blogCircuitBreaker = createCircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 30000
});

// Enhanced error handling wrapper
const handleBlogApiError = (error, response = null, context = '') => {
  const errorType = classifyError(error, response);
  const userMessage = getUserFriendlyMessage(error, response);
  
  logger.error(`${context} error:`, {
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
      // Use page-based pagination as expected by backend
      const backendParams = { ...params };
      // Backend expects 'page' and 'per_page' parameters
      if (params.limit) {
        backendParams.per_page = params.limit;
        delete backendParams.limit;
      }
      
      const searchParams = new URLSearchParams();
      Object.keys(backendParams).forEach(key => {
        if (backendParams[key] !== undefined && backendParams[key] !== null) {
          searchParams.append(key, backendParams[key].toString());
        }
      });
      
      const url = `${BLOG_API_URL}${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
      logger.debug('🔍 Fetching blog posts from:', url);
      
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
      
      // DEBUGGING: Log the full response structure
      logger.info('📦 Raw API response:', data);
      logger.info('📊 Response type:', Array.isArray(data) ? 'Array' : typeof data);
      logger.info('📊 Response keys:', Array.isArray(data) ? 'N/A (is array)' : Object.keys(data));
      
      // Handle different response formats
      let normalizedData;
      if (Array.isArray(data)) {
        // If API returns an array directly, wrap it
        logger.info('⚠️ API returned array directly, normalizing...');
        normalizedData = {
          posts: data,
          total: data.length,
          limit: params.limit || 12,
          offset: params.offset || 0
        };
      } else if (data && data.posts && Array.isArray(data.posts)) {
        // Standard format
        normalizedData = data;
      } else {
        // Unknown format
        logger.error('❌ Unexpected API response format:', data);
        normalizedData = {
          posts: [],
          total: 0,
          limit: params.limit || 12,
          offset: params.offset || 0
        };
      }
      
      logger.info('✅ Blog posts fetched:', normalizedData.posts?.length || 0, 'posts, total:', normalizedData.total);
      return normalizedData;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Posts');
    }
  });
};

export const fetchBlogPost = async (postId) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('📖 Fetching blog post:', postId);
      logger.info('🔗 Blog API URL:', BLOG_API_URL);
      logger.info('🎯 Full URL:', `${BLOG_API_URL}/${postId}`);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/${postId}`, {
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
      logger.info('✅ Blog post fetched:', data.title);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Post');
    }
  });
};

// Admin CRUD operations
export const createBlogPost = async (postData) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('✨ Creating new blog post:', postData.title);
      
      const response = await fetchWithRetry(BLOG_API_URL, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(postData),
        timeout: 15000
      }, {
        maxAttempts: 2,
        baseDelay: 1000
      });
      
      const data = await response.json();
      logger.info('✅ Blog post created:', data.id);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Create Blog Post');
    }
  });
};

export const updateBlogPost = async (postId, postData) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('💾 Updating blog post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/${postId}`, {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(postData),
        timeout: 15000
      }, {
        maxAttempts: 2,
        baseDelay: 1000
      });
      
      const data = await response.json();
      logger.info('✅ Blog post updated:', postId);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Update Blog Post');
    }
  });
};

export const deleteBlogPost = async (postId) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('🗑️ Deleting blog post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/${postId}`, {
        method: 'DELETE',
        headers: { 
          'Accept': 'application/json'
        },
        timeout: 10000
      }, {
        maxAttempts: 2,
        baseDelay: 1000
      });
      
      const data = await response.json();
      logger.info('✅ Blog post deleted:', postId);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Delete Blog Post');
    }
  });
};

// Alias for compatibility with different naming conventions
export const getBlogPosts = fetchBlogPosts;

export const uploadBlogImage = async (file) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('📸 Uploading blog image:', file.name);
      
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
      logger.info('✅ Image uploaded:', data.image_url);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Upload Blog Image');
    }
  });
};

export const likeBlogPost = async (postId, userIdentifier = null) => {
  return blogCircuitBreaker.call(async () => {
    try {
      // Generate or use stored user identifier
      if (!userIdentifier) {
        userIdentifier = localStorage.getItem('user_identifier') || `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        localStorage.setItem('user_identifier', userIdentifier);
      }
      
      logger.info('❤️ Toggling like for post:', postId, 'User:', userIdentifier.substr(0, 10) + '...');
      
      // Use the new database-backed endpoint
      const response = await fetchWithRetry(`${BLOG_API_URL}/${postId}/like`, {
        method: 'POST',
        headers: { 
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_identifier: userIdentifier }),
        timeout: 5000
      }, {
        maxAttempts: 2,
        baseDelay: 500
      });
      
      const data = await response.json();
      logger.info('✅ Post like toggled:', data);
      return data; // { success: true, is_liked: boolean, likes_count: number }
      
    } catch (error) {
      logger.error('❌ Like failed for post ID:', postId, 'Error:', error.message);
      throw handleBlogApiError(error, null, `Like Blog Post (ID: ${postId})`);
    }
  });
};

export const checkLikeStatus = async (postId, userIdentifier = 'default_user') => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info(`🔍 Checking like status for post ${postId}`);
      
      // Use the new database-backed endpoint with query parameter
      const response = await fetchWithRetry(`${BLOG_API_URL}/${postId}/like-status?user_identifier=${encodeURIComponent(userIdentifier)}`, {
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
      logger.info('✅ Like status checked:', data);
      return data; // { success: true, isLiked: boolean, likes_count: number }
      
    } catch (error) {
      logger.error('❌ Error checking like status:', error);
      // Return default values on error
      return { success: false, isLiked: false, likes_count: 0 };
    }
  });
};

export const fetchBlogDistricts = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('📍 Fetching blog districts');
      
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
      logger.info('✅ Blog districts fetched:', data.length, 'districts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Districts');
    }
  });
};

export const fetchBlogTags = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('📋 Fetching blog tags');
      
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
      logger.info('✅ Blog tags fetched:', data.length, 'tags');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Tags');
    }
  });
};

export const fetchFeaturedPosts = async (limit = 3) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('⭐ Fetching featured posts');
      
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
      logger.info('✅ Featured posts fetched:', data.featured_posts?.length || 0, 'posts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Featured Posts');
    }
  });
};

export const fetchTrendingPosts = async (limit = 5) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('🔥 Fetching trending posts');
      
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
      logger.info('✅ Trending posts fetched:', data.trending_posts?.length || 0, 'posts');
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Trending Posts');
    }
  });
};

export const fetchBlogStats = async () => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('📊 Fetching blog stats');
      
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
      logger.info('✅ Blog stats fetched:', data);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Blog Stats');
    }
  });
};

export const fetchRelatedPosts = async (postId, limit = 4) => {
  try {
    logger.info('🔗 Fetching related posts for:', postId);
    
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
    
    logger.info('✅ Related posts fetched:', related.length, 'posts');
    return { related_posts: related };
    
  } catch (error) {
    logger.error('❌ Failed to fetch related posts:', error);
    return { related_posts: [] };
  }
};

export const seedSamplePosts = async () => {
  // Backend automatically seeds sample posts on startup
  logger.info('🌱 Sample posts are automatically seeded by backend');
  return { message: 'Sample posts already available' };
};

// Circuit breaker utilities
export const resetBlogCircuitBreaker = () => {
  blogCircuitBreaker.reset();
  logger.info('🔄 Blog circuit breaker has been reset');
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
    logger.info('🔍 Validating post exists:', postId);
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
    logger.warn('⚠️ Post validation failed for ID:', postId, error.message);
    return false;
  }
};

export const safeLikeBlogPost = async (postId, userIdentifier = 'default_user') => {
  // First validate the post exists
  const exists = await validatePostExists(postId);
  if (!exists) {
    logger.error('❌ Cannot like non-existent post:', postId);
    throw new Error(`Blog post with ID ${postId} does not exist`);
  }
  
  return likeBlogPost(postId, userIdentifier);
};

// =============================
// COMMENT API FUNCTIONS
// =============================

export const fetchComments = async (postId) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('💬 Fetching comments for post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}/${postId}/comments`, {
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
      logger.info('✅ Comments fetched:', data.comments?.length || 0, 'comments');
      return data; // { comments: [...], total: number }
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Fetch Comments');
    }
  });
};

export const createComment = async (postId, commentData) => {
  return blogCircuitBreaker.call(async () => {
    try {
      logger.info('✍️ Creating comment on post:', postId);
      
      const response = await fetchWithRetry(`${BLOG_API_URL}${postId}/comments`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(commentData),
        timeout: 10000
      }, {
        maxAttempts: 2,
        baseDelay: 1000
      });
      
      const data = await response.json();
      logger.info('✅ Comment created:', data.id);
      return data;
      
    } catch (error) {
      throw handleBlogApiError(error, null, 'Create Comment');
    }
  });
};
