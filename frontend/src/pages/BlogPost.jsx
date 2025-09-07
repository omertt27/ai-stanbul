import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { 
  fetchBlogPost, 
  likeBlogPost, 
  checkLikeStatus, 
  fetchRelatedPosts 
} from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import { trackBlogEvent } from '../utils/analytics';
import Comments from '../components/Comments';
import '../App.css';

const BlogPost = () => {
  const { darkMode } = useTheme();
  const { id } = useParams();
  const navigate = useNavigate();
  const [post, setPost] = useState(null);
  const [relatedPosts, setRelatedPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [relatedLoading, setRelatedLoading] = useState(true);
  const [error, setError] = useState(null);
  const [likeLoading, setLikeLoading] = useState(false);
  const [alreadyLiked, setAlreadyLiked] = useState(false);
  const [likeError, setLikeError] = useState(null);

  useEffect(() => {
    console.log('🔄 BlogPost: Loading post with ID:', id);
    loadPost();
    checkUserLikeStatus();
  }, [id]);

  useEffect(() => {
    if (post) {
      console.log('🔄 BlogPost: Loading related posts for:', post.title);
      loadRelatedPosts();
    }
  }, [post]);

  const loadPost = async () => {
    console.log('📖 BlogPost: Loading post with ID:', id);
    setLoading(true);
    setError(null);
    
    try {
      const fetchedPost = await fetchBlogPost(id);
      setPost(fetchedPost);
      console.log('✅ BlogPost: Post loaded successfully:', fetchedPost?.title);

      // Track blog post view event
      if (fetchedPost) {
        trackBlogEvent('view_post', fetchedPost.title);
      }
      trackBlogEvent('view', {
        id: fetchedPost.id,
        title: fetchedPost.title,
        author: fetchedPost.author_name,
        category: fetchedPost.category || 'Uncategorized',
        tags: fetchedPost.tags || [],
        url: window.location.href
      });
    } catch (err) {
      setError(err.message || 'Failed to load blog post');
      console.error('❌ BlogPost: Failed to load post:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadRelatedPosts = async () => {
    setRelatedLoading(true);
    try {
      const response = await fetchRelatedPosts(id, 4);
      setRelatedPosts(response.related_posts || []);
    } catch (err) {
      console.error('Failed to load related posts:', err);
    } finally {
      setRelatedLoading(false);
    }
  };

  const handleLike = async () => {
    if (likeLoading || alreadyLiked) return;
    
    setLikeLoading(true);
    setLikeError(null);
    
    try {
      const result = await likeBlogPost(id);
      
      // Update the post with new like count
      setPost(prev => ({
        ...prev,
        likes_count: result.likes_count
      }));
      
      // Mark as already liked
      setAlreadyLiked(true);
      
      // Track like event
      trackBlogEvent('like_post', post?.title || 'Unknown Post');
      
    } catch (err) {
      console.error('Failed to like post:', err);
      
      // Check if it's a "already liked" error
      if (err.message && err.message.includes('already liked')) {
        setAlreadyLiked(true);
        setLikeError('You have already liked this post');
      } else {
        setLikeError('Failed to like post. Please try again.');
      }
    } finally {
      setLikeLoading(false);
    }
  };

  const checkUserLikeStatus = async () => {
    if (!id) return;
    
    try {
      const status = await checkLikeStatus(id);
      setAlreadyLiked(status.already_liked);
    } catch (err) {
      console.error('Failed to check like status:', err);
      // Don't show error for like status check
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatContent = (content) => {
    return content.split('\n').map((paragraph, index) => (
      <p key={index} className={`mb-4 leading-relaxed ${
        darkMode ? 'text-gray-300' : 'text-gray-700'
      }`}>
        {paragraph}
      </p>
    ));
  };

  if (loading) {
    return (
      <div className={`min-h-screen pt-20 sm:pt-28 md:pt-36 px-2 sm:px-4 pb-8 transition-colors duration-200 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-center items-center py-20">
            <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${
              darkMode ? 'border-indigo-500' : 'border-indigo-600'
            }`}></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`min-h-screen pt-20 sm:pt-28 md:pt-36 px-2 sm:px-4 pb-8 transition-colors duration-200 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-20">
            <div className={`mb-8 p-6 rounded-lg border transition-colors duration-200 ${
              darkMode 
                ? 'bg-red-900/20 border-red-500/20'
                : 'bg-red-50 border-red-200'
            }`}>
              <p className={`mb-4 ${darkMode ? 'text-red-400' : 'text-red-700'}`}>{error}</p>
              <button
                onClick={loadPost}
                className={`px-4 py-2 rounded-lg transition-colors duration-200 ${
                  darkMode
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-red-600 hover:bg-red-700 text-white'
                }`}
              >
                Try Again
              </button>
            </div>
            <Link
              to="/blog"
              className={`inline-flex items-center px-6 py-3 font-semibold rounded-lg transition-colors duration-200 ${
                darkMode
                  ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Blog
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!post) {
    return (
      <div className={`min-h-screen pt-20 sm:pt-28 md:pt-36 px-2 sm:px-4 pb-8 transition-colors duration-200 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-20">
            <h2 className={`text-2xl font-bold mb-4 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>Post not found</h2>
            <Link
              to="/blog"
              className={`inline-flex items-center px-6 py-3 font-semibold rounded-lg transition-colors duration-200 ${
                darkMode
                  ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Blog
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      darkMode ? 'bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800' : 'bg-gradient-to-br from-gray-50 via-white to-gray-100'
    }`}>
      {/* Static Header with Logo and Navigation */}
      <header className={`w-full px-4 py-4 border-b transition-colors duration-200 backdrop-blur-sm ${
        darkMode ? 'bg-gray-900/90 border-gray-700/50' : 'bg-white/90 border-gray-200/50'
      }`}>
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          {/* AI Istanbul Logo */}
          <Link to="/" style={{textDecoration: 'none'}}>
            <div className="chat-title logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
          
          {/* Navigation Links */}
          <nav className="flex items-center gap-6">
            <Link 
              to="/blog" 
              className={`font-medium transition-colors duration-200 ${
                darkMode 
                  ? 'text-indigo-400 hover:text-indigo-300' 
                  : 'text-indigo-600 hover:text-indigo-700'
              }`}
            >
              Blog
            </Link>
            <Link 
              to="/about" 
              className={`font-medium transition-colors duration-200 ${
                darkMode 
                  ? 'text-gray-300 hover:text-white' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              About
            </Link>
            <Link 
              to="/faq" 
              className={`font-medium transition-colors duration-200 ${
                darkMode 
                  ? 'text-gray-300 hover:text-white' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              FAQ
            </Link>
            <Link 
              to="/donate" 
              className={`font-medium transition-colors duration-200 ${
                darkMode 
                  ? 'text-gray-300 hover:text-white' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Donate
            </Link>
          </nav>
        </div>
      </header>

      <div className="px-2 sm:px-4 pb-8">
      <div className="max-w-4xl mx-auto">
        {/* Navigation */}
        <div className="mb-8 pt-6">
          <Link
            to="/blog"
            className={`inline-flex items-center transition-colors duration-200 ${
              darkMode 
                ? 'text-indigo-400 hover:text-indigo-300' 
                : 'text-indigo-600 hover:text-indigo-500'
            }`}
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Blog
          </Link>
        </div>

        <article className={`rounded-xl overflow-hidden transition-all duration-200 backdrop-blur-sm ${
          darkMode 
            ? 'bg-gray-800/80 border border-gray-700/50 shadow-2xl shadow-black/20' 
            : 'bg-white/90 border border-gray-200/50 shadow-xl shadow-gray-900/10'
        }`}>
          {/* Featured Image */}
          {post.images && post.images.length > 0 && (
            <div className="aspect-video lg:aspect-[2/1] overflow-hidden">
              <img
                src={`http://localhost:8001${post.images[0].url}`}
                alt={post.images[0].alt_text || post.title}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-700"
              />
            </div>
          )}

          <div className="p-6 sm:p-8">
            {/* Author Info */}
            {post.author_name && (
              <div className={`flex items-center mb-6 pb-6 border-b transition-colors duration-200 ${
                darkMode ? 'border-gray-700/50' : 'border-gray-200/50'
              }`}>
                {post.author_photo ? (
                  <img 
                    src={post.author_photo} 
                    alt={post.author_name}
                    className="w-12 h-12 rounded-full mr-4 object-cover border-2 border-indigo-500/20"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'flex';
                    }}
                  />
                ) : null}
                <div className={`w-12 h-12 rounded-full mr-4 bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-lg font-semibold shadow-lg ${post.author_photo ? 'hidden' : ''}`}>
                  {post.author_name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <p className={`text-lg font-semibold transition-colors duration-200 ${
                    darkMode ? 'text-gray-100' : 'text-gray-800'
                  }`}>{post.author_name}</p>
                  <p className={`text-sm transition-colors duration-200 ${
                    darkMode ? 'text-indigo-400' : 'text-indigo-600'
                  }`}>Travel Blogger</p>
                </div>
              </div>
            )}

            {/* Title */}
            <h1 className={`text-3xl md:text-4xl lg:text-5xl font-bold mb-6 leading-tight transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>
              {post.title}
            </h1>

            {/* Metadata */}
            <div className={`flex flex-wrap items-center gap-6 mb-8 pb-6 border-b transition-colors duration-200 ${
              darkMode ? 'text-gray-300 border-gray-700/50' : 'text-gray-600 border-gray-200/50'
            }`}>
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                {formatDate(post.created_at)}
              </div>

              {post.district && (
                <div className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  {post.district}
                </div>
              )}

              <button
                onClick={handleLike}
                disabled={likeLoading || alreadyLiked}
                className={`flex items-center transition-colors duration-200 disabled:opacity-50 ${
                  alreadyLiked 
                    ? 'text-red-500 cursor-not-allowed' 
                    : darkMode 
                      ? 'text-gray-400 hover:text-red-400'
                      : 'text-gray-600 hover:text-red-500'
                }`}
                title={alreadyLiked ? 'You have already liked this post' : 'Like this post'}
              >
                <svg
                  className={`w-5 h-5 mr-2 ${likeLoading ? 'animate-pulse' : ''}`}
                  fill={alreadyLiked ? 'currentColor' : 'none'}
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
                {post.likes_count || 0} {(post.likes_count || 0) === 1 ? 'Like' : 'Likes'}
                {alreadyLiked && <span className="ml-2 text-xs">✓</span>}
              </button>

              {/* Like Error Message */}
              {likeError && (
                <div className="text-red-400 text-sm mt-2">
                  {likeError}
                </div>
              )}
            </div>

            {/* Content */}
            <div className={`prose prose-lg max-w-none transition-colors duration-200 ${
              darkMode ? 'prose-invert' : ''
            }`}>
              {formatContent(post.content)}
            </div>

            {/* Additional Images */}
            {post.images && post.images.length > 1 && (
              <div className={`mt-8 border-t pt-8 transition-colors duration-200 ${
                darkMode ? 'border-gray-700/50' : 'border-gray-200/50'
              }`}>
                <h3 className={`text-xl font-semibold mb-6 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>📸 More Photos</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {post.images.slice(1).map((image, index) => (
                    <div key={index} className="aspect-video overflow-hidden rounded-xl group">
                      <img
                        src={`http://localhost:8001${image.url}`}
                        alt={image.alt_text || `Photo ${index + 2}`}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </article>

        {/* Comments Section */}
        <Comments postId={id} />

        {/* Related Posts */}
        {relatedPosts.length > 0 && (
          <div className={`mt-12 border-t pt-12 transition-colors duration-200 ${
            darkMode ? 'border-gray-700/50' : 'border-gray-200/50'
          }`}>
            <div className="flex items-center justify-between mb-8">
              <h2 className={`text-2xl md:text-3xl font-bold transition-colors duration-200 ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                🔗 Related Stories
              </h2>
              {relatedLoading && (
                <div className={`animate-spin rounded-full h-6 w-6 border-b-2 ${
                  darkMode ? 'border-indigo-500' : 'border-indigo-600'
                }`}></div>
              )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {relatedPosts.map((relatedPost) => (
                <Link
                  key={relatedPost.id}
                  to={`/blog/${relatedPost.id}`}
                  className={`group block rounded-xl overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-xl ${
                    darkMode
                      ? 'bg-gray-800/80 hover:bg-gray-800 border border-gray-700/50 shadow-lg shadow-black/20'
                      : 'bg-white/90 hover:bg-white border border-gray-200/50 shadow-lg shadow-gray-900/10 hover:shadow-2xl'
                  }`}
                >
                  {relatedPost.images && relatedPost.images.length > 0 && (
                    <div className="aspect-video overflow-hidden">
                      <img
                        src={`http://localhost:8001${relatedPost.images[0].url}`}
                        alt={relatedPost.images[0].alt_text || relatedPost.title}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                      />
                    </div>
                  )}
                  
                  <div className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      {relatedPost.district && (
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          darkMode ? 'bg-gray-700/70 text-gray-300' : 'bg-gray-100/80 text-gray-600'
                        }`}>
                          📍 {relatedPost.district}
                        </span>
                      )}
                    </div>
                    
                    <h3 className={`font-bold text-sm mb-2 line-clamp-2 group-hover:text-indigo-500 transition-colors duration-200 ${
                      darkMode ? 'text-white' : 'text-gray-900'
                    }`}>
                      {relatedPost.title}
                    </h3>
                    
                    <p className={`text-xs leading-relaxed line-clamp-2 mb-3 transition-colors duration-200 ${
                      darkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      {relatedPost.content}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <span className={`text-xs font-medium ${
                        darkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        {relatedPost.author_name}
                      </span>
                      <div className="flex items-center gap-1">
                        <svg className="w-3 h-3 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        <span className={`text-xs ${
                          darkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          {relatedPost.likes_count}
                        </span>
                      </div>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}

        {/* Static Footer for Blog Post Page */}
        <footer className={`w-full py-8 px-4 mt-12 border-t transition-colors duration-200 backdrop-blur-sm ${
          darkMode
            ? 'bg-gray-900/70 border-gray-700/50 text-gray-300'
            : 'bg-white/70 border-gray-200/50 text-gray-600'
        }`}>
          <div className="max-w-4xl mx-auto flex justify-center gap-8">
            <Link 
              to="/sources" 
              className={`hover:underline transition-colors duration-200 ${
                darkMode 
                  ? 'hover:text-indigo-400' 
                  : 'hover:text-indigo-600'
              }`}
            >
              Sources
            </Link>
            <Link 
              to="/contact" 
              className={`hover:underline transition-colors duration-200 ${
                darkMode 
                  ? 'hover:text-indigo-400' 
                  : 'hover:text-indigo-600'
              }`}
            >
              Contact
            </Link>
          </div>
        </footer>
      </div>
      </div>
    </div>
  );
};

export default BlogPost;
