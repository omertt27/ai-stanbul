import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link, useLocation } from 'react-router-dom';
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
  const location = useLocation();
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
    } catch (err) {
      console.error('❌ BlogPost: Failed to load post:', err);
      setError('Failed to load blog post. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const loadRelatedPosts = async () => {
    if (!post?.id) return;
    
    console.log('🔗 BlogPost: Loading related posts for post ID:', post.id);
    setRelatedLoading(true);
    
    try {
      const related = await fetchRelatedPosts(post.id);
      setRelatedPosts(related || []);
      console.log('✅ BlogPost: Related posts loaded:', related?.length || 0, 'posts');
    } catch (err) {
      console.error('❌ BlogPost: Failed to load related posts:', err);
      setRelatedPosts([]);
    } finally {
      setRelatedLoading(false);
    }
  };

  const handleLike = async () => {
    if (!post?.id || likeLoading || alreadyLiked) return;
    
    console.log('❤️ BlogPost: Liking post:', post.title);
    setLikeLoading(true);
    setLikeError(null);
    
    try {
      const result = await likeBlogPost(post.id);
      console.log('✅ BlogPost: Post liked successfully:', result);
      
      // Update the post with new like count
      setPost(prev => ({
        ...prev,
        likes: result.likes || (prev.likes || 0) + 1
      }));
      
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
      <div className={`blog-page min-h-screen flex items-center justify-center ${
        darkMode ? 'dark' : ''
      }`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <h3 className={`text-lg font-medium ${
            darkMode ? 'text-gray-200' : 'text-gray-600'
          }`}>Loading post...</h3>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`blog-page min-h-screen flex items-center justify-center ${
        darkMode ? 'dark' : ''
      }`}>
        <div className="text-center">
          <h2 className={`text-2xl font-bold mb-4 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>Error loading post</h2>
          <p className={`mb-6 ${
            darkMode ? 'text-gray-400' : 'text-gray-500'
          }`}>{error}</p>
          <Link
            to="/blog"
            className={`blog-back-link ${darkMode ? 'dark' : ''}`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Blog
          </Link>
        </div>
      </div>
    );
  }

  if (!post) {
    return (
      <div className={`blog-page min-h-screen flex items-center justify-center ${
        darkMode ? 'dark' : ''
      }`}>
        <div className="text-center">
          <h2 className={`text-2xl font-bold mb-4 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>Post not found</h2>
          <Link
            to="/blog"
            className={`blog-back-link ${darkMode ? 'dark' : ''}`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Blog
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className={`blog-page min-h-screen transition-colors duration-200 ${
      darkMode ? 'dark' : ''
    }`}>
      {/* Fixed Navigation Header */}
      <header className={`fixed-navbar ${darkMode ? 'dark' : ''}`}>
        {/* AI Istanbul Logo - Centered */}
        <div className="fixed-navbar-logo">
          <Link to="/" style={{textDecoration: 'none'}}>
            <div className="header-logo logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
        </div>
        
        {/* Navigation Links */}
        <nav className="fixed-navbar-links">
          <Link 
            to="/blog" 
            className={`fixed-navbar-link ${location.pathname.startsWith('/blog') ? 'active' : ''}`}
          >
            Blog
          </Link>
          <Link 
            to="/about" 
            className={`fixed-navbar-link ${location.pathname === '/about' ? 'active' : ''}`}
          >
            About
          </Link>
          <Link 
            to="/faq" 
            className={`fixed-navbar-link ${location.pathname === '/faq' ? 'active' : ''}`}
          >
            FAQ
          </Link>
          <Link 
            to="/donate" 
            className={`fixed-navbar-link ${location.pathname === '/donate' ? 'active' : ''}`}
          >
            Donate
          </Link>
        </nav>
      </header>

      {/* Main Content with proper spacing */}
      <main className="page-with-fixed-nav">
        <div className="blog-content">
          {/* Navigation Breadcrumb */}
          <div className="blog-navigation">
            <Link
              to="/blog"
              className={`blog-back-link ${darkMode ? 'dark' : ''}`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Blog
            </Link>
          </div>

          {/* Blog Post Article */}
          <article className={`blog-article ${darkMode ? 'dark' : ''}`}>
            {/* Featured Image - Full Width */}
            {post.images && post.images.length > 0 && (
              <div className="w-full h-64 md:h-80 lg:h-96 overflow-hidden">
                <img
                  src={`${import.meta.env.VITE_API_URL?.replace(/\/ai\/?$/, '') || 'http://localhost:8001'}${post.images[0].url}`}
                  alt={post.images[0].alt_text || post.title}
                  className="w-full h-full object-cover"
                />
              </div>
            )}

            {/* Author Info */}
            {post.author_name && (
              <div className={`blog-author-section ${darkMode ? 'dark' : ''}`}>
                {post.author_photo ? (
                  <img 
                    src={post.author_photo} 
                    alt={post.author_name}
                    className="blog-author-avatar"
                  />
                ) : (
                  <div className={`blog-author-avatar flex items-center justify-center text-lg font-semibold ${
                    darkMode ? 'bg-gray-600 text-gray-200' : 'bg-indigo-100 text-indigo-600'
                  }`}>
                    {post.author_name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div className={`blog-author-info ${darkMode ? 'dark' : ''}`}>
                  <h4>{post.author_name}</h4>
                  <span>{formatDate(post.created_at)}</span>
                </div>
              </div>
            )}

            <div className={`blog-text-content ${darkMode ? 'dark' : ''}`}>
              {/* Title */}
              <h1 className={`text-3xl md:text-4xl font-bold mb-6 leading-tight ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                {post.title}
              </h1>

              {/* Summary */}
              {post.summary && (
                <div className={`text-lg mb-8 p-4 rounded-lg italic border-l-4 ${
                  darkMode 
                    ? 'bg-gray-700 border-indigo-400 text-gray-300' 
                    : 'bg-indigo-50 border-indigo-400 text-indigo-800'
                }`}>
                  {post.summary}
                </div>
              )}

              {/* Content */}
              <div className="prose prose-lg max-w-none">
                {formatContent(post.content)}
              </div>

              {/* Tags and District */}
              {(post.tags || post.district) && (
                <div className="mt-8 pt-6 border-t border-gray-200">
                  {post.district && (
                    <div className="mb-4">
                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                        darkMode 
                          ? 'bg-blue-900 text-blue-200' 
                          : 'bg-blue-100 text-blue-800'
                      }`}>
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        {post.district}
                      </span>
                    </div>
                  )}
                  
                  {post.tags && (
                    <div className="flex flex-wrap gap-2">
                      {post.tags.split(',').map((tag, index) => (
                        <span 
                          key={index}
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            darkMode 
                              ? 'bg-gray-700 text-gray-300' 
                              : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          #{tag.trim()}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Like Button */}
              <div className="mt-8 pt-6 border-t border-gray-200">
                <button
                  onClick={handleLike}
                  disabled={likeLoading || alreadyLiked}
                  className={`inline-flex items-center px-6 py-3 font-semibold rounded-lg transition-all duration-200 ${
                    alreadyLiked
                      ? (darkMode ? 'bg-green-800 text-green-200' : 'bg-green-100 text-green-800')
                      : (darkMode 
                          ? 'bg-indigo-600 hover:bg-indigo-700 text-white' 
                          : 'bg-indigo-600 hover:bg-indigo-700 text-white')
                  } ${(likeLoading || alreadyLiked) ? 'opacity-75 cursor-not-allowed' : 'hover:transform hover:scale-105'}`}
                >
                  <svg className="w-5 h-5 mr-2" fill={alreadyLiked ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                  {likeLoading ? 'Liking...' : alreadyLiked ? 'Liked!' : `Like (${post.likes || 0})`}
                </button>
                {likeError && (
                  <p className="mt-2 text-sm text-red-600">{likeError}</p>
                )}
              </div>
            </div>
          </article>

          {/* Related Posts */}
          {relatedPosts.length > 0 && (
            <section className="mt-12">
              <h2 className={`text-2xl font-bold mb-6 ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                Related Posts
              </h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {relatedPosts.map(relatedPost => (
                  <Link
                    key={relatedPost.id}
                    to={`/blog/${relatedPost.id}`}
                    className={`block rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-all duration-200 hover:transform hover:scale-105 ${
                      darkMode 
                        ? 'bg-gray-800 border border-gray-700' 
                        : 'bg-white border border-gray-200'
                    }`}
                  >
                    {relatedPost.images && relatedPost.images.length > 0 && (
                      <img
                        src={`${import.meta.env.VITE_API_URL?.replace(/\/ai\/?$/, '') || 'http://localhost:8001'}${relatedPost.images[0].url}`}
                        alt={relatedPost.images[0].alt_text || relatedPost.title}
                        className="w-full h-48 object-cover"
                      />
                    )}
                    <div className="p-4">
                      <h3 className={`font-semibold mb-2 line-clamp-2 ${
                        darkMode ? 'text-gray-100' : 'text-gray-900'
                      }`}>
                        {relatedPost.title}
                      </h3>
                      {relatedPost.summary && (
                        <p className={`text-sm line-clamp-3 ${
                          darkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>
                          {relatedPost.summary}
                        </p>
                      )}
                    </div>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {/* Comments Section */}
          <section className="mt-12">
            <Comments postId={id} darkMode={darkMode} />
          </section>
        </div>
      </main>
    </div>
  );
};

export default BlogPost;
