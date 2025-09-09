import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link, useLocation } from 'react-router-dom';
// import { 
//   fetchBlogPost, 
//   likeBlogPost, 
//   checkLikeStatus, 
//   fetchRelatedPosts 
// } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import { trackBlogEvent } from '../utils/analytics';
import Comments from '../components/Comments';
import '../App.css';

// Mock blog data to avoid circuit breaker errors
const mockBlogPosts = [
  {
    id: 1,
    title: "Hidden Gems in Sultanahmet: Beyond the Tourist Trail",
    content: `Discover the secret courtyards, ancient cisterns, and local eateries that most visitors miss in Istanbul's historic heart. From the peaceful Soğukçeşme Sokağı to the underground wonders of Şerefiye Cistern.

Istanbul's Sultanahmet district is much more than the Blue Mosque and Hagia Sophia. While these iconic landmarks deserve their fame, the real magic happens in the quiet corners and hidden passages that most tourists never discover.

**Secret Courtyards and Hidden Gardens**

Start your exploration at Soğukçeşme Sokağı, a cobblestone street lined with restored Ottoman houses. The narrow alley connects Hagia Sophia to Topkapi Palace, but few visitors take the time to appreciate its authentic atmosphere. Early morning is the best time to visit – the golden light filtering through the old windows creates an almost mystical ambiance.

**Underground Wonders**

While everyone knows about the Basilica Cistern, the Şerefiye Cistern offers an equally impressive but far less crowded experience. Built in the 4th century, this underground marvel features beautiful lighting and interactive exhibits that bring Byzantine history to life.

**Local Eating Spots**

Skip the tourist restaurants around the main attractions and head to Pandeli, a historic Ottoman restaurant hidden above the Spice Bazaar. The hand-painted tiles and traditional dishes haven't changed in over a century.

For a more casual experience, find the small çay bahçesi (tea gardens) tucked behind the neighborhood's residential streets. These local gathering spots offer the best Turkish tea and a chance to observe daily life in old Istanbul.`,
    author_name: "Mehmet Yılmaz",
    district: "Sultanahmet",
    created_at: "2024-12-01T10:00:00Z",
    likes: 47,
    likes_count: 47,
    images: []
  },
  {
    id: 2,
    title: "Best Rooftop Views for Sunset in Galata",
    content: `Experience Istanbul's magic hour from the best rooftop terraces in Galata. From trendy bars to quiet cafes, here are the spots where locals go to watch the sun set over the Golden Horn.

Galata offers some of Istanbul's most spectacular sunset views, and knowing where to find them can transform your evening into an unforgettable experience.

**The Hidden Rooftop of Anemon Hotel**

While not a secret per se, the rooftop bar at Anemon Galata offers 360-degree views of the city. Arrive early to secure a table facing the Golden Horn, and watch as the sun sets behind the minarets of the old city.

**Local Favorites**

For a more authentic experience, try the small rooftop cafes along Galip Dede Street. These family-run establishments offer Turkish coffee and baklava with views that rival any luxury hotel.

**Photography Tips**

The best light occurs about 30 minutes before sunset. Position yourself facing southwest for the classic silhouette shots of the historical peninsula. Don't forget to capture the moment when the call to prayer echoes across the water – it's pure Istanbul magic.`,
    author_name: "Ayşe Demir",
    district: "Galata",
    created_at: "2024-11-28T15:30:00Z",
    likes: 73,
    likes_count: 73,
    images: []
  },
  {
    id: 3,
    title: "Street Food Paradise: Kadıköy's Culinary Adventures",
    content: `Dive into the vibrant food scene of Kadıköy, where traditional Turkish flavors meet modern creativity. From the famous fish sandwich vendors to hidden meyhanes serving authentic mezze.

Kadıköy's food scene is a testament to Istanbul's culinary evolution. This Asian-side neighborhood has become a foodie destination where tradition meets innovation.

**The Famous Fish Sandwich**

Start your culinary journey at the ferry terminal, where fishermen grill fresh catch right on their boats. The balık ekmek (fish sandwich) here is legendary – simple ingredients prepared with generations of expertise.

**Hidden Meyhanes**

Venture into the backstreets to discover meyhanes that have been serving the same families for decades. These traditional taverns offer an extensive selection of mezze, from fresh seafood to pickled vegetables.

**Modern Twists on Classic Dishes**

Young chefs in Kadıköy are reimagining Turkish cuisine. Look for restaurants serving contemporary interpretations of Ottoman dishes, using local ingredients in surprising ways.

**Market Adventures**

Don't miss the Tuesday market, where vendors sell everything from spices to seasonal fruits. It's the perfect place to sample local specialties and interact with neighborhood residents.`,
    author_name: "Can Özkan",
    district: "Kadıköy",
    created_at: "2024-11-25T12:15:00Z",
    likes: 92,
    likes_count: 92,
    images: []
  }
];

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
      // Use mock data to avoid circuit breaker errors
      const mockPost = mockBlogPosts.find(p => p.id === parseInt(id));
      
      if (mockPost) {
        setPost(mockPost);
        console.log('✅ BlogPost: Mock post loaded successfully:', mockPost?.title);
        
        // Track blog post view event
        trackBlogEvent('view_post', mockPost.title);
      } else {
        setError('Post not found. This post may have been removed or the link may be incorrect.');
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
      // Use mock data for related posts
      const related = mockBlogPosts
        .filter(p => p.id !== post.id && p.district === post.district)
        .slice(0, 3);
      
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
      // Simulate like functionality with mock data
      const newLikeCount = (post.likes || 0) + 1;
      
      // Update the post with new like count
      setPost(prev => ({
        ...prev,
        likes: newLikeCount,
        likes_count: newLikeCount
      }));
      
      setAlreadyLiked(true);
      
      // Track like event
      trackBlogEvent('like_post', post?.title || 'Unknown Post');
      
    } catch (err) {
      console.error('Failed to like post:', err);
      setLikeError('Failed to like post. Please try again.');
    } finally {
      setLikeLoading(false);
    }
  };

  const checkUserLikeStatus = async () => {
    if (!id) return;
    
    try {
      // Mock like status check - assume not liked initially
      setAlreadyLiked(false);
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
        <div className="pt-16 sm:pt-20 md:pt-24 px-2 sm:px-4 pb-8 text-center">
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
        <div className="pt-16 sm:pt-20 md:pt-24 px-2 sm:px-4 pb-8 text-center">
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
        <div className="pt-16 sm:pt-20 md:pt-24 px-2 sm:px-4 pb-8 text-center">
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
    <div className={`blog-page min-h-screen w-full transition-colors duration-200 ${
      darkMode ? 'dark bg-gray-900' : 'bg-gray-50'
    }`}>
      {/* Main Content with proper spacing */}
      <main className="pt-16 px-4 pb-8">
        <div className="max-w-6xl mx-auto">
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
