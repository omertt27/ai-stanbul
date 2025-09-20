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

// Mock blog data to match BlogList - expanded with full content
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

For a more casual experience, find the small çay bahçesi (tea gardens) tucked behind the neighborhood's residential streets. These local gathering spots offer the best Turkish tea and a chance to observe daily life in old Istanbul.

**Practical Tips**

- Visit early morning (7-9 AM) for the best photo opportunities and fewer crowds
- Wear comfortable walking shoes for cobblestone streets
- Learn a few Turkish phrases – locals appreciate the effort
- Always ask permission before photographing people`,
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

**Galata Tower Area**

Skip the crowded tower itself and explore the surrounding neighborhood. Several cafes and restaurants on Serdar-ı Ekrem Street have terraces with equally stunning views at a fraction of the cost.

**Photography Tips**

The best light occurs about 30 minutes before sunset. Position yourself facing southwest for the classic silhouette shots of the historical peninsula. Don't forget to capture the moment when the call to prayer echoes across the water – it's pure Istanbul magic.

**Best Times to Visit**

- Sunset times vary by season, so check local times
- Weekdays are less crowded than weekends
- Arrive 45 minutes early to secure the best spots
- Bring a light jacket – it can get breezy on rooftops`,
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
      // Try real API first
      const response = await fetchBlogPost(id);
      const postData = response.post || response;
      
      if (postData) {
        setPost(postData);
        console.log('✅ BlogPost: Post loaded successfully from API:', postData?.title);
        trackBlogEvent('view_post', postData.title);
        return;
      }
    } catch (err) {
      console.warn('⚠️ BlogPost: API failed, trying mock data:', err);
    }
    
    // Fallback to mock data
    try {
      const postId = parseInt(id);
      const mockPost = mockBlogPosts.find(post => post.id === postId);
      
      if (mockPost) {
        setPost(mockPost);
        console.log('✅ BlogPost: Post loaded successfully from mock data:', mockPost.title);
        trackBlogEvent('view_post', mockPost.title);
      } else {
        setError('Post not found. This post may have been removed or the link may be incorrect.');
      }
    } catch (err) {
      console.error('❌ BlogPost: Failed to load post from mock data:', err);
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
      const relatedData = await fetchRelatedPosts(post.id, 3);
      const related = relatedData.related_posts || [];
      setRelatedPosts(related);
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
    
    console.log('⭐ BlogPost: Liking post:', post.title);
    setLikeLoading(true);
    setLikeError(null);
    
    try {
      const likeResult = await likeBlogPost(post.id);
      setPost(prev => ({
        ...prev,
        likes: likeResult.likes || (prev.likes || 0) + 1
      }));
      setAlreadyLiked(true);
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
      setAlreadyLiked(false);
    } catch (err) {
      console.error('Failed to check like status:', err);
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
    const paragraphs = content.split('\n').filter(line => line.trim());
    
    return paragraphs.map((paragraph, index) => {
      if (paragraph.startsWith('**') && paragraph.endsWith('**')) {
        const text = paragraph.slice(2, -2);
        return (
          <h3 key={index} className="text-xl font-bold mb-4 mt-6 text-white">
            {text}
          </h3>
        );
      }
      
      if (paragraph.startsWith('- ')) {
        return (
          <li key={index} className="mb-2 ml-4">
            {paragraph.slice(2)}
          </li>
        );
      }
      
      const formattedText = paragraph.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      
      return (
        <p 
          key={index} 
          className="mb-4 leading-relaxed"
          dangerouslySetInnerHTML={{ __html: formattedText }}
        />
      );
    });
  };

  if (loading) {
    return (
      <div className="blog-page min-h-screen flex items-center justify-center">
        <div className="pt-16 sm:pt-20 md:pt-24 px-2 sm:px-4 pb-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <h3 className="text-lg font-medium text-gray-200">Loading post...</h3>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="blog-page min-h-screen flex items-center justify-center">
        <div className="pt-16 sm:pt-20 md:pt-24 px-2 sm:px-4 pb-8 text-center">
          <h2 className="text-2xl font-bold mb-4 text-gray-300">Error loading post</h2>
          <p className="mb-6 text-gray-400">{error}</p>
          <Link to="/blog" className="blog-back-link">
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
      <div className="blog-page min-h-screen flex items-center justify-center">
        <div className="pt-16 sm:pt-20 md:pt-24 px-2 sm:px-4 pb-8 text-center">
          <h2 className="text-2xl font-bold mb-4 text-gray-300">Post not found</h2>
          <Link to="/blog" className="blog-back-link">
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
    <div className="blog-page">
      <main className="pt-20 px-4 pb-8">
        <div className="max-w-4xl mx-auto">
          {/* Navigation Breadcrumb */}
          <div className="mb-6">
            <Link to="/blog" className="blog-back-link">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Blog
            </Link>
          </div>

          {/* Blog Post Container */}
          <article className="blog-post-container">
            {/* Blog Post Header */}
            <header className="blog-post-header">
              <h1 className="blog-post-title">{post.title}</h1>
              
              <div className="blog-post-meta">
                <div className="blog-post-author">
                  <div className="blog-post-author-avatar">
                    {(post.author_name || 'Unknown').charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <div className="text-white font-medium">{post.author_name || 'Unknown Author'}</div>
                    <div className="text-sm text-gray-400">{formatDate(post.created_at)}</div>
                  </div>
                </div>
                
                {post.district && (
                  <div className="blog-post-district">📍 {post.district}</div>
                )}
              </div>
            </header>

            {/* Blog Post Content */}
            <div className="blog-post-content">
              {formatContent(post.content)}
            </div>

            {/* Blog Post Footer */}
            <footer className="blog-post-footer">
              <div className="blog-post-actions">
                <button
                  onClick={handleLike}
                  disabled={likeLoading || alreadyLiked}
                  className="blog-like-button"
                >
                  <svg className="w-5 h-5" fill={alreadyLiked ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                  {likeLoading ? 'Liking...' : alreadyLiked ? 'Liked!' : `${post.likes || post.likes_count || 0} likes`}
                </button>
              </div>
              
              {likeError && (
                <p className="mt-2 text-sm text-red-400">{likeError}</p>
              )}
            </footer>
          </article>

          {/* Comments Section */}
          <section className="mt-12">
            <Comments postId={id} />
          </section>
        </div>
      </main>
    </div>
  );
};

export default BlogPost;
