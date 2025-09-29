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
import { formatLikesCount, getNumberTextSize } from '../utils/formatNumbers';
import Comments from '../components/Comments';
import '../App.css';

// Mock blog data to match BlogList - expanded with full content
const mockBlogPosts = [
  {
    id: 1,
    title: "Hidden Gems in Sultanahmet: Beyond the Tourist Trail",
    content: `Discover the secret courtyards, ancient cisterns, and local eateries that most visitors miss in Istanbul's historic heart. From the peaceful Soğukçeşme Sokağı to the underground wonders of Şerefiye Cistern.

Istanbul's Sultanahmet district is much more than the Blue Mosque and Hagia Sophia. While these iconic landmarks deserve their fame, the ***real magic*** happens in the quiet corners and hidden passages that most tourists never discover.

**Secret Courtyards and Hidden Gardens**

Start your exploration at *Soğukçeşme Sokağı*, a cobblestone street lined with restored Ottoman houses. The narrow alley connects Hagia Sophia to Topkapi Palace, but **few visitors** take the time to appreciate its authentic atmosphere. Early morning is the ***best time*** to visit – the golden light filtering through the old windows creates an almost mystical ambiance.

**Underground Wonders**

While everyone knows about the Basilica Cistern, the Şerefiye Cistern offers an equally impressive but far less crowded experience. Built in the 4th century, this underground marvel features beautiful lighting and interactive exhibits that bring Byzantine history to life.

**Local Eating Spots**

Skip the tourist restaurants around the main attractions and head to ***Pandeli***, a historic Ottoman restaurant hidden above the Spice Bazaar. The hand-painted tiles and traditional dishes haven't changed in over a century.

For a more casual experience, find the small *çay bahçesi* (tea gardens) tucked behind the neighborhood's residential streets. These **local gathering spots** offer the best Turkish tea and a chance to observe daily life in old Istanbul.

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
    images: [
      {
        url: "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?w=800&h=600&fit=crop",
        alt_text: "Soğukçeşme Sokağı cobblestone street in Sultanahmet",
        caption: "The peaceful Soğukçeşme Sokağı street connecting Hagia Sophia to Topkapi Palace"
      },
      {
        url: "https://images.unsplash.com/photo-1541432901042-2d8bd64b4a9b?w=800&h=600&fit=crop",
        alt_text: "Interior of historic Şerefiye Cistern",
        caption: "The mysterious underground world of Şerefiye Cistern with its beautiful lighting"
      },
      {
        url: "https://images.unsplash.com/photo-1564069114553-7215e1ff1890?w=800&h=600&fit=crop",
        alt_text: "Traditional Turkish tea garden",
        caption: "A traditional çay bahçesi (tea garden) hidden in Sultanahmet's backstreets"
      }
    ]
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
    images: [
      {
        url: "https://images.unsplash.com/photo-1520637836862-4d197d17c72a?w=800&h=600&fit=crop",
        alt_text: "Galata Tower and surrounding rooftops at sunset",
        caption: "The iconic Galata Tower surrounded by historic rooftops during golden hour"
      },
      {
        url: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=600&fit=crop",
        alt_text: "Rooftop terrace view of Golden Horn",
        caption: "Panoramic view of the Golden Horn from a Galata rooftop terrace"
      }
    ]
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
    images: [
      {
        url: "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=800&h=600&fit=crop",
        alt_text: "Fresh fish sandwich (balık ekmek) being prepared",
        caption: "Traditional balık ekmek being prepared by fishermen at Kadıköy ferry terminal"
      },
      {
        url: "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&h=600&fit=crop",
        alt_text: "Colorful spices at Kadıköy market",
        caption: "Vibrant spices and local specialties at the famous Tuesday market in Kadıköy"
      },
      {
        url: "https://images.unsplash.com/photo-1574126154517-d1e0d89ef734?w=800&h=600&fit=crop",
        alt_text: "Traditional meyhane interior with mezze",
        caption: "A traditional meyhane with authentic mezze selection in Kadıköy's backstreets"
      }
    ]
  },
  {
    id: 4,
    title: "Text Formatting Guide: Writing Beautiful Blog Posts",
    content: `Learn how to use different text formatting options to make your blog posts more engaging and readable. This guide demonstrates all available formatting features.

***Text formatting*** is essential for creating engaging and readable blog posts. Here's how you can use different styles to enhance your content and guide your readers through your story.

**Basic Text Formatting**

You can use *italic text* to emphasize important words or phrases. For stronger emphasis, use **bold text** to make key points stand out. When you need maximum impact, combine both styles for ***bold italic text***.

**Creating Visual Hierarchy**

Use headings to organize your content and make it scannable:

**Main Section Title**
Start each major section with a bold heading like this.

**Subsection Title** 
Break down complex topics into smaller, digestible parts.

**Practical Examples**

Here are some real-world examples of effective formatting:

- Use *italics* for foreign words like *çay* (tea) or *meze* (appetizers)
- Make **important warnings** or **key takeaways** bold
- Combine styles for ***absolutely critical information***
- Keep regular text unformatted for easy reading

**Best Practices for Readability**

Remember that formatting should enhance, not overwhelm your content:

- Use formatting sparingly – ***too much emphasis*** loses its impact
- Be consistent with your *formatting choices* throughout the post
- **Bold text** works best for short phrases and key terms
- Save *italics* for subtle emphasis and foreign terms

**When to Use Each Style**

Choose your formatting based on the message:

- *Italic*: Subtle emphasis, book titles, foreign words, thoughts
- **Bold**: Strong emphasis, important facts, warnings, key terms  
- ***Bold Italic***: Maximum emphasis, critical information, standout quotes

**Conclusion**

Effective text formatting makes your blog posts more professional and easier to read. Experiment with these styles to find what works best for your writing style and audience.`,
    author_name: "Editorial Team",
    district: "Digital",
    created_at: "2024-12-15T14:30:00Z",
    likes: 156,
    likes_count: 156,
    images: [
      {
        url: "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=800&h=600&fit=crop",
        alt_text: "Writing and text formatting on computer screen",
        caption: "Modern blogging requires effective text formatting for better readability"
      },
      {
        url: "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=800&h=600&fit=crop",
        alt_text: "Typography and text design examples",
        caption: "Good typography enhances the reading experience and engagement"
      }
    ]
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
  const [likesCount, setLikesCount] = useState(0); // Separate state for likes count
  const [selectedImage, setSelectedImage] = useState(null);
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);

  // Generate or get user identifier for like functionality
  const getUserIdentifier = () => {
    let userId = localStorage.getItem('blog_user_id');
    if (!userId) {
      userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('blog_user_id', userId);
    }
    return userId;
  };

  useEffect(() => {
    console.log('🔄 BlogPost: Loading post with ID:', id);
    loadPost();
    if (id) {
      console.log('� BlogPost: Checking like status for post:', id);
      checkUserLikeStatus();
    }
  }, [id]);

  useEffect(() => {
    if (post) {
      console.log('� BlogPost: Loading related posts for:', post.title);
      loadRelatedPosts();
    }
  }, [post?.id]); // Only depend on post.id, not the entire post object

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
        setLikesCount(postData.likes || postData.likes_count || 0); // Set initial likes count
        console.log('✅ BlogPost: Post loaded successfully from API:', postData?.title);
        trackBlogEvent('view_post', postData.title);
        setLoading(false); // Set loading to false when API succeeds
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
        setLikesCount(mockPost.likes || mockPost.likes_count || 0); // Set initial likes count
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
      // Try real API first
      const relatedData = await fetchRelatedPosts(post.id, 3);
      const related = relatedData.related_posts || [];
      setRelatedPosts(related);
      console.log('✅ BlogPost: Related posts loaded from API:', related?.length || 0, 'posts');
    } catch (err) {
      console.warn('⚠️ BlogPost: API failed for related posts, using mock data:', err);
      
      // Fallback to mock data - get other posts excluding the current one
      try {
        const currentPostId = parseInt(post.id);
        const otherPosts = mockBlogPosts
          .filter(mockPost => mockPost.id !== currentPostId)
          .slice(0, 3); // Get up to 3 related posts
        
        setRelatedPosts(otherPosts);
        console.log('✅ BlogPost: Related posts loaded from mock data:', otherPosts.length, 'posts');
      } catch (mockErr) {
        console.error('❌ BlogPost: Failed to load related posts from mock data:', mockErr);
        setRelatedPosts([]);
      }
    } finally {
      setRelatedLoading(false);
    }
  };

  const handleLike = async () => {
    if (!post?.id || likeLoading) return;
    
    console.log('⭐ BlogPost: Liking post:', post.title);
    setLikeLoading(true);
    setLikeError(null);
    
    try {
      const userIdentifier = getUserIdentifier();
      const likeResult = await likeBlogPost(post.id, userIdentifier);
      
      // Update the likes count and liked status from the response
      setLikesCount(likeResult.likes_count || likesCount);
      setAlreadyLiked(likeResult.isLiked || false);
      
      trackBlogEvent('like_post', post?.title || 'Unknown Post');
      console.log('✅ BlogPost: Post like updated - count:', likeResult.likes_count, 'isLiked:', likeResult.isLiked);
      
    } catch (err) {
      console.error('❌ BlogPost: Failed to like post:', err);
      setLikeError('Failed to like post. Please try again.');
      
      // Don't update local state on API failure - show error instead
      trackBlogEvent('like_error', post?.title || 'Unknown Post');
    } finally {
      setLikeLoading(false);
    }
  };

  const checkUserLikeStatus = async () => {
    if (!id) return;
    
    try {
      console.log('🔍 BlogPost: Checking like status for post:', id);
      const userIdentifier = getUserIdentifier();
      const likeStatus = await checkLikeStatus(id, userIdentifier);
      
      setAlreadyLiked(likeStatus.isLiked || false);
      
      // Store the likes count separately, don't update post object
      if (likeStatus.likes !== undefined) {
        setLikesCount(likeStatus.likes);
      }
      
      console.log('✅ BlogPost: Like status checked - isLiked:', likeStatus.isLiked, 'likes:', likeStatus.likes);
    } catch (err) {
      console.warn('⚠️ BlogPost: Failed to check like status:', err);
      setAlreadyLiked(false);
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
    const lines = content.split('\n');
    const elements = [];
    let currentList = [];
    
    lines.forEach((line, index) => {
      const trimmedLine = line.trim();
      
      // Skip empty lines
      if (!trimmedLine) {
        // If we were building a list, close it
        if (currentList.length > 0) {
          elements.push(
            <ul key={`list-${index}`} className="mb-4 ml-4 list-disc">
              {currentList}
            </ul>
          );
          currentList = [];
        }
        return;
      }
      
      // Handle different heading levels
      if (trimmedLine.startsWith('# ')) {
        if (currentList.length > 0) {
          elements.push(<ul key={`list-${index}`} className="mb-4 ml-4 list-disc">{currentList}</ul>);
          currentList = [];
        }
        const text = trimmedLine.slice(2);
        elements.push(
          <h1 key={index} className="text-3xl font-bold mb-6 mt-8 text-white" dangerouslySetInnerHTML={{ __html: formatInlineText(text) }} />
        );
      } else if (trimmedLine.startsWith('## ')) {
        if (currentList.length > 0) {
          elements.push(<ul key={`list-${index}`} className="mb-4 ml-4 list-disc">{currentList}</ul>);
          currentList = [];
        }
        const text = trimmedLine.slice(3);
        elements.push(
          <h2 key={index} className="text-2xl font-bold mb-5 mt-7 text-white" dangerouslySetInnerHTML={{ __html: formatInlineText(text) }} />
        );
      } else if (trimmedLine.startsWith('### ')) {
        if (currentList.length > 0) {
          elements.push(<ul key={`list-${index}`} className="mb-4 ml-4 list-disc">{currentList}</ul>);
          currentList = [];
        }
        const text = trimmedLine.slice(4);
        elements.push(
          <h3 key={index} className="text-xl font-bold mb-4 mt-6 text-white" dangerouslySetInnerHTML={{ __html: formatInlineText(text) }} />
        );
      }
      // Handle headers (lines that start and end with ** but no ###)
      else if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**') && !trimmedLine.includes('***') && !trimmedLine.startsWith('###')) {
        if (currentList.length > 0) {
          elements.push(<ul key={`list-${index}`} className="mb-4 ml-4 list-disc">{currentList}</ul>);
          currentList = [];
        }
        const text = trimmedLine.slice(2, -2);
        elements.push(
          <h3 key={index} className="text-xl font-bold mb-4 mt-6 text-white" dangerouslySetInnerHTML={{ __html: formatInlineText(text) }} />
        );
      }
      // Handle list items (- or • or *)
      else if (trimmedLine.match(/^[-•*]\s/)) {
        const listText = trimmedLine.replace(/^[-•*]\s/, '');
        currentList.push(
          <li 
            key={`li-${index}`} 
            className="mb-2 text-gray-200"
            dangerouslySetInnerHTML={{ __html: formatInlineText(listText) }}
          />
        );
      }
      // Handle numbered lists
      else if (trimmedLine.match(/^\d+\.\s/)) {
        if (currentList.length > 0) {
          elements.push(<ul key={`list-${index}`} className="mb-4 ml-4 list-disc">{currentList}</ul>);
          currentList = [];
        }
        const listText = trimmedLine.replace(/^\d+\.\s/, '');
        elements.push(
          <ol key={index} className="mb-4 ml-4 list-decimal">
            <li className="mb-2 text-gray-200" dangerouslySetInnerHTML={{ __html: formatInlineText(listText) }} />
          </ol>
        );
      }
      // Regular paragraphs
      else {
        if (currentList.length > 0) {
          elements.push(<ul key={`list-${index}`} className="mb-4 ml-4 list-disc">{currentList}</ul>);
          currentList = [];
        }
        elements.push(
          <p 
            key={index} 
            className="mb-4 leading-relaxed text-gray-200"
            dangerouslySetInnerHTML={{ __html: formatInlineText(trimmedLine) }}
          />
        );
      }
    });
    
    // Close any remaining list
    if (currentList.length > 0) {
      elements.push(
        <ul key="final-list" className="mb-4 ml-4 list-disc">
          {currentList}
        </ul>
      );
    }
    
    return elements;
  };

  // Helper function to format inline text with bold, italic, and bold-italic
  const formatInlineText = (text) => {
    if (!text) return text;
    
    let result = text;
    
    // 1. Bold italic (***text***)
    result = result.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    
    // 2. Bold (**text**)
    result = result.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 3. Italic (*text*)
    result = result.replace(/\*([^*]+?)\*/g, '<em>$1</em>');
    
    // 4. Underscore italic (_text_)
    result = result.replace(/\b_([^_]+?)_\b/g, '<em>$1</em>');
    
    // 5. Code (`text`)
    result = result.replace(/`([^`]+?)`/g, '<code class="bg-gray-700 text-yellow-300 px-1 py-0.5 rounded text-sm">$1</code>');
    
    // 6. Links [text](url)
    result = result.replace(/\[([^\]]+?)\]\(([^)]+?)\)/g, '<a href="$2" class="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer">$1</a>');
    
    return result;
  };

  const openImageModal = (image) => {
    setSelectedImage(image);
    setIsImageModalOpen(true);
  };

  const closeImageModal = () => {
    setSelectedImage(null);
    setIsImageModalOpen(false);
  };

  // Close modal on escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        closeImageModal();
      }
    };

    if (isImageModalOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isImageModalOpen]);

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

            {/* Featured Image */}
            {post.images && post.images.length > 0 && (
              <div className="blog-featured-image mb-8 relative">
                <img
                  src={post.images[0].url}
                  alt={post.images[0].alt_text || post.title}
                  className="w-full h-64 md:h-80 lg:h-96 object-cover rounded-lg shadow-lg cursor-pointer"
                  onClick={() => openImageModal(post.images[0])}
                />
                {post.images.length > 1 && (
                  <div className="photo-count-indicator">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                    </svg>
                    {post.images.length} photos
                  </div>
                )}
                {post.images[0].caption && (
                  <p className="text-sm text-gray-400 italic mt-2 text-center">
                    {post.images[0].caption}
                  </p>
                )}
              </div>
            )}

            {/* Blog Post Content */}
            <div className="blog-post-content">
              {formatContent(post.content)}
            </div>

            {/* Photo Gallery */}
            {post.images && post.images.length > 1 && (
              <div className="blog-photo-gallery mt-8">
                <h3 className="text-xl font-bold mb-4 text-white">Photo Gallery</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {post.images.slice(1).map((image, index) => (
                    <div key={index} className="blog-gallery-item">
                      <img
                        src={image.url}
                        alt={image.alt_text || `Gallery image ${index + 2}`}
                        className="w-full h-48 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 cursor-pointer"
                        onClick={() => openImageModal(image)}
                      />
                      {image.caption && (
                        <p className="text-sm text-gray-400 italic mt-2">
                          {image.caption}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

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
                  {likeLoading ? 'Liking...' : alreadyLiked ? 'Liked!' : `${formatLikesCount(likesCount)} likes`}
                </button>
              </div>
              
              {likeError && (
                <p className="mt-2 text-sm text-red-400">{likeError}</p>
              )}
            </footer>
          </article>

          {/* Related Posts */}
          {relatedPosts.length > 0 && (
            <section className="mt-12">
              <h2 className="text-2xl font-bold mb-6 text-white">
                Related Posts
              </h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {relatedPosts.map(relatedPost => (
                  <Link
                    key={relatedPost.id}
                    to={`/blog/${relatedPost.id}`}
                    className="block rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-all duration-200 hover:transform hover:scale-105 bg-gray-800/50 border border-gray-700"
                  >
                    {relatedPost.images && relatedPost.images.length > 0 && (
                      <img
                        src={relatedPost.images[0].url}
                        alt={relatedPost.images[0].alt_text || relatedPost.title}
                        className="w-full h-48 object-cover"
                      />
                    )}
                    <div className="p-4">
                      <h3 className="font-semibold mb-2 line-clamp-2 text-gray-100">
                        {relatedPost.title}
                      </h3>
                      <p className="text-sm line-clamp-3 text-gray-300 mb-3">
                        {relatedPost.content.split('\n')[0].substring(0, 120)}...
                      </p>
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>📍 {relatedPost.district}</span>
                        <span>❤️ {formatLikesCount(relatedPost.likes || relatedPost.likes_count || 0)}</span>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {/* Comments Section */}
          <section className="mt-12">
            <Comments postId={id} />
          </section>
        </div>
      </main>

      {/* Image Modal */}
      {isImageModalOpen && selectedImage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50 p-4"
          onClick={closeImageModal}
        >
          <div className="relative max-w-4xl max-h-full">
            <button
              onClick={closeImageModal}
              className="absolute top-4 right-4 text-white text-2xl hover:text-gray-300 z-10 bg-black bg-opacity-50 rounded-full w-10 h-10 flex items-center justify-center"
            >
              ×
            </button>
            <img
              src={selectedImage.url}
              alt={selectedImage.alt_text}
              className="max-w-full max-h-full object-contain rounded-lg"
              onClick={(e) => e.stopPropagation()}
            />
            {selectedImage.caption && (
              <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white p-4 rounded-b-lg">
                <p className="text-center text-sm md:text-base">
                  {selectedImage.caption}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default BlogPost;
