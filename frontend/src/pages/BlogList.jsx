import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
// import { fetchBlogPosts } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import { trackBlogEvent, trackSearch } from '../utils/analytics';
import '../App.css';

// Mock blog data for demo purposes
const mockBlogPosts = [
  {
    id: 1,
    title: "Hidden Gems in Sultanahmet: Beyond the Tourist Trail",
    content: "Discover the secret courtyards, ancient cisterns, and local eateries that most visitors miss in Istanbul's historic heart. From the peaceful Soğukçeşme Sokağı to the underground wonders of Şerefiye Cistern...",
    author_name: "Mehmet Yılmaz",
    district: "Sultanahmet",
    created_at: "2024-12-01T10:00:00Z",
    likes_count: 47,
    images: []
  },
  {
    id: 2,
    title: "Best Rooftop Views for Sunset in Galata",
    content: "Experience Istanbul's magic hour from the best rooftop terraces in Galata. From trendy bars to quiet cafes, here are the spots where locals go to watch the sun set over the Golden Horn...",
    author_name: "Ayşe Demir",
    district: "Galata",
    created_at: "2024-11-28T15:30:00Z",
    likes_count: 73,
    images: []
  },
  {
    id: 3,
    title: "Street Food Paradise: Kadıköy's Culinary Adventures",
    content: "Dive into the vibrant food scene of Kadıköy, where traditional Turkish flavors meet modern creativity. From the famous fish sandwich vendors to hidden meyhanes serving authentic mezze...",
    author_name: "Can Özkan",
    district: "Kadıköy",
    created_at: "2024-11-25T12:15:00Z",
    likes_count: 92,
    images: []
  },
  {
    id: 4,
    title: "Shopping Like a Local: Beyoğlu's Alternative Markets",
    content: "Skip the tourist shops and discover where Istanbulites really shop. From vintage treasures in Çukurcuma to artisan crafts in the backstreets of Galata, here's your insider guide...",
    author_name: "Zeynep Kaya",
    district: "Beyoğlu",
    created_at: "2024-11-20T09:45:00Z",
    likes_count: 64,
    images: []
  },
  {
    id: 5,
    title: "Early Morning Magic: Bosphorus at Dawn",
    content: "Join the fishermen and early risers for a completely different perspective of Istanbul. The city awakens slowly along the Bosphorus shores, offering peaceful moments and stunning photography opportunities...",
    author_name: "Emre Şahin",
    district: "Beşiktaş",
    created_at: "2024-11-18T06:00:00Z",
    likes_count: 38,
    images: []
  },
  {
    id: 6,
    title: "Traditional Hammam Experience: A First-Timer's Guide",
    content: "Nervous about trying a Turkish bath? This comprehensive guide covers everything from what to expect to proper etiquette, helping you enjoy this centuries-old Istanbul tradition with confidence...",
    author_name: "Fatma Arslan",
    district: "Fatih",
    created_at: "2024-11-15T14:20:00Z",
    likes_count: 156,
    images: []
  }
];

const BlogList = () => {
  console.log('🔧 BlogList: Component instance created');
  const { darkMode } = useTheme();
  const [posts, setPosts] = useState(mockBlogPosts);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalPosts, setTotalPosts] = useState(mockBlogPosts.length);

  const postsPerPage = 6;

  // Istanbul districts that are supported by the AI chatbot
  const chatbotDistricts = [
    'Istanbul (general)', 'Beyoğlu', 'Sultanahmet', 'Fatih', 'Kadıköy', 'Beşiktaş', 'Şişli', 
    'Üsküdar', 'Bakırköy', 'Galata', 'Taksim', 'Ortaköy', 'Karaköy', 'Eminönü'
  ];

  const loadPosts = async () => {
    console.log('🔄 BlogList: Loading posts with params:', { 
      currentPage, 
      postsPerPage, 
      searchTerm, 
      selectedDistrict
    });
    
    setLoading(true);
    setError(null);
    
    try {
      // Simulate loading delay for better UX
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Filter mock posts based on search and district
      let filteredPosts = mockBlogPosts;
      
      if (searchTerm) {
        filteredPosts = filteredPosts.filter(post => 
          post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          post.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
          post.author_name.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }
      
      if (selectedDistrict && selectedDistrict !== 'Istanbul (general)') {
        filteredPosts = filteredPosts.filter(post => 
          post.district === selectedDistrict
        );
      }
      
      // Pagination
      const startIndex = (currentPage - 1) * postsPerPage;
      const endIndex = startIndex + postsPerPage;
      const paginatedPosts = filteredPosts.slice(startIndex, endIndex);
      
      setPosts(paginatedPosts);
      setTotalPosts(filteredPosts.length);
      setTotalPages(Math.ceil(filteredPosts.length / postsPerPage));
      
      console.log('✅ BlogList: Posts loaded successfully', {
        total: filteredPosts.length,
        page: currentPage,
        showing: paginatedPosts.length
      });
      
    } catch (err) {
      console.error('❌ BlogList: Error loading posts:', err);
      setError('Failed to load blog posts. Please try again.');
      setPosts([]);
    } finally {
      setLoading(false);
      console.log('✅ BlogList: loadPosts completed');
    }
  };

  // Load posts when component mounts or search parameters change
  useEffect(() => {
    console.log('🏠 BlogList: Component mounted or params changed, loading posts');
    // Clear any existing state first
    setPosts([]);
    setError(null);
    setLoading(true);
    
    // Small delay to ensure state is reset
    const timer = setTimeout(() => {
      loadPosts();
    }, 10);
    
    return () => clearTimeout(timer);
  }, [currentPage, searchTerm, selectedDistrict]); // Dependencies for reloading

  // Also load on component mount
  useEffect(() => {
    console.log('🔄 BlogList: Initial mount, loading posts');
    loadPosts();
  }, []); // Run once on mount

  const handleSearch = (e) => {
    e.preventDefault();
    setCurrentPage(1);
    
    // Track blog search
    trackSearch(`blog: ${searchTerm} ${selectedDistrict}`.trim());
    trackBlogEvent('search', `${searchTerm} ${selectedDistrict}`.trim());
    
    loadPosts();
  };

  const resetFilters = () => {
    setSearchTerm('');
    setSelectedDistrict('');
    setCurrentPage(1);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const truncateText = (text, maxLength = 150) => {
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength) + '...';
  };

  if (loading && posts.length === 0) {
    return (
      <div className={`min-h-screen w-full pt-16 px-4 pb-8 transition-colors duration-200 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-center items-center py-20">
            <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${
              darkMode ? 'border-indigo-500' : 'border-indigo-600'
            }`}></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen w-full pt-16 px-4 pb-8 transition-colors duration-200 ${
      darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
    }`}>
      <div className="max-w-6xl mx-auto">
        {/* Search and Filters */}
        <div className={`mb-4 sm:mb-6 rounded-lg p-3 sm:p-4 transition-colors duration-200 relative mt-4 sm:mt-6 ${
          darkMode ? 'bg-gray-800' : 'bg-white shadow-lg border border-gray-200'
        }`}>
          <form onSubmit={handleSearch} className="mb-2 sm:mb-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:gap-3">
              <div className="flex-1">
                <label htmlFor="search-posts" className="sr-only">Search posts</label>
                <input
                  type="text"
                  id="search-posts"
                  name="search-posts"
                  placeholder="Search posts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={`w-full px-3 py-2 border rounded-lg focus:outline-none transition-colors duration-200 text-sm ${
                    darkMode 
                      ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500' 
                      : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
                  }`}
                  autoComplete="off"
                />
              </div>
              <button
                type="submit"
                className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 whitespace-nowrap text-sm ${
                  darkMode
                    ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                }`}
              >
                Search
              </button>
            </div>
          </form>

          <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-3">
            <div className="flex-1 min-w-0 sm:min-w-40 sm:max-w-48">
              <label htmlFor="district-filter" className="sr-only">Filter by district</label>
              <select
                id="district-filter"
                name="district-filter"
                value={selectedDistrict}
                onChange={(e) => {
                  setSelectedDistrict(e.target.value);
                  setCurrentPage(1);
                }}
                className={`w-full px-3 py-2 border rounded-lg focus:outline-none transition-colors duration-200 text-sm ${
                  darkMode 
                    ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500'
                    : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
                }`}
                autoComplete="address-level2"
              >
                <option value="">All Districts</option>
                {chatbotDistricts.map((district) => (
                  <option key={district} value={district}>{district}</option>
                ))}
              </select>
            </div>

            {(searchTerm || selectedDistrict) && (
              <button
                type="button"
                onClick={resetFilters}
                className={`px-3 py-2 rounded-lg font-medium transition-colors duration-200 whitespace-nowrap text-sm ${
                  darkMode
                    ? 'bg-gray-600 hover:bg-gray-700 text-white'
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                }`}
              >
                Clear Filters
              </button>
            )}
          </div>
        </div>

        {/* Share Your Story Button */}
        <div className="text-center mb-4 sm:mb-6">
          <Link
            to="/blog/new"
            className="inline-flex items-center px-4 sm:px-6 py-2 sm:py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition-colors duration-200 text-sm sm:text-base shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Share Your Story
          </Link>
        </div>

        {/* Error State */}
        {error && (
          <div className={`mb-8 p-4 rounded-lg border transition-colors duration-200 ${
            darkMode 
              ? 'bg-red-900/20 border-red-500/20'
              : 'bg-red-50 border-red-200'
          }`}>
            <p className={`mb-2 ${darkMode ? 'text-red-400' : 'text-red-700'}`}>{error}</p>
            <button
              onClick={loadPosts}
              className={`px-4 py-2 rounded text-sm transition-colors duration-200 ${
                darkMode
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-red-600 hover:bg-red-700 text-white'
              }`}
            >
              Try Again
            </button>
          </div>
        )}

        {/* Blog Posts Grid */}
        <div style={{minHeight: '400px'}}> {/* Debug wrapper */}
        {posts.length === 0 && !loading && !error ? (
          <div className="text-center py-20">
            <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h3 className={`text-xl font-semibold mb-2 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>No posts found</h3>
            <p className={`mb-4 transition-colors duration-200 ${
              darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>Be the first to share your Istanbul experience!</p>
            <Link
              to="/blog/new"
              className="inline-flex items-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition-colors duration-200"
            >
              Write a Post
            </Link>
          </div>
        ) : (
          <>
            <div className="text-center mb-4">
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {totalPosts === 0 ? 'No posts found' : 
                 totalPosts === 1 ? '1 post total' : 
                 `${totalPosts} posts total`}
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4 sm:gap-6 mb-4 sm:mb-6">
            {posts.map((post) => (            <article
              key={post.id}
              className={`rounded-xl overflow-hidden transition-all duration-300 hover:scale-105 ${
                darkMode
                  ? 'bg-gray-800 hover:bg-gray-750 shadow-xl'
                  : 'bg-white hover:bg-gray-50 shadow-lg hover:shadow-2xl border border-gray-100'
              }`}
            >
                {post.images && post.images.length > 0 && (
                  <div className="aspect-video overflow-hidden">
                    <img
                      src={`${import.meta.env.VITE_API_URL?.replace(/\/ai\/?$/, '') || 'http://localhost:8001'}${post.images[0].url}`}
                      alt={post.images[0].alt_text || post.title}
                      className="w-full h-full object-cover hover:scale-110 transition-transform duration-500"
                    />
                  </div>
                )}
                
                <div className="p-5 sm:p-7">
                  {/* Author Info */}
                  {post.author_name && (
                    <div className="flex items-center mb-4">
                      {post.author_photo ? (
                        <img 
                          src={post.author_photo} 
                          alt={post.author_name}
                          className="w-8 h-8 sm:w-10 sm:h-10 rounded-full mr-3 sm:mr-4 object-cover ring-2 ring-indigo-500/20"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                      ) : null}
                      <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full mr-3 sm:mr-4 bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-content text-white text-sm sm:text-base font-bold shadow-lg ${post.author_photo ? 'hidden' : ''}`}>
                        {post.author_name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className={`text-sm sm:text-base font-semibold transition-colors duration-200 ${
                          darkMode ? 'text-gray-200' : 'text-gray-800'
                        }`}>{post.author_name}</p>
                      </div>
                    </div>
                  )}

                  <h2 className={`text-xl sm:text-2xl font-bold mb-3 leading-tight transition-colors duration-200 ${
                    darkMode 
                      ? 'text-white hover:text-indigo-300' 
                      : 'text-gray-900 hover:text-indigo-600'
                  }`}>
                    <Link to={`/blog/${post.id}`} className="hover:underline">
                      {post.title}
                    </Link>
                  </h2>
                  
                  <p className={`mb-4 sm:mb-5 text-base sm:text-lg leading-relaxed transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {truncateText(post.content, window.innerWidth < 640 ? 120 : 180)}
                  </p>
                  
                  <div className={`flex items-center justify-between text-sm sm:text-base transition-colors duration-200 ${
                    darkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    <div className="flex items-center">
                      <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      <span className="truncate font-medium">{post.district || 'Istanbul'}</span>
                    </div>
                    
                    <div className="flex items-center gap-3 sm:gap-5 flex-shrink-0">
                      <div className="flex items-center">
                        <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-1 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        <span className="font-medium">{post.likes_count || 0}</span>
                      </div>
                      <div className="flex items-center hidden sm:flex">
                        <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-1 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <span className="font-medium">{formatDate(post.created_at)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
          </>
        )}
        </div> {/* End debug wrapper */}
      </div>
    </div>
  );
};

export default BlogList;
