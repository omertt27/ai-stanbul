import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fetchBlogPosts } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import { trackBlogEvent, trackSearch } from '../utils/analytics';
import WeatherAwareBlogRecommendations from '../components/WeatherAwareBlogRecommendations';
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
  },
  {
    id: 7,
    title: "Art & Culture in Karaköy: Where Creativity Meets History",
    content: "Explore the vibrant art scene in Karaköy district, where contemporary galleries blend with historic Ottoman architecture. From the Istanbul Modern to hidden artist studios...",
    author_name: "Selin Yücel",
    district: "Karaköy",
    created_at: "2024-11-12T16:45:00Z",
    likes_count: 82,
    images: []
  },
  {
    id: 8,
    title: "Ferry Adventures: Island Hopping from Eminönü",
    content: "Take the scenic route to Büyükada and Heybeliada. This guide covers the best ferry schedules, what to see on each island, and local seafood restaurants you shouldn't miss...",
    author_name: "Burak Şen",
    district: "Eminönü",
    created_at: "2024-11-10T11:30:00Z",
    likes_count: 95,
    images: []
  },
  {
    id: 9,
    title: "Night Markets and Street Life in Şişli",
    content: "When the sun goes down, Şişli comes alive with bustling night markets, late-night eateries, and vibrant street culture. Here's your guide to experiencing Istanbul after dark...",
    author_name: "Deniz Aktaş",
    district: "Şişli",
    created_at: "2024-11-08T20:15:00Z",
    likes_count: 67,
    images: []
  },
  {
    id: 10,
    title: "Historic Churches and Mosques: A Spiritual Journey",
    content: "Discover the religious heritage of Istanbul through its magnificent churches and mosques. From Hagia Sophia to Chora Church, explore the spiritual heart of the city...",
    author_name: "Prof. Ahmet Güler",
    district: "Fatih",
    created_at: "2024-11-05T13:00:00Z",
    likes_count: 134,
    images: []
  },
  {
    id: 11,
    title: "Coffee Culture: From Traditional to Third Wave",
    content: "Journey through Istanbul's evolving coffee scene, from traditional Turkish coffee ceremonies to modern specialty coffee shops. Discover the best cafes in every district...",
    author_name: "Elif Özdemir",
    district: "Beyoğlu",
    created_at: "2024-11-02T08:30:00Z",
    likes_count: 113,
    images: []
  },
  {
    id: 12,
    title: "Weekend Escape: Üsküdar's Asian Side Charm",
    content: "Cross the Bosphorus to discover Üsküdar's peaceful atmosphere, historic sites, and stunning views of the European side. Perfect for a relaxing weekend exploration...",
    author_name: "Murat Kaya",
    district: "Üsküdar",
    created_at: "2024-10-30T15:45:00Z",
    likes_count: 78,
    images: []
  }
];

const BlogList = () => {
  console.log('🔧 BlogList: Component instance created');
  const { darkMode } = useTheme();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalPosts, setTotalPosts] = useState(0);

  const postsPerPage = 8; // Show 8 posts per page

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
      // Use mock data with pagination (8 posts per page)
      let filteredPosts = mockBlogPosts;
      
      // Apply search filter
      if (searchTerm) {
        filteredPosts = filteredPosts.filter(post => 
          post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          post.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (post.author_name && post.author_name.toLowerCase().includes(searchTerm.toLowerCase()))
        );
      }
      
      // Apply district filter
      if (selectedDistrict && selectedDistrict !== 'Istanbul (general)') {
        filteredPosts = filteredPosts.filter(post => 
          post.district && post.district.toLowerCase() === selectedDistrict.toLowerCase()
        );
      }
      
      // Calculate pagination
      const startIndex = (currentPage - 1) * postsPerPage;
      const endIndex = startIndex + postsPerPage;
      const paginatedPosts = filteredPosts.slice(startIndex, endIndex);
      
      setPosts(paginatedPosts);
      setTotalPosts(filteredPosts.length);
      setTotalPages(Math.ceil(filteredPosts.length / postsPerPage));
      
      console.log('✅ BlogList: Posts loaded successfully', {
        total: filteredPosts.length,
        page: currentPage,
        showing: paginatedPosts.length,
        totalPages: Math.ceil(filteredPosts.length / postsPerPage)
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
    
    // Add debounce delay to prevent search on every keystroke
    const timer = setTimeout(() => {
      loadPosts();
    }, 500);
    
    return () => clearTimeout(timer);
  }, [currentPage, searchTerm, selectedDistrict]); // Added currentPage back for pagination

  const handleSearch = (e) => {
    e.preventDefault();
    setCurrentPage(1); // Reset to first page when searching
    
    // Track blog search
    trackSearch(`blog: ${searchTerm} ${selectedDistrict}`.trim());
    trackBlogEvent('search', `${searchTerm} ${selectedDistrict}`.trim());
    
    loadPosts();
  };

  const resetFilters = () => {
    setSearchTerm('');
    setSelectedDistrict('');
    setCurrentPage(1); // Reset to first page when clearing filters
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
      <div className={`min-h-screen w-full pt-24 px-4 pb-8 transition-colors duration-300 ${
        darkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50'
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
    <div className="min-h-screen w-full transition-colors duration-300 bg-gray-900" style={{ marginTop: '0px', paddingLeft: '1rem', paddingRight: '1rem', paddingBottom: '2rem' }}>
      <div className="max-w-6xl mx-auto">

      {/* Scrollable Content */}
      <div className="pt-4 pb-8">
        {/* Header Section */}
        <div className="text-center py-8 sm:py-12">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-4 pt-28 transition-colors duration-200 text-white">
            Istanbul Stories
          </h1>
          <p className="text-lg sm:text-xl mb-6 max-w-3xl mx-auto transition-colors duration-200 text-gray-300">
            Discover authentic experiences and hidden gems through the eyes of locals and travelers
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <span className="text-sm font-medium transition-colors duration-200 text-gray-400">
              {totalPosts} stories shared • Join the community
            </span>
          </div>
        </div>

        {/* Recent/Featured Posts Section */}
        {posts.length > 0 && !searchTerm && !selectedDistrict && (
          <div className="mb-8">
            <h2 className="text-2xl sm:text-3xl font-bold mb-6 transition-colors duration-200 text-white">
              Recent Stories
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {posts.slice(0, 2).map((post) => (
                <article
                  key={`featured-${post.id}`}
                  className={`rounded-2xl overflow-hidden transition-all duration-300 hover:scale-105 ${
                    darkMode
                      ? 'bg-gray-800 hover:bg-gray-750 shadow-2xl'
                      : 'bg-white hover:bg-gray-50 shadow-xl hover:shadow-2xl border border-gray-100'
                  }`}
                >
                  <div className="p-6 sm:p-8">
                    {/* Author Info */}
                    <div className="flex items-center mb-4">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-lg font-bold shadow-lg">
                        {(post.author || post.author_name || 'Unknown Author').charAt(0).toUpperCase()}
                      </div>
                      <div className="ml-4">
                        <p className={`font-semibold transition-colors duration-200 ${
                          darkMode ? 'text-gray-200' : 'text-gray-800'
                        }`}>{post.author || post.author_name || 'Unknown Author'}</p>
                        <p className={`text-sm transition-colors duration-200 ${
                          darkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>{formatDate(post.created_at)}</p>
                      </div>
                    </div>

                    <h3 className={`text-2xl font-bold mb-3 leading-tight transition-colors duration-200 ${
                      darkMode 
                        ? 'text-white hover:text-indigo-300' 
                        : 'text-gray-900 hover:text-indigo-600'
                    }`}>
                      <Link to={`/blog/${post.id}`} className="hover:underline">
                        {post.title}
                      </Link>
                    </h3>
                    
                    <p className={`mb-4 text-lg leading-relaxed transition-colors duration-200 ${
                      darkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>
                      {truncateText(post.content, 200)}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span className="font-medium">{post.district}</span>
                      </div>
                      
                      <div className="flex items-center">
                        <svg className="w-5 h-5 mr-1 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                        </svg>
                        <span className="font-medium">{post.likes_count || 0}</span>
                      </div>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </div>
        )}

        {/* Weather-Aware Blog Recommendations */}
        <WeatherAwareBlogRecommendations />

        {/* Search and Filters */}
        <div className={`mb-6 sm:mb-8 rounded-xl p-4 sm:p-6 transition-colors duration-200 relative ${
          darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white shadow-xl border border-gray-200'
        }`}>
          <form onSubmit={handleSearch} className="mb-3 sm:mb-4">
            <div className="text-center mb-4">
              <h3 className={`text-xl font-semibold transition-colors duration-200 ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>Find Your Perfect Story</h3>
              <p className={`text-sm transition-colors duration-200 ${
                darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>Search by location, experience, or keyword</p>
            </div>
            
            <div className="flex flex-col gap-3 sm:flex-row sm:gap-4">
              <div className="flex-1">
                <label htmlFor="search-posts" className="sr-only">Search posts</label>
                <input
                  type="text"
                  id="search-posts"
                  name="search-posts"
                  placeholder="Search stories, locations, experiences..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full px-4 py-3 border rounded-xl focus:outline-none transition-all duration-200 bg-gray-700 text-white border-gray-600 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 placeholder-gray-400"
                  autoComplete="off"
                />
              </div>
              <button
                type="submit"
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold rounded-xl transition-all duration-200 whitespace-nowrap shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                <svg className="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Search
              </button>
            </div>
          </form>

          <div className="flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:gap-6 sm:items-center">
            <div className="flex-1 min-w-0 sm:min-w-48 sm:max-w-72">
              <label htmlFor="district-filter" className={`block text-sm font-medium mb-2 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>Filter by District</label>
              <select
                id="district-filter"
                name="district-filter"
                value={selectedDistrict}
                onChange={(e) => {
                  setSelectedDistrict(e.target.value);
                  setCurrentPage(1); // Reset to first page when changing district
                }}
                className="w-full px-4 py-3 border rounded-xl focus:outline-none transition-all duration-200 bg-gray-700 text-white border-gray-600 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20"
                autoComplete="address-level2"
              >
                <option value="">All Districts</option>
                {chatbotDistricts.map((district) => (
                  <option key={district} value={district}>{district}</option>
                ))}
              </select>
            </div>

            {/* Share Your Story Button - moved here next to district filter */}
            <div className="sm:self-end">
              <Link
                to="/blog/new"
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 whitespace-nowrap"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
                Share Your Story
              </Link>
            </div>

            {(searchTerm || selectedDistrict) && (
              <div className="sm:self-end">
                <button
                  type="button"
                  onClick={resetFilters}
                  className={`px-4 py-3 rounded-xl font-medium transition-all duration-200 whitespace-nowrap ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-700 text-white border border-gray-500'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-700 border border-gray-300'
                  }`}
                >
                  <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Clear Filters
                </button>
              </div>
            )}
          </div>
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
                      src={`${import.meta.env.VITE_API_URL?.replace(/\/ai\/?$/, '') || 'http://localhost:8000'}${post.images[0].url}`}
                      alt={post.images[0].alt_text || post.title}
                      className="w-full h-full object-cover hover:scale-110 transition-transform duration-500"
                    />
                  </div>
                )}
                
                <div className="p-5 sm:p-7">
                  {/* Author Info */}
                  {(post.author || post.author_name) && (
                    <div className="flex items-center mb-4">
                      {post.author_photo ? (
                        <img 
                          src={post.author_photo} 
                          alt={post.author || post.author_name}
                          className="w-8 h-8 sm:w-10 sm:h-10 rounded-full mr-3 sm:mr-4 object-cover ring-2 ring-indigo-500/20"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                      ) : null}
                      <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full mr-3 sm:mr-4 bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-content text-white text-sm sm:text-base font-bold shadow-lg ${post.author_photo ? 'hidden' : ''}`}>
                        {(post.author || post.author_name || 'Unknown Author').charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className={`text-sm sm:text-base font-semibold transition-colors duration-200 ${
                          darkMode ? 'text-gray-200' : 'text-gray-800'
                        }`}>{post.author || post.author_name || 'Unknown Author'}</p>
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
                        <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-1 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
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
          
          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center mt-8 gap-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  currentPage === 1
                    ? darkMode 
                      ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : darkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                      : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-300'
                }`}
              >
                <svg className="w-4 h-4 mr-1 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Previous
              </button>
              
              <div className="flex gap-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = currentPage - 2 + i;
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`px-3 py-2 rounded-lg font-medium transition-all duration-200 ${
                        currentPage === pageNum
                          ? 'bg-indigo-600 text-white shadow-lg'
                          : darkMode
                            ? 'bg-gray-700 hover:bg-gray-600 text-white'
                            : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-300'
                      }`}
                    >
                      {pageNum}
                    </button>
                  );
                })}
              </div>
              
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  currentPage === totalPages
                    ? darkMode 
                      ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : darkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                      : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-300'
                }`}
              >
                Next
                <svg className="w-4 h-4 ml-1 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
          </>
        )}
        </div> {/* End debug wrapper */}
      </div>
      </div>
    </div>
  );
};

export default BlogList;
