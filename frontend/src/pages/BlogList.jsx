import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fetchBlogPosts, likeBlogPost } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import { useBlog } from '../contexts/BlogContext';
import { useTranslation } from 'react-i18next';
import { trackBlogEvent, trackSearch } from '../utils/analytics';
import WeatherAwareBlogRecommendations from '../components/WeatherAwareBlogRecommendations';
import { formatLikesCount, getNumberTextSize } from '../utils/formatNumbers';
import SEOHead from '../components/SEOHead';
import '../App.css';
import '../styles/blog-responsive.css';

const BlogList = () => {
  console.log('🔧 BlogList: Component instance created');
  const { darkMode } = useTheme();
  const { t } = useTranslation();
  const { updatePosts, updatePostLikes, getPostLikes } = useBlog();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalPosts, setTotalPosts] = useState(0);
  const [sortBy, setSortBy] = useState('newest');
  const [likingPosts, setLikingPosts] = useState(new Set()); // Track which posts are being liked

  const postsPerPage = 12; // Show 12 posts per page

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
      selectedDistrict,
      sortBy
    });
    
    setLoading(true);
    setError(null);
    
    try {
      // Try to fetch from API first
      let filteredPosts = [];
      
      try {
        console.log('🔗 BlogList: Attempting to fetch from backend API...');
        const apiResponse = await fetchBlogPosts({
          page: currentPage,
          limit: 100, // Get all posts for client-side filtering
          search: searchTerm,
          district: selectedDistrict,
          sort_by: sortBy
        });
        
        if (apiResponse && apiResponse.posts && Array.isArray(apiResponse.posts)) {
          // Transform API posts to match frontend expectations
          filteredPosts = apiResponse.posts.map(post => ({
            ...post,
            author_name: post.author_name || post.author || 'Unknown Author',
            district: post.district || 'Istanbul (general)',
            images: post.images || []
          }));
          console.log('✅ BlogList: Successfully fetched', filteredPosts.length, 'posts from API');
        } else {
          throw new Error('Invalid API response structure');
        }
      } catch (apiErr) {
        console.error('❌ BlogList: API failed:', apiErr);
        setError('Failed to load blog posts. Please try again later.');
        filteredPosts = []; // Empty array instead of mock data
      }
      
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
      
      // Apply sorting
      if (sortBy === 'newest') {
        filteredPosts.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      } else if (sortBy === 'oldest') {
        filteredPosts.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
      } else if (sortBy === 'most_liked') {
        filteredPosts.sort((a, b) => (b.likes_count || 0) - (a.likes_count || 0));
      } else if (sortBy === 'most_popular') {
        // Sort by view count (simulate with likes as proxy)
        filteredPosts.sort((a, b) => (b.likes_count || 0) - (a.likes_count || 0));
      } else if (sortBy === 'time_spent') {
        // Sort by content length as proxy for reading time
        filteredPosts.sort((a, b) => (b.content?.length || 0) - (a.content?.length || 0));
      }
      
      // Calculate pagination
      const startIndex = (currentPage - 1) * postsPerPage;
      const endIndex = startIndex + postsPerPage;
      const paginatedPosts = filteredPosts.slice(startIndex, endIndex);
      
      setPosts(paginatedPosts);
      setTotalPosts(filteredPosts.length);
      setTotalPages(Math.ceil(filteredPosts.length / postsPerPage));
      updatePosts(filteredPosts); // Update global context with all posts
      
      console.log('✅ BlogList: Posts loaded successfully', {
        total: filteredPosts.length,
        page: currentPage,
        showing: paginatedPosts.length,
        totalPages: Math.ceil(filteredPosts.length / postsPerPage),
        sortedBy: sortBy
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
  }, [currentPage, searchTerm, selectedDistrict, sortBy]); // Added sortBy

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
    setWeatherSort(false);
    setCurrentPage(1); // Reset to first page when clearing filters
  };

  const handleLike = async (postId) => {
    if (likingPosts.has(postId)) return; // Prevent multiple clicks

    setLikingPosts(prev => new Set([...prev, postId]));
    
    try {
      const result = await likeBlogPost(postId);
      
      // Update the post in the current posts array
      setPosts(prevPosts => 
        prevPosts.map(post => 
          post.id === postId 
            ? { ...post, likes_count: result.likes_count }
            : post
        )
      );
      
      // Update global context
      updatePostLikes(postId, result.likes_count, result.isLiked);
      
      console.log('✅ Post liked successfully:', result);
    } catch (error) {
      console.error('❌ Error liking post:', error);
    } finally {
      setLikingPosts(prev => {
        const newSet = new Set(prev);
        newSet.delete(postId);
        return newSet;
      });
    }
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
      <SEOHead
        title="Istanbul Travel Blog"
        description="Discover authentic Istanbul experiences through local insights, travel guides, and insider tips. From hidden gems in Sultanahmet to rooftop views in Galata."
        keywords={['Istanbul blog', 'travel guide', 'Istanbul tips', 'Turkey travel', 'Ottoman history', 'Turkish culture', 'Bosphorus', 'Grand Bazaar']}
        url="/blog"
        type="website"
        structuredData={{
          "@type": "Blog",
          "name": "AI Istanbul Travel Blog",
          "description": "Your guide to authentic Istanbul experiences",
          "blogPost": posts.slice(0, 5).map(post => ({
            "@type": "BlogPosting",
            "headline": post.title,
            "url": `${window.location.origin}/blog/${post.id}`,
            "datePublished": post.created_at,
            "author": {
              "@type": "Person", 
              "name": post.author_name || post.author
            }
          }))
        }}
      />
      <div className="max-w-6xl mx-auto">

      {/* Scrollable Content */}
      <div className="pt-4 pb-8">
        {/* Header Section */}
        <div className="text-center py-8 sm:py-12">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-4 pt-28 transition-colors duration-200 text-white">
            {t('blog.title')}
          </h1>
          <p className="text-lg sm:text-xl mb-6 max-w-3xl mx-auto transition-colors duration-200 text-gray-300">
            {t('blog.subtitle')}
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <span className="text-sm font-medium transition-colors duration-200 text-gray-400">
              {t('blog.storiesShared', { count: totalPosts })} • {t('blog.joinCommunity')}
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
                    
                    <div className="blog-post-meta flex items-center justify-between flex-wrap gap-3">
                      <div className="district-info flex items-center text-sm">
                        <svg className="w-4 h-4 mr-1.5 text-indigo-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span className="font-medium text-gray-600 dark:text-gray-400 truncate">{post.district}</span>
                      </div>
                      
                      <button
                        onClick={() => handleLike(post.id)}
                        disabled={likingPosts.has(post.id)}
                        className={`like-button-container like-button blog-like-button flex items-center gap-1.5 px-2.5 py-1.5 rounded-full transition-all duration-200 hover:scale-105 border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30 flex-shrink-0 ${
                          likingPosts.has(post.id)
                            ? 'opacity-50 cursor-not-allowed'
                            : 'hover:shadow-md active:scale-95'
                        }`}
                        title="Like this post"
                      >
                        {likingPosts.has(post.id) ? (
                          <div className="w-4 h-4 animate-spin">
                            <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </div>
                        ) : (
                          <svg className="like-button-heart w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                          </svg>
                        )}
                        <span className="likes-count font-semibold text-sm text-red-600 dark:text-red-400">
                          {formatLikesCount(post.likes_count || 0)}
                        </span>
                      </button>
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

            {/* Sort By Dropdown */}
            <div className="flex-1 min-w-0 sm:min-w-48 sm:max-w-72">
              <label htmlFor="sort-by" className={`block text-sm font-medium mb-2 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>Sort By</label>
              <select
                id="sort-by"
                name="sort-by"
                value={sortBy}
                onChange={(e) => {
                  setSortBy(e.target.value);
                  setCurrentPage(1); // Reset to first page when changing sort
                }}
                className="w-full px-4 py-3 border rounded-xl focus:outline-none transition-all duration-200 bg-gray-700 text-white border-gray-600 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20"
              >
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="most_liked">Most Liked</option>
                <option value="most_popular">Most Popular</option>
                <option value="time_spent">Longest Read</option>
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
            <div className="blog-post-grid grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 sm:gap-6 mb-4 sm:mb-6">
            {posts.map((post) => (            <article
              key={post.id}
              className={`blog-post-card blog-list-card rounded-xl overflow-hidden transition-all duration-300 hover:scale-105 ${
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

                  {/* Weather Context Display */}
                  {post.weather_context && (
                    <div className={`mb-3 p-3 rounded-lg border transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-blue-900/20 border-blue-500/30 text-blue-300'
                        : 'bg-blue-50 border-blue-200 text-blue-700'
                    }`}>
                      <div className="flex items-center gap-2 text-sm">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                            d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
                        </svg>
                        <span className="font-medium">{post.weather_context.reason}</span>
                      </div>
                      {post.weather_context.weather_info && (
                        <div className="text-xs mt-1 opacity-80">
                          {post.weather_context.weather_info}
                        </div>
                      )}
                    </div>
                  )}
                  
                  <p className={`mb-4 sm:mb-5 text-base sm:text-lg leading-relaxed transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {truncateText(post.content, window.innerWidth < 640 ? 120 : 180)}
                  </p>
                  
                  <div className={`flex items-center justify-between text-sm sm:text-base transition-colors duration-200 ${
                    darkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    <div className="flex items-center">
                      <svg className="w-5 h-5 sm:w-6 sm:h-6 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      <span className="truncate font-medium text-base sm:text-lg">{post.district || 'Istanbul'}</span>
                    </div>
                    
                    <div className="flex items-center gap-3 sm:gap-5 flex-shrink-0">
                      <button
                        onClick={() => handleLike(post.id)}
                        disabled={likingPosts.has(post.id)}
                        className={`flex items-center gap-2 px-3 py-2 min-w-[70px] rounded-lg transition-all duration-200 hover:scale-105 bg-transparent ${
                          likingPosts.has(post.id)
                            ? 'opacity-50 cursor-not-allowed'
                            : 'hover:opacity-80 active:scale-95'
                        }`}
                        title="Like this post"
                      >
                        {likingPosts.has(post.id) ? (
                          <div className="w-5 h-5 sm:w-6 sm:h-6 animate-spin">
                            <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </div>
                        ) : (
                          <svg className="w-5 h-5 sm:w-6 sm:h-6 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                          </svg>
                        )}
                        <span className={`font-medium text-base sm:text-lg ${getNumberTextSize(post.likes_count || 0)}`}>
                          {formatLikesCount(post.likes_count || 0)}
                        </span>
                      </button>
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
