import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fetchBlogPosts } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';

const BlogList = () => {
  const { darkMode } = useTheme();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const postsPerPage = 6;

  // Istanbul districts that are supported by the AI chatbot
  const chatbotDistricts = [
    'Beyoğlu', 'Sultanahmet', 'Fatih', 'Kadıköy', 'Beşiktaş', 'Şişli', 
    'Üsküdar', 'Bakırköy', 'Galata', 'Taksim', 'Ortaköy', 'Karaköy', 'Eminönü'
  ];

  useEffect(() => {
    loadPosts();
  }, [searchTerm, selectedDistrict, currentPage]);

  const loadPosts = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = {
        page: currentPage,
        limit: postsPerPage,
        ...(searchTerm && { search: searchTerm }),
        ...(selectedDistrict && { district: selectedDistrict })
      };

      const response = await fetchBlogPosts(params);
      setPosts(response.posts);
      setTotalPages(Math.ceil(response.total / postsPerPage));
    } catch (err) {
      setError(err.message || 'Failed to load blog posts');
      console.error('Failed to load posts:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setCurrentPage(1);
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
      <div className={`min-h-screen pt-24 px-4 transition-colors duration-200 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        <div className="max-w-7xl mx-auto">
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
    <div className={`min-h-screen pt-24 px-4 transition-colors duration-200 ${
      darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
    }`}>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Istanbul Travel Blog
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Discover Istanbul through the eyes of fellow travelers. Share your experiences, get inspired, and find hidden gems in this magnificent city.
          </p>
          <Link
            to="/blog/new"
            className="inline-flex items-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition-colors duration-200"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Share Your Story
          </Link>
        </div>

        {/* Search and Filters */}
        <div className={`mb-8 rounded-lg p-6 transition-colors duration-200 ${
          darkMode ? 'bg-gray-800' : 'bg-white shadow-lg border border-gray-200'
        }`}>
          <form onSubmit={handleSearch} className="mb-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <input
                  type="text"
                  placeholder="Search posts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={`w-full px-4 py-2 border rounded-lg focus:outline-none transition-colors duration-200 ${
                    darkMode 
                      ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500' 
                      : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
                  }`}
                />
              </div>
              <button
                type="submit"
                className={`px-6 py-2 rounded-lg transition-colors duration-200 ${
                  darkMode
                    ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                }`}
              >
                Search
              </button>
            </div>
          </form>

          <div className="flex flex-wrap gap-4">
            <select
              value={selectedDistrict}
              onChange={(e) => {
                setSelectedDistrict(e.target.value);
                setCurrentPage(1);
              }}
              className={`px-4 py-2 border rounded-lg focus:outline-none transition-colors duration-200 ${
                darkMode 
                  ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500'
                  : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
              }`}
            >
              <option value="">All Districts</option>
              {chatbotDistricts.map((district) => (
                <option key={district} value={district}>{district}</option>
              ))}
            </select>

            {(searchTerm || selectedDistrict) && (
              <button
                onClick={resetFilters}
                className={`px-4 py-2 rounded-lg transition-colors duration-200 ${
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
        {posts.length === 0 && !loading && !error ? (
          <div className="text-center py-20">
            <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h3 className="text-xl font-semibold text-gray-300 mb-2">No posts found</h3>
            <p className="text-gray-400 mb-4">Be the first to share your Istanbul experience!</p>
            <Link
              to="/blog/new"
              className="inline-flex items-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition-colors duration-200"
            >
              Write a Post
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {posts.map((post) => (            <article
              key={post.id}
              className={`rounded-lg overflow-hidden transition-colors duration-200 ${
                darkMode
                  ? 'bg-gray-800 hover:bg-gray-750'
                  : 'bg-white hover:bg-gray-50 shadow-lg hover:shadow-xl border border-gray-200'
              }`}
            >
                {post.images && post.images.length > 0 && (
                  <div className="aspect-video overflow-hidden">
                    <img
                      src={`http://localhost:8001${post.images[0].url}`}
                      alt={post.images[0].alt_text || post.title}
                      className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
                    />
                  </div>
                )}
                
                <div className="p-6">
                  {/* Author Info */}
                  {post.author_name && (
                    <div className="flex items-center mb-3">
                      {post.author_photo ? (
                        <img 
                          src={post.author_photo} 
                          alt={post.author_name}
                          className="w-8 h-8 rounded-full mr-3 object-cover"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                      ) : null}
                      <div className={`w-8 h-8 rounded-full mr-3 bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-sm font-semibold ${post.author_photo ? 'hidden' : ''}`}>
                        {post.author_name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className={`text-sm font-medium transition-colors duration-200 ${
                          darkMode ? 'text-gray-300' : 'text-gray-700'
                        }`}>{post.author_name}</p>
                      </div>
                    </div>
                  )}

                  <h2 className={`text-xl font-bold mb-2 transition-colors duration-200 ${
                    darkMode 
                      ? 'hover:text-indigo-300' 
                      : 'hover:text-indigo-600'
                  }`}>
                    <Link to={`/blog/${post.id}`}>
                      {post.title}
                    </Link>
                  </h2>
                  
                  <p className={`mb-4 transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {truncateText(post.content)}
                  </p>
                  
                  <div className={`flex items-center justify-between text-sm transition-colors duration-200 ${
                    darkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    <div className="flex items-center">
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      {post.district || 'Istanbul'}
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="flex items-center">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        {post.likes_count || 0}
                      </div>
                      <div className="flex items-center">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        {formatDate(post.created_at)}
                      </div>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center items-center gap-2 pb-8">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className={`px-4 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
                darkMode
                  ? 'bg-gray-700 text-white hover:bg-gray-600'
                  : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
              }`}
            >
              Previous
            </button>
            
            <span className={`px-4 py-2 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Page {currentPage} of {totalPages}
            </span>
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className={`px-4 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
                darkMode
                  ? 'bg-gray-700 text-white hover:bg-gray-600'
                  : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
              }`}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default BlogList;
