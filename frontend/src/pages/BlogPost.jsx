import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { fetchBlogPost, likeBlogPost, checkLikeStatus } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';

const BlogPost = () => {
  const { darkMode } = useTheme();
  const { id } = useParams();
  const navigate = useNavigate();
  const [post, setPost] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [likeLoading, setLikeLoading] = useState(false);
  const [alreadyLiked, setAlreadyLiked] = useState(false);
  const [likeError, setLikeError] = useState(null);

  useEffect(() => {
    loadPost();
    checkUserLikeStatus();
  }, [id]);

  const loadPost = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const fetchedPost = await fetchBlogPost(id);
      setPost(fetchedPost);
    } catch (err) {
      setError(err.message || 'Failed to load blog post');
      console.error('Failed to load post:', err);
    } finally {
      setLoading(false);
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
      <p key={index} className="mb-4 text-gray-300 leading-relaxed">
        {paragraph}
      </p>
    ));
  };

  if (loading) {
    return (
      <div className={`min-h-screen pt-24 px-4 transition-colors duration-200 ${
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
      <div className={`min-h-screen pt-24 px-4 transition-colors duration-200 ${
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
      <div className="min-h-screen bg-gray-900 text-white pt-24 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-20">
            <h2 className="text-2xl font-bold text-gray-300 mb-4">Post not found</h2>
            <Link
              to="/blog"
              className="inline-flex items-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition-colors duration-200"
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
    <div className="min-h-screen bg-gray-900 text-white pt-24 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Navigation */}
        <div className="mb-8">
          <Link
            to="/blog"
            className="inline-flex items-center text-indigo-400 hover:text-indigo-300 transition-colors duration-200"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Blog
          </Link>
        </div>

        <article className="bg-gray-800 rounded-lg overflow-hidden">
          {/* Featured Image */}
          {post.images && post.images.length > 0 && (
            <div className="aspect-video lg:aspect-[2/1] overflow-hidden">
              <img
                src={`http://localhost:8001${post.images[0].url}`}
                alt={post.images[0].alt_text || post.title}
                className="w-full h-full object-cover"
              />
            </div>
          )}

          <div className="p-8">
            {/* Author Info */}
            {post.author_name && (
              <div className="flex items-center mb-6 pb-6 border-b border-gray-700">
                {post.author_photo ? (
                  <img 
                    src={post.author_photo} 
                    alt={post.author_name}
                    className="w-12 h-12 rounded-full mr-4 object-cover"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'flex';
                    }}
                  />
                ) : null}
                <div className={`w-12 h-12 rounded-full mr-4 bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-lg font-semibold ${post.author_photo ? 'hidden' : ''}`}>
                  {post.author_name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <p className="text-lg font-semibold text-gray-200">{post.author_name}</p>
                  <p className="text-sm text-gray-400">Travel Blogger</p>
                </div>
              </div>
            )}

            {/* Title */}
            <h1 className="text-3xl md:text-4xl font-bold mb-6 text-white leading-tight">
              {post.title}
            </h1>

            {/* Metadata */}
            <div className="flex flex-wrap items-center gap-6 mb-8 text-gray-400 border-b border-gray-700 pb-6">
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
                    : 'text-gray-400 hover:text-red-400'
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
            <div className="prose prose-lg max-w-none">
              {formatContent(post.content)}
            </div>

            {/* Additional Images */}
            {post.images && post.images.length > 1 && (
              <div className="mt-8 border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold mb-4 text-white">More Photos</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {post.images.slice(1).map((image, index) => (
                    <div key={index} className="aspect-video overflow-hidden rounded-lg">
                      <img
                        src={`http://localhost:8001${image.url}`}
                        alt={image.alt_text || `Photo ${index + 2}`}
                        className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </article>

        {/* Call to Action */}
        <div className="mt-12 text-center bg-gray-800 rounded-lg p-8">
          <h2 className="text-2xl font-bold mb-4 text-white">
            Inspired by this story?
          </h2>
          <p className="text-gray-300 mb-6">
            Share your own Istanbul experience and help other travelers discover amazing places.
          </p>
          <Link
            to="/blog/new"
            className="inline-flex items-center px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition-colors duration-200"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Write Your Story
          </Link>
        </div>
      </div>
    </div>
  );
};

export default BlogPost;
