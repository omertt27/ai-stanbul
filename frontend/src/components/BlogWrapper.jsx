import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import BlogList from '../pages/BlogList';

/**
 * BlogWrapper component that ensures BlogList is completely remounted
 * and data is freshly loaded every time user navigates to /blog
 */
const BlogWrapper = () => {
  const location = useLocation();
  const [blogKey, setBlogKey] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    console.log('🔄 BlogWrapper: Navigation detected to', location.pathname);
    
    // Only handle blog routes
    if (location.pathname === '/blog') {
      setIsLoading(true);
      
      // Force complete remount by changing key
      setBlogKey(prev => prev + 1);
      
      // Clear any potential cached data
      if (window.blogCache) {
        delete window.blogCache;
      }
      
      // Scroll to top
      window.scrollTo(0, 0);
      
      // Small delay to ensure clean state
      const timer = setTimeout(() => {
        setIsLoading(false);
        console.log('✅ BlogWrapper: Ready to render BlogList');
      }, 50);
      
      return () => clearTimeout(timer);
    }
  }, [location.pathname, location.search]);

  // Show loading during transition to prevent stale content
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-300">Loading blog...</p>
        </div>
      </div>
    );
  }

  // Render BlogList with unique key to force complete remount
  return <BlogList key={`blog-${blogKey}-${location.pathname}-${location.search}`} />;
};

export default BlogWrapper;
