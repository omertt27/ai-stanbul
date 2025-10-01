import React, { createContext, useContext, useState, useCallback } from 'react';

const BlogContext = createContext();

export const useBlog = () => {
  const context = useContext(BlogContext);
  if (!context) {
    throw new Error('useBlog must be used within a BlogProvider');
  }
  return context;
};

export const BlogProvider = ({ children }) => {
  const [posts, setPosts] = useState([]);
  const [postLikesCache, setPostLikesCache] = useState(new Map());

  // Update likes count for a specific post
  const updatePostLikes = useCallback((postId, newLikesCount, isLiked) => {
    console.log('🔄 BlogContext: Updating post likes:', { postId, newLikesCount, isLiked });
    
    // Update the cache
    setPostLikesCache(prev => {
      const newCache = new Map(prev);
      newCache.set(postId, { likesCount: newLikesCount, isLiked });
      return newCache;
    });

    // Update posts array if it exists
    setPosts(prevPosts => 
      prevPosts.map(post => 
        post.id === postId || post.id === parseInt(postId)
          ? { ...post, likes_count: newLikesCount, likes: newLikesCount }
          : post
      )
    );
  }, []);

  // Get cached likes for a post
  const getPostLikes = useCallback((postId) => {
    return postLikesCache.get(postId) || null;
  }, [postLikesCache]);

  // Update the entire posts array
  const updatePosts = useCallback((newPosts) => {
    setPosts(newPosts);
  }, []);

  // Clear cache (useful for refreshing)
  const clearCache = useCallback(() => {
    setPostLikesCache(new Map());
  }, []);

  const value = {
    posts,
    updatePosts,
    updatePostLikes,
    getPostLikes,
    clearCache,
    postLikesCache
  };

  return (
    <BlogContext.Provider value={value}>
      {children}
    </BlogContext.Provider>
  );
};

export default BlogContext;
