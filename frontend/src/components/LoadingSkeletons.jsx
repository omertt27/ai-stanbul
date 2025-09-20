import React from 'react';

/**
 * Base Skeleton Component
 * Provides the fundamental skeleton loading animation
 */
const Skeleton = ({ 
  className = "",
  width = "100%",
  height = "1rem",
  rounded = "rounded",
  animated = true
}) => {
  return (
    <div 
      className={`
        bg-gray-300 
        ${rounded} 
        ${animated ? 'animate-pulse' : ''} 
        ${className}
      `}
      style={{ width, height }}
    />
  );
};

/**
 * Restaurant Card Skeleton
 * Loading skeleton for restaurant search results
 */
const RestaurantSkeleton = ({ count = 3 }) => {
  return (
    <div className="space-y-4">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="bg-white rounded-lg p-6 shadow-md">
          {/* Restaurant Header */}
          <div className="flex items-start space-x-4 mb-4">
            <Skeleton width="60px" height="60px" rounded="rounded-lg" />
            <div className="flex-1 space-y-2">
              <Skeleton width="70%" height="1.5rem" />
              <Skeleton width="40%" height="1rem" />
              <div className="flex space-x-2">
                <Skeleton width="60px" height="1rem" rounded="rounded-full" />
                <Skeleton width="80px" height="1rem" rounded="rounded-full" />
              </div>
            </div>
          </div>
          
          {/* Restaurant Details */}
          <div className="space-y-2">
            <Skeleton width="100%" height="1rem" />
            <Skeleton width="85%" height="1rem" />
            <Skeleton width="60%" height="1rem" />
          </div>
          
          {/* Rating and Price */}
          <div className="flex justify-between items-center mt-4">
            <Skeleton width="120px" height="1.5rem" />
            <Skeleton width="80px" height="1.5rem" />
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Museum Card Skeleton
 * Loading skeleton for museum information
 */
const MuseumSkeleton = ({ count = 2 }) => {
  return (
    <div className="space-y-6">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="bg-white rounded-lg p-6 shadow-md">
          {/* Museum Header */}
          <div className="flex items-center space-x-4 mb-4">
            <Skeleton width="80px" height="80px" rounded="rounded-lg" />
            <div className="flex-1 space-y-3">
              <Skeleton width="60%" height="2rem" />
              <Skeleton width="45%" height="1rem" />
              <Skeleton width="35%" height="1rem" />
            </div>
          </div>
          
          {/* Museum Description */}
          <div className="space-y-2 mb-4">
            <Skeleton width="100%" height="1rem" />
            <Skeleton width="90%" height="1rem" />
            <Skeleton width="75%" height="1rem" />
          </div>
          
          {/* Museum Details */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Skeleton width="60%" height="1rem" />
              <Skeleton width="80%" height="1rem" />
            </div>
            <div className="space-y-2">
              <Skeleton width="70%" height="1rem" />
              <Skeleton width="50%" height="1rem" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Blog Post Skeleton
 * Loading skeleton for blog posts
 */
const BlogPostSkeleton = ({ count = 3, variant = "card" }) => {
  if (variant === "list") {
    return (
      <div className="space-y-4">
        {Array.from({ length: count }, (_, i) => (
          <div key={i} className="flex space-x-4 p-4 bg-white rounded-lg shadow-sm">
            <Skeleton width="120px" height="80px" rounded="rounded-lg" />
            <div className="flex-1 space-y-2">
              <Skeleton width="80%" height="1.5rem" />
              <Skeleton width="100%" height="1rem" />
              <Skeleton width="60%" height="1rem" />
              <div className="flex space-x-4 mt-3">
                <Skeleton width="60px" height="0.875rem" />
                <Skeleton width="80px" height="0.875rem" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="bg-white rounded-lg shadow-md overflow-hidden">
          {/* Blog Image */}
          <Skeleton width="100%" height="200px" rounded="rounded-none" />
          
          {/* Blog Content */}
          <div className="p-6 space-y-4">
            <Skeleton width="85%" height="1.5rem" />
            <div className="space-y-2">
              <Skeleton width="100%" height="1rem" />
              <Skeleton width="90%" height="1rem" />
              <Skeleton width="70%" height="1rem" />
            </div>
            
            {/* Blog Meta */}
            <div className="flex justify-between items-center pt-2">
              <Skeleton width="60px" height="1rem" />
              <Skeleton width="80px" height="1rem" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Chat Message Skeleton
 * Loading skeleton for chat messages
 */
const ChatMessageSkeleton = () => {
  return (
    <div className="flex space-x-3 p-4">
      <Skeleton width="40px" height="40px" rounded="rounded-full" />
      <div className="flex-1 space-y-2">
        <Skeleton width="25%" height="1rem" />
        <div className="space-y-1">
          <Skeleton width="90%" height="1rem" />
          <Skeleton width="75%" height="1rem" />
          <Skeleton width="60%" height="1rem" />
        </div>
      </div>
    </div>
  );
};

/**
 * Search Results Skeleton
 * Generic skeleton for search results
 */
const SearchResultsSkeleton = ({ count = 5 }) => {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="p-4 bg-white rounded-lg shadow-sm">
          <div className="flex items-start space-x-3">
            <Skeleton width="50px" height="50px" rounded="rounded-lg" />
            <div className="flex-1 space-y-2">
              <Skeleton width="70%" height="1.25rem" />
              <Skeleton width="100%" height="1rem" />
              <Skeleton width="45%" height="1rem" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Typing Indicator Skeleton
 * Shows when AI is thinking/typing
 */
const TypingIndicator = ({ className = "" }) => {
  return (
    <div className={`flex items-center space-x-2 p-4 ${className}`}>
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
             style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
             style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" 
             style={{ animationDelay: '300ms' }} />
      </div>
      <span className="text-sm text-gray-500 ml-2">AI is thinking...</span>
    </div>
  );
};

/**
 * Loading Page Skeleton
 * Full page loading state
 */
const PageSkeleton = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <Skeleton width="300px" height="2.5rem" className="mx-auto" />
          <Skeleton width="500px" height="1.5rem" className="mx-auto" />
        </div>
        
        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <RestaurantSkeleton count={3} />
          </div>
          <div className="space-y-4">
            <MuseumSkeleton count={2} />
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * General Loading Skeleton Component
 * Flexible loading skeleton that adapts to different use cases
 */
const LoadingSkeleton = ({ variant = "default", count = 1, className = "" }) => {
  switch (variant) {
    case "message":
      return <ChatMessageSkeleton count={count} className={className} />;
    case "restaurant":
      return <RestaurantSkeleton count={count} className={className} />;
    case "museum":
      return <MuseumSkeleton count={count} className={className} />;
    case "blog":
      return <BlogPostSkeleton count={count} className={className} />;
    case "search":
      return <SearchResultsSkeleton count={count} className={className} />;
    case "typing":
      return <TypingIndicator className={className} />;
    default:
      return <Skeleton className={className} />;
  }
};

export {
  Skeleton,
  RestaurantSkeleton,
  MuseumSkeleton,
  BlogPostSkeleton,
  ChatMessageSkeleton,
  SearchResultsSkeleton,
  TypingIndicator,
  PageSkeleton,
  LoadingSkeleton
};
