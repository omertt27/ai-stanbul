import React from 'react';

/**
 * Loading skeleton component for blog posts
 * Provides visual feedback while posts are loading
 */
const BlogPostSkeleton = ({ darkMode = false, count = 6 }) => {
  return (
    <>
      {[...Array(count)].map((_, index) => (
        <div
          key={`skeleton-${index}`}
          className={`rounded-2xl overflow-hidden transition-all duration-300 ${
            darkMode
              ? 'bg-gray-800 shadow-xl'
              : 'bg-white shadow-xl border border-gray-100'
          }`}
        >
          <div className="p-6">
            {/* Author skeleton */}
            <div className="flex items-center mb-4">
              <div className={`w-10 h-10 rounded-full animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
              <div className="ml-3 flex-1">
                <div className={`h-4 w-24 rounded animate-pulse ${
                  darkMode ? 'bg-gray-700' : 'bg-gray-200'
                }`}></div>
              </div>
            </div>

            {/* Title skeleton */}
            <div className="mb-3 space-y-2">
              <div className={`h-6 w-full rounded animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
              <div className={`h-6 w-3/4 rounded animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
            </div>

            {/* Content skeleton */}
            <div className="mb-4 space-y-2">
              <div className={`h-4 w-full rounded animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
              <div className={`h-4 w-full rounded animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
              <div className={`h-4 w-2/3 rounded animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
            </div>

            {/* Meta skeleton */}
            <div className="flex items-center justify-between">
              <div className={`h-8 w-24 rounded-full animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
              <div className={`h-8 w-16 rounded-full animate-pulse ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}></div>
            </div>
          </div>
        </div>
      ))}
    </>
  );
};

export default BlogPostSkeleton;
