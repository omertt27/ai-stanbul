/**
 * Number formatting utilities for better UI display
 */

/**
 * Format large numbers for better readability
 * @param {number} num - The number to format
 * @param {object} options - Formatting options
 * @returns {string} Formatted number string
 */
export const formatCount = (num, options = {}) => {
  const {
    useCompactNotation = true,
    showPlusForLarge = false,
    threshold = 1000
  } = options;

  if (!num || num === 0) return '0';
  
  const numValue = parseInt(num);
  
  // For numbers less than 100, show exact count
  if (numValue < 100) {
    return numValue.toString();
  }
  
  // For numbers 100-999, show with + suffix if enabled
  if (numValue < threshold) {
    return showPlusForLarge ? `${numValue}+` : numValue.toString();
  }
  
  // For larger numbers, use compact notation
  if (useCompactNotation) {
    if (numValue >= 1000000) {
      return `${(numValue / 1000000).toFixed(1)}M`;
    }
    if (numValue >= 1000) {
      return `${(numValue / 1000).toFixed(1)}K`;
    }
  }
  
  return numValue.toString();
};

/**
 * Format likes count specifically
 * @param {number} likesCount - Number of likes
 * @returns {string} Formatted likes count
 */
export const formatLikesCount = (likesCount) => {
  return formatCount(likesCount, {
    useCompactNotation: true,
    showPlusForLarge: false,
    threshold: 1000
  });
};

/**
 * Format view count specifically
 * @param {number} viewCount - Number of views
 * @returns {string} Formatted view count
 */
export const formatViewCount = (viewCount) => {
  return formatCount(viewCount, {
    useCompactNotation: true,
    showPlusForLarge: false,
    threshold: 1000
  });
};

/**
 * Get appropriate text size class based on number length
 * @param {number} num - The number to check
 * @returns {string} Tailwind CSS class for text size
 */
export const getNumberTextSize = (num) => {
  const str = formatCount(num);
  
  if (str.length <= 2) return 'text-base';
  if (str.length <= 3) return 'text-sm';
  return 'text-xs';
};

export default {
  formatCount,
  formatLikesCount,
  formatViewCount,
  getNumberTextSize
};
