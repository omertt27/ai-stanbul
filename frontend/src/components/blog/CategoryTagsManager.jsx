/**
 * Category & Tags Manager Component
 * Features:
 * - Multiple category selection
 * - Hierarchical categories
 * - Custom tag creation
 * - AI-powered tag suggestions
 * - Popular tags display
 */
import React, { useState, useMemo, useCallback } from 'react';

// Istanbul-specific categories
const BLOG_CATEGORIES = [
  {
    id: 'neighborhoods',
    name: 'Neighborhoods',
    icon: '🏘️',
    subcategories: [
      'Sultanahmet', 'Beyoğlu', 'Kadıköy', 'Beşiktaş', 'Fatih',
      'Üsküdar', 'Şişli', 'Bakırköy', 'Galata', 'Taksim',
      'Ortaköy', 'Karaköy', 'Eminönü', 'Balat', 'Fener'
    ]
  },
  {
    id: 'attractions',
    name: 'Attractions',
    icon: '🏛️',
    subcategories: [
      'Museums', 'Mosques', 'Palaces', 'Bazaars', 'Parks',
      'Historical Sites', 'Viewpoints', 'Towers', 'Bridges'
    ]
  },
  {
    id: 'food-drink',
    name: 'Food & Drink',
    icon: '🍽️',
    subcategories: [
      'Restaurants', 'Street Food', 'Cafes', 'Rooftop Bars',
      'Traditional Turkish', 'Seafood', 'Breakfast Spots', 'Desserts'
    ]
  },
  {
    id: 'activities',
    name: 'Activities',
    icon: '🎯',
    subcategories: [
      'Bosphorus Cruises', 'Walking Tours', 'Shopping', 'Nightlife',
      'Photography', 'Cooking Classes', 'Turkish Bath', 'Day Trips'
    ]
  },
  {
    id: 'practical',
    name: 'Practical Info',
    icon: '📍',
    subcategories: [
      'Transportation', 'Accommodation', 'Money & Budgeting',
      'Safety Tips', 'Local Customs', 'Weather', 'Language'
    ]
  },
  {
    id: 'culture',
    name: 'Culture & History',
    icon: '📚',
    subcategories: [
      'Ottoman History', 'Byzantine Era', 'Art & Architecture',
      'Traditions', 'Festivals', 'Music', 'Religion'
    ]
  }
];

// Popular tags
const POPULAR_TAGS = [
  'istanbul', 'travel', 'turkey', 'hidden gems', 'budget travel',
  'food tour', 'photography', 'bosphorus', 'history', 'architecture',
  'solo travel', 'family friendly', 'romantic', 'adventure', 'local tips'
];

const CategoryTagsManager = ({
  selectedCategory,
  selectedSubcategories = [],
  tags = [],
  onCategoryChange,
  onSubcategoriesChange,
  onTagsChange,
  onAutoGenerateTags,
  className = ''
}) => {
  const [tagInput, setTagInput] = useState('');
  const [showAllCategories, setShowAllCategories] = useState(false);

  // Handle category selection
  const handleCategoryClick = useCallback((categoryId) => {
    if (selectedCategory === categoryId) {
      onCategoryChange(null);
      onSubcategoriesChange([]);
    } else {
      onCategoryChange(categoryId);
    }
  }, [selectedCategory, onCategoryChange, onSubcategoriesChange]);

  // Handle subcategory toggle
  const handleSubcategoryToggle = useCallback((subcategory) => {
    if (selectedSubcategories.includes(subcategory)) {
      onSubcategoriesChange(selectedSubcategories.filter(s => s !== subcategory));
    } else {
      onSubcategoriesChange([...selectedSubcategories, subcategory]);
    }
  }, [selectedSubcategories, onSubcategoriesChange]);

  // Handle tag addition
  const handleAddTag = useCallback((tag) => {
    const cleanTag = tag.toLowerCase().trim();
    if (cleanTag && !tags.includes(cleanTag)) {
      onTagsChange([...tags, cleanTag]);
    }
    setTagInput('');
  }, [tags, onTagsChange]);

  // Handle tag removal
  const handleRemoveTag = useCallback((tagToRemove) => {
    onTagsChange(tags.filter(t => t !== tagToRemove));
  }, [tags, onTagsChange]);

  // Handle tag input keypress
  const handleTagKeyPress = useCallback((e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      handleAddTag(tagInput);
    }
  }, [tagInput, handleAddTag]);

  // Get current category object
  const currentCategory = useMemo(() => {
    return BLOG_CATEGORIES.find(c => c.id === selectedCategory);
  }, [selectedCategory]);

  // Filter suggested tags (exclude already selected)
  const suggestedTags = useMemo(() => {
    return POPULAR_TAGS.filter(tag => !tags.includes(tag));
  }, [tags]);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Category Selection */}
      <div className="bg-gray-800 rounded-xl border-2 border-gray-700 p-5">
        <label className="block text-sm font-medium text-gray-200 mb-3">
          Category
        </label>

        {/* Main Categories Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
          {BLOG_CATEGORIES.slice(0, showAllCategories ? undefined : 6).map((category) => (
            <button
              key={category.id}
              type="button"
              onClick={() => handleCategoryClick(category.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all text-left ${
                selectedCategory === category.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <span className="text-lg">{category.icon}</span>
              <span className="text-sm font-medium truncate">{category.name}</span>
            </button>
          ))}
        </div>

        {BLOG_CATEGORIES.length > 6 && (
          <button
            type="button"
            onClick={() => setShowAllCategories(!showAllCategories)}
            className="text-sm text-blue-400 hover:text-blue-300"
          >
            {showAllCategories ? 'Show less' : `Show ${BLOG_CATEGORIES.length - 6} more`}
          </button>
        )}

        {/* Subcategories (when category selected) */}
        {currentCategory && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              {currentCategory.name} Subcategories
            </label>
            <div className="flex flex-wrap gap-2">
              {currentCategory.subcategories.map((sub) => (
                <button
                  key={sub}
                  type="button"
                  onClick={() => handleSubcategoryToggle(sub)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                    selectedSubcategories.includes(sub)
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {sub}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Tags Section */}
      <div className="bg-gray-800 rounded-xl border-2 border-gray-700 p-5">
        <div className="flex items-center justify-between mb-3">
          <label className="text-sm font-medium text-gray-200">
            Tags
          </label>
          {onAutoGenerateTags && (
            <button
              type="button"
              onClick={onAutoGenerateTags}
              className="flex items-center gap-1 px-2 py-1 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              AI Generate
            </button>
          )}
        </div>

        {/* Current Tags */}
        <div className="flex flex-wrap gap-2 mb-3 min-h-[32px]">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 px-2 py-1 bg-blue-600/20 text-blue-400 text-xs rounded-full border border-blue-600/30"
            >
              #{tag}
              <button
                type="button"
                onClick={() => handleRemoveTag(tag)}
                className="hover:text-blue-300 ml-1"
              >
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </span>
          ))}
          {tags.length === 0 && (
            <span className="text-xs text-gray-500">No tags added yet</span>
          )}
        </div>

        {/* Tag Input */}
        <input
          type="text"
          value={tagInput}
          onChange={(e) => setTagInput(e.target.value)}
          onKeyDown={handleTagKeyPress}
          placeholder="Add a tag and press Enter..."
          className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-3"
        />

        {/* Popular Tags */}
        {suggestedTags.length > 0 && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Popular tags:</p>
            <div className="flex flex-wrap gap-1">
              {suggestedTags.slice(0, 10).map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => handleAddTag(tag)}
                  className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded hover:bg-gray-600 transition-colors"
                >
                  +{tag}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CategoryTagsManager;
