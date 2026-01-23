/**
 * SEO & Metadata Editor Component
 * Features:
 * - Meta description with character count
 * - Custom URL slug editor
 * - Social media preview (OpenGraph)
 * - SEO score indicator
 * - Focus keyword tracking
 */
import React, { useState, useEffect, useMemo } from 'react';

const SEOMetadataEditor = ({
  title,
  content,
  metaDescription,
  slug,
  focusKeyword,
  featuredImage,
  onMetaDescriptionChange,
  onSlugChange,
  onFocusKeywordChange,
  siteUrl = 'https://aistanbul.net'
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [slugTouched, setSlugTouched] = useState(false);

  // Auto-generate slug from title
  useEffect(() => {
    if (!slugTouched && title) {
      const generatedSlug = title
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .substring(0, 60);
      onSlugChange(generatedSlug);
    }
  }, [title, slugTouched, onSlugChange]);

  // Auto-generate meta description from content
  useEffect(() => {
    if (!metaDescription && content) {
      const plainText = content.replace(/<[^>]*>/g, '').trim();
      const autoDescription = plainText.substring(0, 155);
      onMetaDescriptionChange(autoDescription + (plainText.length > 155 ? '...' : ''));
    }
  }, [content, metaDescription, onMetaDescriptionChange]);

  // Calculate SEO score
  const seoScore = useMemo(() => {
    let score = 0;
    const checks = [];

    // Title checks
    if (title) {
      if (title.length >= 30 && title.length <= 60) {
        score += 15;
        checks.push({ name: 'Title length', status: 'good', message: 'Perfect length (30-60 chars)' });
      } else if (title.length > 0) {
        score += 5;
        checks.push({ name: 'Title length', status: 'warning', message: title.length < 30 ? 'Too short' : 'Too long' });
      } else {
        checks.push({ name: 'Title length', status: 'error', message: 'Missing title' });
      }

      if (focusKeyword && title.toLowerCase().includes(focusKeyword.toLowerCase())) {
        score += 15;
        checks.push({ name: 'Keyword in title', status: 'good', message: 'Focus keyword found in title' });
      } else if (focusKeyword) {
        checks.push({ name: 'Keyword in title', status: 'warning', message: 'Focus keyword not in title' });
      }
    }

    // Meta description checks
    if (metaDescription) {
      if (metaDescription.length >= 120 && metaDescription.length <= 160) {
        score += 15;
        checks.push({ name: 'Meta description', status: 'good', message: 'Perfect length (120-160 chars)' });
      } else if (metaDescription.length > 0) {
        score += 5;
        checks.push({ name: 'Meta description', status: 'warning', message: metaDescription.length < 120 ? 'Too short' : 'Too long' });
      }

      if (focusKeyword && metaDescription.toLowerCase().includes(focusKeyword.toLowerCase())) {
        score += 10;
        checks.push({ name: 'Keyword in meta', status: 'good', message: 'Focus keyword in description' });
      } else if (focusKeyword) {
        checks.push({ name: 'Keyword in meta', status: 'warning', message: 'Focus keyword not in description' });
      }
    } else {
      checks.push({ name: 'Meta description', status: 'error', message: 'Missing meta description' });
    }

    // Content checks
    if (content) {
      const plainText = content.replace(/<[^>]*>/g, '');
      const wordCount = plainText.split(/\s+/).filter(w => w.length > 0).length;
      
      if (wordCount >= 300) {
        score += 20;
        checks.push({ name: 'Content length', status: 'good', message: `${wordCount} words (great!)` });
      } else if (wordCount >= 100) {
        score += 10;
        checks.push({ name: 'Content length', status: 'warning', message: `${wordCount} words (aim for 300+)` });
      } else {
        checks.push({ name: 'Content length', status: 'error', message: `${wordCount} words (too short)` });
      }

      if (focusKeyword) {
        const keywordCount = (plainText.toLowerCase().match(new RegExp(focusKeyword.toLowerCase(), 'g')) || []).length;
        const keywordDensity = ((keywordCount / wordCount) * 100).toFixed(1);
        
        if (keywordDensity >= 0.5 && keywordDensity <= 2.5) {
          score += 15;
          checks.push({ name: 'Keyword density', status: 'good', message: `${keywordDensity}% (ideal)` });
        } else if (keywordCount > 0) {
          score += 5;
          checks.push({ name: 'Keyword density', status: 'warning', message: `${keywordDensity}% (${keywordDensity < 0.5 ? 'too low' : 'too high'})` });
        } else {
          checks.push({ name: 'Keyword density', status: 'error', message: 'Keyword not found in content' });
        }
      }
    }

    // Slug check
    if (slug) {
      if (slug.length <= 60 && !slug.includes('--')) {
        score += 10;
        checks.push({ name: 'URL slug', status: 'good', message: 'Clean and concise' });
      } else {
        score += 5;
        checks.push({ name: 'URL slug', status: 'warning', message: 'Could be improved' });
      }
    }

    return { score: Math.min(score, 100), checks };
  }, [title, content, metaDescription, slug, focusKeyword]);

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getScoreBg = (score) => {
    if (score >= 80) return 'bg-green-600';
    if (score >= 50) return 'bg-yellow-600';
    return 'bg-red-600';
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'good':
        return <span className="text-green-400">✓</span>;
      case 'warning':
        return <span className="text-yellow-400">!</span>;
      case 'error':
        return <span className="text-red-400">✗</span>;
      default:
        return null;
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl border-2 border-gray-700 overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <span className="font-medium text-white">SEO & Social Media</span>
        </div>
        <div className="flex items-center gap-3">
          {/* Score Badge */}
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${getScoreBg(seoScore.score)}/20`}>
            <div className={`w-2 h-2 rounded-full ${getScoreBg(seoScore.score)}`} />
            <span className={`text-sm font-medium ${getScoreColor(seoScore.score)}`}>
              {seoScore.score}/100
            </span>
          </div>
          <svg 
            className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-5 pb-5 space-y-5 border-t border-gray-700">
          {/* Focus Keyword */}
          <div className="pt-4">
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Focus Keyword
            </label>
            <input
              type="text"
              value={focusKeyword || ''}
              onChange={(e) => onFocusKeywordChange(e.target.value)}
              placeholder="e.g., istanbul travel guide"
              className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-400 mt-1">
              The main keyword you want this page to rank for
            </p>
          </div>

          {/* URL Slug */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              URL Slug
            </label>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">{siteUrl}/blog/</span>
              <input
                type="text"
                value={slug || ''}
                onChange={(e) => {
                  setSlugTouched(true);
                  onSlugChange(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'));
                }}
                placeholder="your-post-url"
                className="flex-1 px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Meta Description */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Meta Description
            </label>
            <textarea
              value={metaDescription || ''}
              onChange={(e) => onMetaDescriptionChange(e.target.value)}
              placeholder="A compelling description of your post for search engines..."
              rows={3}
              maxLength={160}
              className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            />
            <div className="flex justify-between mt-1">
              <p className="text-xs text-gray-400">
                Recommended: 120-160 characters
              </p>
              <p className={`text-xs ${
                (metaDescription?.length || 0) > 160 ? 'text-red-400' : 
                (metaDescription?.length || 0) >= 120 ? 'text-green-400' : 'text-gray-400'
              }`}>
                {metaDescription?.length || 0}/160
              </p>
            </div>
          </div>

          {/* Google Preview */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Google Preview
            </label>
            <div className="bg-white rounded-lg p-4">
              <div className="text-xs text-green-700 mb-1">
                {siteUrl}/blog/{slug || 'your-post-url'}
              </div>
              <div className="text-blue-800 text-lg hover:underline cursor-pointer line-clamp-1">
                {title || 'Your Post Title'}
              </div>
              <div className="text-sm text-gray-600 line-clamp-2">
                {metaDescription || 'Your meta description will appear here. Make it compelling to increase click-through rates from search results.'}
              </div>
            </div>
          </div>

          {/* Social Media Preview */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Social Media Preview
            </label>
            <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
              {featuredImage ? (
                <div className="aspect-video bg-gray-700">
                  <img src={featuredImage} alt="Preview" className="w-full h-full object-cover" />
                </div>
              ) : (
                <div className="aspect-video bg-gray-700 flex items-center justify-center">
                  <span className="text-gray-500 text-sm">No featured image</span>
                </div>
              )}
              <div className="p-3">
                <div className="text-xs text-gray-400 uppercase mb-1">aistanbul.net</div>
                <div className="text-white font-medium line-clamp-2">
                  {title || 'Your Post Title'}
                </div>
                <div className="text-sm text-gray-400 line-clamp-2 mt-1">
                  {metaDescription || 'Your meta description...'}
                </div>
              </div>
            </div>
          </div>

          {/* SEO Checklist */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              SEO Checklist
            </label>
            <div className="space-y-2">
              {seoScore.checks.map((check, index) => (
                <div 
                  key={index}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg ${
                    check.status === 'good' ? 'bg-green-900/20' :
                    check.status === 'warning' ? 'bg-yellow-900/20' :
                    'bg-red-900/20'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {getStatusIcon(check.status)}
                    <span className="text-sm text-gray-300">{check.name}</span>
                  </div>
                  <span className={`text-xs ${
                    check.status === 'good' ? 'text-green-400' :
                    check.status === 'warning' ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {check.message}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SEOMetadataEditor;
