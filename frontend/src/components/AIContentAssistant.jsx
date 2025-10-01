import React, { useState, useEffect } from 'react';
import { 
  generateAutoTags, 
  analyzeContentQuality, 
  getDistrictContentSuggestions,
  getSEOSuggestions 
} from '../utils/contentAI';

const AIContentAssistant = ({ 
  title, 
  content, 
  district, 
  currentTags = [],
  onTagSuggestion,
  onContentSuggestion,
  className = '' 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('tags');
  const [qualityAnalysis, setQualityAnalysis] = useState(null);
  const [seoSuggestions, setSeoSuggestions] = useState(null);
  const [suggestedTags, setSuggestedTags] = useState([]);
  const [districtSuggestions, setDistrictSuggestions] = useState([]);

  // Update analysis whenever content changes
  useEffect(() => {
    if (title || content) {
      const analysis = analyzeContentQuality(title, content);
      const seo = getSEOSuggestions(title, content, currentTags);
      const tags = generateAutoTags(title, content, district);
      const suggestions = getDistrictContentSuggestions(district);

      setQualityAnalysis(analysis);
      setSeoSuggestions(seo);
      setSuggestedTags(tags);
      setDistrictSuggestions(suggestions);
    }
  }, [title, content, district, currentTags]);

  const handleTagClick = (tag) => {
    if (onTagSuggestion && !currentTags.includes(tag)) {
      onTagSuggestion(tag);
    }
  };

  const getQualityColor = (quality) => {
    switch (quality) {
      case 'Excellent': return 'text-green-400';
      case 'Good': return 'text-blue-400';
      case 'Fair': return 'text-yellow-400';
      default: return 'text-red-400';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 65) return 'bg-blue-500';
    if (score >= 50) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (!title && !content) {
    return (
      <div className={`ai-assistant-placeholder ${className}`}>
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center gap-2 text-gray-400">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            <span className="text-sm">AI Assistant will help optimize your content once you start writing</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`ai-content-assistant ${className}`}>
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        {/* Header */}
        <div className="bg-gray-900 px-4 py-3 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <h3 className="text-white font-medium">AI Content Assistant</h3>
              {qualityAnalysis && (
                <span className={`text-sm ${getQualityColor(qualityAnalysis.quality)}`}>
                  {qualityAnalysis.quality} ({qualityAnalysis.score}/100)
                </span>
              )}
            </div>
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <svg 
                className={`w-5 h-5 transform transition-transform ${isOpen ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>

          {/* Quick Stats */}
          {qualityAnalysis && (
            <div className="mt-2 flex items-center gap-4 text-xs text-gray-400">
              <span>{qualityAnalysis.wordCount} words</span>
              <span>{qualityAnalysis.paragraphs} paragraphs</span>
              {seoSuggestions && (
                <span>{seoSuggestions.estimatedReadTime} min read</span>
              )}
            </div>
          )}
        </div>

        {/* Content */}
        {isOpen && (
          <div className="p-4">
            {/* Tabs */}
            <div className="flex space-x-1 mb-4">
              {['tags', 'quality', 'seo', 'suggestions'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1 text-sm rounded-md transition-colors ${
                    activeTab === tab
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>

            {/* Tag Suggestions */}
            {activeTab === 'tags' && (
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-white">Suggested Tags</h4>
                <div className="flex flex-wrap gap-2">
                  {suggestedTags.map((tag, index) => (
                    <button
                      key={index}
                      onClick={() => handleTagClick(tag)}
                      className={`px-2 py-1 text-xs rounded-full transition-colors ${
                        currentTags.includes(tag)
                          ? 'bg-green-600 text-white cursor-default'
                          : 'bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white'
                      }`}
                      disabled={currentTags.includes(tag)}
                    >
                      {currentTags.includes(tag) && (
                        <svg className="w-3 h-3 inline mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                      {tag}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Quality Analysis */}
            {activeTab === 'quality' && qualityAnalysis && (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-white">Content Quality</span>
                      <span className={`text-sm ${getQualityColor(qualityAnalysis.quality)}`}>
                        {qualityAnalysis.score}/100
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${getScoreColor(qualityAnalysis.score)}`}
                        style={{ width: `${qualityAnalysis.score}%` }}
                      />
                    </div>
                  </div>
                </div>

                {qualityAnalysis.feedback.length > 0 && (
                  <div>
                    <h5 className="text-sm font-medium text-green-400 mb-2">✓ Strengths</h5>
                    <ul className="text-sm text-gray-300 space-y-1">
                      {qualityAnalysis.feedback.map((item, index) => (
                        <li key={index}>• {item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {qualityAnalysis.suggestions.length > 0 && (
                  <div>
                    <h5 className="text-sm font-medium text-yellow-400 mb-2">💡 Suggestions</h5>
                    <ul className="text-sm text-gray-300 space-y-1">
                      {qualityAnalysis.suggestions.map((item, index) => (
                        <li key={index}>• {item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* SEO Analysis */}
            {activeTab === 'seo' && seoSuggestions && (
              <div className="space-y-4">
                <div>
                  <h5 className="text-sm font-medium text-blue-400 mb-2">📈 SEO Analysis</h5>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Title Length:</span>
                      <span className={`ml-2 ${seoSuggestions.titleLength <= 60 ? 'text-green-400' : 'text-red-400'}`}>
                        {seoSuggestions.titleLength}/60
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Read Time:</span>
                      <span className="ml-2 text-white">{seoSuggestions.estimatedReadTime} min</span>
                    </div>
                  </div>
                </div>

                {seoSuggestions.suggestions.length > 0 && (
                  <div>
                    <h5 className="text-sm font-medium text-green-400 mb-2">💡 SEO Suggestions</h5>
                    <ul className="text-sm text-gray-300 space-y-1">
                      {seoSuggestions.suggestions.map((item, index) => (
                        <li key={index}>• {item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {seoSuggestions.warnings.length > 0 && (
                  <div>
                    <h5 className="text-sm font-medium text-red-400 mb-2">⚠️ Warnings</h5>
                    <ul className="text-sm text-gray-300 space-y-1">
                      {seoSuggestions.warnings.map((item, index) => (
                        <li key={index}>• {item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* District Suggestions */}
            {activeTab === 'suggestions' && (
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-white">
                  Content Suggestions {district && `for ${district}`}
                </h4>
                <ul className="text-sm text-gray-300 space-y-2">
                  {districtSuggestions.map((suggestion, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-blue-400 mt-1">•</span>
                      <span>{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AIContentAssistant;
